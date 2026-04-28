"""
repo_kg_builder.py

Builds structural Knowledge Graphs (KGs) from Python repository source code.

Given a GitHub repo and a commit SHA, this module:
  1. Clones the repo as a bare git mirror (cached locally)
  2. Extracts the source tree at that commit via git archive
  3. Parses every .py file with Python's ast module in parallel
  4. Emits nodes (file, class, function, method, test_function, import) and
     edges (contains, imports, calls, inherits, tests, uses, overrides,
     depends_on, module_depends_on) into a JSON KG

Node metadata includes: signatures, type annotations, default values,
decorators, docstrings, raised/caught exceptions, branch counts,
assert patterns (for test functions), class attributes, module constants,
and __all__ exports.

Output format:
    {
        "nodes": [{"id": ..., "type": ..., "label": ..., "metadata": {...}}, ...],
        "edges": [{"source": ..., "target": ..., "relation": ..., "metadata": {...}}, ...],
        "metadata": {"repo": ..., "base_commit": ..., "file_count": ..., "parse_mode": "source"}
    }

Usage:
    builder = RepoKGBuilder()
    kg = builder.build("psf/requests", "<commit_sha>")
    builder.save("psf/requests", kg)

Module layout (post-split):
  - ast_helpers.py   pure AST-in/data-out utilities (_get_signature,
                     _build_func_metadata, etc.) Re-exported here for
                     back-compat with code that imports them from
                     repo_kg_builder directly.
  - repo_manager.py  git clone / archive extraction (RepoManager).
  - repo_kg_builder  this module — KGNode/KGEdge data types, _parse_file
                     (per-file AST → nodes+edges), RepoASTParser (parallel
                     driver + second-pass edge resolution), RepoKGBuilder
                     (top-level entry point).
"""

import ast
import json
import tempfile
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Set, Optional, Tuple, Union

# Re-exported from ast_helpers so test code and external callers can still
# `from repo_kg_builder import _get_signature` etc. without knowing about
# the split. These are imported as names (not *) so linters resolve them.
from ast_helpers import (
    _make_id,
    _is_test_file,
    _safe_unparse,
    _extract_callee_name,
    _extract_call_receiver,
    _collect_local_types,
    _get_docstring,
    _get_decorators,
    _get_signature,
    _get_exceptions,
    _count_branches,
    _get_assert_patterns,
    _get_return_types,
    _get_base_names,
    _get_class_attributes,
    _get_instantiated_classes_in_class,
    _get_attribute_accesses,
    _get_used_imports,
    _get_instantiated_classes,
    _get_test_target,
    _build_func_metadata,
    _collect_file_level_info,
)
from repo_manager import RepoManager


# Directories to skip during repo traversal — typically non-source content
SKIP_DIRS = {'docs', 'doc', 'examples', 'example', 'vendor', 'migrations', '.git'}

# Files over this line count are skipped to avoid pathological parse times (e.g. generated files)
MAX_FILE_LINES = 5000


@dataclass
class KGNode:
    """A node in the knowledge graph representing a code entity.

    Attributes:
        id: Deterministic 8-char MD5 hash of the entity's qualified name.
            Identical entities across issues/commits map to the same ID.
        type: One of 'file', 'test_file', 'class', 'function', 'method',
              'test_function', 'import'.
        label: Human-readable short name (e.g. filename, class name, function name).
        metadata: Entity-specific data. See _build_func_metadata and _parse_file
                  for the full set of keys per node type.
    """
    id: str
    type: str
    label: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class KGEdge:
    """A directed edge in the knowledge graph representing a relationship.

    Attributes:
        source: ID of the source node.
        target: ID of the target node.
        relation: Relationship type. One of:
            - 'contains': file→class, file→function, class→method
            - 'imports':  file→import module
            - 'calls':    function/method→function/method (best-effort static analysis)
            - 'inherits': class→parent class
        metadata: Edge-specific data, e.g. confidence ('exact'/'ambiguous') for
                  calls and inherits edges resolved in the second pass.
    """
    source: str
    target: str
    relation: str
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-file parsing (runs in worker processes)
# ---------------------------------------------------------------------------

def _parse_file(args: Tuple[str, str, str]) -> Optional[Dict]:
    """Parse a single .py file and return its nodes and edges.

    This function runs inside worker processes spawned by ProcessPoolExecutor,
    so it must be a module-level function (not a method or closure) to be
    picklable. All helpers it calls must also be importable from the module.

    Args:
        args: Tuple of (repo, rel_path, abs_path).
            repo: e.g. 'psf/requests'
            rel_path: path relative to repo root, e.g. 'requests/sessions.py'
            abs_path: absolute path on disk in the temp extract directory

    Returns:
        Dict with 'nodes' and 'edges' lists, or None if the file should be
        skipped (unreadable, over line limit, or has a SyntaxError).

    Node types emitted: file/test_file, import, class, function/method/test_function
    Edge types emitted: contains, imports, calls (unresolved — resolved in second pass)
    """
    repo, rel_path, abs_path = args
    try:
        source = Path(abs_path).read_text(encoding='utf-8', errors='replace')
    except OSError:
        return None

    if source.count('\n') > MAX_FILE_LINES:
        return None

    try:
        tree = ast.parse(source, filename=abs_path)
    except SyntaxError:
        return None

    nodes = []
    edges = []
    # Deduplicate call edges within this file at creation time to avoid
    # accumulating one edge per call-site for frequently called functions
    seen_call_targets: Set[Tuple[str, str]] = set()

    file_id = _make_id(f"file_{repo}_{rel_path}")
    file_type = 'test_file' if _is_test_file(rel_path) else 'file'
    import_map, exports, constants = _collect_file_level_info(tree)

    nodes.append(asdict(KGNode(
        id=file_id, type=file_type, label=Path(rel_path).name,
        metadata={'path': rel_path, 'repo': repo, 'constants': constants, 'exports': exports}
    )))

    # Emit import nodes and edges
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_id = _make_id(f"import_{repo}_{alias.name}")
                nodes.append(asdict(KGNode(id=mod_id, type='import', label=alias.name,
                                           metadata={'repo': repo})))
                edges.append(asdict(KGEdge(source=file_id, target=mod_id, relation='imports')))
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ''
            for alias in node.names:
                full_name = f"{mod_name}.{alias.name}" if mod_name else alias.name
                imp_id = _make_id(f"import_{repo}_{full_name}")
                nodes.append(asdict(KGNode(id=imp_id, type='import', label=full_name,
                                           metadata={'repo': repo, 'module': mod_name,
                                                     'name': alias.name})))
                edges.append(asdict(KGEdge(source=file_id, target=imp_id, relation='imports')))

    def _emit_call_edges(func_id: str, func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
                         class_name: Optional[str] = None):
        """Emit one 'calls' edge per unique (caller, callee_name) pair.

        Edges are marked unresolved=True because callee_name is just a string
        at this point — cross-file resolution happens in RepoASTParser.parse_repo
        after all files are parsed and a global name index is built.

        Each edge carries resolution hints so pass 2 can disambiguate:
            class_hint:        enclosing class name when the call is self.method()
            local_type_hint:   inferred class for receivers like x.method() when
                               x = SomeClass() was seen earlier in the function
            import_resolved:   fully-qualified module path if the bare callee
                               name was imported (e.g. 'json' → 'json'),
                               or if the receiver was imported (e.g. 'json' for
                               json.loads → 'json.loads')
            receiver:          raw receiver expression for attribute calls
                               (kept for debugging / future heuristics)
        """
        local_types = _collect_local_types(func_node)
        for call in ast.walk(func_node):
            if not isinstance(call, ast.Call):
                continue
            callee = _extract_callee_name(call)
            if not callee or (func_id, callee) in seen_call_targets:
                continue
            seen_call_targets.add((func_id, callee))

            receiver = _extract_call_receiver(call)
            class_hint: Optional[str] = None
            local_type_hint: Optional[str] = None
            import_resolved: Optional[str] = None

            if receiver == 'self' and class_name is not None:
                class_hint = class_name
            elif receiver and receiver in local_types:
                local_type_hint = local_types[receiver]
            elif receiver and receiver in import_map:
                # e.g. json.loads where 'json' was imported
                import_resolved = f"{import_map[receiver]}.{callee}"
            else:
                # Bare-name call: foo() where foo was imported
                import_resolved = import_map.get(callee)

            edges.append(asdict(KGEdge(
                source=func_id, target=callee, relation='calls',
                metadata={
                    'unresolved': True,
                    'receiver': receiver,
                    'class_hint': class_hint,
                    'local_type_hint': local_type_hint,
                    'import_resolved': import_resolved,
                }
            )))

    def _emit_func_edges(func_id: str, func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
                         class_name: Optional[str] = None):
        """Emit semantic edges for a function or method beyond call relationships.

        Complements _emit_call_edges with four additional edge types:
            reads/writes:  self.attr accesses within the body, emitted as
                           unresolved attribute name strings. Dropped in pass 2
                           because attribute names are not graph nodes.
            returns:       return type annotation (if present) plus inferred
                           types from return statements. Also unresolved/dropped.
            depends_on:    imports actually referenced in the function body,
                           resolved directly to import node IDs at emit time
                           (no second pass needed — import nodes already exist).
            tests:         for test_* functions only, emits a self-referential
                           edge keyed on the function's own name so pass 2 can
                           strip the 'test_' prefix and link to the target.

        Args:
            func_id: Node ID of the function/method being processed.
            func_node: The AST function node.
            class_name: Name of the enclosing class, passed through to
                        _get_attribute_accesses for self.attr scoping.
        """
        # reads/writes: attribute accesses on self or other objects
        reads, writes = _get_attribute_accesses(func_node, class_name)
        for attr in reads:
            edges.append(asdict(KGEdge(source=func_id, target=attr, relation='reads',
                                       metadata={'unresolved': True})))
        for attr in writes:
            edges.append(asdict(KGEdge(source=func_id, target=attr, relation='writes',
                                       metadata={'unresolved': True})))

        # returns: return type annotations and inferred return expressions
        for ret_type in _get_return_types(func_node):
            edges.append(asdict(KGEdge(source=func_id, target=ret_type, relation='returns',
                                       metadata={'unresolved': True})))

        # depends_on: imports actually used in this function body
        for qualified in _get_used_imports(func_node, import_map):
            imp_id = _make_id(f"import_{repo}_{qualified}")
            edges.append(asdict(KGEdge(source=func_id, target=imp_id, relation='depends_on')))

        # tests: test function → function under test (resolved in second pass)
        if func_node.name.startswith('test_'):
            edges.append(asdict(KGEdge(
                source=func_id, target=func_node.name, relation='tests',
                metadata={'unresolved': True}
            )))

    # Emit class, method, and top-level function nodes
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            bases = _get_base_names(node)
            class_id = _make_id(f"class_{repo}_{rel_path}_{node.name}")
            nodes.append(asdict(KGNode(
                id=class_id, type='class', label=node.name,
                metadata={
                    'filepath': rel_path, 'repo': repo, 'lineno': node.lineno,
                    'bases': bases,
                    'decorators': _get_decorators(node),
                    'docstring': _get_docstring(node),
                    'attributes': _get_class_attributes(node),
                }
            )))
            edges.append(asdict(KGEdge(source=file_id, target=class_id, relation='contains')))

            # Inheritance edges are emitted unresolved here; the second pass
            # in parse_repo resolves base names to actual class node IDs
            for base in bases:
                edges.append(asdict(KGEdge(
                    source=class_id, target=base, relation='inherits',
                    metadata={'unresolved': True}
                )))

            # uses: classes instantiated within any method of this class
            for inst_cls in _get_instantiated_classes_in_class(node):
                edges.append(asdict(KGEdge(source=class_id, target=inst_cls, relation='uses',
                                           metadata={'unresolved': True})))

            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_type = 'test_function' if child.name.startswith('test_') else 'method'
                    func_id = _make_id(f"func_{repo}_{rel_path}_{node.name}_{child.name}")
                    nodes.append(asdict(KGNode(
                        id=func_id, type=func_type, label=child.name,
                        metadata=_build_func_metadata(child, rel_path, repo,
                                                      parent_class=node.name,
                                                      import_map=import_map)
                    )))
                    edges.append(asdict(KGEdge(source=class_id, target=func_id, relation='contains')))
                    # overrides: if method name matches a known base class method (resolved in pass 2)
                    if child.name != '__init__':
                        for base in bases:
                            edges.append(asdict(KGEdge(
                                source=func_id, target=f"{base}.{child.name}", relation='overrides',
                                metadata={'unresolved': True}
                            )))
                    _emit_call_edges(func_id, child, class_name=node.name)
                    _emit_func_edges(func_id, child, class_name=node.name)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_type = 'test_function' if node.name.startswith('test_') else 'function'
            func_id = _make_id(f"func_{repo}_{rel_path}_{node.name}")
            nodes.append(asdict(KGNode(
                id=func_id, type=func_type, label=node.name,
                metadata=_build_func_metadata(node, rel_path, repo, import_map=import_map)
            )))
            edges.append(asdict(KGEdge(source=file_id, target=func_id, relation='contains')))
            _emit_call_edges(func_id, node)
            _emit_func_edges(func_id, node)

    return {'nodes': nodes, 'edges': edges}


# ---------------------------------------------------------------------------
# Parallel driver and second-pass resolution
# ---------------------------------------------------------------------------

class RepoASTParser:
    """Parses all Python files in a repo directory and assembles a structural KG.

    File parsing runs in parallel via ProcessPoolExecutor. After all files
    are parsed, a second pass resolves unresolved 'calls' and 'inherits'
    edges by matching callee/base names against a global node index.

    Call resolution is best-effort:
        - 'qualified': resolved via class_hint, local_type_hint, or import_resolved
        - 'exact': only one function with that name exists in the repo
        - 'ambiguous': multiple functions share the name (e.g. common names
          like 'get' or '__init__'); all candidates are linked
        - Calls to external libraries (no match in repo) are dropped
    """

    def __init__(self, max_workers: int = 4):
        """
        Args:
            max_workers: Number of parallel worker processes for file parsing.
                         Set to 1 for debugging to get synchronous tracebacks.
        """
        self.max_workers = max_workers

    def parse_repo(self, repo: str, repo_dir: Path) -> Dict:
        """Walk repo_dir, parse all .py files in parallel, and return the KG dict.

        Two-pass algorithm:
            Pass 1 (parallel): Each file is parsed independently, emitting nodes
                and unresolved call/inherits edges where targets are name strings.
            Pass 2 (sequential): Build a global name→id index from all nodes,
                then resolve unresolved edges to actual node IDs.

        Args:
            repo: Repository name used as namespace in node IDs (e.g. 'psf/requests').
            repo_dir: Root of the extracted source tree.

        Returns:
            KG dict: {'nodes': [...], 'edges': [...], 'metadata': {...}}
        """
        file_args = []
        for py_file in repo_dir.rglob('*.py'):
            rel = py_file.relative_to(repo_dir)
            if any(part in SKIP_DIRS for part in rel.parts):
                continue
            file_args.append((repo, str(rel), str(py_file)))

        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_parse_file, args): args for args in file_args}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        # --- Pass 2: aggregate nodes and build resolution indices ---
        all_nodes: List[Dict] = []
        seen_node_ids: Set[str] = set()
        # Maps function/method label → list of node IDs (multiple if name is shared)
        label_to_ids: Dict[str, List[str]] = defaultdict(list)
        # Separate index for class nodes used in inheritance resolution
        class_label_to_ids: Dict[str, List[str]] = defaultdict(list)
        # (class_name, method_name) → method node IDs — used for self.method and
        # local-type-hint resolution. A class can technically appear in multiple
        # files (overloaded across modules), so the value is a list.
        class_method_to_ids: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        # Fully-qualified path 'pkg.mod.Thing' → node ID(s). Lets us resolve
        # imports directly when the import_map gave us a qualified target.
        qualified_to_ids: Dict[str, List[str]] = defaultdict(list)

        for result in results:
            for node in result['nodes']:
                if node['id'] not in seen_node_ids:
                    all_nodes.append(node)
                    seen_node_ids.add(node['id'])
                    ntype = node['type']
                    label = node['label']
                    if ntype in ('function', 'method', 'test_function'):
                        label_to_ids[label].append(node['id'])
                        parent = node['metadata'].get('class')
                        if parent:
                            class_method_to_ids[(parent, label)].append(node['id'])
                    elif ntype == 'class':
                        class_label_to_ids[label].append(node['id'])

                    # Index by qualified path: filepath without .py + label.
                    # 'requests/auth.py' + 'HTTPBasicAuth' → 'requests.auth.HTTPBasicAuth'
                    fp = node.get('metadata', {}).get('filepath')
                    if fp and ntype in ('function', 'method', 'class', 'test_function'):
                        mod = fp.removesuffix('.py').replace('/', '.')
                        qualified_to_ids[f"{mod}.{label}"].append(node['id'])
                        parent = node.get('metadata', {}).get('class')
                        if parent:
                            qualified_to_ids[f"{mod}.{parent}.{label}"].append(node['id'])

        nodes_by_id: Dict[str, Dict] = {n['id']: n for n in all_nodes}

        def _resolve_call(meta: Dict, callee_name: str) -> Tuple[List[str], str]:
            """Resolve a call-edge target using the metadata hints.

            Returns (matched_node_ids, confidence_label). Tries hints in order:
                1. class_hint: self.method() inside a method of class C → C.method
                2. local_type_hint: x.method() where x = SomeClass() → SomeClass.method
                3. import_resolved: bare or receiver-qualified name found via imports
                4. fallback: bare name lookup (the original ambiguous behavior)

            Confidence is 'qualified' for hits 1-3 (high precision, single owner)
            and 'exact'/'ambiguous' for the bare-name fallback.
            """
            class_hint = meta.get('class_hint')
            if class_hint:
                hits = class_method_to_ids.get((class_hint, callee_name), [])
                if hits:
                    return hits, 'qualified'

            local_hint = meta.get('local_type_hint')
            if local_hint:
                hits = class_method_to_ids.get((local_hint, callee_name), [])
                if hits:
                    return hits, 'qualified'

            qualified = meta.get('import_resolved')
            if qualified:
                hits = qualified_to_ids.get(qualified, [])
                if hits:
                    return hits, 'qualified'
                # Try just the last component as a class qualifier:
                # 'pkg.mod.Foo.bar' → look up ('Foo', 'bar')
                parts = qualified.rsplit('.', 2)
                if len(parts) == 3:
                    hits = class_method_to_ids.get((parts[1], parts[2]), [])
                    if hits:
                        return hits, 'qualified'

            hits = label_to_ids.get(callee_name, [])
            if not hits:
                return [], 'unresolved'
            return hits, 'exact' if len(hits) == 1 else 'ambiguous'

        # --- Pass 2: resolve edges ---
        all_edges: List[Dict] = []
        seen_edges: Set[Tuple] = set()

        for result in results:
            for edge in result['edges']:
                meta = edge.get('metadata', {})

                if edge['relation'] == 'calls' and meta.get('unresolved'):
                    # Replace callee name string with actual node ID(s)
                    callee_name = edge['target']
                    matches, confidence = _resolve_call(meta, callee_name)
                    if not matches:
                        # External library call — no node in repo, drop the edge
                        continue
                    for target_id in matches:
                        key = (edge['source'], target_id, 'calls')
                        if key not in seen_edges:
                            seen_edges.add(key)
                            all_edges.append(asdict(KGEdge(
                                source=edge['source'], target=target_id, relation='calls',
                                metadata={'confidence': confidence,
                                          'import_resolved': meta.get('import_resolved')}
                            )))

                elif edge['relation'] == 'inherits' and meta.get('unresolved'):
                    # Base names may be dotted (pkg.Base) — take the last component
                    base_name = edge['target'].split('.')[-1]
                    matches = class_label_to_ids.get(base_name, [])
                    if not matches:
                        continue
                    confidence = 'exact' if len(matches) == 1 else 'ambiguous'
                    for target_id in matches:
                        key = (edge['source'], target_id, 'inherits')
                        if key not in seen_edges:
                            seen_edges.add(key)
                            all_edges.append(asdict(KGEdge(
                                source=edge['source'], target=target_id, relation='inherits',
                                metadata={'confidence': confidence}
                            )))

                elif edge['relation'] == 'tests' and meta.get('unresolved'):
                    # Resolve test function → target function by name
                    target_name = edge['target']
                    # Strip 'test_' prefix to find candidate (stored as original func name)
                    if target_name.startswith('test_'):
                        target_name = target_name[5:]
                    matches = label_to_ids.get(target_name, [])
                    if not matches:
                        continue
                    confidence = 'exact' if len(matches) == 1 else 'ambiguous'
                    for target_id in matches:
                        key = (edge['source'], target_id, 'tests')
                        if key not in seen_edges:
                            seen_edges.add(key)
                            all_edges.append(asdict(KGEdge(
                                source=edge['source'], target=target_id, relation='tests',
                                metadata={'confidence': confidence}
                            )))

                elif edge['relation'] == 'uses' and meta.get('unresolved'):
                    # class → uses → class: resolve instantiated class name to class node ID
                    matches = class_label_to_ids.get(edge['target'], [])
                    if not matches:
                        continue
                    confidence = 'exact' if len(matches) == 1 else 'ambiguous'
                    for target_id in matches:
                        key = (edge['source'], target_id, 'uses')
                        if key not in seen_edges:
                            seen_edges.add(key)
                            all_edges.append(asdict(KGEdge(
                                source=edge['source'], target=target_id, relation='uses',
                                metadata={'confidence': confidence}
                            )))

                elif edge['relation'] == 'overrides' and meta.get('unresolved'):
                    # method → overrides → parent method: target is "BaseClass.method_name"
                    parts = edge['target'].rsplit('.', 1)
                    if len(parts) != 2:
                        continue
                    base_name, method_name = parts
                    base_simple = base_name.split('.')[-1]
                    if class_label_to_ids.get(base_simple):
                        for target_id in label_to_ids.get(method_name, []):
                            node = nodes_by_id.get(target_id, {})
                            if node.get('metadata', {}).get('class') == base_simple:
                                key = (edge['source'], target_id, 'overrides')
                                if key not in seen_edges:
                                    seen_edges.add(key)
                                    all_edges.append(asdict(KGEdge(
                                        source=edge['source'], target=target_id,
                                        relation='overrides'
                                    )))

                elif meta.get('unresolved'):
                    # reads/writes/returns reference attribute strings, not node IDs — drop
                    continue

                else:
                    key = (edge['source'], edge['target'], edge['relation'])
                    if key not in seen_edges:
                        seen_edges.add(key)
                        all_edges.append(edge)

        # module_depends_on: file → file edges derived from file→imports→import chains
        # Build: import_id → [file_id that imports it], file_label → file_id
        file_label_to_id: Dict[str, str] = {
            n['label']: n['id'] for n in all_nodes
            if n['type'] in ('file', 'test_file')
        }
        import_to_files: Dict[str, List[str]] = defaultdict(list)
        for edge in all_edges:
            if edge['relation'] == 'imports':
                import_to_files[edge['target']].append(edge['source'])

        for imp_node in all_nodes:
            if imp_node['type'] != 'import':
                continue
            module = imp_node['metadata'].get('module', '')
            # Map dotted module path to a file node: 'requests.sessions' → 'sessions.py'
            parts = (module or imp_node['label']).split('.')
            for i in range(len(parts), 0, -1):
                candidate = parts[i - 1] + '.py'
                target_file_id = file_label_to_id.get(candidate)
                if target_file_id:
                    for src_file_id in import_to_files.get(imp_node['id'], []):
                        if src_file_id != target_file_id:
                            key = (src_file_id, target_file_id, 'module_depends_on')
                            if key not in seen_edges:
                                seen_edges.add(key)
                                all_edges.append(asdict(KGEdge(
                                    source=src_file_id, target=target_file_id,
                                    relation='module_depends_on'
                                )))
                    break

        # --- Add call context metadata to function nodes ---
        # Build caller map: target_node_id → [source_node_ids calling it]
        callers: Dict[str, List[str]] = defaultdict(list)
        for edge in all_edges:
            if edge['relation'] == 'calls':
                callers[edge['target']].append(edge['source'])

        # Build node_id → node dict for quick lookup
        node_by_id = {node['id']: node for node in all_nodes}

        # Annotate each function/method with call context
        for node in all_nodes:
            if node['type'] not in ('function', 'method', 'test_function'):
                continue

            node_id = node['id']
            caller_ids = callers.get(node_id, [])
            caller_count = len(caller_ids)

            # Build direct_callers list with caller details
            direct_callers = []
            for caller_id in caller_ids:
                caller_node = node_by_id.get(caller_id)
                if caller_node:
                    direct_callers.append({
                        'id': caller_id,
                        'label': caller_node['label'],
                        'type': caller_node['type'],
                    })

            # Compute call importance based on caller diversity
            # High: called from many different functions/classes
            # Medium: called from a few places
            # Low: internal only (1-2 callers)
            if caller_count == 0:
                importance = 'internal'
            elif caller_count <= 2:
                importance = 'low'
            elif caller_count <= 10:
                importance = 'medium'
            else:
                importance = 'high'

            # Check if this is a public API (exported or called externally)
            is_public = (
                node['metadata'].get('name', node['label']) in
                node['metadata'].get('exports', [])
            ) or caller_count >= 5

            # Add call context to metadata
            node['metadata']['caller_count'] = caller_count
            node['metadata']['direct_callers'] = direct_callers
            node['metadata']['call_importance'] = importance
            node['metadata']['is_public_api'] = is_public

        return {
            'nodes': all_nodes,
            'edges': all_edges,
            'metadata': {
                'repo': repo,
                'file_count': len(file_args),
                'parse_mode': 'source',
            }
        }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

class RepoKGBuilder:
    """Top-level entry point for building, saving, and loading repo KGs.

    Orchestrates RepoManager (git operations) and RepoASTParser (source
    parsing) into a single build() call. Output is saved as JSON to
    kg_output/kg_{repo}.json.

    Example:
        builder = RepoKGBuilder()
        kg = builder.build('psf/requests', 'a0df2cbb...')
        builder.save('psf/requests', kg)

        # Later
        kg = builder.load('psf/requests')
        engine = KGQueryEngine(kg)
    """

    def __init__(self,
                 output_dir: Path = Path('kg_output'),
                 cache_dir: Path = Path('repo_cache'),
                 max_workers: int = 4):
        """
        Args:
            output_dir: Where KG JSON files are saved.
            cache_dir: Where bare git clones are cached.
            max_workers: Parallel workers for file parsing.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.repo_manager = RepoManager(cache_dir)
        self.ast_parser = RepoASTParser(max_workers=max_workers)

    def build(self, repo: str, commit: str) -> Dict:
        """Build a structural KG for a repo at a specific commit.

        Clones (if needed), extracts source at the commit, parses all .py
        files, resolves edges, and returns the KG dict. The source tree is
        cleaned up automatically via tempfile.TemporaryDirectory.

        Args:
            repo: GitHub repo in 'owner/name' format.
            commit: Commit SHA to build the KG from.

        Returns:
            KG dict ready for saving or querying.
        """
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'src'
            print(f"Extracting {repo}@{commit[:8]}...")
            self.repo_manager.extract_at_commit(repo, commit, dest)
            print("Parsing source...")
            kg = self.ast_parser.parse_repo(repo, dest)

        kg['metadata']['base_commit'] = commit
        return kg

    def save(self, repo: str, kg: Dict):
        """Serialize and save a KG to kg_output/kg_{repo}.json.

        Slashes, dashes, and dots in repo names are replaced with underscores
        to produce a safe filename.

        Args:
            repo: Repository name (used to derive the output filename).
            kg: KG dict as returned by build().
        """
        safe_name = repo.replace('/', '_').replace('-', '_').replace('.', '_')
        output_file = self.output_dir / f"kg_{safe_name}.json"
        with open(output_file, 'w') as f:
            json.dump(kg, f, indent=2)
        print(f"Saved: {repo} -> {output_file} "
              f"({len(kg['nodes'])} nodes, {len(kg['edges'])} edges)")

    def load(self, repo: str) -> Optional[Dict]:
        """Load a previously saved KG from disk.

        Args:
            repo: Repository name (e.g. 'psf/requests').

        Returns:
            KG dict, or None if no saved KG exists for this repo.
        """
        safe_name = repo.replace('/', '_').replace('-', '_').replace('.', '_')
        kg_file = self.output_dir / f"kg_{safe_name}.json"
        if kg_file.exists():
            with open(kg_file) as f:
                return json.load(f)
        return None
