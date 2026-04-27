"""
test_context.py

Extract KG subgraphs from code changes for LLM test generation.

Given a dataset instance with {repo, base_commit, patch, code_file, test_file},
this module:
  1. Parses the patch to identify changed functions/classes
  2. Loads the pre-built KG at base_commit
  3. Finds corresponding nodes in the KG
  4. Performs BFS to extract surrounding context
  5. Returns a structured TestContext

Works generically across any dataset with the standard schema.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import deque

from kg_query import KGQueryEngine


@dataclass
class TestContext:
    """Structured subgraph extracted for test generation.

    Attributes:
        seeds: Nodes representing the changed functions/classes under test.
        context_nodes: BFS-expanded neighbors providing surrounding context.
        edges: Edges within the subgraph (source and target both in context).
        test_nodes: Existing test functions found via 'tests' edges.
        repo: Repository name for reference.
        base_commit: Commit SHA the KG was built from.
    """
    seeds: List[Dict]
    context_nodes: List[Dict]
    edges: List[Dict]
    test_nodes: List[Dict]
    repo: str
    base_commit: str


class PatchParser:
    """Parse unified diffs to extract changed function/class names."""

    @staticmethod
    def extract_changed_functions(patch: str, code_file: str) -> Set[str]:
        """Extract function and class names changed in a specific file's hunks.

        Parses unified diff hunks for code_file and identifies function/class
        definitions that appear in the changed lines (those with + or context
        lines near the hunk start). Returns a set of changed names.

        Args:
            patch: Unified diff string (multi-file).
            code_file: Relative path to the file to extract changes from
                      (e.g. 'requests/sessions.py').

        Returns:
            Set of function/class name strings that changed in code_file.
        """
        changed_names: Set[str] = set()

        # Split patch into individual file diffs (--- a/path and +++ b/path)
        file_diff_pattern = r'^--- a/.*?\n\+\+\+ b/(.+?)$'
        current_file = None
        current_hunk = []

        lines = patch.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for file boundary: +++ b/path
            if line.startswith('+++'):
                # Extract the file path from '+++ b/path'
                match = re.match(r'^\+\+\+ b/(.+)$', line)
                if match:
                    current_file = match.group(1)
                    current_hunk = []

            # Within the target file, collect hunk lines
            if current_file == code_file:
                # Hunk header: @@ -start,count +start,count @@
                if line.startswith('@@'):
                    if current_hunk:
                        changed_names.update(
                            PatchParser._extract_defs_from_hunk(current_hunk)
                        )
                    current_hunk = []
                # Accumulate changed lines (+ or context, not -)
                elif line.startswith(('+', ' ')) and not line.startswith('+++'):
                    current_hunk.append(line)

            i += 1

        # Process final hunk
        if current_hunk:
            changed_names.update(
                PatchParser._extract_defs_from_hunk(current_hunk)
            )

        return changed_names

    @staticmethod
    def _extract_defs_from_hunk(hunk_lines: List[str]) -> Set[str]:
        """Extract function/class definitions from a list of hunk lines.

        Looks for lines matching:
          def function_name(...)
          async def function_name(...)
          class ClassName(...)

        Returns the set of names found.
        """
        names: Set[str] = set()

        for line in hunk_lines:
            # Strip the leading +/space marker
            content = line[1:] if line and line[0] in ('+', ' ') else line

            # Match: def/async def/class name(
            def_match = re.match(r'^\s*(async\s+)?def\s+(\w+)\s*\(', content)
            if def_match:
                names.add(def_match.group(2))

            class_match = re.match(r'^\s*class\s+(\w+)\s*[\(:]', content)
            if class_match:
                names.add(class_match.group(1))

        return names


class TestContextExtractor:
    """Extract KG subgraphs from dataset instances for test generation."""

    def __init__(self, engine: KGQueryEngine):
        """
        Args:
            engine: Loaded KGQueryEngine on the pre-built KG.
        """
        self.engine = engine
        self.patch_parser = PatchParser()

    def extract(
        self,
        instance: Dict,
        depth: int = 2,
        edge_filter: Optional[Set[str]] = None,
    ) -> TestContext:
        """Extract a KG subgraph from a dataset instance.

        Args:
            instance: Dict with keys:
                - repo: Repository name (e.g. 'psf/requests')
                - base_commit: Commit SHA the KG was built from
                - patch: Unified diff of code changes
                - code_file: Relative path to code file (e.g. 'requests/sessions.py')
                - test_file: Relative path to test file (e.g. 'tests/test_sessions.py')
            depth: BFS depth for context expansion (default 2).
            edge_filter: Set of edge relations to traverse during BFS.
                        If None, uses smart defaults (contains, calls, inherits,
                        tests, uses, depends_on).

        Returns:
            TestContext with seeds, context_nodes, edges, test_nodes.
        """
        if edge_filter is None:
            edge_filter = {'contains', 'calls', 'inherits', 'tests', 'uses', 'depends_on'}

        # Extract changed function/class names from the patch
        changed_names = self.patch_parser.extract_changed_functions(
            instance['patch'],
            instance['code_file']
        )

        # Find the code file node
        code_file_results = self.engine.find_file_by_path(instance['code_file'])
        if not code_file_results:
            raise ValueError(f"Code file not found: {instance['code_file']}")
        code_file_node = code_file_results[0]

        # Find test file node (may not exist in KG)
        test_file_node = None
        test_file_results = self.engine.find_file_by_path(instance['test_file'])
        if test_file_results:
            test_file_node = test_file_results[0]

        # Find seed nodes: changed functions in code_file
        seed_ids: List[str] = []
        for name in changed_names:
            funcs = self.engine.find_function_by_name(name)
            # Filter to only those in the code_file
            for func in funcs:
                if func['metadata'].get('filepath') == instance['code_file']:
                    seed_ids.append(func['id'])

        # If no changed functions found, use the code_file itself as seed
        if not seed_ids:
            seed_ids = [code_file_node['id']]

        # Add test file as seed if it exists
        if test_file_node:
            seed_ids.append(test_file_node['id'])

        # BFS to extract subgraph
        subgraph_nodes, subgraph_edges = self._bfs(
            seed_ids,
            depth=depth,
            edge_filter=edge_filter
        )

        # Find test functions via 'tests' edges
        test_nodes = []
        for edge in subgraph_edges:
            if edge['relation'] == 'tests':
                test_node = self.engine.nodes_by_id.get(edge['source'])
                if test_node:
                    test_nodes.append(test_node)

        # Separate seeds from context
        seed_node_ids = set(seed_ids)
        seed_nodes = [n for n in subgraph_nodes if n['id'] in seed_node_ids]
        context_nodes = [n for n in subgraph_nodes if n['id'] not in seed_node_ids]

        return TestContext(
            seeds=seed_nodes,
            context_nodes=context_nodes,
            edges=subgraph_edges,
            test_nodes=test_nodes,
            repo=instance['repo'],
            base_commit=instance['base_commit'],
        )

    def _bfs(
        self,
        seed_ids: List[str],
        depth: int = 2,
        edge_filter: Optional[Set[str]] = None,
    ) -> tuple:
        """BFS traversal from seeds, filtered by edge type.

        Expands outward from seed nodes up to `depth` hops, including only
        edges whose relation is in edge_filter. Both directions are traversed
        (incoming and outgoing).

        Args:
            seed_ids: Starting node IDs.
            depth: Maximum hop distance.
            edge_filter: Set of edge relations to include.

        Returns:
            (nodes, edges) — lists of node and edge dicts in the subgraph.
        """
        if edge_filter is None:
            edge_filter = set()

        visited_nodes: Set[str] = set()
        visited_edges: Set[tuple] = set()
        nodes_list: List[Dict] = []
        edges_list: List[Dict] = []

        frontier = deque([(nid, 0) for nid in seed_ids if nid in self.engine.nodes_by_id])

        while frontier:
            node_id, current_depth = frontier.popleft()

            if node_id in visited_nodes:
                continue

            visited_nodes.add(node_id)
            nodes_list.append(self.engine.nodes_by_id[node_id])

            if current_depth >= depth:
                continue

            # Outgoing edges
            for edge in self.engine.edges_by_source.get(node_id, []):
                if edge['relation'] not in edge_filter:
                    continue

                target_id = edge['target']
                edge_key = (edge['source'], target_id, edge['relation'])

                if edge_key not in visited_edges and target_id in self.engine.nodes_by_id:
                    visited_edges.add(edge_key)
                    edges_list.append(edge)

                    if target_id not in visited_nodes:
                        frontier.append((target_id, current_depth + 1))

            # Incoming edges (for call graph, tests, etc.)
            for edge in self.engine.edges_by_target.get(node_id, []):
                if edge['relation'] not in edge_filter:
                    continue

                source_id = edge['source']
                edge_key = (source_id, edge['target'], edge['relation'])

                if edge_key not in visited_edges and source_id in self.engine.nodes_by_id:
                    visited_edges.add(edge_key)
                    edges_list.append(edge)

                    if source_id not in visited_nodes:
                        frontier.append((source_id, current_depth + 1))

        return nodes_list, edges_list
