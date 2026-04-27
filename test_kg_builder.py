"""
test_kg_builder.py

Unit tests for repo_kg_builder helpers and edge emission.

Runs against synthetic Python source strings so no git clone is needed.
Uses only stdlib (unittest + ast) — no external dependencies required.

Run with: python3 -m pytest test_kg_builder.py -v
       or: python3 test_kg_builder.py
"""

import ast
import json
import tempfile
import unittest
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from repo_kg_builder import (
    _parse_file,
    _get_attribute_accesses,
    _get_return_types,
    _get_used_imports,
    _get_instantiated_classes,
    _get_instantiated_classes_in_class,
    _get_test_target,
    _collect_file_level_info,
    _count_branches,
    _build_func_metadata,
    _get_signature,
    _get_decorators,
    _get_exceptions,
    _get_class_attributes,
    _get_assert_patterns,
    _is_test_file,
    _safe_unparse,
    RepoASTParser,
    KGNode,
    KGEdge,
    _make_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_source(source: str, repo: str = 'test/repo', rel_path: str = 'mod.py') -> dict:
    """Write source to a temp file and call _parse_file on it."""
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
        f.write(source)
        tmp = f.name
    result = _parse_file((repo, rel_path, tmp))
    Path(tmp).unlink()
    return result


def _edges_of(result: dict, relation: str) -> list:
    return [e for e in result['edges'] if e['relation'] == relation]


def _node_of(result: dict, label: str) -> dict:
    return next((n for n in result['nodes'] if n['label'] == label), None)


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------

class TestGetAttributeAccesses(unittest.TestCase):
    def _func(self, src: str):
        tree = ast.parse(src)
        return tree.body[0]

    def test_reads_self_attr(self):
        f = self._func("def f(self):\n    return self.x")
        reads, writes = _get_attribute_accesses(f, 'MyClass')
        self.assertIn('x', reads)
        self.assertEqual(writes, [])

    def test_writes_self_attr(self):
        f = self._func("def f(self):\n    self.x = 1")
        reads, writes = _get_attribute_accesses(f, 'MyClass')
        self.assertIn('x', writes)

    def test_no_duplicates(self):
        f = self._func("def f(self):\n    return self.x + self.x")
        reads, _ = _get_attribute_accesses(f, 'MyClass')
        self.assertEqual(reads.count('x'), 1)

    def test_ignores_nested_function(self):
        src = "def f(self):\n    def inner(self):\n        self.y = 2\n    return self.x"
        f = self._func(src)
        reads, writes = _get_attribute_accesses(f, 'MyClass')
        self.assertIn('x', reads)
        self.assertNotIn('y', writes)


class TestGetReturnTypes(unittest.TestCase):
    def _func(self, src: str):
        return ast.parse(src).body[0]

    def test_annotation(self):
        f = self._func("def f() -> int: pass")
        self.assertIn('int', _get_return_types(f))

    def test_no_annotation(self):
        f = self._func("def f(): pass")
        self.assertEqual(_get_return_types(f), [])

    def test_complex_annotation(self):
        f = self._func("def f() -> Optional[str]: pass")
        types = _get_return_types(f)
        self.assertTrue(len(types) > 0)


class TestGetUsedImports(unittest.TestCase):
    def test_used_import(self):
        src = "def f():\n    return os.path.join('a', 'b')"
        tree = ast.parse(src)
        func = tree.body[0]
        import_map = {'os': 'os'}
        result = _get_used_imports(func, import_map)
        self.assertIn('os', result)

    def test_unused_import(self):
        src = "def f():\n    return 42"
        tree = ast.parse(src)
        func = tree.body[0]
        import_map = {'os': 'os', 're': 're'}
        result = _get_used_imports(func, import_map)
        self.assertEqual(result, [])

    def test_no_duplicates(self):
        src = "def f():\n    os.path.join('a'); os.getcwd()"
        tree = ast.parse(src)
        func = tree.body[0]
        import_map = {'os': 'os'}
        result = _get_used_imports(func, import_map)
        self.assertEqual(result.count('os'), 1)


class TestGetInstantiatedClasses(unittest.TestCase):
    def _func(self, src: str):
        return ast.parse(src).body[0]

    def test_direct_instantiation(self):
        f = self._func("def f():\n    x = MyClass()")
        self.assertIn('MyClass', _get_instantiated_classes(f))

    def test_attribute_instantiation(self):
        f = self._func("def f():\n    x = mod.Foo()")
        self.assertIn('Foo', _get_instantiated_classes(f))

    def test_lowercase_ignored(self):
        f = self._func("def f():\n    x = helper()")
        self.assertEqual(_get_instantiated_classes(f), [])


class TestGetInstantiatedClassesInClass(unittest.TestCase):
    def test_aggregates_across_methods(self):
        src = (
            "class A:\n"
            "    def m1(self):\n        x = Foo()\n"
            "    def m2(self):\n        y = Bar()\n"
        )
        node = ast.parse(src).body[0]
        result = _get_instantiated_classes_in_class(node)
        self.assertIn('Foo', result)
        self.assertIn('Bar', result)

    def test_no_duplicates(self):
        src = "class A:\n    def m1(self):\n        Foo()\n    def m2(self):\n        Foo()\n"
        node = ast.parse(src).body[0]
        result = _get_instantiated_classes_in_class(node)
        self.assertEqual(result.count('Foo'), 1)


class TestGetTestTarget(unittest.TestCase):
    def _func(self, src: str):
        return ast.parse(src).body[0]

    def test_name_convention(self):
        f = self._func("def test_send(self):\n    pass")
        target = _get_test_target(f, {'send'})
        self.assertEqual(target, 'send')

    def test_name_convention_multipart(self):
        f = self._func("def test_send_request(self):\n    pass")
        target = _get_test_target(f, {'send_request'})
        self.assertEqual(target, 'send_request')

    def test_no_match(self):
        f = self._func("def test_nothing(self):\n    pass")
        target = _get_test_target(f, {'send', 'get'})
        self.assertIsNone(target)


# ---------------------------------------------------------------------------
# _parse_file integration tests
# ---------------------------------------------------------------------------

class TestParseFileCalls(unittest.TestCase):
    SOURCE = """\
def caller():
    callee()

def callee():
    pass
"""

    def test_emits_unresolved_calls_edge(self):
        result = _parse_source(self.SOURCE)
        call_edges = _edges_of(result, 'calls')
        self.assertTrue(len(call_edges) > 0)
        self.assertTrue(all(e['metadata'].get('unresolved') for e in call_edges))

    def test_caller_target_is_name_string(self):
        result = _parse_source(self.SOURCE)
        call_edges = _edges_of(result, 'calls')
        targets = [e['target'] for e in call_edges]
        self.assertIn('callee', targets)


class TestParseFileDependsOn(unittest.TestCase):
    SOURCE = """\
import os

def f():
    return os.getcwd()
"""

    def test_emits_depends_on_edge(self):
        result = _parse_source(self.SOURCE)
        dep_edges = _edges_of(result, 'depends_on')
        self.assertTrue(len(dep_edges) > 0)

    def test_depends_on_points_to_import_node(self):
        result = _parse_source(self.SOURCE)
        dep_edges = _edges_of(result, 'depends_on')
        import_ids = {n['id'] for n in result['nodes'] if n['type'] == 'import'}
        for e in dep_edges:
            self.assertIn(e['target'], import_ids)


class TestParseFileTestsEdge(unittest.TestCase):
    SOURCE = """\
def send():
    pass

def test_send():
    send()
"""

    def test_emits_unresolved_tests_edge(self):
        result = _parse_source(self.SOURCE, rel_path='tests/test_mod.py')
        tests_edges = _edges_of(result, 'tests')
        self.assertTrue(len(tests_edges) > 0)
        self.assertTrue(all(e['metadata'].get('unresolved') for e in tests_edges))


class TestParseFileUsesEdge(unittest.TestCase):
    SOURCE = """\
class Dependency:
    pass

class Consumer:
    def method(self):
        Dependency()
"""

    def test_emits_unresolved_uses_edge(self):
        result = _parse_source(self.SOURCE)
        uses_edges = _edges_of(result, 'uses')
        self.assertTrue(len(uses_edges) > 0)

    def test_uses_source_is_class(self):
        result = _parse_source(self.SOURCE)
        uses_edges = _edges_of(result, 'uses')
        class_ids = {n['id'] for n in result['nodes'] if n['type'] == 'class'}
        for e in uses_edges:
            self.assertIn(e['source'], class_ids)


class TestParseFileOverridesEdge(unittest.TestCase):
    SOURCE = """\
class Base:
    def method(self):
        pass

class Child(Base):
    def method(self):
        pass
"""

    def test_emits_unresolved_overrides_edge(self):
        result = _parse_source(self.SOURCE)
        ov_edges = _edges_of(result, 'overrides')
        self.assertTrue(len(ov_edges) > 0)

    def test_overrides_target_encodes_base_and_method(self):
        result = _parse_source(self.SOURCE)
        ov_edges = _edges_of(result, 'overrides')
        targets = [e['target'] for e in ov_edges]
        self.assertIn('Base.method', targets)


class TestParseFileInherits(unittest.TestCase):
    SOURCE = """\
class Base:
    pass

class Child(Base):
    pass
"""

    def test_emits_unresolved_inherits_edge(self):
        result = _parse_source(self.SOURCE)
        inh_edges = _edges_of(result, 'inherits')
        self.assertTrue(len(inh_edges) > 0)
        self.assertTrue(all(e['metadata'].get('unresolved') for e in inh_edges))


# ---------------------------------------------------------------------------
# RepoASTParser second-pass resolution tests
# ---------------------------------------------------------------------------

class TestSecondPassResolution(unittest.TestCase):
    """Test that parse_repo correctly resolves unresolved edges across files."""

    CALLER_SRC = """\
def caller():
    callee()
"""

    CALLEE_SRC = """\
def callee():
    pass
"""

    INHERIT_SRC = """\
class Base:
    pass

class Child(Base):
    pass
"""

    TEST_SRC = """\
def send():
    pass

def test_send():
    send()
"""

    USES_SRC = """\
class Dep:
    pass

class User:
    def run(self):
        Dep()
"""

    def _build_kg(self, files: dict) -> dict:
        """Write files to a temp dir and run RepoASTParser on them."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for rel, src in files.items():
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(src)
            parser = RepoASTParser(max_workers=1)
            return parser.parse_repo('test/repo', root)

    def test_calls_resolved(self):
        kg = self._build_kg({'caller.py': self.CALLER_SRC, 'callee.py': self.CALLEE_SRC})
        call_edges = [e for e in kg['edges'] if e['relation'] == 'calls']
        node_ids = {n['id'] for n in kg['nodes']}
        self.assertTrue(len(call_edges) > 0)
        for e in call_edges:
            self.assertIn(e['target'], node_ids, "call edge target must be a node ID")
            self.assertFalse(e.get('metadata', {}).get('unresolved'), "resolved edges must not have unresolved=True")

    def test_inherits_resolved(self):
        kg = self._build_kg({'classes.py': self.INHERIT_SRC})
        inh_edges = [e for e in kg['edges'] if e['relation'] == 'inherits']
        node_ids = {n['id'] for n in kg['nodes']}
        self.assertTrue(len(inh_edges) > 0)
        for e in inh_edges:
            self.assertIn(e['target'], node_ids)

    def test_tests_edge_resolved(self):
        kg = self._build_kg({'test_mod.py': self.TEST_SRC})
        tests_edges = [e for e in kg['edges'] if e['relation'] == 'tests']
        node_ids = {n['id'] for n in kg['nodes']}
        self.assertTrue(len(tests_edges) > 0, "expected at least one 'tests' edge")
        for e in tests_edges:
            self.assertIn(e['target'], node_ids)

    def test_uses_edge_resolved(self):
        kg = self._build_kg({'uses.py': self.USES_SRC})
        uses_edges = [e for e in kg['edges'] if e['relation'] == 'uses']
        node_ids = {n['id'] for n in kg['nodes']}
        self.assertTrue(len(uses_edges) > 0, "expected at least one 'uses' edge")
        for e in uses_edges:
            self.assertIn(e['target'], node_ids)

    def test_overrides_edge_resolved(self):
        src = (
            "class Base:\n    def method(self): pass\n\n"
            "class Child(Base):\n    def method(self): pass\n"
        )
        kg = self._build_kg({'override.py': src})
        ov_edges = [e for e in kg['edges'] if e['relation'] == 'overrides']
        node_ids = {n['id'] for n in kg['nodes']}
        self.assertTrue(len(ov_edges) > 0, "expected at least one 'overrides' edge")
        for e in ov_edges:
            self.assertIn(e['target'], node_ids)

    def test_module_depends_on_cross_file(self):
        importer = "from sessions import Session\n\ndef f():\n    pass\n"
        imported = "class Session:\n    pass\n"
        kg = self._build_kg({'client.py': importer, 'sessions.py': imported})
        mod_edges = [e for e in kg['edges'] if e['relation'] == 'module_depends_on']
        self.assertTrue(len(mod_edges) > 0, "expected at least one module_depends_on edge")
        node_ids = {n['id'] for n in kg['nodes']}
        for e in mod_edges:
            self.assertIn(e['source'], node_ids)
            self.assertIn(e['target'], node_ids)

    def test_no_unresolved_edges_in_output(self):
        """After parse_repo, no edge should still carry unresolved=True."""
        kg = self._build_kg({
            'a.py': self.CALLER_SRC,
            'b.py': self.CALLEE_SRC,
            'c.py': self.INHERIT_SRC,
        })
        for edge in kg['edges']:
            self.assertFalse(
                edge.get('metadata', {}).get('unresolved'),
                f"edge {edge['relation']} still has unresolved=True"
            )


# ---------------------------------------------------------------------------
# Signature extraction tests
# ---------------------------------------------------------------------------

class TestGetSignature(unittest.TestCase):
    def _func(self, src: str):
        return ast.parse(src).body[0]

    def test_no_params(self):
        sig = _get_signature(self._func("def f(): pass"))
        self.assertEqual(sig['params'], [])
        self.assertNotIn('returns', sig)

    def test_positional_args(self):
        sig = _get_signature(self._func("def f(a, b): pass"))
        names = [p['name'] for p in sig['params']]
        self.assertEqual(names, ['a', 'b'])
        # No defaults present, so 'default' key should be absent for both
        self.assertTrue(all('default' not in p for p in sig['params']))

    def test_default_right_alignment(self):
        # defaults align to the rightmost args: a has no default, b=1, c=2
        sig = _get_signature(self._func("def f(a, b=1, c=2): pass"))
        params = {p['name']: p for p in sig['params']}
        self.assertNotIn('default', params['a'])
        self.assertEqual(params['b']['default'], '1')
        self.assertEqual(params['c']['default'], '2')

    def test_annotations(self):
        sig = _get_signature(self._func("def f(a: int, b: str = 'x') -> bool: pass"))
        params = {p['name']: p for p in sig['params']}
        self.assertEqual(params['a']['annotation'], 'int')
        self.assertEqual(params['b']['annotation'], 'str')
        self.assertEqual(params['b']['default'], "'x'")
        self.assertEqual(sig['returns'], 'bool')

    def test_vararg_and_kwarg(self):
        sig = _get_signature(self._func("def f(*args, **kwargs): pass"))
        names = [p['name'] for p in sig['params']]
        self.assertIn('*args', names)
        self.assertIn('**kwargs', names)

    def test_vararg_kwarg_annotations(self):
        sig = _get_signature(self._func("def f(*args: int, **kwargs: str): pass"))
        params = {p['name']: p for p in sig['params']}
        self.assertEqual(params['*args']['annotation'], 'int')
        self.assertEqual(params['**kwargs']['annotation'], 'str')

    def test_kwonly_args(self):
        sig = _get_signature(self._func("def f(*, x, y=2): pass"))
        params = {p['name']: p for p in sig['params']}
        self.assertIn('x', params)
        self.assertNotIn('default', params['x'])
        self.assertEqual(params['y']['default'], '2')

    def test_posonly_args(self):
        sig = _get_signature(self._func("def f(a, b, /, c): pass"))
        names = [p['name'] for p in sig['params']]
        self.assertEqual(names, ['a', 'b', 'c'])

    def test_default_none_is_recorded(self):
        # Literal None defaults are stored as the string 'None' (the AST node
        # is ast.Constant(value=None), which is not Python's None sentinel).
        sig = _get_signature(self._func("def f(x=None): pass"))
        params = {p['name']: p for p in sig['params']}
        self.assertEqual(params['x']['default'], 'None')


# ---------------------------------------------------------------------------
# Decorator extraction tests
# ---------------------------------------------------------------------------

class TestGetDecorators(unittest.TestCase):
    def _node(self, src: str):
        return ast.parse(src).body[0]

    def test_no_decorators(self):
        self.assertEqual(_get_decorators(self._node("def f(): pass")), [])

    def test_plain_decorator(self):
        decs = _get_decorators(self._node("@staticmethod\ndef f(): pass"))
        self.assertEqual(decs, ['staticmethod'])

    def test_attribute_decorator(self):
        decs = _get_decorators(self._node("@pytest.mark.skip\ndef f(): pass"))
        self.assertIn('pytest.mark.skip', decs)

    def test_call_decorator(self):
        src = "@pytest.mark.parametrize('x', [1, 2])\ndef f(x): pass"
        decs = _get_decorators(self._node(src))
        self.assertEqual(len(decs), 1)
        self.assertIn('parametrize', decs[0])

    def test_multiple_decorators_preserve_order(self):
        src = "@a\n@b\n@c\ndef f(): pass"
        self.assertEqual(_get_decorators(self._node(src)), ['a', 'b', 'c'])

    def test_class_decorators(self):
        decs = _get_decorators(self._node("@dataclass\nclass C: pass"))
        self.assertEqual(decs, ['dataclass'])


# ---------------------------------------------------------------------------
# Exception extraction tests
# ---------------------------------------------------------------------------

class TestGetExceptions(unittest.TestCase):
    def _func(self, src: str):
        return ast.parse(src).body[0]

    def test_no_exceptions(self):
        exc = _get_exceptions(self._func("def f(): return 1"))
        self.assertEqual(exc, {'raises': [], 'catches': []})

    def test_simple_raise(self):
        exc = _get_exceptions(self._func("def f():\n    raise ValueError('bad')"))
        self.assertEqual(len(exc['raises']), 1)
        self.assertIn('ValueError', exc['raises'][0])

    def test_bare_raise_ignored(self):
        # `raise` with no exception (re-raise) has child.exc=None and is skipped
        src = "def f():\n    try:\n        pass\n    except:\n        raise"
        exc = _get_exceptions(self._func(src))
        self.assertEqual(exc['raises'], [])

    def test_except_handler(self):
        src = "def f():\n    try:\n        x = 1\n    except KeyError:\n        pass"
        exc = _get_exceptions(self._func(src))
        self.assertIn('KeyError', exc['catches'])

    def test_bare_except_ignored(self):
        src = "def f():\n    try:\n        x = 1\n    except:\n        pass"
        exc = _get_exceptions(self._func(src))
        self.assertEqual(exc['catches'], [])

    def test_dedup(self):
        src = (
            "def f():\n"
            "    raise ValueError('a')\n"
            "    raise ValueError('a')\n"
        )
        exc = _get_exceptions(self._func(src))
        self.assertEqual(len(exc['raises']), 1)

    def test_multiple_handlers(self):
        src = (
            "def f():\n"
            "    try:\n        x = 1\n"
            "    except KeyError:\n        pass\n"
            "    except TypeError:\n        pass"
        )
        exc = _get_exceptions(self._func(src))
        self.assertIn('KeyError', exc['catches'])
        self.assertIn('TypeError', exc['catches'])


# ---------------------------------------------------------------------------
# Class attribute extraction tests
# ---------------------------------------------------------------------------

class TestGetClassAttributes(unittest.TestCase):
    def _cls(self, src: str):
        return ast.parse(src).body[0]

    def test_init_assignments(self):
        src = (
            "class C:\n"
            "    def __init__(self):\n"
            "        self.x = 1\n"
            "        self.y = 2\n"
        )
        attrs = _get_class_attributes(self._cls(src))
        self.assertEqual(attrs, ['x', 'y'])

    def test_annotated_assignment(self):
        src = (
            "class C:\n"
            "    def __init__(self):\n"
            "        self.x: int = 0\n"
        )
        self.assertEqual(_get_class_attributes(self._cls(src)), ['x'])

    def test_other_methods_ignored(self):
        # Only __init__ is scanned per the docstring contract
        src = (
            "class C:\n"
            "    def __init__(self):\n        self.x = 1\n"
            "    def setup(self):\n        self.y = 2\n"
        )
        attrs = _get_class_attributes(self._cls(src))
        self.assertIn('x', attrs)
        self.assertNotIn('y', attrs)

    def test_no_init(self):
        src = "class C:\n    pass"
        self.assertEqual(_get_class_attributes(self._cls(src)), [])

    def test_dedup_preserves_order(self):
        src = (
            "class C:\n"
            "    def __init__(self):\n"
            "        self.x = 1\n"
            "        self.y = 2\n"
            "        self.x = 3\n"
        )
        self.assertEqual(_get_class_attributes(self._cls(src)), ['x', 'y'])


# ---------------------------------------------------------------------------
# Assert pattern extraction tests
# ---------------------------------------------------------------------------

class TestGetAssertPatterns(unittest.TestCase):
    def _func(self, src: str):
        return ast.parse(src).body[0]

    def test_assert_statement(self):
        patterns = _get_assert_patterns(self._func("def t():\n    assert x == 1"))
        self.assertEqual(len(patterns), 1)
        self.assertIn('x == 1', patterns[0])

    def test_unittest_style_call(self):
        src = "def t(self):\n    self.assertEqual(a, b)"
        patterns = _get_assert_patterns(self._func(src))
        self.assertTrue(any('assertEqual' in p for p in patterns))

    def test_non_assert_call_ignored(self):
        src = "def t(self):\n    self.helper(a, b)"
        patterns = _get_assert_patterns(self._func(src))
        self.assertEqual(patterns, [])

    def test_no_asserts(self):
        self.assertEqual(_get_assert_patterns(self._func("def t():\n    return 1")), [])


# ---------------------------------------------------------------------------
# Branch counting tests
# ---------------------------------------------------------------------------

class TestCountBranches(unittest.TestCase):
    def _func(self, src: str):
        return ast.parse(src).body[0]

    def test_no_branches(self):
        self.assertEqual(_count_branches(self._func("def f():\n    return 1")), 0)

    def test_if(self):
        self.assertEqual(_count_branches(self._func("def f(x):\n    if x: return")), 1)

    def test_for_and_while(self):
        src = "def f():\n    for i in range(3):\n        while True:\n            break"
        self.assertEqual(_count_branches(self._func(src)), 2)

    def test_mixed(self):
        src = (
            "def f(x):\n"
            "    if x:\n        pass\n"
            "    for i in range(3):\n        pass\n"
            "    while x:\n        x -= 1"
        )
        self.assertEqual(_count_branches(self._func(src)), 3)

    def test_excludes_nested_function_branches(self):
        # Branches inside a nested function belong to that function's complexity,
        # not the outer one's. The walker skips into FunctionDef children entirely.
        src = (
            "def f():\n"
            "    if True: pass\n"
            "    def inner():\n        if True: pass\n        if True: pass"
        )
        self.assertEqual(_count_branches(self._func(src)), 1)


# ---------------------------------------------------------------------------
# Test file detection tests
# ---------------------------------------------------------------------------

class TestIsTestFile(unittest.TestCase):
    def test_test_prefix(self):
        self.assertTrue(_is_test_file('test_foo.py'))

    def test_test_suffix(self):
        self.assertTrue(_is_test_file('foo_test.py'))

    def test_tests_dir(self):
        self.assertTrue(_is_test_file('pkg/tests/helpers.py'))

    def test_test_dir(self):
        self.assertTrue(_is_test_file('pkg/test/helpers.py'))

    def test_non_test_file(self):
        self.assertFalse(_is_test_file('pkg/utils.py'))


# ---------------------------------------------------------------------------
# _safe_unparse failure path
# ---------------------------------------------------------------------------

class TestSafeUnparse(unittest.TestCase):
    def test_returns_string_for_valid_node(self):
        node = ast.parse("x + 1").body[0].value
        self.assertEqual(_safe_unparse(node), 'x + 1')

    def test_returns_none_on_failure(self):
        # An empty Name node missing required fields makes ast.unparse raise
        broken = ast.Name()
        self.assertIsNone(_safe_unparse(broken))


# ---------------------------------------------------------------------------
# Async function handling
# ---------------------------------------------------------------------------

class TestAsyncFunctionSupport(unittest.TestCase):
    def _afunc(self, src: str):
        return ast.parse(src).body[0]

    def test_signature_on_async(self):
        sig = _get_signature(self._afunc("async def f(x: int) -> str: pass"))
        self.assertEqual(sig['params'][0]['name'], 'x')
        self.assertEqual(sig['returns'], 'str')

    def test_exceptions_on_async(self):
        src = "async def f():\n    raise ValueError('x')"
        exc = _get_exceptions(self._afunc(src))
        self.assertTrue(any('ValueError' in r for r in exc['raises']))

    def test_branches_on_async(self):
        src = "async def f(x):\n    if x:\n        return"
        self.assertEqual(_count_branches(self._afunc(src)), 1)

    def test_build_func_metadata_marks_is_async(self):
        node = self._afunc("async def f(): pass")
        meta = _build_func_metadata(node, 'mod.py', 'r/r')
        self.assertTrue(meta['is_async'])

    def test_parse_file_emits_async_node(self):
        result = _parse_source("async def go():\n    pass\n")
        node = _node_of(result, 'go')
        self.assertIsNotNone(node)
        self.assertTrue(node['metadata']['is_async'])


# ---------------------------------------------------------------------------
# _build_func_metadata tests
# ---------------------------------------------------------------------------

class TestBuildFuncMetadata(unittest.TestCase):
    def _func(self, src: str):
        return ast.parse(src).body[0]

    def test_basic_function(self):
        node = self._func("def f(x: int) -> int:\n    '''doc'''\n    return x")
        meta = _build_func_metadata(node, 'a/b.py', 'org/repo')
        self.assertEqual(meta['filepath'], 'a/b.py')
        self.assertEqual(meta['repo'], 'org/repo')
        self.assertEqual(meta['returns'], 'int')
        self.assertEqual(meta['docstring'], 'doc')
        self.assertFalse(meta['is_async'])
        self.assertNotIn('class', meta)

    def test_method_includes_class(self):
        node = self._func("def m(self): pass")
        meta = _build_func_metadata(node, 'm.py', 'r/r', parent_class='C')
        self.assertEqual(meta['class'], 'C')

    def test_assert_patterns_extracted_in_test_files(self):
        src = "def f():\n    assert x == 1"
        node = self._func(src)
        # Test file: extracted (covers test functions and their helper classes)
        meta_test_file = _build_func_metadata(node, 'tests/test_m.py', 'r/r')
        # Production file: skipped to avoid noise from invariant asserts
        meta_prod = _build_func_metadata(node, 'src/m.py', 'r/r')
        self.assertTrue(len(meta_test_file['assert_patterns']) > 0)
        self.assertEqual(meta_prod['assert_patterns'], [])

    def test_external_deps_from_annotation(self):
        node = self._func("def f(x: MyClass) -> DomNode:\n    pass")
        meta = _build_func_metadata(node, 'a.py', 'r/r',
                                     import_map={'MyClass': 'pkg.MyClass',
                                                 'DomNode': 'utils.dom.DomNode'})
        self.assertIn('pkg.MyClass', meta['external_deps'])
        self.assertIn('utils.dom.DomNode', meta['external_deps'])

    def test_external_deps_excludes_builtins(self):
        node = self._func("def f(x: int) -> str:\n    pass")
        meta = _build_func_metadata(node, 'a.py', 'r/r', import_map={})
        self.assertEqual(meta['external_deps'], [])

    def test_external_deps_from_body_instantiation(self):
        node = self._func("def f():\n    x = MyClass()\n    return x")
        meta = _build_func_metadata(node, 'a.py', 'r/r',
                                     import_map={'MyClass': 'mod.MyClass'})
        self.assertIn('mod.MyClass', meta['external_deps'])

    def test_side_effects_direct_write(self):
        node = self._func("def m(self):\n    self.x = 1")
        meta = _build_func_metadata(node, 'a.py', 'r/r')
        self.assertIn('x', meta['side_effects'])

    def test_side_effects_subscript_write(self):
        node = self._func("def m(self):\n    self.cache[k] = v")
        meta = _build_func_metadata(node, 'a.py', 'r/r')
        self.assertIn('cache', meta['side_effects'])

    def test_side_effects_empty_when_no_write(self):
        node = self._func("def m(self):\n    return self.x")
        meta = _build_func_metadata(node, 'a.py', 'r/r')
        self.assertEqual(meta['side_effects'], [])


# ---------------------------------------------------------------------------
# _collect_file_level_info tests
# ---------------------------------------------------------------------------

class TestCollectFileLevelInfo(unittest.TestCase):
    def test_plain_imports(self):
        tree = ast.parse("import os\nimport sys")
        imap, _, _ = _collect_file_level_info(tree)
        self.assertEqual(imap['os'], 'os')
        self.assertEqual(imap['sys'], 'sys')

    def test_dotted_import_uses_root_local(self):
        tree = ast.parse("import os.path")
        imap, _, _ = _collect_file_level_info(tree)
        self.assertEqual(imap['os'], 'os.path')

    def test_aliased_import(self):
        tree = ast.parse("import numpy as np")
        imap, _, _ = _collect_file_level_info(tree)
        self.assertEqual(imap['np'], 'numpy')

    def test_from_import_qualifies_with_module(self):
        tree = ast.parse("from pkg.mod import Thing")
        imap, _, _ = _collect_file_level_info(tree)
        self.assertEqual(imap['Thing'], 'pkg.mod.Thing')

    def test_exports(self):
        tree = ast.parse("__all__ = ['a', 'b']")
        _, exports, _ = _collect_file_level_info(tree)
        self.assertEqual(exports, ['a', 'b'])

    def test_uppercase_constants(self):
        tree = ast.parse("MAX = 100\nMIN: int = 0\nlower = 5")
        _, _, consts = _collect_file_level_info(tree)
        self.assertEqual(consts.get('MAX'), '100')
        self.assertEqual(consts.get('MIN'), '0')
        self.assertNotIn('lower', consts)


# ---------------------------------------------------------------------------
# Tier 1 resolution improvements: precision tests
# ---------------------------------------------------------------------------

class TestResolutionPrecision(unittest.TestCase):
    """Verify Tier 1 resolution heuristics disambiguate correctly.

    Each test constructs a scenario with two same-named methods/functions
    and asserts the resolver picks the right one based on the available hint.
    """

    def _build_kg(self, files: dict) -> dict:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for rel, src in files.items():
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(src)
            parser = RepoASTParser(max_workers=1)
            return parser.parse_repo('test/repo', root)

    def _calls_from(self, kg: dict, caller_label: str) -> list:
        caller = next(n for n in kg['nodes'] if n['label'] == caller_label)
        return [
            (e, next(n for n in kg['nodes'] if n['id'] == e['target']))
            for e in kg['edges']
            if e['relation'] == 'calls' and e['source'] == caller['id']
        ]

    def test_self_method_resolves_to_own_class(self):
        # Two classes both define save(); A.run() calls self.save() and must
        # resolve to A.save, NOT B.save.
        src = (
            "class A:\n"
            "    def run(self):\n        self.save()\n"
            "    def save(self):\n        pass\n"
            "class B:\n"
            "    def save(self):\n        pass\n"
        )
        kg = self._build_kg({'mod.py': src})
        calls = self._calls_from(kg, 'run')
        self.assertEqual(len(calls), 1, "expected exactly one resolved call")
        edge, target_node = calls[0]
        self.assertEqual(target_node['label'], 'save')
        self.assertEqual(target_node['metadata']['class'], 'A')
        self.assertEqual(edge['metadata']['confidence'], 'qualified')

    def test_local_type_hint_resolves_constructor_assignment(self):
        # x = Foo(); x.do() should resolve to Foo.do, not Bar.do
        src = (
            "class Foo:\n    def do(self): pass\n"
            "class Bar:\n    def do(self): pass\n"
            "def caller():\n"
            "    x = Foo()\n"
            "    x.do()\n"
        )
        kg = self._build_kg({'mod.py': src})
        calls = self._calls_from(kg, 'caller')
        do_targets = [t for _, t in calls if t['label'] == 'do']
        self.assertEqual(len(do_targets), 1, "expected exactly one 'do' resolution")
        self.assertEqual(do_targets[0]['metadata']['class'], 'Foo')

    def test_import_map_disambiguates_module_qualified_call(self):
        # json.loads vs pickle.loads: the import map tells us which module
        # owns the receiver. Here we put a homonym 'loads' function in a
        # local module 'json.py' so it can resolve.
        importer = (
            "import json\n"
            "def caller():\n"
            "    json.loads('{}')\n"
        )
        json_mod = "def loads(s):\n    return s\n"
        pickle_mod = "def loads(s):\n    return s\n"
        kg = self._build_kg({
            'app.py': importer,
            'json.py': json_mod,
            'pickle.py': pickle_mod,
        })
        calls = self._calls_from(kg, 'caller')
        loads_targets = [t for _, t in calls if t['label'] == 'loads']
        self.assertEqual(len(loads_targets), 1, "expected single resolution via import")
        # Confirm it picked the json.py one, not pickle.py
        self.assertIn('json', loads_targets[0]['metadata']['filepath'])

    def test_bare_call_falls_back_to_label(self):
        # When no hints apply (bare unimported call to a local function),
        # resolution should still work via the label index.
        src = (
            "def helper(): pass\n"
            "def caller():\n    helper()\n"
        )
        kg = self._build_kg({'mod.py': src})
        calls = self._calls_from(kg, 'caller')
        self.assertEqual(len(calls), 1)
        edge, target = calls[0]
        self.assertEqual(target['label'], 'helper')
        self.assertEqual(edge['metadata']['confidence'], 'exact')

    def test_qualified_confidence_label_emitted(self):
        # The new 'qualified' confidence value should appear when hints fire.
        src = (
            "class A:\n"
            "    def run(self):\n        self.go()\n"
            "    def go(self):\n        pass\n"
        )
        kg = self._build_kg({'mod.py': src})
        run_calls = [e for e in kg['edges']
                     if e['relation'] == 'calls'
                     and any(n['id'] == e['source'] and n['label'] == 'run' for n in kg['nodes'])]
        self.assertTrue(any(e['metadata']['confidence'] == 'qualified' for e in run_calls))


class TestResolutionHelpers(unittest.TestCase):
    """Unit tests for the new helper functions."""

    def test_extract_call_receiver_for_attribute(self):
        from repo_kg_builder import _extract_call_receiver
        call = ast.parse("obj.method()").body[0].value
        self.assertEqual(_extract_call_receiver(call), 'obj')

    def test_extract_call_receiver_for_dotted(self):
        from repo_kg_builder import _extract_call_receiver
        call = ast.parse("self.foo.bar()").body[0].value
        self.assertEqual(_extract_call_receiver(call), 'self.foo')

    def test_extract_call_receiver_returns_none_for_bare_call(self):
        from repo_kg_builder import _extract_call_receiver
        call = ast.parse("foo()").body[0].value
        self.assertIsNone(_extract_call_receiver(call))

    def test_collect_local_types_simple(self):
        from repo_kg_builder import _collect_local_types
        func = ast.parse(
            "def f():\n"
            "    x = Foo()\n"
            "    y = Bar()\n"
        ).body[0]
        types = _collect_local_types(func)
        self.assertEqual(types, {'x': 'Foo', 'y': 'Bar'})

    def test_collect_local_types_ignores_lowercase_call(self):
        from repo_kg_builder import _collect_local_types
        func = ast.parse(
            "def f():\n"
            "    x = helper()\n"
        ).body[0]
        self.assertEqual(_collect_local_types(func), {})

    def test_collect_local_types_attribute_constructor(self):
        from repo_kg_builder import _collect_local_types
        func = ast.parse(
            "def f():\n"
            "    x = mod.Foo()\n"
        ).body[0]
        self.assertEqual(_collect_local_types(func), {'x': 'Foo'})

    def test_collect_local_types_skips_nested_function(self):
        from repo_kg_builder import _collect_local_types
        func = ast.parse(
            "def f():\n"
            "    x = Foo()\n"
            "    def inner():\n"
            "        y = Bar()\n"
        ).body[0]
        types = _collect_local_types(func)
        self.assertIn('x', types)
        self.assertNotIn('y', types)

    def test_collect_local_types_reassignment(self):
        from repo_kg_builder import _collect_local_types
        func = ast.parse(
            "def f():\n"
            "    x = Foo()\n"
            "    x = Bar()\n"
        ).body[0]
        # Last assignment wins
        self.assertEqual(_collect_local_types(func)['x'], 'Bar')


if __name__ == '__main__':
    unittest.main()
