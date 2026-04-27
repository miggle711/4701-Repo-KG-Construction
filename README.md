# Repository Knowledge Graph Construction

Build structural Knowledge Graphs (KGs) from Python repository source code. Given a GitHub repo and a commit SHA, the system clones the repo, parses every `.py` file with the AST module, and emits a queryable JSON graph of nodes and edges representing the code's structure.

Designed as a foundation for test generation using SWE-bench data.

## Quick Start

```bash
pip install datasets pyvis
python3 run.py
```

`run.py` loads SWE-bench Lite, lists available repos, prompts for one, and builds its KG.

```python
from repo_kg_builder import RepoKGBuilder
from kg_query import KGQueryEngine
import json

# Build from a specific commit
builder = RepoKGBuilder()
kg = builder.build('psf/requests', 'a0df2cbb')
builder.save('psf/requests', kg)

# Load and query
with open('kg_output/kg_psf_requests.json') as f:
    kg = json.load(f)

engine = KGQueryEngine(kg)

# Find a file and explore its contents
files = engine.find_file_by_path('sessions.py')
contents = engine.get_file_contents(files[0]['id'])
print(contents['classes'], contents['functions'])

# Find what calls a function
callers = engine.find_callers(contents['functions'][0]['id'])

# Visualise a subgraph in the browser
engine.visualize([files[0]['id']], depth=2, output_path='sessions.html')
```

## Files

| File | Purpose |
|------|---------|
| `repo_kg_builder.py` | Clone repo, parse AST, emit KG nodes and edges |
| `kg_query.py` | In-memory query engine and pyvis visualisation |
| `run.py` | Interactive entry point using SWE-bench Lite |
| `test_kg_builder.py` | Unit and integration tests (no git clone needed) |

## Graph Structure

### Node types

| Type | Represents |
|------|-----------|
| `file` | A `.py` source file |
| `test_file` | A test file (`test_*.py` or `*_test.py`) |
| `class` | A class definition |
| `function` | A top-level function |
| `method` | A method inside a class |
| `test_function` | A `test_*` function or method |
| `import` | An imported module or name |

### Edge types

| Relation | Meaning |
|----------|---------|
| `contains` | File/class contains a class, function, or method |
| `imports` | File imports a module or name |
| `calls` | Function calls another function (confidence: `exact`/`ambiguous`) |
| `inherits` | Class inherits from another class |
| `tests` | Test function targets a specific function |
| `uses` | Class instantiates another class |
| `overrides` | Method overrides a parent class method |
| `depends_on` | Function uses a specific import |
| `module_depends_on` | File depends on another file via imports |

### Node metadata

Every function/method node carries: parameter list with defaults and annotations, return type annotation, decorators, docstring, raised and caught exceptions, branch count, and whether it is async. Test functions additionally store assert patterns. Class nodes include base classes, decorators, docstring, and class-level attributes. File nodes include module constants and `__all__` exports.

## Query Engine

```python
engine = KGQueryEngine(kg)

# Node accessors
engine.get_files()                          # all file/test_file nodes
engine.get_functions()                      # all function/method/test_function nodes

# Structural queries
engine.get_file_contents(file_id)           # {file, classes, functions}
engine.get_class_methods(class_id)          # list of method nodes

# Call graph
engine.find_callers(func_id)                # functions that call this one
engine.find_callees(func_id)                # functions this one calls
engine.find_test_functions_for(func_id)     # test functions covering this function

# Search
engine.find_file_by_path('sessions.py')     # substring match on path
engine.find_function_by_name('send')        # exact label match

# Export
engine.export_subgraph([node_id, ...])      # nodes + 1-hop edges as dict

# Visualise (requires pyvis)
engine.visualize([node_id], depth=2, output_path='graph.html')
```

## Running Tests

```bash
python3 test_kg_builder.py -v
```

Tests run entirely on synthetic Python source — no git clone or network access required. 35 tests covering all helper functions and edge types end-to-end through `parse_repo`.

## Installation

```bash
pip install datasets   # for run.py / SWE-bench data
pip install pyvis      # for visualize()
```

No other dependencies beyond the Python standard library.

## See Also

- [SWE-bench](https://github.com/princeton-nlp/SWE-bench) — the benchmark dataset used as input
