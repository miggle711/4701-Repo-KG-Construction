"""
kg_query.py

Query engine for structural knowledge graphs produced by repo_kg_builder.py.

Loads a KG dict and builds four in-memory indices for O(1) lookups:
    - nodes_by_id:       {node_id: node}
    - nodes_by_type:     {type: [node, ...]}
    - edges_by_source:   {source_id: [edge, ...]}
    - edges_by_target:   {target_id: [edge, ...]}
    - edges_by_relation: {relation: [edge, ...]}

Example:
    from kg_query import KGQueryEngine
    import json

    with open('kg_output/kg_psf_requests.json') as f:
        kg = json.load(f)

    engine = KGQueryEngine(kg)
    files = engine.find_file_by_path('sessions.py')
    methods = engine.get_class_methods(class_id)
    callers = engine.find_callers(func_id)
"""

from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional


class KGQueryEngine:
    """In-memory query interface for a structural knowledge graph.

    All query methods operate on pre-built indices and run in O(1) or O(k)
    time where k is the number of results, not the total graph size.
    """

    def __init__(self, kg_data: Dict):
        """Load a KG dict and build lookup indices.

        Args:
            kg_data: KG dict as produced by RepoKGBuilder.build() or loaded
                     from a saved JSON file. Expected keys: 'nodes', 'edges', 'metadata'.
        """
        self.kg = kg_data
        self._build_indices()

    def _build_indices(self):
        """Build all lookup indices from the raw node and edge lists.

        Called once at init. After this, all query methods use the indices
        rather than scanning the full node/edge lists.
        """
        self.nodes_by_id: Dict[str, Dict] = {}
        self.nodes_by_type: Dict[str, List[Dict]] = defaultdict(list)
        self.edges_by_source: Dict[str, List[Dict]] = defaultdict(list)
        self.edges_by_target: Dict[str, List[Dict]] = defaultdict(list)
        self.edges_by_relation: Dict[str, List[Dict]] = defaultdict(list)

        for node in self.kg['nodes']:
            self.nodes_by_id[node['id']] = node
            self.nodes_by_type[node['type']].append(node)

        for edge in self.kg['edges']:
            self.edges_by_source[edge['source']].append(edge)
            self.edges_by_target[edge['target']].append(edge)
            self.edges_by_relation[edge['relation']].append(edge)

    # ------------------------------------------------------------------
    # Node accessors
    # ------------------------------------------------------------------

    def get_files(self) -> List[Dict]:
        """Return all file and test_file nodes in the KG."""
        return self.nodes_by_type.get('file', []) + self.nodes_by_type.get('test_file', [])

    def get_functions(self) -> List[Dict]:
        """Return all function, method, and test_function nodes in the KG."""
        return (self.nodes_by_type.get('function', []) +
                self.nodes_by_type.get('method', []) +
                self.nodes_by_type.get('test_function', []))

    # ------------------------------------------------------------------
    # Structural queries
    # ------------------------------------------------------------------

    def get_file_contents(self, file_id: str) -> Dict:
        """Return the classes and top-level functions defined in a file.

        Args:
            file_id: Node ID of the file node.

        Returns:
            Dict with keys:
                file: the file node
                classes: list of class nodes directly contained in the file
                functions: list of function/test_function nodes directly in the file
            Returns {} if file_id is not found.
        """
        file_node = self.nodes_by_id.get(file_id)
        if not file_node:
            return {}

        classes = []
        functions = []
        for edge in self.edges_by_source.get(file_id, []):
            if edge['relation'] != 'contains':
                continue
            node = self.nodes_by_id.get(edge['target'])
            if not node:
                continue
            if node['type'] == 'class':
                classes.append(node)
            elif node['type'] in ('function', 'test_function'):
                functions.append(node)

        return {'file': file_node, 'classes': classes, 'functions': functions}

    def get_class_methods(self, class_id: str) -> List[Dict]:
        """Return all method and test_function nodes belonging to a class.

        Args:
            class_id: Node ID of the class node.

        Returns:
            List of method/test_function nodes contained by the class.
        """
        return [
            self.nodes_by_id[e['target']]
            for e in self.edges_by_source.get(class_id, [])
            if e['relation'] == 'contains' and e['target'] in self.nodes_by_id
        ]

    # ------------------------------------------------------------------
    # Call graph queries
    # ------------------------------------------------------------------

    def find_callers(self, func_id: str) -> List[Dict]:
        """Return all functions/methods that call a given function.

        Only returns nodes that exist in the KG (external callers are excluded).
        Confidence of the underlying call edge is not filtered — both 'exact'
        and 'ambiguous' matches are included.

        Args:
            func_id: Node ID of the target function.

        Returns:
            List of function/method/test_function nodes that call this function.
        """
        return [
            self.nodes_by_id[e['source']]
            for e in self.edges_by_target.get(func_id, [])
            if e['relation'] == 'calls' and e['source'] in self.nodes_by_id
        ]

    def find_callees(self, func_id: str) -> List[Dict]:
        """Return all functions/methods called by a given function.

        Args:
            func_id: Node ID of the calling function.

        Returns:
            List of function/method/test_function nodes called by this function.
        """
        return [
            self.nodes_by_id[e['target']]
            for e in self.edges_by_source.get(func_id, [])
            if e['relation'] == 'calls' and e['target'] in self.nodes_by_id
        ]

    def find_test_functions_for(self, func_id: str) -> List[Dict]:
        """Return test functions that call a given function.

        Filters find_callers to only test_function nodes. Useful for finding
        existing test coverage for a function when generating new tests.

        Args:
            func_id: Node ID of the function under test.

        Returns:
            List of test_function nodes that call this function.
        """
        return [
            caller for caller in self.find_callers(func_id)
            if caller['type'] == 'test_function'
        ]

    # ------------------------------------------------------------------
    # Search queries
    # ------------------------------------------------------------------

    def find_file_by_path(self, path: str, exact: bool = False) -> List[Dict]:
        """Find file nodes by path substring or exact match.

        Args:
            path: Path string to search for (e.g. 'sessions.py' or
                  'requests/sessions.py').
            exact: If True, match the full path exactly. If False (default),
                   match any file whose path contains the given string.

        Returns:
            List of matching file/test_file nodes.
        """
        results = []
        for node in self.get_files():
            file_path = node['metadata'].get('path', '')
            if exact and file_path == path:
                results.append(node)
            elif not exact and path in file_path:
                results.append(node)
        return results

    def find_function_by_name(self, name: str) -> List[Dict]:
        """Find all function, method, and test_function nodes with a given name.

        Args:
            name: Exact function name to search for (e.g. 'send', '__init__').

        Returns:
            List of matching nodes. Multiple results are expected for common
            names like '__init__' or 'get' that appear in many classes.
        """
        return [n for n in self.get_functions() if n['label'] == name]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_subgraph(self, node_ids: List[str]) -> Dict:
        """Export a subgraph containing the given nodes and their immediate neighbors.

        For each node in node_ids, all outgoing edges and their target nodes
        are included. This gives one hop of context around each seed node,
        useful for feeding a focused slice of the KG to a test generator.

        Args:
            node_ids: List of node IDs to use as seeds.

        Returns:
            Dict with 'nodes' and 'edges' keys, ready for JSON serialization
            or passing to another KGQueryEngine instance.
        """
        nodes = []
        edges = []
        seen_nodes = set()

        for node_id in node_ids:
            if node_id not in self.nodes_by_id:
                continue
            if node_id not in seen_nodes:
                nodes.append(self.nodes_by_id[node_id])
                seen_nodes.add(node_id)
            for edge in self.edges_by_source.get(node_id, []):
                target_id = edge['target']
                if target_id not in seen_nodes and target_id in self.nodes_by_id:
                    nodes.append(self.nodes_by_id[target_id])
                    seen_nodes.add(target_id)
                edges.append(edge)

        return {'nodes': nodes, 'edges': edges}

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    # Colours per node type — consistent across renders
    _NODE_COLOURS = {
        'file':          '#4A90D9',
        'test_file':     '#7B68EE',
        'class':         '#E67E22',
        'function':      '#27AE60',
        'method':        '#2ECC71',
        'test_function': '#E74C3C',
        'import':        '#95A5A6',
    }

    def visualize(
        self,
        seed_ids: List[str],
        depth: int = 2,
        output_path: str = 'kg_vis.html',
        max_nodes: int = 200,
    ) -> str:
        """Render a subgraph around seed nodes as an interactive HTML file.

        Performs a BFS from each seed node up to `depth` hops, collecting
        nodes and edges. Renders with pyvis as a force-directed graph saved
        to an HTML file you can open in any browser.

        Node colours indicate type; hovering a node shows its full metadata.
        Edge labels show the relation type.

        Args:
            seed_ids: Node IDs to start the BFS from. Use find_file_by_path()
                      or find_function_by_name() to get IDs.
            depth: Number of hops to expand from each seed. depth=1 gives
                   immediate neighbours only; depth=2 gives neighbours of
                   neighbours. Keep low for dense graphs.
            output_path: Path to write the HTML file to.
            max_nodes: Stop expanding once this many nodes are collected,
                       to prevent unrenderable hairballs on dense graphs.

        Returns:
            Absolute path to the generated HTML file.

        Example:
            files = engine.find_file_by_path('sessions.py')
            engine.visualize([files[0]['id']], depth=2, output_path='sessions.html')
        """
        try:
            from pyvis.network import Network
        except ImportError:
            raise ImportError("pyvis is required for visualisation: pip install pyvis")

        # BFS to collect nodes and edges up to `depth` hops
        visited_nodes: set = set()
        visited_edges: set = set()
        subgraph_nodes: List[Dict] = []
        subgraph_edges: List[Dict] = []

        frontier = [(nid, 0) for nid in seed_ids if nid in self.nodes_by_id]
        while frontier and len(visited_nodes) < max_nodes:
            node_id, current_depth = frontier.pop(0)
            if node_id in visited_nodes:
                continue
            visited_nodes.add(node_id)
            subgraph_nodes.append(self.nodes_by_id[node_id])

            if current_depth >= depth:
                continue

            for edge in self.edges_by_source.get(node_id, []):
                target_id = edge['target']
                edge_key = (edge['source'], edge['target'], edge['relation'])
                if edge_key not in visited_edges and target_id in self.nodes_by_id:
                    visited_edges.add(edge_key)
                    subgraph_edges.append(edge)
                    if target_id not in visited_nodes:
                        frontier.append((target_id, current_depth + 1))

        net = Network(height='900px', width='100%', directed=True, bgcolor='#1a1a2e',
                      font_color='white')
        net.barnes_hut(spring_length=120, spring_strength=0.04, damping=0.09)

        for node in subgraph_nodes:
            colour = self._NODE_COLOURS.get(node['type'], '#BDC3C7')
            # Build tooltip from metadata — skip None values and long lists
            meta = node['metadata']
            tooltip_lines = [f"<b>{node['label']}</b>", f"type: {node['type']}"]
            for k, v in meta.items():
                if v is None or v == [] or v == {}:
                    continue
                if isinstance(v, list):
                    v = ', '.join(str(i) for i in v[:5])
                    if len(meta.get(k, [])) > 5:
                        v += '...'
                tooltip_lines.append(f"{k}: {v}")
            net.add_node(
                node['id'],
                label=node['label'],
                title='<br>'.join(tooltip_lines),
                color=colour,
                shape='dot' if node['type'] in ('function', 'method', 'test_function') else 'box',
                size=20 if node['id'] in seed_ids else 12,
            )

        for edge in subgraph_edges:
            if edge['source'] in visited_nodes and edge['target'] in visited_nodes:
                net.add_edge(
                    edge['source'],
                    edge['target'],
                    label=edge['relation'],
                    color={'color': '#666688', 'highlight': '#ffffff'},
                    font={'color': '#aaaacc', 'size': 9},
                )

        output_path = str(Path(output_path).resolve())
        net.save_graph(output_path)
        print(f"Visualisation saved: {output_path} ({len(subgraph_nodes)} nodes, {len(subgraph_edges)} edges)")
        return output_path
