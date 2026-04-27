"""
kg_to_rdf.py

Convert a KG JSON file (produced by repo_kg_builder.py) into an RDF graph
and serialize it to Turtle.

Mapping:
  - Each node becomes an RDF resource at kg:<id>
  - Node 'type' becomes rdf:type kg:<Type> (e.g. kg:Method, kg:TestFunction)
  - Node 'label' -> rdfs:label
  - Scalar metadata fields (lineno, filepath, returns, is_async, branches,
    docstring, class) become datatype properties under the kg: namespace
  - List metadata (decorators, raises, catches, assert_patterns) become
    repeated literals
  - Params become blank nodes with kg:paramName / kg:paramAnnotation /
    kg:paramDefault
  - Each edge becomes a triple: <source> kg:<relation> <target>
  - Edge metadata (confidence, import_resolved) is attached via an
    rdf:Statement reification when present

Usage:
    python kg_to_rdf.py kg_output/kg_psf_requests.json
    python kg_to_rdf.py kg_output/kg_psf_requests.json --out kg_output/kg_psf_requests.ttl
"""

import argparse
import json
from pathlib import Path

from rdflib import Graph, Literal, Namespace, URIRef, BNode
from rdflib.namespace import RDF, RDFS, XSD


KG = Namespace("https://example.org/kg/")
NODE = Namespace("https://example.org/kg/node/")


# Map node type strings to camel-cased RDF class names.
TYPE_TO_CLASS = {
    "file": "File",
    "test_file": "TestFile",
    "class": "Class",
    "function": "Function",
    "test_function": "TestFunction",
    "method": "Method",
    "import": "Import",
}


def _node_uri(node_id: str) -> URIRef:
    return NODE[node_id]


def _add_params(g: Graph, subject: URIRef, params: list) -> None:
    for p in params:
        bn = BNode()
        g.add((subject, KG.param, bn))
        if p.get("name") is not None:
            g.add((bn, KG.paramName, Literal(p["name"])))
        if p.get("annotation") is not None:
            g.add((bn, KG.paramAnnotation, Literal(p["annotation"])))
        if p.get("default") is not None:
            g.add((bn, KG.paramDefault, Literal(p["default"])))


def _add_node(g: Graph, node: dict) -> None:
    uri = _node_uri(node["id"])
    cls_name = TYPE_TO_CLASS.get(node["type"], node["type"].title())
    g.add((uri, RDF.type, KG[cls_name]))
    g.add((uri, RDFS.label, Literal(node.get("label", ""))))

    md = node.get("metadata") or {}

    # Scalars
    for key in ("filepath", "repo", "returns", "docstring", "class"):
        val = md.get(key)
        if val is not None:
            g.add((uri, KG[key if key != "class" else "className"], Literal(val)))

    if md.get("lineno") is not None:
        g.add((uri, KG.lineno, Literal(md["lineno"], datatype=XSD.integer)))
    if md.get("branches") is not None:
        g.add((uri, KG.branches, Literal(md["branches"], datatype=XSD.integer)))
    if md.get("is_async") is not None:
        g.add((uri, KG.isAsync, Literal(bool(md["is_async"]), datatype=XSD.boolean)))

    # Lists of strings
    for key, prop in (
        ("decorators", "decorator"),
        ("raises", "raises"),
        ("catches", "catches"),
        ("assert_patterns", "assertPattern"),
    ):
        for item in md.get(key, []) or []:
            g.add((uri, KG[prop], Literal(item)))

    # Params (list of dicts)
    if md.get("params"):
        _add_params(g, uri, md["params"])


def _add_edge(g: Graph, edge: dict) -> None:
    s = _node_uri(edge["source"])
    o = _node_uri(edge["target"])
    p = KG[edge["relation"]]
    g.add((s, p, o))

    md = edge.get("metadata") or {}
    if md.get("confidence") or md.get("import_resolved"):
        # Reify so we can attach edge-level annotations.
        stmt = BNode()
        g.add((stmt, RDF.type, RDF.Statement))
        g.add((stmt, RDF.subject, s))
        g.add((stmt, RDF.predicate, p))
        g.add((stmt, RDF.object, o))
        if md.get("confidence"):
            g.add((stmt, KG.confidence, Literal(md["confidence"])))
        if md.get("import_resolved"):
            g.add((stmt, KG.importResolved, Literal(md["import_resolved"])))


def kg_to_rdf(kg: dict) -> Graph:
    g = Graph()
    g.bind("kg", KG)
    g.bind("node", NODE)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)

    md = kg.get("metadata") or {}
    if md.get("repo"):
        repo_uri = URIRef(f"https://github.com/{md['repo']}")
        g.add((repo_uri, RDF.type, KG.Repo))
        g.add((repo_uri, RDFS.label, Literal(md["repo"])))
        if md.get("base_commit"):
            g.add((repo_uri, KG.baseCommit, Literal(md["base_commit"])))
        if md.get("file_count") is not None:
            g.add((repo_uri, KG.fileCount, Literal(md["file_count"], datatype=XSD.integer)))
        if md.get("parse_mode"):
            g.add((repo_uri, KG.parseMode, Literal(md["parse_mode"])))

    for node in kg.get("nodes", []):
        _add_node(g, node)
    for edge in kg.get("edges", []):
        _add_edge(g, edge)
    return g


def convert(json_path: Path, out_path: Path) -> tuple[int, int]:
    kg = json.loads(json_path.read_text())
    g = kg_to_rdf(kg)
    g.serialize(destination=str(out_path), format="turtle")
    return len(kg.get("nodes", [])), len(kg.get("edges", []))


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert KG JSON to RDF Turtle.")
    ap.add_argument("json_path", type=Path, help="Path to KG JSON file.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output .ttl path (defaults to <input>.ttl).")
    args = ap.parse_args()

    out = args.out or args.json_path.with_suffix(".ttl")
    n_nodes, n_edges = convert(args.json_path, out)
    triples = sum(1 for _ in Graph().parse(str(out), format="turtle"))
    print(f"Wrote {out}  ({n_nodes} nodes, {n_edges} edges -> {triples} triples)")


if __name__ == "__main__":
    main()
