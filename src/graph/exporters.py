"""
Export functions for UML knowledge graphs.

This module provides functions to export NetworkX graphs to various
formats including JSON (for Notebook 05), GraphML (for visualization tools),
and optionally RDF/OWL for semantic queries.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx


def export_to_json(graph: nx.DiGraph,
                   output_path: Path,
                   metadata: Optional[Dict] = None,
                   indent: int = 2) -> None:
    """
    Export knowledge graph to JSON format.

    This format is optimized for consumption by Notebook 05 (GraphRAG queries).

    Args:
        graph: NetworkX DiGraph to export
        output_path: Path to output JSON file
        metadata: Optional additional metadata to include
        indent: JSON indentation level
    """
    # Collect source images from nodes
    source_images = list(set(
        data.get('source_image', '')
        for _, data in graph.nodes(data=True)
        if data.get('source_image')
    ))

    data = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'node_count': graph.number_of_nodes(),
            'edge_count': graph.number_of_edges(),
            'source_images': source_images,
            **(metadata or {})
        },
        'nodes': [],
        'edges': []
    }

    # Export nodes
    for node_id, attrs in graph.nodes(data=True):
        node_data = {'id': node_id}
        # Convert all attributes, handling non-serializable types
        for key, value in attrs.items():
            node_data[key] = _make_serializable(value)
        data['nodes'].append(node_data)

    # Export edges
    for source, target, attrs in graph.edges(data=True):
        edge_data = {
            'source': source,
            'target': target
        }
        for key, value in attrs.items():
            edge_data[key] = _make_serializable(value)
        data['edges'].append(edge_data)

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)

    print(f"Exported graph to: {output_path}")


def load_from_json(input_path: Path) -> nx.DiGraph:
    """
    Load a knowledge graph from JSON format.

    Args:
        input_path: Path to JSON file

    Returns:
        NetworkX DiGraph reconstructed from JSON
    """
    with open(input_path, 'r') as f:
        data = json.load(f)

    graph = nx.DiGraph()

    # Load nodes
    for node in data.get('nodes', []):
        node_id = node.pop('id')
        graph.add_node(node_id, **node)

    # Load edges
    for edge in data.get('edges', []):
        source = edge.pop('source')
        target = edge.pop('target')
        graph.add_edge(source, target, **edge)

    return graph


def export_to_graphml(graph: nx.DiGraph, output_path: Path) -> None:
    """
    Export knowledge graph to GraphML format.

    GraphML is compatible with visualization tools like Gephi, yEd, and Cytoscape.

    Note: Complex attributes (lists, dicts) are converted to JSON strings
    since GraphML only supports scalar types.

    Args:
        graph: NetworkX DiGraph to export
        output_path: Path to output GraphML file
    """
    # Create a copy to modify for export
    export_graph = graph.copy()

    # Convert complex attributes to JSON strings for GraphML compatibility
    for node in export_graph.nodes():
        attrs = export_graph.nodes[node]
        for key, value in list(attrs.items()):
            if isinstance(value, (list, dict)):
                attrs[key] = json.dumps(value)

    for u, v in export_graph.edges():
        attrs = export_graph.edges[u, v]
        for key, value in list(attrs.items()):
            if isinstance(value, (list, dict)):
                attrs[key] = json.dumps(value)

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nx.write_graphml(export_graph, str(output_path))
    print(f"Exported GraphML to: {output_path}")


def export_statistics(stats: Dict[str, Any], output_path: Path) -> None:
    """
    Export graph statistics to JSON.

    Args:
        stats: Statistics dictionary from UMLKnowledgeGraphBuilder
        output_path: Path to output JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"Exported statistics to: {output_path}")


def export_by_image(graph: nx.DiGraph,
                    output_dir: Path,
                    indent: int = 2) -> Dict[str, Path]:
    """
    Export separate JSON files for each source image.

    Useful for analyzing individual UML diagrams.

    Args:
        graph: NetworkX DiGraph to export
        output_dir: Directory to save per-image JSON files
        indent: JSON indentation level

    Returns:
        Dictionary mapping image paths to output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group nodes by source image
    images = {}
    for node_id, data in graph.nodes(data=True):
        source_image = data.get('source_image', 'unknown')
        if source_image not in images:
            images[source_image] = []
        images[source_image].append(node_id)

    output_files = {}

    for image_path, node_ids in images.items():
        # Create subgraph for this image
        subgraph = graph.subgraph(node_ids).copy()

        # Generate output filename from image name
        if image_path and image_path != 'unknown':
            img_name = Path(image_path).stem
        else:
            img_name = 'unknown'

        # Sanitize filename
        safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in img_name)
        output_path = output_dir / f"{safe_name}_graph.json"

        # Export subgraph
        export_to_json(
            subgraph,
            output_path,
            metadata={'source_image': image_path},
            indent=indent
        )

        output_files[image_path] = output_path

    print(f"Exported {len(output_files)} per-image graphs to: {output_dir}")
    return output_files


def _make_serializable(value: Any) -> Any:
    """
    Convert a value to a JSON-serializable type.

    Args:
        value: Any Python value

    Returns:
        JSON-serializable version of the value
    """
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    elif isinstance(value, (list, tuple)):
        return [_make_serializable(v) for v in value]
    elif isinstance(value, dict):
        return {str(k): _make_serializable(v) for k, v in value.items()}
    else:
        return str(value)


# Optional RDF export (only if rdflib is available)
try:
    from rdflib import Graph as RDFGraph, Namespace, Literal, URIRef
    from rdflib.namespace import RDF, RDFS, OWL
    RDF_AVAILABLE = True
except ImportError:
    RDF_AVAILABLE = False


def export_to_rdf(graph: nx.DiGraph, output_path: Path) -> bool:
    """
    Export knowledge graph to RDF/OWL format.

    Enables SPARQL queries and semantic reasoning.

    Args:
        graph: NetworkX DiGraph to export
        output_path: Path to output RDF file

    Returns:
        True if export succeeded, False if rdflib is not available
    """
    if not RDF_AVAILABLE:
        print("rdflib not available, skipping RDF export")
        print("Install with: pip install rdflib")
        return False

    # Define namespace
    UML = Namespace("http://example.org/uml#")

    rdf_graph = RDFGraph()
    rdf_graph.bind("uml", UML)

    # Add nodes as classes
    for node_id, attrs in graph.nodes(data=True):
        node_uri = URIRef(UML[node_id])
        rdf_graph.add((node_uri, RDF.type, UML.Class))
        rdf_graph.add((node_uri, RDFS.label, Literal(attrs.get('name', ''))))

        # Add attributes
        for i, attr in enumerate(attrs.get('attributes', [])):
            attr_uri = URIRef(UML[f"{node_id}_attr_{i}"])
            rdf_graph.add((node_uri, UML.hasAttribute, attr_uri))
            rdf_graph.add((attr_uri, UML.attributeName, Literal(attr.get('name', ''))))
            rdf_graph.add((attr_uri, UML.attributeType, Literal(attr.get('type', ''))))
            rdf_graph.add((attr_uri, UML.visibility, Literal(attr.get('visibility', ''))))

        # Add methods
        for i, method in enumerate(attrs.get('methods', [])):
            method_uri = URIRef(UML[f"{node_id}_method_{i}"])
            rdf_graph.add((node_uri, UML.hasMethod, method_uri))
            rdf_graph.add((method_uri, UML.methodName, Literal(method.get('name', ''))))
            rdf_graph.add((method_uri, UML.returnType, Literal(method.get('return_type', ''))))
            rdf_graph.add((method_uri, UML.visibility, Literal(method.get('visibility', ''))))

    # Add edges as relationships
    for source, target, attrs in graph.edges(data=True):
        source_uri = URIRef(UML[source])
        target_uri = URIRef(UML[target])
        rel_type = attrs.get('relationship_type', 'association')

        # Map UML relationships to RDF predicates
        if rel_type == 'inheritance':
            rdf_graph.add((source_uri, RDFS.subClassOf, target_uri))
        elif rel_type == 'composition':
            rdf_graph.add((source_uri, UML.composedOf, target_uri))
        elif rel_type == 'aggregation':
            rdf_graph.add((source_uri, UML.aggregates, target_uri))
        else:
            rdf_graph.add((source_uri, UML.associatedWith, target_uri))

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rdf_graph.serialize(destination=str(output_path), format='xml')
    print(f"Exported RDF to: {output_path}")
    return True
