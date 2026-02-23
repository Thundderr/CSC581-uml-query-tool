"""
Knowledge Graph Construction for UML Class Diagrams.

This module provides classes and functions for building knowledge graphs
from detected UML class diagrams, including arrow detection, class matching,
and export to various formats.
"""

from .arrow_matching import (
    ArrowEndpoints,
    ARROW_ID,
    CLASS_BOX_ID,
    estimate_arrow_endpoints,
    find_arrow_endpoints_from_pixels,
    distance_to_bbox_edge,
    distance_to_bbox_center,
    bbox_to_bbox_distance,
    determine_direction,
    find_closest_class,
    detect_arrows,
    match_arrows_to_classes,
    infer_relationship_type,
    extract_relationships_from_image,
)

from .builder import UMLKnowledgeGraphBuilder

from .exporters import (
    export_to_json,
    load_from_json,
    export_to_graphml,
    export_statistics,
    export_by_image,
    export_to_rdf,
    RDF_AVAILABLE,
)

from .neo4j_store import (
    Neo4jConfig,
    get_config,
    get_driver,
    ensure_schema,
    graph_exists,
    list_graphs,
    upsert_graph,
    load_graph,
)

__all__ = [
    # Arrow matching
    'ArrowEndpoints',
    'ARROW_ID',
    'CLASS_BOX_ID',
    'estimate_arrow_endpoints',
    'find_arrow_endpoints_from_pixels',
    'distance_to_bbox_edge',
    'distance_to_bbox_center',
    'bbox_to_bbox_distance',
    'determine_direction',
    'find_closest_class',
    'detect_arrows',
    'match_arrows_to_classes',
    'infer_relationship_type',
    'extract_relationships_from_image',
    # Builder
    'UMLKnowledgeGraphBuilder',
    # Exporters
    'export_to_json',
    'load_from_json',
    'export_to_graphml',
    'export_statistics',
    'export_by_image',
    'export_to_rdf',
    'RDF_AVAILABLE',
    # Neo4j
    'Neo4jConfig',
    'get_config',
    'get_driver',
    'ensure_schema',
    'graph_exists',
    'list_graphs',
    'upsert_graph',
    'load_graph',
]
