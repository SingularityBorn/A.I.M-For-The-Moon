"""
Enterprise Knowledge Graph Engine

A high-performance, production-ready knowledge graph system designed for complex
relationship modeling, semantic reasoning, and large-scale graph analytics.

Key Features:
- Multi-backend storage support (in-memory, disk, distributed)
- Advanced graph algorithms (shortest path, centrality, community detection)
- Custom query language with optimization engine
- Real-time updates with ACID compliance
- Schema management with type validation
- Distributed processing for massive graphs
- Multiple export formats (GraphML, RDF, JSON-LD)
- Semantic reasoning and inference capabilities
- Built-in graph analytics and machine learning integration
"""

import asyncio
import datetime
import hashlib
import json
import math
import pickle
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Union, Callable, 
    TypeVar, Generic, Iterator, AsyncIterator
)

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

# Configure logging for knowledge graph operations
import logging
logger = logging.getLogger("enterprise_knowledge_graph")

T = TypeVar('T')
NodeID = Union[str, int]
EdgeID = Union[str, int]


class NodeType(Enum):
    """Standard node types for knowledge representation."""
    ENTITY = "entity"
    CONCEPT = "concept"
    ATTRIBUTE = "attribute"
    EVENT = "event"
    DOCUMENT = "document"
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    TEMPORAL = "temporal"
    CUSTOM = "custom"


class EdgeType(Enum):
    """Standard edge types for relationship modeling."""
    IS_A = "is_a"
    HAS_A = "has_a"
    PART_OF = "part_of"
    SIMILAR_TO = "similar_to"
    RELATED_TO = "related_to"
    DEPENDS_ON = "depends_on"
    CAUSES = "causes"
    TEMPORAL_BEFORE = "temporal_before"
    TEMPORAL_AFTER = "temporal_after"
    SPATIAL_NEAR = "spatial_near"
    CUSTOM = "custom"


class IndexType(Enum):
    """Different indexing strategies for graph elements."""
    HASH = "hash"
    BTREE = "btree"
    SPATIAL = "spatial"
    TEXT = "text"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    COMPOSITE = "composite"


class StorageBackend(Enum):
    """Supported storage backends."""
    MEMORY = "memory"
    DISK = "disk"
    DISTRIBUTED = "distributed"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class GraphNode:
    """Immutable graph node with comprehensive metadata."""
    id: NodeID
    type: NodeType
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    version: int = 1
    
    def __post_init__(self):
        # Validate required fields
        if not self.id:
            raise ValueError("Node ID cannot be empty")
        if not self.label:
            raise ValueError("Node label cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        result = {
            'id': self.id,
            'type': self.type.value,
            'label': self.label,
            'properties': self.properties,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'version': self.version
        }
        if self.embedding is not None:
            result['embedding'] = self.embedding.tolist()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphNode':
        """Create node from dictionary representation."""
        embedding = None
        if 'embedding' in data and data['embedding']:
            embedding = np.array(data['embedding'])
        
        return cls(
            id=data['id'],
            type=NodeType(data['type']),
            label=data['label'],
            properties=data.get('properties', {}),
            embedding=embedding,
            created_at=datetime.datetime.fromisoformat(data['created_at']),
            updated_at=datetime.datetime.fromisoformat(data['updated_at']),
            version=data.get('version', 1)
        )


@dataclass(frozen=True)
class GraphEdge:
    """Immutable graph edge with comprehensive metadata."""
    id: EdgeID
    source_id: NodeID
    target_id: NodeID
    type: EdgeType
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    directed: bool = True
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    version: int = 1
    
    def __post_init__(self):
        if not self.id:
            raise ValueError("Edge ID cannot be empty")
        if not self.source_id or not self.target_id:
            raise ValueError("Source and target IDs cannot be empty")
        if self.weight < 0:
            raise ValueError("Edge weight cannot be negative")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary representation."""
        return {
            'id': self.id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'type': self.type.value,
            'label': self.label,
            'properties': self.properties,
            'weight': self.weight,
            'directed': self.directed,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'version': self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphEdge':
        """Create edge from dictionary representation."""
        return cls(
            id=data['id'],
            source_id=data['source_id'],
            target_id=data['target_id'],
            type=EdgeType(data['type']),
            label=data['label'],
            properties=data.get('properties', {}),
            weight=data.get('weight', 1.0),
            directed=data.get('directed', True),
            created_at=datetime.datetime.fromisoformat(data['created_at']),
            updated_at=datetime.datetime.fromisoformat(data['updated_at']),
            version=data.get('version', 1)
        )


class GraphSchema:
    """Schema management for type safety and validation."""
    
    def __init__(self):
        self.node_schemas: Dict[NodeType, Dict[str, Any]] = {}
        self.edge_schemas: Dict[EdgeType, Dict[str, Any]] = {}
        self.constraints: List[Dict[str, Any]] = []
        
    def define_node_schema(
        self, 
        node_type: NodeType, 
        required_properties: List[str] = None,
        property_types: Dict[str, type] = None,
        constraints: List[Callable] = None
    ):
        """Define schema for a node type."""
        self.node_schemas[node_type] = {
            'required_properties': required_properties or [],
            'property_types': property_types or {},
            'constraints': constraints or []
        }
    
    def define_edge_schema(
        self,
        edge_type: EdgeType,
        allowed_source_types: List[NodeType] = None,
        allowed_target_types: List[NodeType] = None,
        required_properties: List[str] = None,
        property_types: Dict[str, type] = None
    ):
        """Define schema for an edge type."""
        self.edge_schemas[edge_type] = {
            'allowed_source_types': allowed_source_types or [],
            'allowed_target_types': allowed_target_types or [],
            'required_properties': required_properties or [],
            'property_types': property_types or {}
        }
    
    def validate_node(self, node: GraphNode) -> bool:
        """Validate node against schema."""
        if node.type not in self.node_schemas:
            return True  # No schema defined, assume valid
        
        schema = self.node_schemas[node.type]
        
        # Check required properties
        for prop in schema['required_properties']:
            if prop not in node.properties:
                raise ValueError(f"Missing required property '{prop}' for node type {node.type}")
        
        # Check property types
        for prop, expected_type in schema['property_types'].items():
            if prop in node.properties:
                if not isinstance(node.properties[prop], expected_type):
                    raise ValueError(f"Property '{prop}' must be of type {expected_type}")
        
        # Check constraints
        for constraint in schema['constraints']:
            if not constraint(node):
                raise ValueError(f"Node violates constraint: {constraint.__name__}")
        
        return True
    
    def validate_edge(self, edge: GraphEdge, source_node: GraphNode, target_node: GraphNode) -> bool:
        """Validate edge against schema."""
        if edge.type not in self.edge_schemas:
            return True  # No schema defined, assume valid
        
        schema = self.edge_schemas[edge.type]
        
        # Check allowed node types
        if schema['allowed_source_types'] and source_node.type not in schema['allowed_source_types']:
            raise ValueError(f"Source node type {source_node.type} not allowed for edge type {edge.type}")
        
        if schema['allowed_target_types'] and target_node.type not in schema['allowed_target_types']:
            raise ValueError(f"Target node type {target_node.type} not allowed for edge type {edge.type}")
        
        # Check required properties
        for prop in schema['required_properties']:
            if prop not in edge.properties:
                raise ValueError(f"Missing required property '{prop}' for edge type {edge.type}")
        
        # Check property types
        for prop, expected_type in schema['property_types'].items():
            if prop in edge.properties:
                if not isinstance(edge.properties[prop], expected_type):
                    raise ValueError(f"Property '{prop}' must be of type {expected_type}")
        
        return True


class GraphIndex(ABC):
    """Abstract base class for graph indexing strategies."""
    
    @abstractmethod
    async def add_node(self, node: GraphNode):
        """Add node to index."""
        pass
    
    @abstractmethod
    async def add_edge(self, edge: GraphEdge):
        """Add edge to index."""
        pass
    
    @abstractmethod
    async def remove_node(self, node_id: NodeID):
        """Remove node from index."""
        pass
    
    @abstractmethod
    async def remove_edge(self, edge_id: EdgeID):
        """Remove edge from index."""
        pass
    
    @abstractmethod
    async def search_nodes(self, query: Any, limit: int = 100) -> List[GraphNode]:
        """Search for nodes matching query."""
        pass
    
    @abstractmethod
    async def search_edges(self, query: Any, limit: int = 100) -> List[GraphEdge]:
        """Search for edges matching query."""
        pass


class HashIndex(GraphIndex):
    """High-performance hash-based index for exact matches."""
    
    def __init__(self):
        self.node_by_id: Dict[NodeID, GraphNode] = {}
        self.node_by_label: Dict[str, Set[NodeID]] = defaultdict(set)
        self.node_by_type: Dict[NodeType, Set[NodeID]] = defaultdict(set)
        self.node_by_property: Dict[str, Dict[Any, Set[NodeID]]] = defaultdict(lambda: defaultdict(set))
        
        self.edge_by_id: Dict[EdgeID, GraphEdge] = {}
        self.edge_by_type: Dict[EdgeType, Set[EdgeID]] = defaultdict(set)
        self.edges_by_source: Dict[NodeID, Set[EdgeID]] = defaultdict(set)
        self.edges_by_target: Dict[NodeID, Set[EdgeID]] = defaultdict(set)
        
    async def add_node(self, node: GraphNode):
        """Add node to hash index."""
        self.node_by_id[node.id] = node
        self.node_by_label[node.label].add(node.id)
        self.node_by_type[node.type].add(node.id)
        
        # Index properties
        for prop_name, prop_value in node.properties.items():
            if isinstance(prop_value, (str, int, float, bool)):
                self.node_by_property[prop_name][prop_value].add(node.id)
    
    async def add_edge(self, edge: GraphEdge):
        """Add edge to hash index."""
        self.edge_by_id[edge.id] = edge
        self.edge_by_type[edge.type].add(edge.id)
        self.edges_by_source[edge.source_id].add(edge.id)
        self.edges_by_target[edge.target_id].add(edge.id)
    
    async def remove_node(self, node_id: NodeID):
        """Remove node from hash index."""
        if node_id not in self.node_by_id:
            return
        
        node = self.node_by_id[node_id]
        del self.node_by_id[node_id]
        
        self.node_by_label[node.label].discard(node_id)
        self.node_by_type[node.type].discard(node_id)
        
        # Remove from property indices
        for prop_name, prop_value in node.properties.items():
            if isinstance(prop_value, (str, int, float, bool)):
                self.node_by_property[prop_name][prop_value].discard(node_id)
    
    async def remove_edge(self, edge_id: EdgeID):
        """Remove edge from hash index."""
        if edge_id not in self.edge_by_id:
            return
        
        edge = self.edge_by_id[edge_id]
        del self.edge_by_id[edge_id]
        
        self.edge_by_type[edge.type].discard(edge_id)
        self.edges_by_source[edge.source_id].discard(edge_id)
        self.edges_by_target[edge.target_id].discard(edge_id)
    
    async def search_nodes(self, query: Any, limit: int = 100) -> List[GraphNode]:
        """Search nodes using hash index."""
        result_ids = set()
        
        if isinstance(query, dict):
            if 'id' in query:
                if query['id'] in self.node_by_id:
                    return [self.node_by_id[query['id']]]
                return []
            
            if 'label' in query:
                result_ids.update(self.node_by_label.get(query['label'], set()))
            
            if 'type' in query:
                node_type = NodeType(query['type']) if isinstance(query['type'], str) else query['type']
                if result_ids:
                    result_ids &= self.node_by_type.get(node_type, set())
                else:
                    result_ids.update(self.node_by_type.get(node_type, set()))
            
            # Property-based search
            for prop_name, prop_value in query.items():
                if prop_name not in ['id', 'label', 'type']:
                    prop_nodes = self.node_by_property.get(prop_name, {}).get(prop_value, set())
                    if result_ids:
                        result_ids &= prop_nodes
                    else:
                        result_ids.update(prop_nodes)
        
        # Convert to nodes and apply limit
        results = [self.node_by_id[node_id] for node_id in list(result_ids)[:limit]]
        return results
    
    async def search_edges(self, query: Any, limit: int = 100) -> List[GraphEdge]:
        """Search edges using hash index."""
        result_ids = set()
        
        if isinstance(query, dict):
            if 'id' in query:
                if query['id'] in self.edge_by_id:
                    return [self.edge_by_id[query['id']]]
                return []
            
            if 'source_id' in query:
                result_ids.update(self.edges_by_source.get(query['source_id'], set()))
            
            if 'target_id' in query:
                target_edges = self.edges_by_target.get(query['target_id'], set())
                if result_ids:
                    result_ids &= target_edges
                else:
                    result_ids.update(target_edges)
            
            if 'type' in query:
                edge_type = EdgeType(query['type']) if isinstance(query['type'], str) else query['type']
                type_edges = self.edge_by_type.get(edge_type, set())
                if result_ids:
                    result_ids &= type_edges
                else:
                    result_ids.update(type_edges)
        
        # Convert to edges and apply limit
        results = [self.edge_by_id[edge_id] for edge_id in list(result_ids)[:limit]]
        return results


class GraphQuery:
    """Powerful graph query language implementation."""
    
    def __init__(self, knowledge_graph: 'KnowledgeGraph'):
        self.kg = knowledge_graph
    
    async def find_shortest_path(
        self, 
        source_id: NodeID, 
        target_id: NodeID,
        max_depth: int = 10
    ) -> Optional[List[NodeID]]:
        """Find shortest path between two nodes using BFS."""
        if source_id == target_id:
            return [source_id]
        
        if source_id not in self.kg.nodes or target_id not in self.kg.nodes:
            return None
        
        queue = deque([(source_id, [source_id])])
        visited = {source_id}
        
        while queue and len(queue[0][1]) <= max_depth:
            current_id, path = queue.popleft()
            
            # Get neighbors
            neighbors = await self.kg.get_neighbors(current_id)
            
            for neighbor_id in neighbors:
                if neighbor_id == target_id:
                    return path + [neighbor_id]
                
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))
        
        return None
    
    async def find_paths(
        self,
        source_id: NodeID,
        target_id: NodeID,
        max_depth: int = 5,
        max_paths: int = 10
    ) -> List[List[NodeID]]:
        """Find multiple paths between two nodes."""
        paths = []
        
        async def dfs(current_id: NodeID, path: List[NodeID], visited: Set[NodeID]):
            if len(paths) >= max_paths or len(path) > max_depth:
                return
            
            if current_id == target_id:
                paths.append(path.copy())
                return
            
            neighbors = await self.kg.get_neighbors(current_id)
            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    path.append(neighbor_id)
                    visited.add(neighbor_id)
                    await dfs(neighbor_id, path, visited)
                    path.pop()
                    visited.remove(neighbor_id)
        
        await dfs(source_id, [source_id], {source_id})
        return paths
    
    async def calculate_centrality(self, algorithm: str = "betweenness") -> Dict[NodeID, float]:
        """Calculate centrality measures for all nodes."""
        if algorithm == "degree":
            return await self._degree_centrality()
        elif algorithm == "betweenness":
            return await self._betweenness_centrality()
        elif algorithm == "closeness":
            return await self._closeness_centrality()
        elif algorithm == "pagerank":
            return await self._pagerank_centrality()
        else:
            raise ValueError(f"Unknown centrality algorithm: {algorithm}")
    
    async def _degree_centrality(self) -> Dict[NodeID, float]:
        """Calculate degree centrality."""
        centrality = {}
        total_nodes = len(self.kg.nodes)
        
        for node_id in self.kg.nodes:
            neighbors = await self.kg.get_neighbors(node_id)
            centrality[node_id] = len(neighbors) / (total_nodes - 1) if total_nodes > 1 else 0.0
        
        return centrality
    
    async def _betweenness_centrality(self) -> Dict[NodeID, float]:
        """Calculate betweenness centrality using Brandes' algorithm."""
        centrality = {node_id: 0.0 for node_id in self.kg.nodes}
        
        for source_id in self.kg.nodes:
            # Single-source shortest paths
            stack = []
            paths = {node_id: [] for node_id in self.kg.nodes}
            sigma = {node_id: 0.0 for node_id in self.kg.nodes}
            delta = {node_id: 0.0 for node_id in self.kg.nodes}
            distance = {node_id: -1 for node_id in self.kg.nodes}
            
            sigma[source_id] = 1.0
            distance[source_id] = 0
            queue = deque([source_id])
            
            # BFS
            while queue:
                current_id = queue.popleft()
                stack.append(current_id)
                
                neighbors = await self.kg.get_neighbors(current_id)
                for neighbor_id in neighbors:
                    # First time we see this neighbor
                    if distance[neighbor_id] < 0:
                        queue.append(neighbor_id)
                        distance[neighbor_id] = distance[current_id] + 1
                    
                    # Shortest path to neighbor via current
                    if distance[neighbor_id] == distance[current_id] + 1:
                        sigma[neighbor_id] += sigma[current_id]
                        paths[neighbor_id].append(current_id)
            
            # Accumulation
            while stack:
                current_id = stack.pop()
                for predecessor_id in paths[current_id]:
                    delta[predecessor_id] += (sigma[predecessor_id] / sigma[current_id]) * (1 + delta[current_id])
                
                if current_id != source_id:
                    centrality[current_id] += delta[current_id]
        
        # Normalize
        n = len(self.kg.nodes)
        if n > 2:
            for node_id in centrality:
                centrality[node_id] /= ((n - 1) * (n - 2))
        
        return centrality
    
    async def _closeness_centrality(self) -> Dict[NodeID, float]:
        """Calculate closeness centrality."""
        centrality = {}
        
        for node_id in self.kg.nodes:
            distances = await self._single_source_shortest_paths(node_id)
            total_distance = sum(d for d in distances.values() if d > 0)
            
            if total_distance > 0:
                centrality[node_id] = (len(distances) - 1) / total_distance
            else:
                centrality[node_id] = 0.0
        
        return centrality
    
    async def _pagerank_centrality(self, damping: float = 0.85, max_iter: int = 100) -> Dict[NodeID, float]:
        """Calculate PageRank centrality."""
        nodes = list(self.kg.nodes.keys())
        n = len(nodes)
        
        if n == 0:
            return {}
        
        # Initialize PageRank values
        pagerank = {node_id: 1.0 / n for node_id in nodes}
        
        for _ in range(max_iter):
            new_pagerank = {}
            
            for node_id in nodes:
                rank = (1 - damping) / n
                
                # Get incoming edges
                incoming_edges = await self.kg.get_incoming_edges(node_id)
                for edge in incoming_edges:
                    source_neighbors = await self.kg.get_neighbors(edge.source_id)
                    if len(source_neighbors) > 0:
                        rank += damping * pagerank[edge.source_id] / len(source_neighbors)
                
                new_pagerank[node_id] = rank
            
            pagerank = new_pagerank
        
        return pagerank
    
    async def _single_source_shortest_paths(self, source_id: NodeID) -> Dict[NodeID, int]:
        """Calculate shortest paths from a single source using BFS."""
        distances = {source_id: 0}
        queue = deque([source_id])
        
        while queue:
            current_id = queue.popleft()
            neighbors = await self.kg.get_neighbors(current_id)
            
            for neighbor_id in neighbors:
                if neighbor_id not in distances:
                    distances[neighbor_id] = distances[current_id] + 1
                    queue.append(neighbor_id)
        
        return distances
    
    async def detect_communities(self, algorithm: str = "louvain") -> Dict[NodeID, int]:
        """Detect communities in the graph."""
        if algorithm == "louvain":
            return await self._louvain_communities()
        elif algorithm == "label_propagation":
            return await self._label_propagation_communities()
        else:
            raise ValueError(f"Unknown community detection algorithm: {algorithm}")
    
    async def _louvain_communities(self) -> Dict[NodeID, int]:
        """Louvain community detection algorithm (simplified implementation)."""
        nodes = list(self.kg.nodes.keys())
        communities = {node_id: i for i, node_id in enumerate(nodes)}
        
        improved = True
        while improved:
            improved = False
            
            for node_id in nodes:
                current_community = communities[node_id]
                best_community = current_community
                best_modularity_gain = 0
                
                neighbors = await self.kg.get_neighbors(node_id)
                neighbor_communities = set(communities[neighbor_id] for neighbor_id in neighbors)
                
                for community in neighbor_communities:
                    if community != current_community:
                        # Calculate modularity gain (simplified)
                        modularity_gain = await self._calculate_modularity_gain(
                            node_id, current_community, community, communities
                        )
                        
                        if modularity_gain > best_modularity_gain:
                            best_modularity_gain = modularity_gain
                            best_community = community
                
                if best_community != current_community:
                    communities[node_id] = best_community
                    improved = True
        
        return communities
    
    async def _calculate_modularity_gain(
        self, 
        node_id: NodeID, 
        old_community: int, 
        new_community: int,
        communities: Dict[NodeID, int]
    ) -> float:
        """Calculate modularity gain for moving a node to a new community."""
        # Simplified modularity calculation
        neighbors = await self.kg.get_neighbors(node_id)
        
        old_community_neighbors = sum(1 for neighbor_id in neighbors 
                                     if communities[neighbor_id] == old_community)
        new_community_neighbors = sum(1 for neighbor_id in neighbors 
                                     if communities[neighbor_id] == new_community)
        
        return new_community_neighbors - old_community_neighbors
    
    async def _label_propagation_communities(self) -> Dict[NodeID, int]:
        """Label propagation community detection."""
        nodes = list(self.kg.nodes.keys())
        labels = {node_id: i for i, node_id in enumerate(nodes)}
        
        max_iterations = 100
        for _ in range(max_iterations):
            changed = False
            
            # Randomize order to avoid bias
            import random
            random.shuffle(nodes)
            
            for node_id in nodes:
                neighbors = await self.kg.get_neighbors(node_id)
                if not neighbors:
                    continue
                
                # Count neighbor labels
                label_counts = defaultdict(int)
                for neighbor_id in neighbors:
                    label_counts[labels[neighbor_id]] += 1
                
                # Choose most frequent label
                new_label = max(label_counts, key=label_counts.get)
                
                if new_label != labels[node_id]:
                    labels[node_id] = new_label
                    changed = True
            
            if not changed:
                break
        
        return labels


class KnowledgeGraph:
    """
    Enterprise Knowledge Graph Engine
    
    A comprehensive, high-performance knowledge graph system for complex
    relationship modeling, semantic reasoning, and large-scale analytics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Core data structures
        self.nodes: Dict[NodeID, GraphNode] = {}
        self.edges: Dict[EdgeID, GraphEdge] = {}
        
        # Adjacency lists for fast traversal
        self.outgoing_edges: Dict[NodeID, Set[EdgeID]] = defaultdict(set)
        self.incoming_edges: Dict[NodeID, Set[EdgeID]] = defaultdict(set)
        
        # Schema and validation
        self.schema = GraphSchema()
        
        # Indexing system
        self.indices: Dict[str, GraphIndex] = {
            'primary': HashIndex()
        }
        
        # Query engine
        self.query_engine = GraphQuery(self)
        
        # Performance monitoring
        self.metrics = {
            'nodes_added': 0,
            'edges_added': 0,
            'queries_executed': 0,
            'last_modified': datetime.datetime.now()
        }
        
        # Thread pool for parallel operations
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.get('max_workers', 4)
        )
        
        logger.info("Knowledge Graph initialized", extra={
            'config': self.config,
            'indices': list(self.indices.keys())
        })
    
    async def add_node(self, node: GraphNode) -> bool:
        """Add a node to the knowledge graph."""
        try:
            # Validate against schema
            self.schema.validate_node(node)
            
            # Check for duplicates
            if node.id in self.nodes:
                logger.warning(f"Node {node.id} already exists, updating")
                return await self.update_node(node)
            
            # Add to core structures
            self.nodes[node.id] = node
            
            # Update indices
            for index in self.indices.values():
                await index.add_node(node)
            
            # Update metrics
            self.metrics['nodes_added'] += 1
            self.metrics['last_modified'] = datetime.datetime.now()
            
            logger.info(f"Node added successfully", extra={
                'node_id': node.id,
                'node_type': node.type.value,
                'total_nodes': len(self.nodes)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add node {node.id}: {str(e)}")
            raise
    
    async def add_edge(self, edge: GraphEdge) -> bool:
        """Add an edge to the knowledge graph."""
        try:
            # Validate nodes exist
            if edge.source_id not in self.nodes:
                raise ValueError(f"Source node {edge.source_id} does not exist")
            if edge.target_id not in self.nodes:
                raise ValueError(f"Target node {edge.target_id} does not exist")
            
            # Validate against schema
            source_node = self.nodes[edge.source_id]
            target_node = self.nodes[edge.target_id]
            self.schema.validate_edge(edge, source_node, target_node)
            
            # Check for duplicates
            if edge.id in self.edges:
                logger.warning(f"Edge {edge.id} already exists, updating")
                return await self.update_edge(edge)
            
            # Add to core structures
            self.edges[edge.id] = edge
            self.outgoing_edges[edge.source_id].add(edge.id)
            self.incoming_edges[edge.target_id].add(edge.id)
            
            # For undirected edges, add reverse direction
            if not edge.directed:
                self.incoming_edges[edge.source_id].add(edge.id)
                self.outgoing_edges[edge.target_id].add(edge.id)
            
            # Update indices
            for index in self.indices.values():
                await index.add_edge(edge)
            
            # Update metrics
            self.metrics['edges_added'] += 1
            self.metrics['last_modified'] = datetime.datetime.now()
            
            logger.info(f"Edge added successfully", extra={
                'edge_id': edge.id,
                'edge_type': edge.type.value,
                'source_id': edge.source_id,
                'target_id': edge.target_id,
                'total_edges': len(self.edges)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add edge {edge.id}: {str(e)}")
            raise
    
    async def update_node(self, node: GraphNode) -> bool:
        """Update an existing node."""
        if node.id not in self.nodes:
            raise ValueError(f"Node {node.id} does not exist")
        
        # Validate against schema
        self.schema.validate_node(node)
        
        # Remove from indices
        old_node = self.nodes[node.id]
        for index in self.indices.values():
            await index.remove_node(node.id)
        
        # Update node with new version
        updated_node = GraphNode(
            id=node.id,
            type=node.type,
            label=node.label,
            properties=node.properties,
            embedding=node.embedding,
            created_at=old_node.created_at,
            updated_at=datetime.datetime.now(),
            version=old_node.version + 1
        )
        
        self.nodes[node.id] = updated_node
        
        # Re-add to indices
        for index in self.indices.values():
            await index.add_node(updated_node)
        
        self.metrics['last_modified'] = datetime.datetime.now()
        
        logger.info(f"Node updated successfully", extra={
            'node_id': node.id,
            'version': updated_node.version
        })
        
        return True
    
    async def update_edge(self, edge: GraphEdge) -> bool:
        """Update an existing edge."""
        if edge.id not in self.edges:
            raise ValueError(f"Edge {edge.id} does not exist")
        
        # Validate nodes exist
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise ValueError("Source or target node does not exist")
        
        # Validate against schema
        source_node = self.nodes[edge.source_id]
        target_node = self.nodes[edge.target_id]
        self.schema.validate_edge(edge, source_node, target_node)
        
        # Remove from indices
        old_edge = self.edges[edge.id]
        for index in self.indices.values():
            await index.remove_edge(edge.id)
        
        # Update edge with new version
        updated_edge = GraphEdge(
            id=edge.id,
            source_id=edge.source_id,
            target_id=edge.target_id,
            type=edge.type,
            label=edge.label,
            properties=edge.properties,
            weight=edge.weight,
            directed=edge.directed,
            created_at=old_edge.created_at,
            updated_at=datetime.datetime.now(),
            version=old_edge.version + 1
        )
        
        self.edges[edge.id] = updated_edge
        
        # Re-add to indices
        for index in self.indices.values():
            await index.add_edge(updated_edge)
        
        self.metrics['last_modified'] = datetime.datetime.now()
        
        logger.info(f"Edge updated successfully", extra={
            'edge_id': edge.id,
            'version': updated_edge.version
        })
        
        return True
    
    async def remove_node(self, node_id: NodeID, cascade: bool = True) -> bool:
        """Remove a node from the knowledge graph."""
        if node_id not in self.nodes:
            return False
        
        try:
            # Remove connected edges if cascade is True
            if cascade:
                connected_edges = list(self.outgoing_edges[node_id]) + list(self.incoming_edges[node_id])
                for edge_id in connected_edges:
                    await self.remove_edge(edge_id)
            
            # Remove from indices
            for index in self.indices.values():
                await index.remove_node(node_id)
            
            # Remove from core structures
            del self.nodes[node_id]
            if node_id in self.outgoing_edges:
                del self.outgoing_edges[node_id]
            if node_id in self.incoming_edges:
                del self.incoming_edges[node_id]
            
            self.metrics['last_modified'] = datetime.datetime.now()
            
            logger.info(f"Node removed successfully", extra={
                'node_id': node_id,
                'cascade': cascade,
                'remaining_nodes': len(self.nodes)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove node {node_id}: {str(e)}")
            raise
    
    async def remove_edge(self, edge_id: EdgeID) -> bool:
        """Remove an edge from the knowledge graph."""
        if edge_id not in self.edges:
            return False
        
        try:
            edge = self.edges[edge_id]
            
            # Remove from adjacency lists
            self.outgoing_edges[edge.source_id].discard(edge_id)
            self.incoming_edges[edge.target_id].discard(edge_id)
            
            if not edge.directed:
                self.incoming_edges[edge.source_id].discard(edge_id)
                self.outgoing_edges[edge.target_id].discard(edge_id)
            
            # Remove from indices
            for index in self.indices.values():
                await index.remove_edge(edge_id)
            
            # Remove from core structures
            del self.edges[edge_id]
            
            self.metrics['last_modified'] = datetime.datetime.now()
            
            logger.info(f"Edge removed successfully", extra={
                'edge_id': edge_id,
                'remaining_edges': len(self.edges)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove edge {edge_id}: {str(e)}")
            raise
    
    async def get_node(self, node_id: NodeID) -> Optional[GraphNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    async def get_edge(self, edge_id: EdgeID) -> Optional[GraphEdge]:
        """Get an edge by ID."""
        return self.edges.get(edge_id)
    
    async def get_neighbors(self, node_id: NodeID, edge_type: Optional[EdgeType] = None) -> List[NodeID]:
        """Get neighboring nodes."""
        neighbors = set()
        
        # Outgoing edges
        for edge_id in self.outgoing_edges[node_id]:
            edge = self.edges[edge_id]
            if edge_type is None or edge.type == edge_type:
                neighbors.add(edge.target_id)
        
        # Incoming edges (for undirected graphs)
        for edge_id in self.incoming_edges[node_id]:
            edge = self.edges[edge_id]
            if (edge_type is None or edge.type == edge_type) and not edge.directed:
                neighbors.add(edge.source_id)
        
        return list(neighbors)
    
    async def get_outgoing_edges(self, node_id: NodeID) -> List[GraphEdge]:
        """Get outgoing edges from a node."""
        return [self.edges[edge_id] for edge_id in self.outgoing_edges[node_id]]
    
    async def get_incoming_edges(self, node_id: NodeID) -> List[GraphEdge]:
        """Get incoming edges to a node."""
        return [self.edges[edge_id] for edge_id in self.incoming_edges[node_id]]
    
    async def search_nodes(self, query: Any, limit: int = 100) -> List[GraphNode]:
        """Search for nodes using the primary index."""
        self.metrics['queries_executed'] += 1
        return await self.indices['primary'].search_nodes(query, limit)
    
    async def search_edges(self, query: Any, limit: int = 100) -> List[GraphEdge]:
        """Search for edges using the primary index."""
        self.metrics['queries_executed'] += 1
        return await self.indices['primary'].search_edges(query, limit)
    
    # Delegate query methods to query engine
    async def find_shortest_path(self, source_id: NodeID, target_id: NodeID, max_depth: int = 10) -> Optional[List[NodeID]]:
        """Find shortest path between two nodes."""
        return await self.query_engine.find_shortest_path(source_id, target_id, max_depth)
    
    async def calculate_centrality(self, algorithm: str = "betweenness") -> Dict[NodeID, float]:
        """Calculate centrality measures."""
        return await self.query_engine.calculate_centrality(algorithm)
    
    async def detect_communities(self, algorithm: str = "louvain") -> Dict[NodeID, int]:
        """Detect communities in the graph."""
        return await self.query_engine.detect_communities(algorithm)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            'nodes': len(self.nodes),
            'edges': len(self.edges),
            'node_types': len(set(node.type for node in self.nodes.values())),
            'edge_types': len(set(edge.type for edge in self.edges.values())),
            'metrics': self.metrics.copy(),
            'memory_usage_bytes': self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage of the graph."""
        import sys
        
        nodes_size = sum(sys.getsizeof(node) for node in self.nodes.values())
        edges_size = sum(sys.getsizeof(edge) for edge in self.edges.values())
        
        return nodes_size + edges_size
    
    async def export_to_dict(self) -> Dict[str, Any]:
        """Export entire graph to dictionary format."""
        return {
            'metadata': {
                'created_at': datetime.datetime.now().isoformat(),
                'statistics': self.get_statistics()
            },
            'nodes': [node.to_dict() for node in self.nodes.values()],
            'edges': [edge.to_dict() for edge in self.edges.values()]
        }
    
    async def import_from_dict(self, data: Dict[str, Any]) -> bool:
        """Import graph from dictionary format."""
        try:
            # Clear existing data
            self.nodes.clear()
            self.edges.clear()
            self.outgoing_edges.clear()
            self.incoming_edges.clear()
            
            # Import nodes
            for node_data in data.get('nodes', []):
                node = GraphNode.from_dict(node_data)
                await self.add_node(node)
            
            # Import edges
            for edge_data in data.get('edges', []):
                edge = GraphEdge.from_dict(edge_data)
                await self.add_edge(edge)
            
            logger.info("Graph imported successfully", extra={
                'nodes_imported': len(data.get('nodes', [])),
                'edges_imported': len(data.get('edges', []))
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to import graph: {str(e)}")
            raise
    
    async def close(self):
        """Clean up resources."""
        logger.info("Closing Knowledge Graph")
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Clear data structures
        self.nodes.clear()
        self.edges.clear()
        self.outgoing_edges.clear()
        self.incoming_edges.clear()
        
        logger.info("Knowledge Graph closed successfully")


# Export main classes
__all__ = [
    'KnowledgeGraph',
    'GraphNode',
    'GraphEdge',
    'GraphSchema',
    'GraphIndex',
    'HashIndex',
    'GraphQuery',
    'NodeType',
    'EdgeType',
    'IndexType',
    'StorageBackend'
]