"""
MNEMIA Vector Memory Store

Interface for storing and retrieving semantic embeddings using Qdrant.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    VectorParams, Distance, PointStruct, Filter, 
    FieldCondition, MatchValue, Range
)

logger = logging.getLogger(__name__)

class VectorMemoryStore:
    """Manages vector embeddings for semantic memory."""
    
    def __init__(self, host: str = "localhost", port: int = 6333):
        """Initialize connection to Qdrant."""
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = "mnemia_memories"
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure the memories collection exists."""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,  # Sentence transformer embedding size
                        distance=Distance.COSINE
                    )
                )
                logger.info("Collection created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise
    
    def store_memory(
        self, 
        content: str, 
        embedding: List[float], 
        memory_type: str = "episodic",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a memory with its vector embedding.
        
        Args:
            content: The text content of the memory
            embedding: Vector embedding of the content
            memory_type: Type of memory (episodic, semantic, procedural)
            metadata: Additional metadata
            
        Returns:
            The ID of the stored memory
        """
        try:
            memory_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat()
            
            payload = {
                "content": content,
                "memory_type": memory_type,
                "timestamp": timestamp,
                "id": memory_id,
                **(metadata or {})
            }
            
            point = PointStruct(
                id=memory_id,
                vector=embedding,
                payload=payload
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"Stored memory: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            raise
    
    def search_memories(
        self, 
        query_embedding: List[float], 
        limit: int = 10,
        memory_type: Optional[str] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar memories using vector similarity.
        
        Args:
            query_embedding: Vector to search for
            limit: Maximum number of results
            memory_type: Filter by memory type
            min_score: Minimum similarity score
            
        Returns:
            List of matching memories with scores
        """
        try:
            search_filter = None
            if memory_type:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="memory_type",
                            match=MatchValue(value=memory_type)
                        )
                    ]
                )
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=limit,
                score_threshold=min_score
            )
            
            memories = []
            for result in results:
                memory = {
                    "id": result.id,
                    "score": result.score,
                    **result.payload
                }
                memories.append(memory)
            
            logger.info(f"Found {len(memories)} memories")
            return memories
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            raise
    
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific memory by ID."""
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id]
            )
            
            if result:
                memory = {
                    "id": result[0].id,
                    **result[0].payload
                }
                return memory
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id}: {e}")
            return None
    
    def update_memory_metadata(
        self, 
        memory_id: str, 
        metadata: Dict[str, Any]
    ) -> bool:
        """Update metadata for an existing memory."""
        try:
            # Get current memory
            current = self.get_memory(memory_id)
            if not current:
                return False
            
            # Update payload
            updated_payload = {**current, **metadata}
            updated_payload["updated_at"] = datetime.utcnow().isoformat()
            
            # Get the current vector (we need it for upserting)
            point_data = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id],
                with_vectors=True
            )[0]
            
            # Update the point
            updated_point = PointStruct(
                id=memory_id,
                vector=point_data.vector,
                payload=updated_payload
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[updated_point]
            )
            
            logger.info(f"Updated memory metadata: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating memory {memory_id}: {e}")
            return False
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[memory_id]
                )
            )
            logger.info(f"Deleted memory: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            # Get memory type distribution
            type_counts = {}
            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000  # Adjust based on your needs
            )[0]
            
            for point in results:
                memory_type = point.payload.get("memory_type", "unknown")
                type_counts[memory_type] = type_counts.get(memory_type, 0) + 1
            
            return {
                "total_memories": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.value,
                "memory_types": type_counts,
                "status": collection_info.status.value
            }
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {}
    
    def find_similar_memories(
        self, 
        memory_id: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find memories similar to a given memory."""
        try:
            # Get the vector for the given memory
            point_data = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id],
                with_vectors=True
            )
            
            if not point_data:
                return []
            
            query_vector = point_data[0].vector
            
            # Search for similar memories (excluding the original)
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit + 1  # +1 because the original will be included
            )
            
            # Filter out the original memory
            similar_memories = []
            for result in results:
                if result.id != memory_id:
                    memory = {
                        "id": result.id,
                        "score": result.score,
                        **result.payload
                    }
                    similar_memories.append(memory)
            
            return similar_memories[:limit]
            
        except Exception as e:
            logger.error(f"Error finding similar memories: {e}")
            return []
    
    def cleanup_old_memories(self, days_threshold: int = 30) -> int:
        """Clean up old memories based on age."""
        try:
            cutoff_date = datetime.utcnow().timestamp() - (days_threshold * 24 * 3600)
            
            # Note: This is a simplified cleanup
            # In a real implementation, you'd want more sophisticated retention policies
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="timestamp",
                            range=Range(lt=cutoff_date)
                        )
                    ]
                ),
                limit=1000
            )[0]
            
            if results:
                ids_to_delete = [point.id for point in results]
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=ids_to_delete)
                )
                logger.info(f"Cleaned up {len(ids_to_delete)} old memories")
                return len(ids_to_delete)
            
            return 0
            
        except Exception as e:
            logger.error(f"Error cleaning up memories: {e}")
            return 0 