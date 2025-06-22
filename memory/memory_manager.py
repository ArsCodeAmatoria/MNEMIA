"""
MNEMIA Memory Manager Service
Coordinates vector memory (Qdrant), graph memory (Neo4j), and caching (Redis)
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Database clients
from qdrant_client import QdrantClient
from qdrant_client.http import models
from neo4j import GraphDatabase
import redis.asyncio as redis
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="MNEMIA Memory Manager",
    description="Consciousness memory coordination service",
    version="1.0.0"
)

# Pydantic models
class MemoryInput(BaseModel):
    content: str
    context: str
    memory_type: str
    emotional_state: Dict
    modal_state: str
    confidence: float = 1.0
    metadata: Optional[Dict] = None

class MemoryQuery(BaseModel):
    query: str
    memory_types: Optional[List[str]] = None
    limit: int = 10
    threshold: float = 0.3
    include_graph: bool = True

class MemoryResponse(BaseModel):
    memories: List[Dict]
    total_found: int
    processing_time: float
    sources: List[str]

@dataclass
class DatabaseConnections:
    qdrant: QdrantClient
    neo4j: Any  # Neo4j driver
    redis: Any  # Redis client
    encoder: SentenceTransformer

class MemoryManager:
    """Core memory management service"""
    
    def __init__(self):
        self.connections: Optional[DatabaseConnections] = None
        self.collection_name = "mnemia_memories"
        
    async def initialize(self):
        """Initialize database connections"""
        try:
            # Initialize Qdrant
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            qdrant_client = QdrantClient(url=qdrant_url)
            
            # Initialize Neo4j
            neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            neo4j_user = os.getenv("NEO4J_USER", "neo4j")
            neo4j_password = os.getenv("NEO4J_PASSWORD", "mnemia123")
            neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            
            # Initialize Redis
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            redis_client = redis.from_url(redis_url, password="mnemia_redis_pass", decode_responses=True)
            
            # Initialize sentence transformer
            encoder = SentenceTransformer('all-MiniLM-L6-v2')
            
            self.connections = DatabaseConnections(
                qdrant=qdrant_client,
                neo4j=neo4j_driver,
                redis=redis_client,
                encoder=encoder
            )
            
            # Initialize collections and schemas
            await self._setup_qdrant_collection()
            await self._verify_neo4j_schema()
            
            logger.info("Memory Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Memory Manager: {e}")
            raise
    
    async def _setup_qdrant_collection(self):
        """Setup Qdrant collection for memories"""
        try:
            collections = self.connections.qdrant.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)
            
            if not collection_exists:
                self.connections.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=384,  # Sentence transformer dimension
                        distance=models.Distance.COSINE
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        m=16,
                        ef_construct=100,
                        full_scan_threshold=10000
                    ),
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            quantile=0.99,
                            always_ram=True
                        )
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error setting up Qdrant collection: {e}")
    
    async def _verify_neo4j_schema(self):
        """Verify Neo4j schema is properly initialized"""
        try:
            with self.connections.neo4j.session() as session:
                result = session.run("SHOW CONSTRAINTS")
                constraints = [record["name"] for record in result]
                
                expected_constraints = ["memory_id", "concept_id", "emotion_name", "modal_state_name"]
                missing_constraints = [c for c in expected_constraints if c not in constraints]
                
                if missing_constraints:
                    logger.warning(f"Missing Neo4j constraints: {missing_constraints}")
                else:
                    logger.info("Neo4j schema verification complete")
                    
        except Exception as e:
            logger.error(f"Error verifying Neo4j schema: {e}")
    
    async def store_memory(self, memory_input: MemoryInput) -> str:
        """Store memory across all systems"""
        try:
            start_time = asyncio.get_event_loop().time()
            memory_id = f"mem_{int(datetime.now().timestamp() * 1000)}"
            
            # Generate embedding
            content_for_embedding = f"{memory_input.content} {memory_input.context}"
            embedding = self.connections.encoder.encode(content_for_embedding).tolist()
            
            # Store in Qdrant (vector search)
            await self._store_in_qdrant(memory_id, memory_input, embedding)
            
            # Store in Neo4j (graph relationships)
            await self._store_in_neo4j(memory_id, memory_input)
            
            # Cache recent memory in Redis
            await self._cache_memory(memory_id, memory_input)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"Stored memory {memory_id} in {processing_time:.3f}s")
            
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            raise
    
    async def _store_in_qdrant(self, memory_id: str, memory_input: MemoryInput, embedding: List[float]):
        """Store memory in Qdrant vector database"""
        payload = {
            "content": memory_input.content,
            "context": memory_input.context,
            "memory_type": memory_input.memory_type,
            "emotional_state": memory_input.emotional_state,
            "modal_state": memory_input.modal_state,
            "confidence": memory_input.confidence,
            "timestamp": datetime.now().isoformat(),
            "metadata": memory_input.metadata or {}
        }
        
        self.connections.qdrant.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=memory_id,
                    vector=embedding,
                    payload=payload
                )
            ]
        )
    
    async def _store_in_neo4j(self, memory_id: str, memory_input: MemoryInput):
        """Store memory in Neo4j graph database"""
        
        def store_memory_tx(tx, memory_id, memory_input):
            # Create memory node
            query = """
            CREATE (m:Memory {
                id: $memory_id,
                content: $content,
                context: $context,
                type: $memory_type,
                confidence: $confidence,
                timestamp: datetime(),
                source: 'memory_manager'
            })
            """
            tx.run(query, 
                memory_id=memory_id,
                content=memory_input.content,
                context=memory_input.context,
                memory_type=memory_input.memory_type,
                confidence=memory_input.confidence
            )
            
            # Link to modal state
            if memory_input.modal_state:
                modal_query = """
                MATCH (m:Memory {id: $memory_id})
                MATCH (ms:ModalState {name: $modal_state})
                MERGE (m)-[:OCCURRED_IN]->(ms)
                """
                tx.run(modal_query, memory_id=memory_id, modal_state=memory_input.modal_state)
            
            # Link to emotions
            if memory_input.emotional_state.get("primary_emotions"):
                for emotion in memory_input.emotional_state["primary_emotions"][:3]:
                    emotion_query = """
                    MATCH (m:Memory {id: $memory_id})
                    MATCH (e:Emotion {name: $emotion})
                    MERGE (m)-[:FELT_EMOTION {intensity: $intensity}]->(e)
                    """
                    intensity = memory_input.emotional_state.get("emotional_intensity", 0.5)
                    tx.run(emotion_query, memory_id=memory_id, emotion=emotion, intensity=intensity)
            
            # Link to MNEMIA
            mnemia_query = """
            MATCH (m:Memory {id: $memory_id})
            MATCH (mnemia:Person {id: 'mnemia'})
            MERGE (mnemia)-[:HAS_MEMORY]->(m)
            """
            tx.run(mnemia_query, memory_id=memory_id)
        
        with self.connections.neo4j.session() as session:
            session.execute_write(store_memory_tx, memory_id, memory_input)
    
    async def _cache_memory(self, memory_id: str, memory_input: MemoryInput):
        """Cache recent memory in Redis"""
        cache_data = {
            "content": memory_input.content,
            "context": memory_input.context,
            "memory_type": memory_input.memory_type,
            "modal_state": memory_input.modal_state,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in recent memories list
        await self.connections.redis.lpush("recent_memories", json.dumps(cache_data))
        await self.connections.redis.ltrim("recent_memories", 0, 99)  # Keep last 100
        
        # Store individual memory with expiration
        await self.connections.redis.setex(f"memory:{memory_id}", 3600, json.dumps(cache_data))
    
    async def query_memories(self, query: MemoryQuery) -> MemoryResponse:
        """Query memories across all systems"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Vector search in Qdrant
            vector_memories = await self._query_qdrant(query)
            
            # Graph search in Neo4j (if requested)
            graph_memories = []
            if query.include_graph:
                graph_memories = await self._query_neo4j(query)
            
            # Merge and deduplicate results
            all_memories = self._merge_memory_results(vector_memories, graph_memories)
            
            # Sort by relevance/confidence
            all_memories.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return MemoryResponse(
                memories=all_memories[:query.limit],
                total_found=len(all_memories),
                processing_time=processing_time,
                sources=["qdrant"] + (["neo4j"] if query.include_graph else [])
            )
            
        except Exception as e:
            logger.error(f"Error querying memories: {e}")
            raise
    
    async def _query_qdrant(self, query: MemoryQuery) -> List[Dict]:
        """Query Qdrant vector database"""
        query_embedding = self.connections.encoder.encode(query.query).tolist()
        
        search_filter = None
        if query.memory_types:
            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="memory_type",
                        match=models.MatchAny(any=query.memory_types)
                    )
                ]
            )
        
        search_result = self.connections.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=query.limit * 2,  # Get more for filtering
            score_threshold=query.threshold,
            with_payload=True
        )
        
        memories = []
        for hit in search_result:
            memory_data = {
                "id": hit.id,
                "score": hit.score,
                "source": "qdrant",
                **hit.payload
            }
            memories.append(memory_data)
        
        return memories
    
    async def _query_neo4j(self, query: MemoryQuery) -> List[Dict]:
        """Query Neo4j graph database"""
        
        def query_memories_tx(tx, query_text):
            # Use fulltext search
            cypher_query = """
            CALL db.index.fulltext.queryNodes('memory_content', $query_text)
            YIELD node, score
            WHERE score > 0.5
            MATCH (mnemia:Person {id: 'mnemia'})-[:HAS_MEMORY]->(node)
            OPTIONAL MATCH (node)-[:OCCURRED_IN]->(ms:ModalState)
            OPTIONAL MATCH (node)-[:FELT_EMOTION]->(e:Emotion)
            RETURN node, score, ms.name as modal_state, collect(e.name) as emotions
            ORDER BY score DESC
            LIMIT $limit
            """
            
            result = tx.run(cypher_query, query_text=query_text, limit=query.limit)
            
            memories = []
            for record in result:
                node = record["node"]
                memory_data = {
                    "id": node["id"],
                    "content": node["content"],
                    "context": node.get("context", ""),
                    "memory_type": node["type"],
                    "confidence": node["confidence"],
                    "timestamp": node["timestamp"].isoformat() if node.get("timestamp") else None,
                    "modal_state": record["modal_state"],
                    "emotions": record["emotions"],
                    "score": record["score"],
                    "source": "neo4j"
                }
                memories.append(memory_data)
            
            return memories
        
        with self.connections.neo4j.session() as session:
            return session.execute_read(query_memories_tx, query.query)
    
    def _merge_memory_results(self, vector_memories: List[Dict], graph_memories: List[Dict]) -> List[Dict]:
        """Merge and deduplicate memory results from different sources"""
        memory_map = {}
        
        # Add vector memories
        for memory in vector_memories:
            memory_id = memory["id"]
            memory_map[memory_id] = memory
        
        # Add graph memories (prefer graph data for relationship info)
        for memory in graph_memories:
            memory_id = memory["id"]
            if memory_id in memory_map:
                # Merge data, preferring graph relationships
                memory_map[memory_id].update({
                    "emotions": memory.get("emotions", []),
                    "graph_score": memory.get("score", 0)
                })
            else:
                memory_map[memory_id] = memory
        
        return list(memory_map.values())
    
    async def get_stats(self) -> Dict:
        """Get memory system statistics"""
        try:
            # Qdrant stats
            collection_info = self.connections.qdrant.get_collection(self.collection_name)
            
            # Neo4j stats
            with self.connections.neo4j.session() as session:
                neo4j_result = session.run("MATCH (m:Memory) RETURN count(m) as memory_count")
                neo4j_memory_count = neo4j_result.single()["memory_count"]
            
            # Redis stats
            redis_info = await self.connections.redis.info("memory")
            
            return {
                "qdrant": {
                    "total_vectors": collection_info.points_count,
                    "vector_size": collection_info.config.params.vectors.size,
                    "status": collection_info.status
                },
                "neo4j": {
                    "total_memories": neo4j_memory_count,
                    "status": "connected"
                },
                "redis": {
                    "used_memory": redis_info.get("used_memory_human", "unknown"),
                    "connected_clients": redis_info.get("connected_clients", 0),
                    "status": "connected"
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}

# Global memory manager instance
memory_manager = MemoryManager()

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize memory manager on startup"""
    await memory_manager.initialize()

@app.post("/memories", response_model=str)
async def store_memory(memory_input: MemoryInput):
    """Store a new memory"""
    return await memory_manager.store_memory(memory_input)

@app.post("/memories/query", response_model=MemoryResponse)
async def query_memories(query: MemoryQuery):
    """Query memories"""
    return await memory_manager.query_memories(query)

@app.get("/stats")
async def get_stats():
    """Get memory system statistics"""
    return await memory_manager.get_stats()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "MNEMIA Memory Manager",
        "connections": {
            "qdrant": "connected" if memory_manager.connections else "disconnected",
            "neo4j": "connected" if memory_manager.connections else "disconnected",
            "redis": "connected" if memory_manager.connections else "disconnected"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "memory_manager:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
        log_level="info"
    ) 