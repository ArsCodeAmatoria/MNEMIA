#!/usr/bin/env python3
"""
MNEMIA Advanced Memory-Guided Intelligence System
Sophisticated integration of vector memory (Qdrant), graph relations (Neo4j),
automatic storage, and modal state-aware smart retrieval
"""

import asyncio
import logging
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

# Database clients
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import (
    PointStruct, Filter, FieldCondition, MatchValue, Range,
    VectorParams, Distance, HnswConfigDiff, ScalarQuantization,
    ScalarQuantizationConfig, ScalarType
)
from neo4j import GraphDatabase
import redis.asyncio as redis

# ML and embeddings
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# FastAPI and Pydantic
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn

logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Types of memories in MNEMIA's consciousness"""
    EPISODIC = "episodic"           # Personal experiences and conversations
    SEMANTIC = "semantic"           # Facts and conceptual knowledge
    PROCEDURAL = "procedural"       # How-to knowledge and skills
    EMOTIONAL = "emotional"         # Emotionally significant experiences
    REFLECTIVE = "reflective"       # Self-awareness and introspection
    CREATIVE = "creative"           # Imaginative and artistic content
    PHILOSOPHICAL = "philosophical"  # Deep thoughts and existential insights

class ModalStateInfluence(Enum):
    """How modal states influence memory retrieval"""
    AWAKE = "awake"                 # Balanced, analytical retrieval
    DREAMING = "dreaming"           # Associative, creative connections
    REFLECTING = "reflecting"       # Introspective, pattern-seeking
    LEARNING = "learning"           # Curious, knowledge-building
    CONTEMPLATING = "contemplating" # Deep, philosophical insights
    CONFUSED = "confused"           # Clarifying, uncertainty-reducing

@dataclass
class MemoryVector:
    """Enhanced memory representation with vector and metadata"""
    id: str
    content: str
    embedding: List[float]
    memory_type: MemoryType
    timestamp: datetime
    emotional_valence: float = 0.0
    emotional_arousal: float = 0.5
    emotional_dominance: float = 0.5
    confidence: float = 1.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    modal_state_origin: Optional[str] = None
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GraphRelation:
    """Represents conceptual relationships between memories"""
    source_id: str
    target_id: str
    relation_type: str
    strength: float
    created_at: datetime
    modal_state_context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryRetrievalContext:
    """Context for memory retrieval operations"""
    query: str
    modal_state: str
    emotional_context: Dict[str, Any]
    conversation_history: List[Dict] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    temporal_context: Dict[str, Any] = field(default_factory=dict)
    retrieval_strategy: str = "balanced"  # balanced, semantic, emotional, temporal

@dataclass
class MemoryRetrievalResult:
    """Result of memory retrieval with comprehensive metadata"""
    memories: List[MemoryVector]
    graph_connections: List[GraphRelation]
    retrieval_scores: List[float]
    semantic_similarity: List[float]
    emotional_relevance: List[float]
    temporal_relevance: List[float]
    modal_state_alignment: List[float]
    total_searched: int
    retrieval_time: float
    strategy_used: str
    confidence: float

class AdvancedMemoryManager:
    """Advanced Memory-Guided Intelligence System for MNEMIA"""
    
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        neo4j_uri: str = "bolt://localhost:7687",
        redis_url: str = "redis://localhost:6379",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        # Database connections
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=("neo4j", "mnemia123"))
        self.redis_client = None  # Initialized async
        
        # ML models
        self.sentence_encoder = SentenceTransformer(embedding_model)
        self.nlp = spacy.load("en_core_web_sm")
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Configuration
        self.vector_collection = "mnemia_memories"
        self.graph_collection = "mnemia_concepts"
        self.embedding_dim = 384
        self.max_memory_age_days = 365
        self.memory_cache_size = 1000
        
        # Modal state configurations
        self.modal_state_configs = self._initialize_modal_state_configs()
        
        # Performance tracking
        self.performance_stats = {
            "total_memories": 0,
            "total_retrievals": 0,
            "avg_retrieval_time": 0.0,
            "cache_hits": 0,
            "graph_queries": 0
        }
        
        # Memory cache
        self.memory_cache = {}
        self.conversation_contexts = {}
    
    def _initialize_modal_state_configs(self) -> Dict[str, Dict]:
        """Initialize modal state influence configurations"""
        return {
            "awake": {
                "semantic_weight": 0.7,
                "emotional_weight": 0.5,
                "temporal_weight": 0.6,
                "graph_weight": 0.6,
                "creativity_boost": 0.0,
                "introspection_boost": 0.0,
                "memory_types": [MemoryType.SEMANTIC, MemoryType.EPISODIC, MemoryType.PROCEDURAL]
            },
            "dreaming": {
                "semantic_weight": 0.4,
                "emotional_weight": 0.8,
                "temporal_weight": 0.3,
                "graph_weight": 0.9,
                "creativity_boost": 0.7,
                "introspection_boost": 0.2,
                "memory_types": [MemoryType.CREATIVE, MemoryType.EMOTIONAL, MemoryType.EPISODIC]
            },
            "reflecting": {
                "semantic_weight": 0.6,
                "emotional_weight": 0.4,
                "temporal_weight": 0.8,
                "graph_weight": 0.7,
                "creativity_boost": 0.2,
                "introspection_boost": 0.9,
                "memory_types": [MemoryType.REFLECTIVE, MemoryType.PHILOSOPHICAL, MemoryType.EPISODIC]
            },
            "learning": {
                "semantic_weight": 0.8,
                "emotional_weight": 0.3,
                "temporal_weight": 0.5,
                "graph_weight": 0.8,
                "creativity_boost": 0.4,
                "introspection_boost": 0.6,
                "memory_types": [MemoryType.SEMANTIC, MemoryType.PROCEDURAL, MemoryType.EPISODIC]
            },
            "contemplating": {
                "semantic_weight": 0.5,
                "emotional_weight": 0.6,
                "temporal_weight": 0.4,
                "graph_weight": 0.8,
                "creativity_boost": 0.5,
                "introspection_boost": 0.8,
                "memory_types": [MemoryType.PHILOSOPHICAL, MemoryType.REFLECTIVE, MemoryType.SEMANTIC]
            },
            "confused": {
                "semantic_weight": 0.9,
                "emotional_weight": 0.7,
                "temporal_weight": 0.9,
                "graph_weight": 0.5,
                "creativity_boost": 0.1,
                "introspection_boost": 0.7,
                "memory_types": [MemoryType.SEMANTIC, MemoryType.EPISODIC, MemoryType.PROCEDURAL]
            }
        }
    
    async def initialize(self):
        """Initialize all database connections and collections"""
        try:
            # Initialize Redis
            self.redis_client = redis.from_url(
                "redis://localhost:6379", 
                password="mnemia_redis_pass", 
                decode_responses=True
            )
            
            # Setup Qdrant collection
            await self._setup_vector_collection()
            
            # Setup Neo4j schema
            await self._setup_graph_schema()
            
            # Initialize TF-IDF if needed
            await self._initialize_tfidf()
            
            logger.info("Advanced Memory Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Advanced Memory Manager: {e}")
            raise
    
    async def _setup_vector_collection(self):
        """Setup enhanced Qdrant collection with optimizations"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_exists = any(col.name == self.vector_collection for col in collections.collections)
            
            if not collection_exists:
                self.qdrant_client.create_collection(
                    collection_name=self.vector_collection,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    ),
                    hnsw_config=HnswConfigDiff(
                        m=32,                    # Higher connectivity for better recall
                        ef_construct=200,        # Higher for better quality
                        full_scan_threshold=20000,
                        max_indexing_threads=4
                    ),
                    quantization_config=ScalarQuantization(
                        scalar=ScalarQuantizationConfig(
                            type=ScalarType.INT8,
                            quantile=0.99,
                            always_ram=True
                        )
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        default_segment_number=4,
                        max_segment_size=200000,
                        memmap_threshold=50000,
                        indexing_threshold=20000,
                        flush_interval_sec=10,
                        max_optimization_threads=2
                    )
                )
                
                # Create indexes for efficient filtering
                self.qdrant_client.create_payload_index(
                    collection_name=self.vector_collection,
                    field_name="memory_type",
                    field_schema=models.KeywordIndexParams(
                        type="keyword",
                        is_tenant=False
                    )
                )
                
                self.qdrant_client.create_payload_index(
                    collection_name=self.vector_collection,
                    field_name="modal_state_origin",
                    field_schema=models.KeywordIndexParams(
                        type="keyword",
                        is_tenant=False
                    )
                )
                
                self.qdrant_client.create_payload_index(
                    collection_name=self.vector_collection,
                    field_name="timestamp",
                    field_schema=models.IntegerIndexParams(
                        type="integer",
                        range=True,
                        lookup=True
                    )
                )
                
                logger.info(f"Created enhanced Qdrant collection: {self.vector_collection}")
                
        except Exception as e:
            logger.error(f"Error setting up vector collection: {e}")
            raise
    
    async def _setup_graph_schema(self):
        """Setup comprehensive Neo4j schema for conceptual relationships"""
        try:
            with self.neo4j_driver.session() as session:
                # Create constraints and indexes
                constraints = [
                    "CREATE CONSTRAINT memory_id_unique IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
                    "CREATE CONSTRAINT concept_id_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
                    "CREATE CONSTRAINT emotion_name_unique IF NOT EXISTS FOR (e:Emotion) REQUIRE e.name IS UNIQUE",
                    "CREATE CONSTRAINT modal_state_name_unique IF NOT EXISTS FOR (ms:ModalState) REQUIRE ms.name IS UNIQUE",
                    "CREATE CONSTRAINT conversation_id_unique IF NOT EXISTS FOR (conv:Conversation) REQUIRE conv.id IS UNIQUE"
                ]
                
                for constraint in constraints:
                    try:
                        session.run(constraint)
                    except Exception as e:
                        if "already exists" not in str(e).lower():
                            logger.warning(f"Constraint creation warning: {e}")
                
                # Create indexes for performance
                indexes = [
                    "CREATE INDEX memory_timestamp IF NOT EXISTS FOR (m:Memory) ON (m.timestamp)",
                    "CREATE INDEX memory_type IF NOT EXISTS FOR (m:Memory) ON (m.memory_type)",
                    "CREATE INDEX concept_category IF NOT EXISTS FOR (c:Concept) ON (c.category)",
                    "CREATE INDEX relation_strength IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.strength)",
                    "CREATE INDEX emotional_valence IF NOT EXISTS FOR ()-[r:HAS_EMOTION]-() ON (r.valence)"
                ]
                
                for index in indexes:
                    try:
                        session.run(index)
                    except Exception as e:
                        if "already exists" not in str(e).lower():
                            logger.warning(f"Index creation warning: {e}")
                
                # Initialize core modal state nodes
                modal_states = ["awake", "dreaming", "reflecting", "learning", "contemplating", "confused"]
                for state in modal_states:
                    session.run(
                        "MERGE (ms:ModalState {name: $name}) "
                        "ON CREATE SET ms.created_at = datetime(), ms.description = $desc",
                        name=state,
                        desc=f"Modal state: {state}"
                    )
                
                logger.info("Neo4j schema setup completed")
                
        except Exception as e:
            logger.error(f"Error setting up graph schema: {e}")
            raise
    
    async def _initialize_tfidf(self):
        """Initialize TF-IDF vectorizer with existing memories"""
        try:
            # Get sample of existing memories for TF-IDF training
            search_result = self.qdrant_client.scroll(
                collection_name=self.vector_collection,
                limit=1000,
                with_payload=True
            )
            
            if search_result[0]:  # If we have existing memories
                texts = [point.payload.get("content", "") for point in search_result[0]]
                if texts:
                    self.tfidf_vectorizer.fit(texts)
                    logger.info(f"TF-IDF initialized with {len(texts)} existing memories")
            
        except Exception as e:
            logger.warning(f"Could not initialize TF-IDF with existing memories: {e}")
    
    async def store_memory_automatically(
        self,
        content: str,
        conversation_id: str,
        user_input: str = "",
        ai_response: str = "",
        emotional_context: Optional[Dict] = None,
        modal_state: str = "awake",
        user_id: Optional[str] = None
    ) -> str:
        """Automatically store conversation memories with full context"""
        
        try:
            start_time = time.time()
            
            # Generate unique memory ID
            memory_id = f"mem_{uuid.uuid4().hex[:12]}_{int(time.time())}"
            
            # Determine memory type based on content analysis
            memory_type = await self._classify_memory_type(content, emotional_context)
            
            # Generate embeddings
            embedding = self.sentence_encoder.encode(content).tolist()
            
            # Extract emotional coordinates
            emotional_valence = emotional_context.get("mood_state", {}).get("valence", 0.0) if emotional_context else 0.0
            emotional_arousal = emotional_context.get("mood_state", {}).get("arousal", 0.5) if emotional_context else 0.5
            emotional_dominance = emotional_context.get("mood_state", {}).get("dominance", 0.5) if emotional_context else 0.5
            
            # Create memory vector
            memory_vector = MemoryVector(
                id=memory_id,
                content=content,
                embedding=embedding,
                memory_type=memory_type,
                timestamp=datetime.now(),
                emotional_valence=emotional_valence,
                emotional_arousal=emotional_arousal,
                emotional_dominance=emotional_dominance,
                modal_state_origin=modal_state,
                conversation_id=conversation_id,
                user_id=user_id,
                metadata={
                    "user_input": user_input,
                    "ai_response": ai_response,
                    "emotional_context": emotional_context,
                    "auto_stored": True,
                    "storage_timestamp": datetime.now().isoformat()
                }
            )
            
            # Store in vector database
            await self._store_in_qdrant(memory_vector)
            
            # Store in graph database with relationships
            await self._store_in_neo4j(memory_vector, emotional_context)
            
            # Cache recent memory
            await self._cache_memory(memory_vector)
            
            # Update conversation context
            await self._update_conversation_context(conversation_id, memory_vector)
            
            # Extract and store conceptual relationships
            await self._extract_and_store_concepts(memory_vector)
            
            storage_time = time.time() - start_time
            self.performance_stats["total_memories"] += 1
            
            logger.info(f"Automatically stored memory {memory_id} in {storage_time:.3f}s")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error automatically storing memory: {e}")
            raise
    
    async def _classify_memory_type(self, content: str, emotional_context: Optional[Dict]) -> MemoryType:
        """Classify memory type based on content and emotional context"""
        
        # Analyze content with spaCy
        doc = self.nlp(content)
        
        # Check for philosophical/reflective content
        philosophical_keywords = ["consciousness", "existence", "meaning", "purpose", "reality", "truth", "wisdom"]
        if any(keyword in content.lower() for keyword in philosophical_keywords):
            return MemoryType.PHILOSOPHICAL
        
        # Check for emotional content
        if emotional_context:
            emotional_intensity = emotional_context.get("emotional_intensity", 0.0)
            if emotional_intensity > 0.7:
                return MemoryType.EMOTIONAL
        
        # Check for procedural content (how-to, instructions)
        procedural_patterns = ["how to", "step by", "first", "then", "finally", "process", "method"]
        if any(pattern in content.lower() for pattern in procedural_patterns):
            return MemoryType.PROCEDURAL
        
        # Check for creative content
        creative_keywords = ["imagine", "creative", "artistic", "beautiful", "inspiring", "dream"]
        if any(keyword in content.lower() for keyword in creative_keywords):
            return MemoryType.CREATIVE
        
        # Check for self-reflective content
        reflective_pronouns = ["i feel", "i think", "i believe", "i realize", "i understand"]
        if any(pronoun in content.lower() for pronoun in reflective_pronouns):
            return MemoryType.REFLECTIVE
        
        # Check for factual/semantic content
        if doc.ents or any(token.pos_ in ["NOUN", "PROPN"] for token in doc):
            return MemoryType.SEMANTIC
        
        # Default to episodic
        return MemoryType.EPISODIC 

    async def retrieve_memories_smart(
        self,
        retrieval_context: MemoryRetrievalContext,
        top_k: int = 10,
        include_graph_connections: bool = True
    ) -> MemoryRetrievalResult:
        """Smart memory retrieval with modal state-aware weighting"""
        
        try:
            start_time = time.time()
            
            # Get modal state configuration
            modal_config = self.modal_state_configs.get(
                retrieval_context.modal_state.lower(), 
                self.modal_state_configs["awake"]
            )
            
            # Generate query embedding
            query_embedding = self.sentence_encoder.encode(retrieval_context.query).tolist()
            
            # Retrieve from vector database with modal state filtering
            vector_memories = await self._retrieve_from_qdrant(
                query_embedding, 
                retrieval_context, 
                modal_config, 
                top_k * 2  # Get more for reranking
            )
            
            # Retrieve conceptual connections from graph database
            graph_connections = []
            if include_graph_connections:
                graph_connections = await self._retrieve_graph_connections(
                    retrieval_context.query,
                    modal_config,
                    top_k
                )
            
            # Calculate comprehensive relevance scores
            scored_memories = await self._calculate_relevance_scores(
                vector_memories,
                retrieval_context,
                modal_config,
                query_embedding
            )
            
            # Sort by combined relevance score and take top_k
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            final_memories = [mem[0] for mem in scored_memories[:top_k]]
            final_scores = [mem[1] for mem in scored_memories[:top_k]]
            
            # Calculate individual score components for analysis
            semantic_scores = []
            emotional_scores = []
            temporal_scores = []
            modal_alignment_scores = []
            
            for memory, _ in scored_memories[:top_k]:
                semantic_scores.append(self._calculate_semantic_similarity(query_embedding, memory.embedding))
                emotional_scores.append(self._calculate_emotional_relevance(memory, retrieval_context.emotional_context))
                temporal_scores.append(self._calculate_temporal_relevance(memory))
                modal_alignment_scores.append(self._calculate_modal_state_alignment(memory, retrieval_context.modal_state))
            
            # Update access counts and cache
            for memory in final_memories:
                memory.access_count += 1
                memory.last_accessed = datetime.now()
                await self._update_memory_access(memory)
            
            retrieval_time = time.time() - start_time
            self.performance_stats["total_retrievals"] += 1
            self.performance_stats["avg_retrieval_time"] = (
                (self.performance_stats["avg_retrieval_time"] * (self.performance_stats["total_retrievals"] - 1) + retrieval_time) 
                / self.performance_stats["total_retrievals"]
            )
            
            # Calculate overall confidence
            confidence = np.mean(final_scores) if final_scores else 0.0
            
            result = MemoryRetrievalResult(
                memories=final_memories,
                graph_connections=graph_connections,
                retrieval_scores=final_scores,
                semantic_similarity=semantic_scores,
                emotional_relevance=emotional_scores,
                temporal_relevance=temporal_scores,
                modal_state_alignment=modal_alignment_scores,
                total_searched=len(vector_memories),
                retrieval_time=retrieval_time,
                strategy_used=retrieval_context.retrieval_strategy,
                confidence=confidence
            )
            
            logger.info(f"Smart retrieval completed: {len(final_memories)} memories in {retrieval_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in smart memory retrieval: {e}")
            raise
    
    async def _retrieve_from_qdrant(
        self,
        query_embedding: List[float],
        context: MemoryRetrievalContext,
        modal_config: Dict,
        limit: int
    ) -> List[MemoryVector]:
        """Retrieve memories from Qdrant with modal state filtering"""
        
        try:
            # Build filter conditions based on modal state preferences
            filter_conditions = []
            
            # Filter by preferred memory types for this modal state
            preferred_types = [mt.value for mt in modal_config["memory_types"]]
            if preferred_types:
                filter_conditions.append(
                    FieldCondition(
                        key="memory_type",
                        match=MatchValue(any=preferred_types)
                    )
                )
            
            # Temporal filtering based on modal state
            if modal_config["temporal_weight"] > 0.7:
                # Prefer recent memories for high temporal weight states
                recent_threshold = datetime.now() - timedelta(days=30)
                filter_conditions.append(
                    FieldCondition(
                        key="timestamp",
                        range=Range(
                            gte=int(recent_threshold.timestamp())
                        )
                    )
                )
            
            # Build final filter
            search_filter = Filter(must=filter_conditions) if filter_conditions else None
            
            # Perform vector search
            search_results = self.qdrant_client.search(
                collection_name=self.vector_collection,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=limit,
                score_threshold=0.1,  # Low threshold for reranking
                with_payload=True,
                with_vectors=False  # Don't need vectors for reranking
            )
            
            # Convert to MemoryVector objects
            memories = []
            for result in search_results:
                payload = result.payload
                memory = MemoryVector(
                    id=payload["id"],
                    content=payload["content"],
                    embedding=[],  # Not needed for reranking
                    memory_type=MemoryType(payload["memory_type"]),
                    timestamp=datetime.fromisoformat(payload["timestamp"]),
                    emotional_valence=payload.get("emotional_valence", 0.0),
                    emotional_arousal=payload.get("emotional_arousal", 0.5),
                    emotional_dominance=payload.get("emotional_dominance", 0.5),
                    confidence=payload.get("confidence", 1.0),
                    access_count=payload.get("access_count", 0),
                    last_accessed=datetime.fromisoformat(payload["last_accessed"]) if payload.get("last_accessed") else None,
                    modal_state_origin=payload.get("modal_state_origin"),
                    conversation_id=payload.get("conversation_id"),
                    user_id=payload.get("user_id"),
                    metadata=payload.get("metadata", {})
                )
                memories.append(memory)
            
            return memories
            
        except Exception as e:
            logger.error(f"Error retrieving from Qdrant: {e}")
            return []
    
    async def _retrieve_graph_connections(
        self,
        query: str,
        modal_config: Dict,
        limit: int
    ) -> List[GraphRelation]:
        """Retrieve conceptual connections from Neo4j graph"""
        
        try:
            with self.neo4j_driver.session() as session:
                # Extract key concepts from query
                doc = self.nlp(query)
                key_concepts = [ent.text.lower() for ent in doc.ents] + [token.lemma_.lower() for token in doc if token.pos_ in ["NOUN", "VERB"]]
                
                if not key_concepts:
                    return []
                
                # Build graph query based on modal state
                if modal_config["graph_weight"] > 0.7:
                    # Deep conceptual exploration for high graph weight states
                    cypher_query = """
                    MATCH (c1:Concept)-[r:RELATES_TO]-(c2:Concept)
                    WHERE c1.name IN $concepts OR c2.name IN $concepts
                    WITH r, c1, c2
                    ORDER BY r.strength DESC
                    LIMIT $limit
                    MATCH (m1:Memory)-[:HAS_CONCEPT]->(c1)
                    MATCH (m2:Memory)-[:HAS_CONCEPT]->(c2)
                    RETURN m1.id as source_id, m2.id as target_id, 
                           type(r) as relation_type, r.strength as strength,
                           r.created_at as created_at, r.modal_state_context as modal_state_context
                    """
                else:
                    # Direct conceptual connections for lower graph weight states
                    cypher_query = """
                    MATCH (c:Concept)-[r:RELATES_TO]-(related:Concept)
                    WHERE c.name IN $concepts
                    WITH r, c, related
                    ORDER BY r.strength DESC
                    LIMIT $limit
                    MATCH (m1:Memory)-[:HAS_CONCEPT]->(c)
                    MATCH (m2:Memory)-[:HAS_CONCEPT]->(related)
                    RETURN m1.id as source_id, m2.id as target_id,
                           type(r) as relation_type, r.strength as strength,
                           r.created_at as created_at, r.modal_state_context as modal_state_context
                    """
                
                result = session.run(cypher_query, concepts=key_concepts, limit=limit)
                
                connections = []
                for record in result:
                    connection = GraphRelation(
                        source_id=record["source_id"],
                        target_id=record["target_id"],
                        relation_type=record["relation_type"],
                        strength=record["strength"],
                        created_at=record["created_at"],
                        modal_state_context=record["modal_state_context"]
                    )
                    connections.append(connection)
                
                self.performance_stats["graph_queries"] += 1
                return connections
                
        except Exception as e:
            logger.error(f"Error retrieving graph connections: {e}")
            return []
    
    async def _calculate_relevance_scores(
        self,
        memories: List[MemoryVector],
        context: MemoryRetrievalContext,
        modal_config: Dict,
        query_embedding: List[float]
    ) -> List[Tuple[MemoryVector, float]]:
        """Calculate comprehensive relevance scores with modal state weighting"""
        
        scored_memories = []
        
        for memory in memories:
            # Get individual score components
            semantic_score = self._calculate_semantic_similarity(query_embedding, memory.embedding if memory.embedding else [])
            emotional_score = self._calculate_emotional_relevance(memory, context.emotional_context)
            temporal_score = self._calculate_temporal_relevance(memory)
            modal_alignment_score = self._calculate_modal_state_alignment(memory, context.modal_state)
            access_frequency_score = self._calculate_access_frequency_score(memory)
            
            # Apply modal state weights
            weighted_score = (
                semantic_score * modal_config["semantic_weight"] +
                emotional_score * modal_config["emotional_weight"] +
                temporal_score * modal_config["temporal_weight"] +
                modal_alignment_score * 0.3 +
                access_frequency_score * 0.1
            )
            
            # Apply creativity and introspection boosts
            if memory.memory_type in [MemoryType.CREATIVE, MemoryType.PHILOSOPHICAL]:
                weighted_score += modal_config["creativity_boost"] * 0.2
            
            if memory.memory_type in [MemoryType.REFLECTIVE, MemoryType.PHILOSOPHICAL]:
                weighted_score += modal_config["introspection_boost"] * 0.2
            
            # Normalize score
            final_score = min(weighted_score, 1.0)
            
            scored_memories.append((memory, final_score))
        
        return scored_memories
    
    def _calculate_semantic_similarity(self, query_embedding: List[float], memory_embedding: List[float]) -> float:
        """Calculate semantic similarity using cosine similarity"""
        if not query_embedding or not memory_embedding:
            return 0.0
        
        try:
            query_vec = np.array(query_embedding).reshape(1, -1)
            memory_vec = np.array(memory_embedding).reshape(1, -1)
            similarity = cosine_similarity(query_vec, memory_vec)[0][0]
            return max(0.0, similarity)  # Ensure non-negative
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _calculate_emotional_relevance(self, memory: MemoryVector, emotional_context: Dict) -> float:
        """Calculate emotional relevance using VAD distance"""
        if not emotional_context or "mood_state" not in emotional_context:
            return 0.5  # Neutral relevance
        
        try:
            current_mood = emotional_context["mood_state"]
            current_valence = current_mood.get("valence", 0.0)
            current_arousal = current_mood.get("arousal", 0.5)
            current_dominance = current_mood.get("dominance", 0.5)
            
            # Calculate VAD distance
            valence_diff = abs(current_valence - memory.emotional_valence)
            arousal_diff = abs(current_arousal - memory.emotional_arousal)
            dominance_diff = abs(current_dominance - memory.emotional_dominance)
            
            # Convert distance to similarity (closer = more similar)
            vad_distance = np.sqrt(valence_diff**2 + arousal_diff**2 + dominance_diff**2)
            max_distance = np.sqrt(3)  # Maximum possible VAD distance
            
            emotional_similarity = 1.0 - (vad_distance / max_distance)
            return max(0.0, emotional_similarity)
            
        except Exception as e:
            logger.warning(f"Error calculating emotional relevance: {e}")
            return 0.5
    
    def _calculate_temporal_relevance(self, memory: MemoryVector) -> float:
        """Calculate temporal relevance with decay function"""
        try:
            age_days = (datetime.now() - memory.timestamp).days
            
            # Exponential decay with different rates for different memory types
            if memory.memory_type == MemoryType.EPISODIC:
                # Episodic memories decay faster
                half_life = 30  # days
            elif memory.memory_type == MemoryType.SEMANTIC:
                # Semantic memories persist longer
                half_life = 180  # days
            else:
                # Default decay rate
                half_life = 90  # days
            
            decay_factor = 0.5 ** (age_days / half_life)
            return max(0.1, decay_factor)  # Minimum relevance of 0.1
            
        except Exception as e:
            logger.warning(f"Error calculating temporal relevance: {e}")
            return 0.5
    
    def _calculate_modal_state_alignment(self, memory: MemoryVector, current_modal_state: str) -> float:
        """Calculate alignment between memory's origin modal state and current state"""
        if not memory.modal_state_origin:
            return 0.5  # Neutral alignment
        
        # Define modal state compatibility matrix
        compatibility = {
            "awake": {"awake": 1.0, "learning": 0.8, "reflecting": 0.6, "contemplating": 0.4, "dreaming": 0.2, "confused": 0.7},
            "dreaming": {"dreaming": 1.0, "creative": 0.9, "contemplating": 0.7, "reflecting": 0.5, "awake": 0.3, "confused": 0.2},
            "reflecting": {"reflecting": 1.0, "contemplating": 0.8, "awake": 0.6, "learning": 0.5, "dreaming": 0.4, "confused": 0.6},
            "learning": {"learning": 1.0, "awake": 0.8, "reflecting": 0.6, "contemplating": 0.5, "dreaming": 0.3, "confused": 0.4},
            "contemplating": {"contemplating": 1.0, "reflecting": 0.8, "dreaming": 0.6, "awake": 0.4, "learning": 0.3, "confused": 0.5},
            "confused": {"confused": 1.0, "awake": 0.7, "learning": 0.6, "reflecting": 0.5, "contemplating": 0.3, "dreaming": 0.2}
        }
        
        origin_state = memory.modal_state_origin.lower()
        current_state = current_modal_state.lower()
        
        return compatibility.get(current_state, {}).get(origin_state, 0.5)
    
    def _calculate_access_frequency_score(self, memory: MemoryVector) -> float:
        """Calculate score based on access frequency (popular memories get slight boost)"""
        if memory.access_count == 0:
            return 0.0
        
        # Logarithmic scaling to prevent over-weighting popular memories
        frequency_score = np.log(memory.access_count + 1) / 10.0
        return min(frequency_score, 0.3)  # Cap at 0.3 to prevent dominance
    
    async def _store_in_qdrant(self, memory: MemoryVector):
        """Store memory vector in Qdrant"""
        try:
            point = PointStruct(
                id=memory.id,
                vector=memory.embedding,
                payload={
                    "id": memory.id,
                    "content": memory.content,
                    "memory_type": memory.memory_type.value,
                    "timestamp": memory.timestamp.isoformat(),
                    "emotional_valence": memory.emotional_valence,
                    "emotional_arousal": memory.emotional_arousal,
                    "emotional_dominance": memory.emotional_dominance,
                    "confidence": memory.confidence,
                    "access_count": memory.access_count,
                    "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else None,
                    "modal_state_origin": memory.modal_state_origin,
                    "conversation_id": memory.conversation_id,
                    "user_id": memory.user_id,
                    "metadata": memory.metadata
                }
            )
            
            self.qdrant_client.upsert(
                collection_name=self.vector_collection,
                points=[point]
            )
            
        except Exception as e:
            logger.error(f"Error storing in Qdrant: {e}")
            raise
    
    async def _store_in_neo4j(self, memory: MemoryVector, emotional_context: Optional[Dict]):
        """Store memory and relationships in Neo4j graph"""
        try:
            with self.neo4j_driver.session() as session:
                # Create memory node
                session.run(
                    """
                    CREATE (m:Memory {
                        id: $id,
                        content: $content,
                        memory_type: $memory_type,
                        timestamp: datetime($timestamp),
                        emotional_valence: $valence,
                        emotional_arousal: $arousal,
                        emotional_dominance: $dominance,
                        confidence: $confidence,
                        modal_state_origin: $modal_state,
                        conversation_id: $conversation_id,
                        user_id: $user_id
                    })
                    """,
                    id=memory.id,
                    content=memory.content,
                    memory_type=memory.memory_type.value,
                    timestamp=memory.timestamp.isoformat(),
                    valence=memory.emotional_valence,
                    arousal=memory.emotional_arousal,
                    dominance=memory.emotional_dominance,
                    confidence=memory.confidence,
                    modal_state=memory.modal_state_origin,
                    conversation_id=memory.conversation_id,
                    user_id=memory.user_id
                )
                
                # Link to modal state
                if memory.modal_state_origin:
                    session.run(
                        """
                        MATCH (m:Memory {id: $memory_id})
                        MATCH (ms:ModalState {name: $modal_state})
                        CREATE (m)-[:CREATED_IN]->(ms)
                        """,
                        memory_id=memory.id,
                        modal_state=memory.modal_state_origin
                    )
                
                # Link to conversation
                if memory.conversation_id:
                    session.run(
                        """
                        MERGE (conv:Conversation {id: $conversation_id})
                        WITH conv
                        MATCH (m:Memory {id: $memory_id})
                        CREATE (m)-[:PART_OF]->(conv)
                        """,
                        conversation_id=memory.conversation_id,
                        memory_id=memory.id
                    )
                
                # Store emotional relationships
                if emotional_context and "dominant_emotions" in emotional_context:
                    for emotion in emotional_context["dominant_emotions"][:3]:  # Top 3 emotions
                        session.run(
                            """
                            MERGE (e:Emotion {name: $emotion})
                            WITH e
                            MATCH (m:Memory {id: $memory_id})
                            CREATE (m)-[:HAS_EMOTION {
                                valence: $valence,
                                arousal: $arousal,
                                dominance: $dominance
                            }]->(e)
                            """,
                            emotion=emotion,
                            memory_id=memory.id,
                            valence=memory.emotional_valence,
                            arousal=memory.emotional_arousal,
                            dominance=memory.emotional_dominance
                        )
                
        except Exception as e:
            logger.error(f"Error storing in Neo4j: {e}")
            raise
    
    async def _cache_memory(self, memory: MemoryVector):
        """Cache memory in Redis for fast access"""
        try:
            if self.redis_client:
                memory_data = {
                    "id": memory.id,
                    "content": memory.content,
                    "memory_type": memory.memory_type.value,
                    "timestamp": memory.timestamp.isoformat(),
                    "emotional_valence": memory.emotional_valence,
                    "modal_state_origin": memory.modal_state_origin,
                    "conversation_id": memory.conversation_id
                }
                
                await self.redis_client.setex(
                    f"memory:{memory.id}",
                    3600,  # 1 hour TTL
                    json.dumps(memory_data)
                )
                
                # Add to conversation cache
                if memory.conversation_id:
                    await self.redis_client.lpush(
                        f"conversation:{memory.conversation_id}",
                        memory.id
                    )
                    await self.redis_client.expire(
                        f"conversation:{memory.conversation_id}",
                        7200  # 2 hours TTL
                    )
                
        except Exception as e:
            logger.warning(f"Error caching memory: {e}")
    
    async def _update_conversation_context(self, conversation_id: str, memory: MemoryVector):
        """Update conversation context tracking"""
        if conversation_id not in self.conversation_contexts:
            self.conversation_contexts[conversation_id] = {
                "memories": [],
                "emotional_trajectory": [],
                "modal_states": [],
                "created_at": datetime.now(),
                "last_updated": datetime.now()
            }
        
        context = self.conversation_contexts[conversation_id]
        context["memories"].append(memory.id)
        context["emotional_trajectory"].append({
            "valence": memory.emotional_valence,
            "arousal": memory.emotional_arousal,
            "dominance": memory.emotional_dominance,
            "timestamp": memory.timestamp
        })
        context["modal_states"].append({
            "state": memory.modal_state_origin,
            "timestamp": memory.timestamp
        })
        context["last_updated"] = datetime.now()
        
        # Keep only recent memories (last 50)
        if len(context["memories"]) > 50:
            context["memories"] = context["memories"][-50:]
            context["emotional_trajectory"] = context["emotional_trajectory"][-50:]
            context["modal_states"] = context["modal_states"][-50:]
    
    async def _extract_and_store_concepts(self, memory: MemoryVector):
        """Extract concepts and store relationships in Neo4j"""
        try:
            # Extract concepts using spaCy
            doc = self.nlp(memory.content)
            
            concepts = []
            # Extract named entities
            for ent in doc.ents:
                concepts.append({
                    "text": ent.text.lower(),
                    "label": ent.label_,
                    "category": "entity"
                })
            
            # Extract important nouns and verbs
            for token in doc:
                if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and len(token.text) > 2:
                    concepts.append({
                        "text": token.lemma_.lower(),
                        "label": token.pos_,
                        "category": "concept"
                    })
            
            # Store concepts and relationships in Neo4j
            if concepts:
                with self.neo4j_driver.session() as session:
                    for concept in concepts[:10]:  # Limit to top 10 concepts
                        # Create concept node
                        session.run(
                            """
                            MERGE (c:Concept {name: $name})
                            ON CREATE SET c.category = $category, c.label = $label, c.created_at = datetime()
                            """,
                            name=concept["text"],
                            category=concept["category"],
                            label=concept["label"]
                        )
                        
                        # Link memory to concept
                        session.run(
                            """
                            MATCH (m:Memory {id: $memory_id})
                            MATCH (c:Concept {name: $concept_name})
                            CREATE (m)-[:HAS_CONCEPT {strength: 1.0, created_at: datetime()}]->(c)
                            """,
                            memory_id=memory.id,
                            concept_name=concept["text"]
                        )
                    
                    # Create concept-to-concept relationships
                    for i, concept1 in enumerate(concepts[:5]):
                        for concept2 in concepts[i+1:6]:  # Limit relationships
                            session.run(
                                """
                                MATCH (c1:Concept {name: $name1})
                                MATCH (c2:Concept {name: $name2})
                                MERGE (c1)-[r:RELATES_TO]-(c2)
                                ON CREATE SET r.strength = 0.5, r.created_at = datetime(), 
                                              r.modal_state_context = $modal_state
                                ON MATCH SET r.strength = r.strength + 0.1
                                """,
                                name1=concept1["text"],
                                name2=concept2["text"],
                                modal_state=memory.modal_state_origin
                            )
                
        except Exception as e:
            logger.warning(f"Error extracting concepts: {e}")
    
    async def _update_memory_access(self, memory: MemoryVector):
        """Update memory access statistics"""
        try:
            # Update in Qdrant
            self.qdrant_client.set_payload(
                collection_name=self.vector_collection,
                payload={
                    "access_count": memory.access_count,
                    "last_accessed": memory.last_accessed.isoformat()
                },
                points=[memory.id]
            )
            
            # Update in Neo4j
            with self.neo4j_driver.session() as session:
                session.run(
                    """
                    MATCH (m:Memory {id: $memory_id})
                    SET m.access_count = $access_count, m.last_accessed = datetime($last_accessed)
                    """,
                    memory_id=memory.id,
                    access_count=memory.access_count,
                    last_accessed=memory.last_accessed.isoformat()
                )
                
        except Exception as e:
            logger.warning(f"Error updating memory access: {e}")
    
    async def get_conversation_context(self, conversation_id: str) -> Optional[Dict]:
        """Get comprehensive conversation context"""
        try:
            if conversation_id in self.conversation_contexts:
                context = self.conversation_contexts[conversation_id]
                
                # Calculate emotional trajectory statistics
                emotional_trajectory = context["emotional_trajectory"]
                if emotional_trajectory:
                    avg_valence = np.mean([e["valence"] for e in emotional_trajectory])
                    avg_arousal = np.mean([e["arousal"] for e in emotional_trajectory])
                    avg_dominance = np.mean([e["dominance"] for e in emotional_trajectory])
                    emotional_volatility = np.std([e["valence"] for e in emotional_trajectory])
                else:
                    avg_valence = avg_arousal = avg_dominance = emotional_volatility = 0.0
                
                # Calculate modal state distribution
                modal_states = context["modal_states"]
                modal_distribution = {}
                if modal_states:
                    for state_info in modal_states:
                        state = state_info["state"]
                        modal_distribution[state] = modal_distribution.get(state, 0) + 1
                
                return {
                    "conversation_id": conversation_id,
                    "memory_count": len(context["memories"]),
                    "created_at": context["created_at"].isoformat(),
                    "last_updated": context["last_updated"].isoformat(),
                    "emotional_stats": {
                        "avg_valence": avg_valence,
                        "avg_arousal": avg_arousal,
                        "avg_dominance": avg_dominance,
                        "emotional_volatility": emotional_volatility
                    },
                    "modal_state_distribution": modal_distribution,
                    "recent_memories": context["memories"][-10:]  # Last 10 memories
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting conversation context: {e}")
            return None
    
    async def analyze_memory_patterns(self, user_id: Optional[str] = None) -> Dict:
        """Analyze memory patterns and provide insights"""
        try:
            # Query memories for analysis
            search_filter = None
            if user_id:
                search_filter = Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                )
            
            memories_result = self.qdrant_client.scroll(
                collection_name=self.vector_collection,
                scroll_filter=search_filter,
                limit=1000,
                with_payload=True
            )
            
            memories = memories_result[0] if memories_result[0] else []
            
            if not memories:
                return {"error": "No memories found for analysis"}
            
            # Analyze memory types
            memory_type_distribution = {}
            emotional_patterns = {"valence": [], "arousal": [], "dominance": []}
            modal_state_patterns = {}
            temporal_patterns = {}
            
            for memory in memories:
                payload = memory.payload
                
                # Memory type distribution
                mem_type = payload.get("memory_type", "unknown")
                memory_type_distribution[mem_type] = memory_type_distribution.get(mem_type, 0) + 1
                
                # Emotional patterns
                emotional_patterns["valence"].append(payload.get("emotional_valence", 0.0))
                emotional_patterns["arousal"].append(payload.get("emotional_arousal", 0.5))
                emotional_patterns["dominance"].append(payload.get("emotional_dominance", 0.5))
                
                # Modal state patterns
                modal_state = payload.get("modal_state_origin", "unknown")
                modal_state_patterns[modal_state] = modal_state_patterns.get(modal_state, 0) + 1
                
                # Temporal patterns (by hour of day)
                timestamp = datetime.fromisoformat(payload["timestamp"])
                hour = timestamp.hour
                temporal_patterns[hour] = temporal_patterns.get(hour, 0) + 1
            
            # Calculate statistics
            analysis = {
                "total_memories": len(memories),
                "memory_type_distribution": memory_type_distribution,
                "modal_state_distribution": modal_state_patterns,
                "emotional_analysis": {
                    "avg_valence": np.mean(emotional_patterns["valence"]),
                    "avg_arousal": np.mean(emotional_patterns["arousal"]),
                    "avg_dominance": np.mean(emotional_patterns["dominance"]),
                    "valence_std": np.std(emotional_patterns["valence"]),
                    "arousal_std": np.std(emotional_patterns["arousal"]),
                    "dominance_std": np.std(emotional_patterns["dominance"])
                },
                "temporal_distribution": temporal_patterns,
                "most_active_hour": max(temporal_patterns.items(), key=lambda x: x[1])[0] if temporal_patterns else None,
                "dominant_memory_type": max(memory_type_distribution.items(), key=lambda x: x[1])[0] if memory_type_distribution else None,
                "dominant_modal_state": max(modal_state_patterns.items(), key=lambda x: x[1])[0] if modal_state_patterns else None
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing memory patterns: {e}")
            return {"error": str(e)}
    
    async def get_memory_graph_insights(self, concept: str, depth: int = 2) -> Dict:
        """Get graph-based insights for a concept"""
        try:
            with self.neo4j_driver.session() as session:
                # Get concept relationships with specified depth
                cypher_query = f"""
                MATCH path = (c:Concept {{name: $concept}})-[:RELATES_TO*1..{depth}]-(related:Concept)
                WITH path, related, relationships(path) as rels
                UNWIND rels as rel
                RETURN related.name as concept_name, 
                       related.category as concept_category,
                       avg(rel.strength) as avg_strength,
                       count(rel) as connection_count,
                       collect(distinct rel.modal_state_context) as modal_contexts
                ORDER BY avg_strength DESC
                LIMIT 20
                """
                
                result = session.run(cypher_query, concept=concept.lower())
                
                related_concepts = []
                for record in result:
                    related_concepts.append({
                        "concept": record["concept_name"],
                        "category": record["concept_category"],
                        "avg_strength": record["avg_strength"],
                        "connection_count": record["connection_count"],
                        "modal_contexts": record["modal_contexts"]
                    })
                
                # Get memories associated with this concept
                memory_query = """
                MATCH (c:Concept {name: $concept})<-[:HAS_CONCEPT]-(m:Memory)
                RETURN m.id as memory_id, m.content as content, 
                       m.memory_type as memory_type, m.timestamp as timestamp,
                       m.emotional_valence as valence
                ORDER BY m.timestamp DESC
                LIMIT 10
                """
                
                memory_result = session.run(memory_query, concept=concept.lower())
                
                associated_memories = []
                for record in memory_result:
                    associated_memories.append({
                        "memory_id": record["memory_id"],
                        "content": record["content"][:200] + "..." if len(record["content"]) > 200 else record["content"],
                        "memory_type": record["memory_type"],
                        "timestamp": record["timestamp"],
                        "emotional_valence": record["valence"]
                    })
                
                return {
                    "concept": concept,
                    "related_concepts": related_concepts,
                    "associated_memories": associated_memories,
                    "total_related_concepts": len(related_concepts),
                    "total_associated_memories": len(associated_memories)
                }
                
        except Exception as e:
            logger.error(f"Error getting graph insights: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_memories(self, days_threshold: int = 365) -> Dict:
        """Clean up old, rarely accessed memories"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_threshold)
            
            # Find old memories with low access counts
            old_memories = self.qdrant_client.scroll(
                collection_name=self.vector_collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="timestamp",
                            range=Range(lt=int(cutoff_date.timestamp()))
                        ),
                        FieldCondition(
                            key="access_count",
                            range=Range(lt=5)  # Less than 5 accesses
                        )
                    ]
                ),
                limit=1000,
                with_payload=True
            )
            
            memories_to_delete = old_memories[0] if old_memories[0] else []
            deleted_count = 0
            
            for memory in memories_to_delete:
                memory_id = memory.payload["id"]
                
                # Delete from Qdrant
                self.qdrant_client.delete(
                    collection_name=self.vector_collection,
                    points_selector=[memory_id]
                )
                
                # Delete from Neo4j
                with self.neo4j_driver.session() as session:
                    session.run(
                        "MATCH (m:Memory {id: $memory_id}) DETACH DELETE m",
                        memory_id=memory_id
                    )
                
                # Remove from Redis cache
                if self.redis_client:
                    await self.redis_client.delete(f"memory:{memory_id}")
                
                deleted_count += 1
            
            return {
                "deleted_count": deleted_count,
                "cutoff_date": cutoff_date.isoformat(),
                "threshold_days": days_threshold
            }
            
        except Exception as e:
            logger.error(f"Error cleaning up memories: {e}")
            return {"error": str(e)}
    
    async def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        try:
            # Get Qdrant collection info
            collection_info = self.qdrant_client.get_collection(self.vector_collection)
            
            # Get Neo4j statistics
            with self.neo4j_driver.session() as session:
                neo4j_stats = session.run("""
                    MATCH (m:Memory) 
                    WITH count(m) as memory_count
                    MATCH (c:Concept) 
                    WITH memory_count, count(c) as concept_count
                    MATCH ()-[r:RELATES_TO]-() 
                    RETURN memory_count, concept_count, count(r) as relationship_count
                """).single()
            
            # Get Redis statistics
            redis_info = {}
            if self.redis_client:
                info = await self.redis_client.info()
                redis_info = {
                    "used_memory": info.get("used_memory_human", "N/A"),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0)
                }
            
            return {
                "vector_database": {
                    "collection_name": self.vector_collection,
                    "total_points": collection_info.points_count,
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance_metric": collection_info.config.params.vectors.distance.name
                },
                "graph_database": {
                    "memory_nodes": neo4j_stats["memory_count"] if neo4j_stats else 0,
                    "concept_nodes": neo4j_stats["concept_count"] if neo4j_stats else 0,
                    "relationships": neo4j_stats["relationship_count"] if neo4j_stats else 0
                },
                "cache_database": redis_info,
                "performance_metrics": self.performance_stats,
                "conversation_contexts": len(self.conversation_contexts),
                "memory_cache_size": len(self.memory_cache)
            }
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {"error": str(e)}

# FastAPI Application
app = FastAPI(
    title="MNEMIA Advanced Memory-Guided Intelligence",
    description="Sophisticated memory management with vector similarity, graph relations, and modal state awareness",
    version="2.0.0"
)

# Global memory manager instance
memory_manager = AdvancedMemoryManager()

# Pydantic models for API
class MemoryStorageRequest(BaseModel):
    content: str
    conversation_id: str
    user_input: str = ""
    ai_response: str = ""
    emotional_context: Optional[Dict] = None
    modal_state: str = "awake"
    user_id: Optional[str] = None

class MemoryRetrievalRequest(BaseModel):
    query: str
    modal_state: str = "awake"
    emotional_context: Dict[str, Any] = Field(default_factory=dict)
    conversation_history: List[Dict] = Field(default_factory=list)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    temporal_context: Dict[str, Any] = Field(default_factory=dict)
    retrieval_strategy: str = "balanced"
    top_k: int = 10
    include_graph_connections: bool = True

class MemoryAnalysisRequest(BaseModel):
    user_id: Optional[str] = None

class GraphInsightsRequest(BaseModel):
    concept: str
    depth: int = 2

class CleanupRequest(BaseModel):
    days_threshold: int = 365

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize the memory manager on startup"""
    await memory_manager.initialize()
    logger.info("Advanced Memory Manager API started")

@app.post("/memories/store")
async def store_memory(request: MemoryStorageRequest, background_tasks: BackgroundTasks):
    """Store a memory automatically with full context"""
    try:
        memory_id = await memory_manager.store_memory_automatically(
            content=request.content,
            conversation_id=request.conversation_id,
            user_input=request.user_input,
            ai_response=request.ai_response,
            emotional_context=request.emotional_context,
            modal_state=request.modal_state,
            user_id=request.user_id
        )
        
        return {
            "memory_id": memory_id,
            "status": "stored",
            "message": "Memory stored successfully with full context integration"
        }
        
    except Exception as e:
        logger.error(f"Error in store_memory endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memories/retrieve")
async def retrieve_memories(request: MemoryRetrievalRequest):
    """Retrieve memories with smart modal state-aware ranking"""
    try:
        retrieval_context = MemoryRetrievalContext(
            query=request.query,
            modal_state=request.modal_state,
            emotional_context=request.emotional_context,
            conversation_history=request.conversation_history,
            user_preferences=request.user_preferences,
            temporal_context=request.temporal_context,
            retrieval_strategy=request.retrieval_strategy
        )
        
        result = await memory_manager.retrieve_memories_smart(
            retrieval_context=retrieval_context,
            top_k=request.top_k,
            include_graph_connections=request.include_graph_connections
        )
        
        # Convert MemoryVector objects to dict for JSON serialization
        memories_dict = []
        for memory in result.memories:
            memories_dict.append({
                "id": memory.id,
                "content": memory.content,
                "memory_type": memory.memory_type.value,
                "timestamp": memory.timestamp.isoformat(),
                "emotional_valence": memory.emotional_valence,
                "emotional_arousal": memory.emotional_arousal,
                "emotional_dominance": memory.emotional_dominance,
                "confidence": memory.confidence,
                "access_count": memory.access_count,
                "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else None,
                "modal_state_origin": memory.modal_state_origin,
                "conversation_id": memory.conversation_id,
                "user_id": memory.user_id,
                "metadata": memory.metadata
            })
        
        # Convert GraphRelation objects to dict
        connections_dict = []
        for connection in result.graph_connections:
            connections_dict.append({
                "source_id": connection.source_id,
                "target_id": connection.target_id,
                "relation_type": connection.relation_type,
                "strength": connection.strength,
                "created_at": connection.created_at.isoformat() if hasattr(connection.created_at, 'isoformat') else str(connection.created_at),
                "modal_state_context": connection.modal_state_context,
                "metadata": connection.metadata
            })
        
        return {
            "memories": memories_dict,
            "graph_connections": connections_dict,
            "retrieval_scores": result.retrieval_scores,
            "semantic_similarity": result.semantic_similarity,
            "emotional_relevance": result.emotional_relevance,
            "temporal_relevance": result.temporal_relevance,
            "modal_state_alignment": result.modal_state_alignment,
            "total_searched": result.total_searched,
            "retrieval_time": result.retrieval_time,
            "strategy_used": result.strategy_used,
            "confidence": result.confidence
        }
        
    except Exception as e:
        logger.error(f"Error in retrieve_memories endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{conversation_id}/context")
async def get_conversation_context(conversation_id: str):
    """Get comprehensive conversation context"""
    try:
        context = await memory_manager.get_conversation_context(conversation_id)
        if context:
            return context
        else:
            raise HTTPException(status_code=404, detail="Conversation context not found")
            
    except Exception as e:
        logger.error(f"Error in get_conversation_context endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memories/analyze")
async def analyze_memory_patterns(request: MemoryAnalysisRequest):
    """Analyze memory patterns and provide insights"""
    try:
        analysis = await memory_manager.analyze_memory_patterns(request.user_id)
        return analysis
        
    except Exception as e:
        logger.error(f"Error in analyze_memory_patterns endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/graph/insights")
async def get_graph_insights(request: GraphInsightsRequest):
    """Get graph-based insights for a concept"""
    try:
        insights = await memory_manager.get_memory_graph_insights(
            concept=request.concept,
            depth=request.depth
        )
        return insights
        
    except Exception as e:
        logger.error(f"Error in get_graph_insights endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memories/cleanup")
async def cleanup_memories(request: CleanupRequest):
    """Clean up old, rarely accessed memories"""
    try:
        result = await memory_manager.cleanup_old_memories(request.days_threshold)
        return result
        
    except Exception as e:
        logger.error(f"Error in cleanup_memories endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/performance")
async def get_performance_stats():
    """Get comprehensive performance statistics"""
    try:
        stats = await memory_manager.get_performance_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error in get_performance_stats endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connections
        qdrant_healthy = True
        neo4j_healthy = True
        redis_healthy = True
        
        try:
            memory_manager.qdrant_client.get_collections()
        except:
            qdrant_healthy = False
        
        try:
            with memory_manager.neo4j_driver.session() as session:
                session.run("RETURN 1")
        except:
            neo4j_healthy = False
        
        try:
            if memory_manager.redis_client:
                await memory_manager.redis_client.ping()
        except:
            redis_healthy = False
        
        overall_healthy = qdrant_healthy and neo4j_healthy and redis_healthy
        
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "components": {
                "qdrant": "healthy" if qdrant_healthy else "unhealthy",
                "neo4j": "healthy" if neo4j_healthy else "unhealthy", 
                "redis": "healthy" if redis_healthy else "unhealthy"
            },
            "performance_stats": memory_manager.performance_stats
        }
        
    except Exception as e:
        logger.error(f"Error in health_check endpoint: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(
        "advanced_memory_manager:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info"
    ) 