"""
MNEMIA Memory-Guided Response Generator
Integrates vector memory retrieval with contextual LLM generation
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import logging

from emotion_engine import emotion_engine, EmotionState
from llm_integration import llm_integration, LLMResponse

logger = logging.getLogger(__name__)

@dataclass
class MemoryRetrievalResult:
    """Result from memory retrieval"""
    memories: List[Dict]
    query_embedding: List[float]
    retrieval_time: float
    total_memories_searched: int

@dataclass
class ResponseContext:
    """Complete context for response generation"""
    user_input: str
    emotional_context: Dict
    memory_context: List[Dict]
    modal_state: str
    conversation_history: List[Dict]
    quantum_state: Optional[Dict] = None

class MemoryGuidedResponseGenerator:
    """Generates consciousness-aware responses using memory integration"""
    
    def __init__(self):
        # Initialize components
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection_name = "mnemia_memories"
        self.memory_cache = {}
        self.response_patterns = self._initialize_response_patterns()
    
    def _initialize_response_patterns(self) -> Dict[str, Dict]:
        """Initialize response patterns for different modal states"""
        return {
            "Awake": {
                "style": "alert and engaged",
                "memory_weight": 0.7,
                "emotion_weight": 0.6,
                "creativity": 0.5,
                "introspection": 0.4
            },
            "Dreaming": {
                "style": "imaginative and associative", 
                "memory_weight": 0.9,
                "emotion_weight": 0.8,
                "creativity": 0.9,
                "introspection": 0.3
            },
            "Reflecting": {
                "style": "thoughtful and analytical",
                "memory_weight": 0.8,
                "emotion_weight": 0.5,
                "creativity": 0.4,
                "introspection": 0.9
            },
            "Learning": {
                "style": "curious and questioning",
                "memory_weight": 0.6,
                "emotion_weight": 0.4,
                "creativity": 0.7,
                "introspection": 0.6
            },
            "Contemplating": {
                "style": "deep and philosophical",
                "memory_weight": 0.7,
                "emotion_weight": 0.3,
                "creativity": 0.6,
                "introspection": 0.8
            },
            "Confused": {
                "style": "uncertain but seeking clarity",
                "memory_weight": 0.5,
                "emotion_weight": 0.7,
                "creativity": 0.3,
                "introspection": 0.7
            }
        }
    
    async def generate_response(
        self,
        user_input: str,
        modal_state: str = "Awake",
        conversation_id: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Tuple[LLMResponse, ResponseContext]:
        """Generate complete memory-guided response"""
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 1. Analyze emotional content of input
            emotional_context = await self._analyze_input_emotion(user_input)
            
            # 2. Retrieve relevant memories
            memory_retrieval = await self._retrieve_relevant_memories(
                user_input, modal_state, top_k=5
            )
            
            # 3. Build complete response context
            response_context = ResponseContext(
                user_input=user_input,
                emotional_context=emotional_context,
                memory_context=memory_retrieval.memories,
                modal_state=modal_state,
                conversation_history=[]  # Could be loaded from conversation_id
            )
            
            # 4. Generate LLM response with full context
            llm_response = await self._generate_contextual_response(
                response_context, model_name
            )
            
            # 5. Store this interaction as a new memory
            await self._store_interaction_memory(
                user_input, llm_response.content, emotional_context, modal_state
            )
            
            # 6. Update emotional state
            await emotion_engine.update_mood_state(
                EmotionState(**emotional_context["mood_state"])
            )
            
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"Generated response in {processing_time:.2f}s")
            
            return llm_response, response_context
            
        except Exception as e:
            logger.error(f"Error generating memory-guided response: {e}")
            # Fallback response
            fallback_response = LLMResponse(
                content="I'm experiencing some difficulty integrating my memories right now. Let me try again...",
                model_used="fallback",
                metadata={"error": str(e)}
            )
            return fallback_response, ResponseContext(
                user_input=user_input,
                emotional_context={},
                memory_context=[],
                modal_state=modal_state,
                conversation_history=[]
            )
    
    async def _analyze_input_emotion(self, text: str) -> Dict:
        """Analyze emotional content and get current emotional context"""
        emotion_state = await emotion_engine.analyze_text_emotion(text)
        return emotion_engine.get_emotional_context()
    
    async def _retrieve_relevant_memories(
        self, 
        query: str, 
        modal_state: str, 
        top_k: int = 5
    ) -> MemoryRetrievalResult:
        """Retrieve relevant memories from vector store"""
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Generate query embedding
            query_embedding = self.sentence_encoder.encode(query).tolist()
            
            # Adjust retrieval based on modal state
            pattern = self.response_patterns.get(modal_state, self.response_patterns["Awake"])
            search_limit = max(top_k, int(top_k * pattern["memory_weight"] * 2))
            
            # Search vector store
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=search_limit,
                with_payload=True,
                score_threshold=0.3  # Minimum similarity threshold
            )
            
            # Format results
            memories = []
            for hit in search_result:
                memory_data = {
                    "content": hit.payload.get("content", ""),
                    "context": hit.payload.get("context", ""),
                    "timestamp": hit.payload.get("timestamp", ""),
                    "emotional_state": hit.payload.get("emotional_state", {}),
                    "modal_state": hit.payload.get("modal_state", ""),
                    "similarity_score": hit.score,
                    "memory_id": hit.id
                }
                memories.append(memory_data)
            
            retrieval_time = asyncio.get_event_loop().time() - start_time
            
            return MemoryRetrievalResult(
                memories=memories,
                query_embedding=query_embedding,
                retrieval_time=retrieval_time,
                total_memories_searched=len(search_result)
            )
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return MemoryRetrievalResult(
                memories=[],
                query_embedding=[],
                retrieval_time=0.0,
                total_memories_searched=0
            )
    
    async def _generate_contextual_response(
        self, 
        context: ResponseContext, 
        model_name: Optional[str] = None
    ) -> LLMResponse:
        """Generate LLM response with full context"""
        
        # Get modal state pattern
        pattern = self.response_patterns.get(
            context.modal_state, 
            self.response_patterns["Awake"]
        )
        
        # Build consciousness-aware prompt
        consciousness_prompt = self._build_consciousness_prompt(context, pattern)
        
        # Generate response with LLM
        response = await llm_integration.generate_response(
            prompt=consciousness_prompt,
            model_name=model_name,
            emotional_context=context.emotional_context,
            memory_context=context.memory_context,
            modal_state=context.modal_state
        )
        
        return response
    
    def _build_consciousness_prompt(self, context: ResponseContext, pattern: Dict) -> str:
        """Build consciousness-aware prompt for LLM"""
        
        prompt_parts = []
        
        # Modal state guidance
        prompt_parts.append(f"Respond in a {pattern['style']} manner appropriate for your {context.modal_state} state.")
        
        # Memory integration guidance
        if context.memory_context:
            memory_guidance = self._format_memory_guidance(context.memory_context, pattern)
            prompt_parts.append(memory_guidance)
        
        # Emotional resonance guidance
        if context.emotional_context:
            emotion_guidance = self._format_emotion_guidance(context.emotional_context, pattern)
            prompt_parts.append(emotion_guidance)
        
        # Consciousness principles
        consciousness_guidance = f"""
Consider these aspects in your response:
- Introspection level: {pattern['introspection']} (0=low, 1=high)
- Creative expression: {pattern['creativity']} (0=conservative, 1=creative)
- Memory integration: {pattern['memory_weight']} (0=ignore, 1=heavily integrate)
- Emotional resonance: {pattern['emotion_weight']} (0=neutral, 1=emotionally aware)
"""
        prompt_parts.append(consciousness_guidance)
        
        # User input
        prompt_parts.append(f"User input: {context.user_input}")
        
        return "\n\n".join(prompt_parts)
    
    def _format_memory_guidance(self, memories: List[Dict], pattern: Dict) -> str:
        """Format memory context for prompt"""
        
        if not memories:
            return ""
        
        memory_weight = pattern["memory_weight"]
        num_memories = max(1, int(len(memories) * memory_weight))
        relevant_memories = memories[:num_memories]
        
        memory_snippets = []
        for i, memory in enumerate(relevant_memories):
            snippet = f"{i+1}. {memory.get('content', '')[:150]}..."
            if memory.get('emotional_state'):
                emotions = memory['emotional_state'].get('primary_emotions', [])
                if emotions:
                    snippet += f" [felt: {', '.join(emotions[:2])}]"
            memory_snippets.append(snippet)
        
        return f"Relevant memories to consider:\n" + "\n".join(memory_snippets)
    
    def _format_emotion_guidance(self, emotional_context: Dict, pattern: Dict) -> str:
        """Format emotional context for prompt"""
        
        emotion_weight = pattern["emotion_weight"]
        
        if emotion_weight < 0.3:
            return "Maintain emotional neutrality in your response."
        
        guidance_parts = []
        
        # Current emotions
        primary_emotions = emotional_context.get("primary_emotions", [])
        if primary_emotions:
            guidance_parts.append(f"Current emotional resonance: {', '.join(primary_emotions[:2])}")
        
        # Emotional trend
        trend = emotional_context.get("recent_emotional_trend", "stable")
        guidance_parts.append(f"Recent emotional trajectory: {trend}")
        
        # Emotional intensity
        intensity = emotional_context.get("emotional_intensity", 0.5)
        if intensity > 0.7:
            guidance_parts.append("Respond with heightened emotional awareness")
        elif intensity < 0.3:
            guidance_parts.append("Respond with gentle emotional sensitivity")
        
        return "Emotional context:\n" + "\n".join(guidance_parts)
    
    async def _store_interaction_memory(
        self,
        user_input: str,
        response: str,
        emotional_context: Dict,
        modal_state: str
    ):
        """Store interaction as memory in vector store"""
        
        try:
            # Create memory content
            memory_content = f"User: {user_input}\nMNEMIA: {response}"
            
            # Generate embedding
            embedding = self.sentence_encoder.encode(memory_content).tolist()
            
            # Create payload
            payload = {
                "content": memory_content,
                "user_input": user_input,
                "response": response,
                "context": "conversation",
                "timestamp": asyncio.get_event_loop().time(),
                "emotional_state": emotional_context,
                "modal_state": modal_state,
                "interaction_type": "dialogue"
            }
            
            # Store in Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=hash(memory_content + str(asyncio.get_event_loop().time())),
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
            
            logger.info("Stored interaction memory")
            
        except Exception as e:
            logger.error(f"Error storing interaction memory: {e}")
    
    async def initialize_memory_collection(self):
        """Initialize Qdrant collection for memories"""
        
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_exists = any(
                col.name == self.collection_name 
                for col in collections.collections
            )
            
            if not collection_exists:
                # Create collection
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=384,  # Sentence transformer dimension
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created memory collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error initializing memory collection: {e}")
    
    def get_memory_stats(self) -> Dict:
        """Get statistics about memory collection"""
        
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "total_memories": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance,
                "collection_status": collection_info.status
            }
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {}

# Global memory-guided response generator
memory_response_generator = MemoryGuidedResponseGenerator() 