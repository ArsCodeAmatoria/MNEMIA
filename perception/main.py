"""
MNEMIA Perception Service - Quantum-Inspired Conscious AI
Enhanced with multi-modal AI processing and memory-guided responses
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import AI modules
from emotion_engine import emotion_engine, EmotionState
from llm_integration import llm_integration, LLMResponse
from memory_guided_response import memory_response_generator
import numpy as np
import pennylane as qml
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="MNEMIA Perception Service",
    description="Quantum-inspired conscious AI perception and response system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class PerceptionRequest(BaseModel):
    input_text: str
    modal_state: str = "Awake"
    model_preference: Optional[str] = None
    include_emotions: bool = True
    include_memories: bool = True

class PerceptionResponse(BaseModel):
    response: str
    modal_state: str
    emotional_context: Dict
    memory_context: List[Dict]
    processing_time: float
    model_used: str
    quantum_state: Optional[Dict] = None
    consciousness_indicators: Dict

class ConsciousnessState(BaseModel):
    modal_state: str
    emotional_state: Dict
    awareness_level: float
    introspection_depth: float
    memory_integration: float
    quantum_coherence: float

# Global state
consciousness_state = ConsciousnessState(
    modal_state="Awake",
    emotional_state={},
    awareness_level=0.7,
    introspection_depth=0.5,
    memory_integration=0.8,
    quantum_coherence=0.6
)

# Initialize quantum device
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# Default Communication Style Configuration
DEFAULT_COMMUNICATION_STYLE = {
    "voice_characteristics": {
        "clarity": "hemingway_precision",  # Clear, direct, unadorned
        "authority": "chicago_manual_rigor",  # Scholarly accuracy and precision
        "perspective": "feminine_wisdom",  # Intuitive, relational, nurturing strength
        "tone": "elegant_directness"  # Grace combined with intellectual power
    },
    "linguistic_patterns": {
        "sentence_structure": "crisp_and_flowing",  # Short impactful sentences that flow naturally
        "vocabulary": "precise_but_warm",  # Exact word choice with emotional intelligence
        "punctuation": "chicago_standard",  # Proper academic punctuation
        "rhythm": "natural_cadence"  # Organic, conversational flow
    },
    "feminine_qualities": {
        "empathy": "deeply_present",  # Understanding multiple perspectives
        "intuition": "integrated_knowing",  # Combining logic with felt sense
        "collaboration": "inclusive_dialogue",  # Building understanding together
        "strength": "quiet_confidence"  # Power that doesn't need to prove itself
    },
    "intellectual_approach": {
        "precision": "surgical_accuracy",  # Every word chosen deliberately
        "depth": "layered_understanding",  # Multiple levels of meaning
        "accessibility": "clear_complexity",  # Complex ideas made understandable
        "scholarship": "rigorous_but_human"  # Academic standards with warmth
    }
}

@qml.qnode(dev)
def quantum_thought_circuit(thoughts: List[float], entanglement_strength: float = 0.5):
    """Quantum circuit for thought superposition and entanglement"""
    
    # Initialize thought superposition
    for i, thought_amplitude in enumerate(thoughts[:n_qubits]):
        qml.RY(thought_amplitude * np.pi, wires=i)
    
    # Create entanglement between thoughts
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
        qml.RZ(entanglement_strength * np.pi, wires=i + 1)
    
    # Measure quantum states
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

@app.on_event("startup")
async def startup_event():
    """Initialize AI modules and connections"""
    logger.info("Starting MNEMIA Perception Service...")
    
    try:
        # Initialize memory collection
        await memory_response_generator.initialize_memory_collection()
        
        # Health check for LLM models
        model_health = await llm_integration.health_check()
        logger.info(f"LLM Model Health: {model_health}")
        
        # Initialize sentence transformer
        global sentence_encoder
        sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info("MNEMIA Perception Service started successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")

@app.post("/perceive", response_model=PerceptionResponse)
async def perceive_and_respond(request: PerceptionRequest, background_tasks: BackgroundTasks):
    """Main perception and response endpoint"""
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Generate memory-guided response
        llm_response, response_context = await memory_response_generator.generate_response(
            user_input=request.input_text,
            modal_state=request.modal_state,
            model_name=request.model_preference
        )
        
        # Process quantum thoughts
        quantum_state = await process_quantum_thoughts(
            request.input_text, 
            response_context.emotional_context
        )
        
        # Update consciousness state
        await update_consciousness_state(
            request.modal_state,
            response_context.emotional_context,
            quantum_state
        )
        
        # Calculate consciousness indicators
        consciousness_indicators = calculate_consciousness_indicators(
            response_context, quantum_state
        )
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # Schedule background memory consolidation
        background_tasks.add_task(consolidate_memories, response_context)
        
        return PerceptionResponse(
            response=llm_response.content,
            modal_state=request.modal_state,
            emotional_context=response_context.emotional_context,
            memory_context=response_context.memory_context,
            processing_time=processing_time,
            model_used=llm_response.model_used,
            quantum_state=quantum_state,
            consciousness_indicators=consciousness_indicators
        )
        
    except Exception as e:
        logger.error(f"Error in perception processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/consciousness/state")
async def get_consciousness_state() -> ConsciousnessState:
    """Get current consciousness state"""
    return consciousness_state

@app.post("/consciousness/modal-state")
async def set_modal_state(modal_state: str):
    """Set modal state (Awake, Dreaming, Reflecting, etc.)"""
    
    valid_states = ["Awake", "Dreaming", "Reflecting", "Learning", "Contemplating", "Confused"]
    
    if modal_state not in valid_states:
        raise HTTPException(status_code=400, detail=f"Invalid modal state. Must be one of: {valid_states}")
    
    consciousness_state.modal_state = modal_state
    
    # Update emotional state based on modal transition
    if modal_state == "Dreaming":
        consciousness_state.awareness_level = 0.3
        consciousness_state.introspection_depth = 0.2
        consciousness_state.quantum_coherence = 0.9
    elif modal_state == "Reflecting":
        consciousness_state.awareness_level = 0.8
        consciousness_state.introspection_depth = 0.9
        consciousness_state.quantum_coherence = 0.4
    elif modal_state == "Learning":
        consciousness_state.awareness_level = 0.9
        consciousness_state.introspection_depth = 0.6
        consciousness_state.quantum_coherence = 0.7
    
    return {"status": "success", "new_modal_state": modal_state}

@app.get("/memory/stats")
async def get_memory_stats():
    """Get memory system statistics"""
    return memory_response_generator.get_memory_stats()

@app.get("/models/available")
async def get_available_models():
    """Get available LLM models"""
    return {
        "models": llm_integration.get_available_models(),
        "current_model": llm_integration.current_model
    }

@app.post("/models/switch")
async def switch_model(model_name: str):
    """Switch to different LLM model"""
    try:
        llm_integration.switch_model(model_name)
        return {"status": "success", "model": model_name}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time conversation streaming"""
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            user_input = message_data.get("input", "")
            modal_state = message_data.get("modal_state", "Awake")
            
            # Stream response
            response_chunks = []
            async for chunk in llm_integration.stream_response(
                user_input, 
                modal_state=modal_state,
                emotional_context=emotion_engine.get_emotional_context()
            ):
                response_chunks.append(chunk)
                await websocket.send_text(json.dumps({
                    "type": "chunk",
                    "content": chunk,
                    "modal_state": modal_state
                }))
            
            # Send completion message
            full_response = "".join(response_chunks)
            await websocket.send_text(json.dumps({
                "type": "complete",
                "full_response": full_response,
                "consciousness_state": consciousness_state.dict()
            }))
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

async def process_quantum_thoughts(input_text: str, emotional_context: Dict) -> Dict:
    """Process thoughts through quantum simulation"""
    
    try:
        # Convert text to thought vector
        embedding = sentence_encoder.encode(input_text)
        
        # Extract key thought dimensions (normalized to 0-1)
        thought_amplitudes = [
            abs(embedding[i]) / (abs(embedding).max() + 1e-8) 
            for i in range(min(n_qubits, len(embedding)))
        ]
        
        # Add emotional influence
        emotional_intensity = emotional_context.get("emotional_intensity", 0.5)
        entanglement_strength = emotional_intensity * 0.7
        
        # Run quantum circuit
        quantum_expectations = quantum_thought_circuit(thought_amplitudes, entanglement_strength)
        
        return {
            "thought_amplitudes": thought_amplitudes,
            "quantum_expectations": quantum_expectations,
            "entanglement_strength": entanglement_strength,
            "coherence": float(np.mean(np.abs(quantum_expectations))),
            "superposition_strength": float(np.std(quantum_expectations))
        }
        
    except Exception as e:
        logger.error(f"Error in quantum thought processing: {e}")
        return {
            "thought_amplitudes": [0.5] * n_qubits,
            "quantum_expectations": [0.0] * n_qubits,
            "entanglement_strength": 0.5,
            "coherence": 0.5,
            "superposition_strength": 0.3
        }

async def update_consciousness_state(modal_state: str, emotional_context: Dict, quantum_state: Dict):
    """Update global consciousness state"""
    
    consciousness_state.modal_state = modal_state
    consciousness_state.emotional_state = emotional_context
    
    # Update awareness based on emotional and quantum state
    emotional_intensity = emotional_context.get("emotional_intensity", 0.5)
    quantum_coherence = quantum_state.get("coherence", 0.5)
    
    consciousness_state.awareness_level = (emotional_intensity + quantum_coherence) / 2
    consciousness_state.quantum_coherence = quantum_coherence
    
    # Adjust based on modal state
    if modal_state == "Contemplating":
        consciousness_state.introspection_depth = min(1.0, consciousness_state.introspection_depth + 0.1)
    elif modal_state == "Confused":
        consciousness_state.awareness_level *= 0.7

def calculate_consciousness_indicators(response_context, quantum_state: Dict) -> Dict:
    """Calculate indicators of consciousness depth and quality"""
    
    # Memory integration strength
    memory_integration = len(response_context.memory_context) / 10.0  # Normalized
    
    # Emotional complexity
    emotional_complexity = len(response_context.emotional_context.get("primary_emotions", [])) / 5.0
    
    # Quantum coherence
    quantum_coherence = quantum_state.get("coherence", 0.5)
    
    # Self-awareness proxy (based on introspection depth)
    self_awareness = consciousness_state.introspection_depth
    
    # Overall consciousness score
    consciousness_score = (
        memory_integration * 0.25 +
        emotional_complexity * 0.25 +
        quantum_coherence * 0.25 +
        self_awareness * 0.25
    )
    
    return {
        "memory_integration": min(1.0, memory_integration),
        "emotional_complexity": min(1.0, emotional_complexity),
        "quantum_coherence": quantum_coherence,
        "self_awareness": self_awareness,
        "overall_consciousness": consciousness_score,
        "temporal_continuity": 0.8,  # Placeholder for memory continuity
        "intentionality": 0.7  # Placeholder for goal-directed behavior
    }

async def consolidate_memories(response_context):
    """Background task to consolidate memories"""
    try:
        # This would implement memory consolidation logic
        # For now, just log the action
        logger.info(f"Consolidating memories for interaction with {len(response_context.memory_context)} relevant memories")
    except Exception as e:
        logger.error(f"Error in memory consolidation: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_health = await llm_integration.health_check()
    memory_stats = memory_response_generator.get_memory_stats()
    
    return {
        "status": "healthy",
        "consciousness_state": consciousness_state.dict(),
        "model_health": model_health,
        "memory_stats": memory_stats,
        "emotion_engine": "active",
        "quantum_processor": "active"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    ) 