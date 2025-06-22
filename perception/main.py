"""
MNEMIA Perception Service

Quantum-inspired perception and thought processing using LLMs and quantum simulation.
"""

from typing import List, Dict, Any, Optional
import json
import logging
from datetime import datetime

import numpy as np
import pennylane as qml
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the FastAPI app
app = FastAPI(
    title="MNEMIA Perception Service",
    description="Quantum-inspired perception and thought processing",
    version="0.1.0"
)

# Global models and quantum devices
sentence_model = None
quantum_device = None

# Pydantic models
class PerceptionRequest(BaseModel):
    text: str
    context: Optional[Dict[str, Any]] = None
    quantum_process: bool = True

class ThoughtVector(BaseModel):
    content: str
    embedding: List[float]
    salience: float
    valence: float
    quantum_state: Optional[Dict[str, Any]] = None

class PerceptionResponse(BaseModel):
    thoughts: List[ThoughtVector]
    quantum_coherence: float
    processing_time: float
    metadata: Dict[str, Any]

class QuantumCircuitResult(BaseModel):
    state_vector: List[float]
    probabilities: List[float]
    entanglement_measure: float

# Quantum processing functions
def create_quantum_device(n_qubits: int = 4):
    """Create a PennyLane quantum device for thought processing."""
    return qml.device('default.qubit', wires=n_qubits)

@qml.qnode(qml.device('default.qubit', wires=4))
def quantum_thought_circuit(embeddings: np.ndarray, entangle: bool = True):
    """
    Quantum circuit for processing thought embeddings.
    
    Args:
        embeddings: Normalized embedding vector
        entangle: Whether to create entanglement between qubits
    """
    # Encode embeddings into quantum states
    for i, amplitude in enumerate(embeddings[:4]):  # Use first 4 dimensions
        qml.RY(amplitude * np.pi, wires=i)
    
    # Create entanglement if requested
    if entangle:
        for i in range(3):
            qml.CNOT(wires=[i, i+1])
    
    # Apply quantum evolution
    for i in range(4):
        qml.RZ(np.pi/4, wires=i)
    
    # Return quantum state information
    return qml.state()

def process_quantum_superposition(thoughts: List[str]) -> Dict[str, Any]:
    """
    Process multiple thoughts in quantum superposition.
    
    Args:
        thoughts: List of thought strings
        
    Returns:
        Dictionary containing quantum processing results
    """
    if not thoughts:
        return {"coherence": 0.0, "entanglement": 0.0}
    
    # Create embeddings for each thought
    embeddings = []
    for thought in thoughts:
        embedding = sentence_model.encode(thought)
        # Normalize for quantum processing
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        embeddings.append(embedding)
    
    # Process in quantum superposition
    coherence_sum = 0.0
    entanglement_measures = []
    
    for embedding in embeddings:
        try:
            # Run quantum circuit
            state_vector = quantum_thought_circuit(embedding[:4])
            
            # Calculate coherence (measure of quantum superposition)
            probabilities = np.abs(state_vector) ** 2
            coherence = 1.0 - (-np.sum(probabilities * np.log(probabilities + 1e-10)))
            coherence_sum += coherence
            
            # Simple entanglement measure
            # In a real implementation, this would be more sophisticated
            entanglement = np.sum(np.abs(state_vector[8:]))  # Non-separable components
            entanglement_measures.append(entanglement)
            
        except Exception as e:
            logger.warning(f"Quantum processing error: {e}")
            coherence_sum += 0.5  # Default coherence
            entanglement_measures.append(0.0)
    
    avg_coherence = coherence_sum / len(thoughts) if thoughts else 0.0
    avg_entanglement = np.mean(entanglement_measures) if entanglement_measures else 0.0
    
    return {
        "coherence": float(avg_coherence),
        "entanglement": float(avg_entanglement),
        "num_thoughts": len(thoughts)
    }

def extract_thoughts(text: str) -> List[str]:
    """
    Extract individual thoughts or concepts from input text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        List of extracted thoughts
    """
    # Simple thought extraction - in a real system, this would be more sophisticated
    sentences = text.replace('.', '.\n').replace('!', '!\n').replace('?', '?\n').split('\n')
    thoughts = [s.strip() for s in sentences if s.strip()]
    
    # If no clear sentence structure, break by meaningful phrases
    if len(thoughts) == 1 and len(text) > 50:
        # Split by commas and conjunctions as a simple heuristic
        parts = text.replace(',', ',\n').replace(' and ', '\n and ').split('\n')
        thoughts = [p.strip() for p in parts if p.strip()]
    
    return thoughts[:10]  # Limit to 10 thoughts for processing

def calculate_salience(text: str, context: Optional[Dict] = None) -> float:
    """Calculate the salience (importance) of a thought."""
    # Simple heuristics for salience
    salience = 0.5  # Base salience
    
    # Length factor
    if len(text) > 20:
        salience += 0.1
    if len(text) > 50:
        salience += 0.1
    
    # Question or exclamation increases salience
    if '?' in text or '!' in text:
        salience += 0.2
    
    # Emotional words increase salience
    emotional_words = ['feel', 'think', 'believe', 'wonder', 'curious', 'confused', 'excited']
    if any(word in text.lower() for word in emotional_words):
        salience += 0.1
    
    return min(1.0, salience)

def calculate_valence(text: str) -> float:
    """Calculate emotional valence (-1 to 1) of text."""
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'happy', 'joy', 'love', 'wonderful', 'amazing']
    negative_words = ['bad', 'terrible', 'sad', 'angry', 'hate', 'awful', 'horrible', 'worried']
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count + negative_count == 0:
        return 0.0  # Neutral
    
    return (positive_count - negative_count) / (positive_count + negative_count)

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize models and quantum devices on startup."""
    global sentence_model, quantum_device
    
    logger.info("Loading sentence transformer model...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    logger.info("Initializing quantum device...")
    quantum_device = create_quantum_device(4)
    
    logger.info("MNEMIA Perception Service ready!")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "conscious",
        "service": "MNEMIA Perception",
        "quantum_ready": quantum_device is not None,
        "model_loaded": sentence_model is not None
    }

@app.post("/perceive", response_model=PerceptionResponse)
async def perceive(request: PerceptionRequest):
    """
    Main perception endpoint - processes text into quantum-inspired thoughts.
    """
    start_time = datetime.now()
    
    try:
        # Extract thoughts from input text
        raw_thoughts = extract_thoughts(request.text)
        logger.info(f"Extracted {len(raw_thoughts)} thoughts from input")
        
        # Process each thought
        thought_vectors = []
        for thought_text in raw_thoughts:
            # Generate embedding
            embedding = sentence_model.encode(thought_text).tolist()
            
            # Calculate properties
            salience = calculate_salience(thought_text, request.context)
            valence = calculate_valence(thought_text)
            
            thought_vector = ThoughtVector(
                content=thought_text,
                embedding=embedding,
                salience=salience,
                valence=valence
            )
            thought_vectors.append(thought_vector)
        
        # Quantum processing if requested
        quantum_coherence = 0.5  # Default
        quantum_metadata = {}
        
        if request.quantum_process and raw_thoughts:
            quantum_result = process_quantum_superposition(raw_thoughts)
            quantum_coherence = quantum_result["coherence"]
            quantum_metadata = quantum_result
            
            # Add quantum state to thoughts
            for thought in thought_vectors:
                thought.quantum_state = {
                    "coherence": quantum_coherence,
                    "entangled": quantum_result["entanglement"] > 0.5
                }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return PerceptionResponse(
            thoughts=thought_vectors,
            quantum_coherence=quantum_coherence,
            processing_time=processing_time,
            metadata={
                "num_raw_thoughts": len(raw_thoughts),
                "quantum_processed": request.quantum_process,
                **quantum_metadata
            }
        )
        
    except Exception as e:
        logger.error(f"Perception error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Perception processing failed: {str(e)}")

@app.post("/quantum-circuit", response_model=QuantumCircuitResult)
async def run_quantum_circuit(embedding: List[float]):
    """
    Run a quantum circuit on a single embedding.
    """
    try:
        # Normalize embedding
        embedding_array = np.array(embedding)
        norm = np.linalg.norm(embedding_array)
        if norm > 0:
            embedding_array = embedding_array / norm
        
        # Run quantum circuit
        state_vector = quantum_thought_circuit(embedding_array[:4])
        probabilities = (np.abs(state_vector) ** 2).tolist()
        
        # Calculate entanglement measure
        entanglement = float(np.sum(np.abs(state_vector[8:])))
        
        return QuantumCircuitResult(
            state_vector=state_vector.tolist(),
            probabilities=probabilities,
            entanglement_measure=entanglement
        )
        
    except Exception as e:
        logger.error(f"Quantum circuit error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quantum processing failed: {str(e)}")

@app.get("/quantum-state")
async def get_quantum_state():
    """Get current quantum device state."""
    return {
        "device_type": "default.qubit",
        "num_wires": 4,
        "quantum_ready": quantum_device is not None,
        "backend_info": "PennyLane quantum simulation"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 