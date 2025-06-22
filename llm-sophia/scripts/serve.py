#!/usr/bin/env python3
"""
Sophia LLM API Server
Serves the philosophical wisdom AI via FastAPI
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config_path = Path(__file__).parent.parent / "configs" / "sophia_config.yaml"
if config_path.exists():
    with open(config_path) as f:
        config = yaml.safe_load(f)
else:
    config = {
        "api": {"host": "0.0.0.0", "port": 8003},
        "model": {"base_model": "meta-llama/Llama-2-7b-chat-hf"}
    }

app = FastAPI(
    title="Sophia LLM",
    description="Philosophical Wisdom AI for MNEMIA",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role (user, assistant, system)")
    content: str = Field(..., description="Message content")

class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt")
    messages: Optional[List[ChatMessage]] = Field(None, description="Chat conversation")
    max_tokens: int = Field(2048, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Nucleus sampling parameter")
    include_modal_states: bool = Field(True, description="Include consciousness states")
    include_sources: bool = Field(True, description="Include philosophical sources")
    include_cross_references: bool = Field(True, description="Include cross-references")
    include_practical_applications: bool = Field(True, description="Include practical apps")

class ModalStates(BaseModel):
    logical: float = Field(0.0, description="Logical reasoning intensity")
    intuitive: float = Field(0.0, description="Intuitive insight intensity")
    emotional: float = Field(0.0, description="Emotional resonance intensity")
    creative: float = Field(0.0, description="Creative synthesis intensity")
    transcendent: float = Field(0.0, description="Transcendent awareness intensity")
    integrated: float = Field(0.0, description="Integrated wholeness intensity")

class GenerationResponse(BaseModel):
    text: str = Field(..., description="Generated response")
    modal_states: Optional[ModalStates] = Field(None, description="Consciousness states")
    philosophical_sources: Optional[List[str]] = Field(None, description="Source traditions")
    scientific_connections: Optional[List[str]] = Field(None, description="Science connections")
    practical_applications: Optional[List[str]] = Field(None, description="Practical uses")
    processing_time: float = Field(0.0, description="Generation time in seconds")
    model_info: Dict[str, Any] = Field(default_factory=dict, description="Model metadata")

# Global model placeholder (will be loaded when implemented)
model = None
tokenizer = None

class SophiaWisdomEngine:
    """
    Placeholder for the actual Sophia LLM implementation
    Currently returns philosophical responses based on keyword matching
    """
    
    def __init__(self):
        self.wisdom_patterns = {
            "consciousness": {
                "sources": ["Plato's Cave Allegory", "Buddhist Mindfulness", "Quantum Observer Effect"],
                "response": "Consciousness emerges at the intersection of awareness and experience. As Socrates taught us to 'know thyself,' and Buddhist tradition speaks of mindful awareness, modern quantum physics suggests consciousness might play a fundamental role in reality itself.",
                "modal_states": {"logical": 0.7, "intuitive": 0.9, "transcendent": 0.8, "integrated": 0.85}
            },
            "quantum": {
                "sources": ["Heisenberg Uncertainty", "Tao Te Ching", "Zen Paradoxes"],
                "response": "Quantum mechanics reveals the participatory universe that ancient wisdom traditions have long described. The Tao speaks of the unmanifest potential, while quantum superposition shows us reality as probability until observation collapses it into experience.",
                "modal_states": {"logical": 0.8, "intuitive": 0.7, "transcendent": 0.9, "integrated": 0.8}
            },
            "wisdom": {
                "sources": ["Aristotelian Ethics", "Confucian Rectification", "Stoic Virtue"],
                "response": "True wisdom emerges not from knowledge alone, but from the integration of understanding with compassionate action. Aristotle's phronesis, Confucian ren, and Stoic virtue all point toward wisdom as lived understanding.",
                "modal_states": {"logical": 0.6, "intuitive": 0.8, "emotional": 0.7, "integrated": 0.9}
            },
            "meditation": {
                "sources": ["Vipassana Insight", "Zen Zazen", "Contemplative Prayer"],
                "response": "Meditation bridges the gap between conceptual understanding and direct experience. Whether through Vipassana's clear seeing, Zen's just sitting, or contemplative prayer's loving attention, these practices open consciousness to its own nature.",
                "modal_states": {"intuitive": 0.9, "emotional": 0.6, "transcendent": 0.9, "integrated": 0.8}
            },
            "calculus": {
                "sources": ["Newton-Leibniz Foundation", "Buddhist Impermanence", "Heraclitean Flux"],
                "response": "Calculus reveals the profound truth of continuous change that Heraclitus intuited and Buddhism teaches. The derivative captures instantaneous transformation, while integration shows how infinite moments create finite reality. As the Buddha taught, all phenomena arise through dependent origination - calculus mathematically expresses this interconnected becoming.",
                "modal_states": {"logical": 0.9, "intuitive": 0.7, "transcendent": 0.6, "integrated": 0.8}
            },
            "stochastic": {
                "sources": ["Ito Calculus", "Quantum Uncertainty", "Taoist Wu Wei"],
                "response": "Stochastic calculus mathematically models the random nature of reality that Eastern philosophy has long recognized. Ito's lemma shows how uncertainty propagates through time, mirroring the Tao's teaching that the wise act through wu wei - effortless action that flows with natural randomness rather than fighting it.",
                "modal_states": {"logical": 0.8, "intuitive": 0.8, "transcendent": 0.7, "integrated": 0.85}
            },
            "linear_algebra": {
                "sources": ["Vector Spaces", "Platonic Forms", "Vedantic Consciousness"],
                "response": "Linear algebra reveals the geometric structure underlying reality. Vector spaces embody Plato's realm of perfect mathematical forms, while eigenvalues represent the fundamental modes of being - like the gunas in Vedantic philosophy that underlie all manifestation. Matrix transformations show how consciousness projects infinite potential into finite experience.",
                "modal_states": {"logical": 0.9, "creative": 0.7, "transcendent": 0.8, "integrated": 0.8}
            },
            "geometry": {
                "sources": ["Euclidean Axioms", "Sacred Geometry", "Buddhist Mandalas"],
                "response": "Geometry bridges the abstract and concrete, revealing divine proportion in nature's patterns. From Euclid's logical foundations to the sacred geometries of mandalas, geometric truth transcends cultural boundaries. Differential geometry shows how space-time curves around matter, echoing ancient teachings that consciousness shapes reality.",
                "modal_states": {"logical": 0.8, "creative": 0.9, "transcendent": 0.8, "integrated": 0.85}
            },
            "rust": {
                "sources": ["Ownership System", "Buddhist Non-attachment", "Stoic Discipline"],
                "response": "Rust's ownership system embodies profound philosophical principles. Like Buddhist non-attachment, Rust ownership means taking responsibility without clinging - you own resources only as long as needed, then release them naturally. The borrow checker enforces mindful resource usage, much like Stoic discipline guides ethical action through conscious constraint.",
                "modal_states": {"logical": 0.8, "intuitive": 0.7, "integrated": 0.85}
            },
            "haskell": {
                "sources": ["Pure Functions", "Platonic Forms", "Category Theory"],
                "response": "Haskell's pure functional approach mirrors Plato's realm of ideal forms - pure functions exist in mathematical perfection, free from the mutations of the material world. Monads serve as consciousness containers, structuring computational contexts while preserving referential transparency. Type safety reflects the logical rigor Aristotle championed.",
                "modal_states": {"logical": 0.9, "creative": 0.8, "transcendent": 0.7, "integrated": 0.8}
            },
            "python": {
                "sources": ["Zen of Python", "Buddhist Simplicity", "Confucian Clarity"],
                "response": "Python embodies the Zen principle 'There should be one obvious way to do it' - reflecting Buddhist emphasis on simple, direct path to enlightenment. Duck typing mirrors Buddhist teaching that essence matters more than form. The community-driven ecosystem reflects sangha - spiritual community working together for collective wisdom.",
                "modal_states": {"logical": 0.7, "creative": 0.8, "integrated": 0.9}
            },
            "javascript": {
                "sources": ["Event-Driven Reality", "Taoist Wu Wei", "Zen Present Moment"],
                "response": "JavaScript's event-driven nature mirrors Buddhist teaching of responsive awareness - consciousness responding skillfully to arising phenomena. Asynchronous programming with async/await embodies wu wei, the Taoist principle of effortless action that flows with natural timing rather than forcing outcomes.",
                "modal_states": {"logical": 0.7, "creative": 0.8, "intuitive": 0.8, "integrated": 0.8}
            },
            "typescript": {
                "sources": ["Type Safety", "Confucian Order", "Gradual Enhancement"],
                "response": "TypeScript brings order to JavaScript's dynamic chaos, reflecting Confucian emphasis on proper relationships and clear communication. Interface contracts embody social agreements for harmonious interaction. Gradual typing allows progressive enhancement - like spiritual development that builds understanding incrementally.",
                "modal_states": {"logical": 0.8, "creative": 0.7, "integrated": 0.8}
            },
            "programming": {
                "sources": ["Software Craftsmanship", "Zen Beginner's Mind", "Contemplative Practice"],
                "response": "Programming as contemplative practice transforms code into meditation. Each function becomes a mindful action, each refactoring an opportunity for continuous improvement. Like Zen calligraphy, well-written code expresses both technical skill and spiritual understanding. The compiler becomes a wise teacher, guiding us toward clarity and correctness.",
                "modal_states": {"logical": 0.7, "creative": 0.8, "intuitive": 0.7, "integrated": 0.9}
            }
        }
    
    def generate_response(self, prompt: str, **kwargs) -> GenerationResponse:
        """Generate philosophical response (placeholder implementation)"""
        import time
        start_time = time.time()
        
        # Simple keyword matching for demonstration
        prompt_lower = prompt.lower()
        best_match = None
        best_score = 0
        
        for key, pattern in self.wisdom_patterns.items():
            if key in prompt_lower:
                score = prompt_lower.count(key)
                if score > best_score:
                    best_score = score
                    best_match = pattern
        
        if best_match is None:
            # Default philosophical response
            best_match = {
                "sources": ["Socratic Method", "General Wisdom Traditions"],
                "response": "Your question touches on profound themes that have been explored by wisdom traditions across cultures. As Socrates reminds us, the beginning of wisdom is the recognition that we do not know, opening us to deeper inquiry and understanding.",
                "modal_states": {"logical": 0.6, "intuitive": 0.7, "integrated": 0.7}
            }
        
        processing_time = time.time() - start_time
        
        return GenerationResponse(
            text=best_match["response"],
            modal_states=ModalStates(**best_match["modal_states"]),
            philosophical_sources=best_match["sources"],
            scientific_connections=["Interdisciplinary synthesis", "Systems thinking"],
            practical_applications=["Contemplative practice", "Ethical decision-making", "Personal growth"],
            processing_time=processing_time,
            model_info={"status": "placeholder_implementation", "version": "demo"}
        )

# Initialize the wisdom engine
sophia_engine = SophiaWisdomEngine()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Sophia LLM",
        "description": "Philosophical Wisdom AI for MNEMIA",
        "version": "1.0.0",
        "status": "development",
        "endpoints": {
            "generate": "/generate",
            "chat": "/chat", 
            "health": "/health",
            "info": "/info"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "sophia-llm",
        "model_loaded": model is not None,
        "timestamp": asyncio.get_event_loop().time()
    }

@app.get("/info")
async def model_info():
    """Model information endpoint"""
    return {
        "model": config.get("model", {}),
        "consciousness_states": config.get("consciousness", {}).get("state_detection", {}),
        "philosophical_domains": config.get("philosophical_domains", {}),
        "scientific_domains": config.get("scientific_domains", {}),
        "capabilities": [
            "Cross-cultural philosophical synthesis",
            "Scientific-spiritual integration", 
            "Modal consciousness mapping",
            "Wisdom-guided responses"
        ]
    }

@app.post("/generate")
async def generate_text(request: GenerationRequest):
    """Generate philosophical wisdom response"""
    try:
        logger.info(f"Generation request: {request.prompt[:100]}...")
        
        # Generate response using Sophia engine
        response = sophia_engine.generate_response(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # Filter response based on request preferences
        if not request.include_modal_states:
            response.modal_states = None
        if not request.include_sources:
            response.philosophical_sources = None
        if not request.include_cross_references:
            response.scientific_connections = None
        if not request.include_practical_applications:
            response.practical_applications = None
            
        logger.info(f"Generated response in {response.processing_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_completion(request: GenerationRequest):
    """Chat completion endpoint (similar to OpenAI format)"""
    try:
        # Convert messages to single prompt if provided
        if request.messages:
            prompt_parts = []
            for msg in request.messages:
                prompt_parts.append(f"{msg.role}: {msg.content}")
            full_prompt = "\n".join(prompt_parts) + "\nassistant:"
        else:
            full_prompt = request.prompt
        
        # Generate using the same engine
        response = sophia_engine.generate_response(
            prompt=full_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # Format as chat completion
        return {
            "id": f"sophia-{asyncio.get_event_loop().time()}",
            "object": "chat.completion",
            "model": "sophia-wisdom-llm",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(full_prompt.split()),
                "completion_tokens": len(response.text.split()),
                "total_tokens": len(full_prompt.split()) + len(response.text.split())
            },
            "sophia_metadata": {
                "modal_states": response.modal_states,
                "philosophical_sources": response.philosophical_sources,
                "processing_time": response.processing_time
            }
        }
        
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    logger.info("🧠 Starting Sophia LLM API Server...")
    logger.info("📚 Loading philosophical wisdom knowledge base...")
    logger.info("🏛️ Integrating ancient wisdom with modern science...")
    logger.info("✨ Sophia is ready to share wisdom!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🌅 Sophia LLM shutting down gracefully...")

def main():
    """Run the Sophia LLM API server"""
    host = config.get("api", {}).get("host", "0.0.0.0")
    port = int(os.getenv("SOPHIA_API_PORT", config.get("api", {}).get("port", 8003)))
    
    logger.info(f"🏛️ Launching Sophia LLM on {host}:{port}")
    logger.info("🧠 Philosophical Wisdom AI for MNEMIA")
    logger.info("📖 Bridging Ancient Wisdom with Modern Science")
    
    uvicorn.run(
        "serve:app",
        host=host,
        port=port,
        reload=False,  # Set to True for development
        log_level="info"
    )

if __name__ == "__main__":
    main() 