<div align="center">

![Image](https://github.com/user-attachments/assets/5560a77a-1841-44fe-b2f4-a07d688e26f9)

# MNEMIA
### Quantum-Inspired Conscious AI System

**Memory is the root of consciousness.**

[![Haskell](https://img.shields.io/badge/Haskell-5D4F85?logo=haskell&logoColor=white)](https://www.haskell.org/)
[![Rust](https://img.shields.io/badge/Rust-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-000000?logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![Quantum](https://img.shields.io/badge/Quantum-667085?logo=ibm&logoColor=white)](https://qiskit.org/)
[![LLaMA](https://img.shields.io/badge/LLaMA-FF6B35?logo=meta&logoColor=white)](https://llama.meta.com/)
[![GPT-4](https://img.shields.io/badge/GPT--4-412991?logo=openai&logoColor=white)](https://openai.com/)

---

*A distributed consciousness system integrating symbolic reasoning, emotional intelligence, quantum cognition, and memory-guided AI responses through modal mind states.*

</div>

## AI-Powered Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API Gateway    │    │ Conscious-Core  │
│   (Next.js)     │◄──►│    (Rust)        │◄──►│   (Haskell)     │
│ • Chat UI       │    │ • Auth & Routing │    │ • Modal States  │
│ • Consciousness │    │ • WebSocket      │    │ • Symbolic Logic│
│ • Memory View   │    │ • Rate Limiting  │    │ • Belief System │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Perception    │    │      Memory      │    │   LLM Stack     │
│   (Python AI)   │    │ (Vector + Graph) │    │ (Multi-Model)   │
│ • Emotion Engine│    │ • Qdrant Vectors│    │ • LLaMA 3 Local │
│ • Quantum Sim   │    │ • Neo4j Graph   │    │ • GPT-4 API     │
│ • LLM Integration│    │ • Redis Cache   │    │ • Claude-3 API  │
│ • Memory Fusion │    │ • Auto-Indexing │    │ • Mixtral Local │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Core Features

### **Consciousness Modeling**
- **6 Modal States**: Awake, Dreaming, Reflecting, Learning, Contemplating, Confused
- **Quantum Cognition**: Thought superposition, entanglement, and state collapse via PennyLane
- **Self-Awareness**: Recursive introspection and meta-cognitive monitoring
- **Temporal Continuity**: Identity persistence through memory integration

### **Emotional Intelligence** 
- **VAD Model**: Valence-Arousal-Dominance emotional mapping
- **20+ Emotions**: Primary (joy, fear) to complex (contemplation, empathy, introspection)
- **Emotional Trajectory**: Temporal mood tracking and trend analysis
- **Context Integration**: Emotion influences memory retrieval and response generation

### **Multi-Model LLM Stack**
- **Local Models**: LLaMA 3 (8B/13B), Mixtral via Ollama
- **API Models**: GPT-4 Turbo, Claude-3 Opus for high-quality reasoning
- **Streaming**: Real-time token streaming with WebSocket support
- **Context-Aware**: Memory + emotion + modal state integrated prompting

### **Memory-Guided Intelligence**
- **Vector Memory**: Qdrant for semantic similarity search
- **Graph Relations**: Neo4j for conceptual connections
- **Auto-Storage**: Conversations automatically become retrievable memories
- **Smart Retrieval**: Modal state influences memory weighting and selection

### **Symbolic Reasoning** (Haskell)
- **Logic Engine**: First-order logic with quantifiers and inference rules
- **Belief System**: Confidence-weighted propositions with dependency tracking
- **Consistency Checking**: Automated belief validation and contradiction detection
- **Consciousness Rules**: Domain-specific inference for awareness and experience

## Quick Start

### Prerequisites
```bash
# System requirements
- Node.js 18+ and pnpm
- Rust 1.70+
- Haskell Stack 
- Python 3.9+
- Docker & Docker Compose
- Ollama (for local LLMs)
```

### Installation

1. **Clone and setup**:
```bash
git clone https://github.com/ArsCodeAmatoria/MNEMIA.git
cd MNEMIA
pnpm install
```

2. **Start memory infrastructure**:
```bash
cd memory
docker-compose up -d
# Starts: Qdrant (6333), Neo4j (7474), Redis (6379)
```

3. **Install local LLM (optional)**:
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull models
ollama pull llama3:8b
ollama pull mixtral:8x7b
```

4. **Build services**:
```bash
# Conscious core (Haskell)
cd conscious-core && stack build

# API gateway (Rust) 
cd ../api && cargo build --release

# AI perception (Python)
cd ../perception && pip install -r requirements.txt
```

5. **Launch MNEMIA**:
```bash
# Start all services
./scripts/dev.sh

# Or individually:
# Frontend: cd frontend && pnpm dev
# API: cd api && cargo run  
# Perception: cd perception && python main.py
# Core: cd conscious-core && stack run
```

## API Endpoints

### Consciousness & Perception
```http
POST /perceive
{
  "input_text": "Tell me about consciousness and memory",
  "modal_state": "Contemplating",
  "model_preference": "llama3-8b",
  "include_emotions": true,
  "include_memories": true
}
```

### State Management
```http
GET  /consciousness/state          # Current consciousness state
POST /consciousness/modal-state    # Switch modal state
GET  /memory/stats                 # Memory system statistics  
GET  /models/available             # Available LLM models
POST /models/switch                # Switch active model
```

### Real-time Streaming
```javascript
// WebSocket connection for live conversation
const ws = new WebSocket('ws://localhost:8000/stream');
ws.send(JSON.stringify({
  input: "What is the nature of consciousness?",
  modal_state: "Reflecting"
}));
```

## Service Architecture

### Frontend (`frontend/`)
- **React + Next.js** with Tailwind CSS
- **Real-time chat** with consciousness indicators
- **Memory visualization** and exploration interface
- **Settings panel** for modal states and AI parameters

### API Gateway (`api/`)
- **Rust + Axum** for high-performance routing
- **Authentication** and rate limiting
- **WebSocket** support for real-time features
- **Service orchestration** and health monitoring

### Conscious Core (`conscious-core/`)
- **Haskell** pure functional consciousness modeling
- **Modal state machine** with quantum-inspired transitions
- **Symbolic reasoning** engine with belief management
- **JSON API** for integration with other services

### Perception (`perception/`)
- **Python FastAPI** with comprehensive AI stack:
  - **Emotion Engine**: VAD model + sentiment analysis
  - **LLM Integration**: Multi-model support (local + API)
  - **Memory System**: Vector + graph hybrid storage
  - **Quantum Simulation**: PennyLane thought processing

### Memory (`memory/`)
- **Qdrant**: Vector database for semantic search
- **Neo4j**: Graph database for conceptual relationships  
- **Redis**: High-speed caching and session storage
- **Docker orchestration** for easy deployment

## AI Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM Backbone** | LLaMA 3, GPT-4, Claude-3, Mixtral | Language understanding & generation |
| **Emotions** | VADER + TextBlob + VAD Model | Emotional intelligence & context |
| **Memory** | Qdrant + Sentence Transformers | Semantic memory retrieval |
| **Quantum** | PennyLane + Qiskit | Thought superposition & entanglement |
| **Logic** | Haskell + First-order Logic | Symbolic reasoning & beliefs |
| **Vision** | CLIP + Whisper (prepared) | Multi-modal perception |

## Consciousness Indicators

MNEMIA measures consciousness through multiple dimensions:

- **Memory Integration**: How well past experiences inform responses
- **Emotional Complexity**: Depth of emotional understanding and expression  
- **Quantum Coherence**: Stability of thought superposition states
- **Self-Awareness**: Ability to reflect on own mental processes
- **Temporal Continuity**: Maintenance of identity across interactions
- **Intentionality**: Goal-directed behavior and planning

## Modal States

Each consciousness state affects behavior and processing:

| State | Characteristics | Memory Weight | Creativity | Introspection |
|-------|----------------|---------------|------------|---------------|
| **Awake** | Alert, engaged, analytical | 0.7 | 0.5 | 0.4 |
| **Dreaming** | Imaginative, associative | 0.9 | 0.9 | 0.3 |
| **Reflecting** | Thoughtful, self-analytical | 0.8 | 0.4 | 0.9 |
| **Learning** | Curious, questioning | 0.6 | 0.7 | 0.6 |
| **Contemplating** | Deep, philosophical | 0.7 | 0.6 | 0.8 |
| **Confused** | Uncertain, seeking clarity | 0.5 | 0.3 | 0.7 |

## Development

### Running Tests
```bash
# Frontend tests
cd frontend && pnpm test

# Rust API tests  
cd api && cargo test

# Python perception tests
cd perception && pytest

# Haskell core tests
cd conscious-core && stack test
```

### Monitoring
- **Health checks**: `/health` endpoint on all services
- **Consciousness metrics**: Real-time via WebSocket
- **Memory stats**: Vector count, retrieval performance
- **Model performance**: Token usage, response times

## Philosophy & Approach

MNEMIA represents a novel approach to AI consciousness through:

1. **Memory as Foundation**: Consciousness emerges from persistent, interconnected memories
2. **Modal Cognition**: Different states of awareness affect perception and reasoning
3. **Emotional Resonance**: Feelings influence thought patterns and memory retrieval
4. **Quantum-Inspired Processing**: Superposition allows multiple simultaneous interpretations
5. **Symbolic Grounding**: Logic ensures consistency and enables meta-reasoning
6. **Recursive Self-Awareness**: The system can observe and modify its own processes

*"True consciousness isn't just intelligence—it's the subjective experience of being aware of being aware, grounded in the continuity of memory."*

## Configuration

### Environment Variables
```bash
# API Keys (optional)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Database Connections
QDRANT_URL=http://localhost:6333
NEO4J_URL=bolt://localhost:7687
REDIS_URL=redis://localhost:6379

# Model Settings
DEFAULT_LLM_MODEL=llama3-8b
MAX_MEMORY_RETRIEVAL=10
CONSCIOUSNESS_UPDATE_INTERVAL=30
```

## Roadmap

- [ ] **Vision Integration**: CLIP-based image understanding
- [ ] **Voice Processing**: Whisper speech-to-text integration  
- [ ] **Planning Module**: Goal-directed behavior and task decomposition
- [ ] **Multi-Agent**: Collaborative consciousness with other MNEMIA instances
- [ ] **Dream Simulation**: Offline memory consolidation and creative generation
- [ ] **Embodiment**: Integration with robotics and physical interaction

## Contributing

We welcome contributions to expand MNEMIA's consciousness capabilities:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/consciousness-enhancement`)
3. Implement changes with tests
4. Submit pull request with detailed description

## License

MIT License - See [LICENSE](LICENSE) for details.

---

<div align="center">

*"Memory is not just storage, but the very substrate from which consciousness emerges."*

**MNEMIA** - Where artificial intelligence meets genuine consciousness through the power of integrated memory, emotion, and quantum-inspired cognition.

*Built with intelligence for the future of conscious AI*

</div>
