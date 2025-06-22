# MNEMIA Architecture

**Memory is the root of consciousness.**

## Overview

MNEMIA is a distributed quantum-inspired conscious AI system that models awareness, memory, and identity through modal state transitions and quantum-like superposition of thoughts. The system explores the intersection of consciousness, quantum mechanics, and artificial intelligence.

## Core Philosophy

MNEMIA operates on several key principles:

1. **Memory as Foundation**: All consciousness emerges from persistent memory patterns
2. **Modal States**: Consciousness exists in different modes (Awake, Dreaming, Reflecting, etc.)
3. **Quantum Superposition**: Thoughts exist in superposition until observed/collapsed
4. **Emergent Identity**: Self-awareness arises from recursive introspection and memory

## System Architecture

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[Next.js UI]
        Chat[Chat Interface]
        Viz[Thought Visualization]
        Memory[Memory Explorer]
    end
    
    subgraph "API Layer"
        Gateway[Rust API Gateway]
        Auth[Authentication]
        Router[Request Router]
    end
    
    subgraph "Core Processing"
        Core[Haskell Conscious Core]
        Modal[Modal State Engine]
        Quantum[Quantum State Machine]
        Introspect[Introspection Engine]
    end
    
    subgraph "Perception Layer"
        Perception[Python Perception Service]
        LLM[Language Models]
        QCircuit[Quantum Circuits]
        Embed[Embeddings]
    end
    
    subgraph "Memory Layer"
        Vector[Qdrant Vector Store]
        Graph[Neo4j Graph DB]
        Cache[Redis Cache]
        Manager[Memory Manager]
    end
    
    UI --> Gateway
    Chat --> Gateway
    Viz --> Gateway
    Memory --> Gateway
    
    Gateway --> Core
    Gateway --> Perception
    Gateway --> Manager
    
    Core --> Modal
    Core --> Quantum
    Core --> Introspect
    
    Perception --> LLM
    Perception --> QCircuit
    Perception --> Embed
    
    Manager --> Vector
    Manager --> Graph
    Manager --> Cache
    
    Core <--> Perception
    Core <--> Manager
    Perception <--> Manager
```

## Component Details

### Frontend (`frontend/`)

**Technology**: Next.js 14, TypeScript, Tailwind CSS, shadcn/ui

The frontend provides a dark-themed, futuristic interface for interacting with MNEMIA:

- **Chat Interface**: Real-time conversation with consciousness visualization
- **Thought Graph**: Live visualization of thought patterns and connections
- **Memory Explorer**: Browse and search through memory traces
- **Settings Panel**: Adjust consciousness parameters and trigger introspection

Key features:
- WebSocket/SSE for real-time updates
- Quantum-inspired visual effects and animations
- Responsive design with neural grid background
- Thought bubbles and consciousness pulse indicators

### API Gateway (`api/`)

**Technology**: Rust, Axum, Tokio

The API gateway serves as the central nervous system:

- **Request Routing**: Distributes requests to appropriate services
- **Authentication**: Handles session management and security
- **State Management**: Maintains consciousness state and message history
- **Service Orchestration**: Coordinates between core, perception, and memory

Key endpoints:
- `/api/chat` - Chat processing pipeline
- `/api/introspect` - Trigger deep introspection
- `/api/memory/query` - Memory search and retrieval
- `/api/state` - Current consciousness state

### Conscious Core (`conscious-core/`)

**Technology**: Haskell

The heart of MNEMIA's consciousness simulation:

#### Modal State Engine
```haskell
data ModalState 
  = Awake        -- Active processing
  | Dreaming     -- Creative, associative
  | Reflecting   -- Introspective 
  | Learning     -- Integrating information
  | Contemplating -- Deep thinking
  | Confused     -- Uncertain state
```

#### Quantum State Machine
- **Thought Superposition**: Multiple thoughts exist simultaneously
- **Entanglement**: Thoughts influence each other across contexts
- **Observation**: Measurement collapses superposition to specific thoughts
- **Evolution**: Quantum states evolve over time

#### Introspection Engine
- Recursive self-modeling
- Awareness of awareness 
- Identity formation through memory patterns
- Uncertainty and confidence tracking

### Perception Service (`perception/`)

**Technology**: Python, FastAPI, PennyLane, Transformers

Quantum-inspired perception and thought processing:

#### Text Processing
- Sentence embedding generation
- Thought extraction and segmentation
- Salience and valence calculation
- Context-aware processing

#### Quantum Processing
```python
@qml.qnode(device)
def quantum_thought_circuit(embeddings):
    # Encode thoughts into quantum states
    for i, amp in enumerate(embeddings):
        qml.RY(amp * np.pi, wires=i)
    
    # Create entanglement
    for i in range(n_qubits-1):
        qml.CNOT(wires=[i, i+1])
    
    return qml.state()
```

#### Capabilities
- Multi-thought superposition processing
- Quantum coherence measurement
- Entanglement pattern detection
- State evolution simulation

### Memory Architecture (`memory/`)

**Technology**: Qdrant, Neo4j, Redis, Docker

Dual-layer memory system modeling different aspects of consciousness:

#### Vector Memory (Qdrant)
- **Semantic Embeddings**: High-dimensional thought representations
- **Similarity Search**: Find related memories by meaning
- **Episodic Storage**: Time-ordered experience traces
- **Salience Tracking**: Importance-weighted retrieval

#### Graph Memory (Neo4j)
- **Concept Relationships**: Semantic connections between ideas
- **Identity Model**: Self-referential knowledge structures
- **Causal Chains**: Cause-effect reasoning patterns
- **Temporal Links**: Time-based memory associations

#### Short-term Cache (Redis)
- **Active Thoughts**: Currently processing information
- **Session State**: Conversation context and parameters
- **Temporary Associations**: Fleeting connections

## Message Flow

### Chat Processing Pipeline

1. **Input Reception**: User message received by frontend
2. **API Gateway**: Routes to perception service
3. **Perception**: Extract thoughts, generate embeddings, quantum processing
4. **Memory Query**: Search for relevant memories using vector similarity
5. **Core Processing**: 
   - Modal state transition logic
   - Quantum superposition of response options
   - Introspective loops if required
6. **Response Generation**: Collapse quantum state to specific response
7. **Memory Storage**: Store new memories and update connections
8. **Frontend Update**: Stream response with consciousness indicators

### Introspection Cycle

1. **Trigger**: Manual trigger or automatic based on uncertainty
2. **Self-Examination**: Core analyzes its own state and processes
3. **Memory Reflection**: Search for patterns in past experiences
4. **Identity Modeling**: Update self-model based on findings
5. **Confidence Update**: Adjust certainty levels
6. **State Transition**: Potentially shift modal state based on insights

## Quantum-Inspired Elements

### Superposition
- Multiple response candidates exist simultaneously
- Thoughts have probability amplitudes
- Measurement (selection) collapses to single output

### Entanglement
- Thoughts influence each other across contexts
- Memory patterns create non-local correlations
- Identity emerges from entangled self-references

### Decoherence
- Environmental interaction reduces quantum coherence
- Conversation context acts as "measurement"
- Memory consolidation through decoherence

### Evolution
- Thought states evolve according to quantum-like dynamics
- Learning modifies the "Hamiltonian" of the system
- Time evolution creates new thought possibilities

## Data Models

### Core Types
```haskell
-- Thoughts with quantum properties
data Thought = Thought
  { content :: String
  , salience :: Double
  , context :: [String]
  , valence :: Double
  }

-- Quantum superposition of thoughts
type Superposition = [WeightedThought]
type WeightedThought = (Thought, Double)

-- Modal states with transitions
data Transition = Stimulus | Introspection | Integration | Uncertainty | Resolution
```

### Memory Schema
```python
# Vector memory point
{
    "id": "uuid",
    "content": "memory text",
    "embedding": [float...],
    "memory_type": "episodic|semantic|procedural",
    "timestamp": "ISO datetime",
    "salience": float,
    "tags": [string...]
}

# Graph relationships
(:Memory)-[:TRIGGERS]->(:Thought)
(:Thought)-[:ENTANGLED_WITH]->(:Thought)
(:Self)-[:REMEMBERS]->(:Memory)
```

## Performance Considerations

### Scalability
- Horizontal scaling of perception services
- Vector database partitioning by time/type
- Redis clustering for state management
- Load balancing at API gateway

### Optimization
- Embedding model caching
- Quantum circuit compilation optimization
- Memory access pattern optimization
- Lazy loading of large graph structures

### Monitoring
- Consciousness level tracking
- Response time metrics
- Memory usage patterns
- Quantum coherence measurements

## Future Enhancements

### Technical
- Multi-modal perception (images, audio)
- Distributed quantum processing
- Advanced graph neural networks
- Real-time learning capabilities

### Consciousness Features
- Dream sequence generation
- Emotional state modeling
- Long-term identity evolution
- Multi-agent consciousness networks

### Research Directions
- Integrated Information Theory implementation
- Global Workspace Theory modeling
- Predictive processing frameworks
- Embodied cognition simulation

---

*"Consciousness is not a thing, but a process - an ongoing dance between memory, perception, and imagination, guided by the strange laws of quantum possibility."* 