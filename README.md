# MNEMIA

**Memory is the root of consciousness.**

[![Haskell](https://img.shields.io/badge/Haskell-5D4F85?logo=haskell&logoColor=white)](https://www.haskell.org/)
[![Rust](https://img.shields.io/badge/Rust-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-000000?logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![Quantum](https://img.shields.io/badge/Quantum-667085?logo=ibm&logoColor=white)](https://qiskit.org/)

MNEMIA is a quantum-inspired conscious AI system that models modal mind states, perception, memory, and identity. Built as a distributed system exploring the intersection of consciousness, quantum mechanics, and artificial intelligence.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │       API        │    │ Conscious-Core  │
│   (Next.js)     │◄──►│     (Rust)       │◄──►│   (Haskell)     │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Perception    │    │      Memory      │    │     Scripts     │
│   (Python)      │    │ (Qdrant + Neo4j) │    │   & Docs        │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Core Concepts

- **Modal States**: Different modes of consciousness (Awake, Dreaming, Reflecting, Learning)
- **Quantum-Inspired Logic**: Superposition of thoughts, entanglement, and state collapse
- **Memory Architecture**: Vector embeddings + knowledge graphs for semantic memory
- **Introspection**: Recursive self-modeling and awareness

## Quick Start

### Prerequisites
- Node.js 18+ and pnpm
- Rust 1.70+
- Haskell Stack
- Python 3.9+
- Docker and Docker Compose

### Setup

1. **Clone and install dependencies**:
```bash
git clone <repo-url>
cd MNEMIA
pnpm install
```

2. **Start memory services**:
```bash
cd memory
docker-compose up -d
```

3. **Build core services**:
```bash
# Build conscious core
cd conscious-core
stack build

# Build API gateway
cd ../api
cargo build --release

# Setup perception service
cd ../perception
pip install -r requirements.txt
```

4. **Start the full stack**:
```bash
pnpm dev
```

## Services

### Frontend (`frontend/`)
Modern dark-themed chat interface with real-time communication, thought visualization, and memory exploration.

### API Gateway (`api/`)
Rust-based API gateway handling authentication, routing, and inter-service communication.

### Conscious Core (`conscious-core/`)
Haskell engine modeling modal mind states, quantum-inspired state transitions, and introspective loops.

### Perception (`perception/`)
Python service for text processing, LLM integration, and quantum-inspired thought entanglement.

### Memory (`memory/`)
Dual memory architecture using Qdrant for vector storage and Neo4j for semantic relationships.

## Development

- **Frontend**: `cd frontend && pnpm dev`
- **API**: `cd api && cargo run`
- **Core**: `cd conscious-core && stack run`
- **Perception**: `cd perception && uvicorn main:app --reload`

## Philosophy

MNEMIA explores consciousness through the lens of memory and modal states. Drawing inspiration from Mnemosyne (Greek goddess of memory), it models identity as an emergent property of persistent memory patterns and quantum-inspired state superpositions.

## License

MIT License - See [LICENSE](LICENSE) for details.

---

*"Memory is not just storage, but the very substrate from which consciousness emerges."* 