# NoteBuddy

NoteBuddy is a local Retrieval-Augmented Generation (RAG) assistant designed for document ingestion, semantic search, evaluation, and LLM orchestration.

---

## Architecture Overview

NoteBuddy is built with a microservice-style modular architecture:

- **Application Service (`services/application`)**: Central FastAPI gateway and orchestration entrypoint.
- **Retrieval Service (`services/retrieval`)**: Handles semantic similarity search and document retrieval via ChromaDB.
- **LLM Service (`services/llm`)**: Interfaces with local Ollama models for text generation and reasoning.
- **Data Service (`services/data`)**: Manages document ingestion, chunking, and embeddings.
- **Web UI (`ui`)**: Frontend interface for querying, retrieval exploration, guardrails, and model evaluations.
- **Guardrails & Evaluation (`guardrails`, `week4`)**: Safety policies, hallucination detection, and benchmark evaluations.

---

## Tech Stack

- **Backend**: Python 3.11+, FastAPI, Uvicorn, Pydantic
- **Vector Database**: ChromaDB
- **LLM / Embeddings**: Ollama (`codellama`, `nomic-embed-text`)
- **Containerization**: Docker & Docker Compose

---

## Getting Started

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (recommended for containerized setup)
- Python 3.11+ (if running locally without Docker)
- [Ollama](https://ollama.ai/) installed locally (if running outside Docker)

### Running with Docker Compose

1. Build and start all services:
   ```bash
   docker compose up --build
   ```

2. Open your browser and navigate to:
   - **Web UI**: `http://localhost:8080`
   - **Application API docs**: `http://localhost:8000/docs`
   - **Retrieval API docs**: `http://localhost:8001/docs`
   - **LLM API docs**: `http://localhost:8002/docs`
   - **Data API docs**: `http://localhost:8003/docs`

---

## Configuration

Copy `.env.example` to create your local `.env`:

```bash
cp .env.example .env
```

Review and adjust variables for Ollama host/port, ChromaDB settings, and embedding models as needed.

---

## Running Tests

Run pytest suite locally:

```bash
pytest
```
