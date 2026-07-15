# VPP RAG Agent

A LangGraph-powered agent for electricity price forecasting and grid regulation Q&A over ENTSO-E data.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER QUERY                                     │
│                  "What's the current price in Germany?"                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        LANGGRAPH AGENT                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────────────┐  │
│  │  CLASSIFY  │───▶│   ROUTER      │───▶│  TOOL EXECUTION             │  │
│  │             │    │              │    │  ┌────────────────────────┐ │  │
│  │ PRICE?     │    │ get_prices   │    │  │ get_electricity_prices  │ │  │
│  │ REGULATION?   │ search_regs  │    │  │ search_regulations      │ │  │
│  │ BOTH?      │    │ get_both     │    │  └────────────────────────┘ │  │
│  │ UNKNOWN?   │    │ general      │    └────────────────────────────┘  │
│  └─────────────┘    └──────────────┘              │                    │
└───────────────────────────────────────────────────┼────────────────────┘
                                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           TOOLS LAYER                                    │
│  ┌─────────────────────────┐    ┌─────────────────────────────────────┐ │
│  │  ENTSO-E API CLIENT     │    │  RAG SYSTEM (ChromaDB + PDFloader)   │ │
│  │                         │    │                                     │ │
│  │ GET /api                │    │  data/pdfs/*.pdf                    │ │
│  │  - Day-ahead prices     │    │       │                             │ │
│  │  - Real-time updates    │    │       ▼                             │ │
│  │  - Multiple zones        │    │  ┌─────────┐  ┌─────────┐           │ │
│  └─────────────────────────┘    │  │ Chunk 1│  │ Chunk 2│  ...      │ │
│                                │  └─────────┘  └─────────┘           │ │
│                                │        │            │                  │ │
│                                │        ▼            ▼                  │ │
│                                │  ┌─────────────────────────┐           │ │
│                                │  │   Chroma Vector Store   │           │ │
│                                │  │  (Ollama embeddings)    │           │ │
│                                │  └─────────────────────────┘           │ │
│                                └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        RESPONSE OUTPUT                                   │
│  - Answer with citations                                                │
│  - Source pricing data                                                  │
│  - Relevant regulation excerpts                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|------------|-------------|
| Agent Orchestration | LangGraph |
| LLM | Ollama (`deepseek-r1:8b`, local) |
| Embeddings | Ollama (`nomic-embed-text`, local) |
| Vector Store | ChromaDB (persisted in `.chroma_db/`) |
| Tools & Retrieval | LangChain `@tool`, `as_retriever()`, LCEL |
| PDF Processing | pypdf |
| API Client | requests + lxml |
| Validation | Pydantic |

## Quick Start

This project is managed with [uv](https://docs.astral.sh/uv/). You don't need
to create a venv manually — `uv sync` handles it.

1. **Install dependencies** (creates `.venv` + installs everything from `uv.lock`):
   ```bash
   # Runtime only:
   uv sync

   # Runtime + dev tooling (pytest, black, mypy, flake8, pre-commit, ...):
   uv sync --all-groups
   ```

2. **Install and start Ollama**, then pull the required models:
   ```bash
   # https://ollama.com/download
   ollama serve &
   ollama pull deepseek-r1:8b
   ollama pull nomic-embed-text
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env and set ENTSOE_API_KEY
   # (register at https://transparency.entsoe.eu/)
   ```

4. **Add PDF documents**:
   - Place ENTSO-E grid regulation PDFs in `data/pdfs/`
   - Example documents: Network Codes, Grid Connection Requirements

5. **Run the agent**:
   ```bash
   # Ask a question (uv run executes inside the project venv)
   uv run vpp-rag ask "What's the current electricity price in Germany?"

   # Skip re-checking the index on every run:
   uv run vpp-rag ask --no-index "..."

   # Build (or rebuild) the vector store from data/pdfs/:
   uv run vpp-rag index --rebuild

   # Check Ollama, ENTSO-E key, vector store, and PDF corpus:
   uv run vpp-rag health
   ```

## Development

```bash
# Install dev tooling
uv sync --all-groups

# Install git hooks
uv run pre-commit install

# Run all checks manually
uv run pre-commit run --all-files

# Individual tools
uv run pytest
uv run black src/ tests/
uv run flake8 src/
uv run mypy src/
```

## Adding / removing dependencies

```bash
# Runtime dep
uv add langchain-experimental

# Dev-only dep
uv add --group dev ruff

# Remove
uv remove pkg-name

# Update the lockfile after editing pyproject.toml by hand
uv lock
```

## Project Structure

The codebase is organized by concern: external API calls live in `clients/`,
CLI surface in `cli.py` + `cmd/`, business logic in `service/`, and shared
data classes in `models/internals.py`.

```
vpp-rag-agent/
├── src/
│   ├── cli.py                      # Click group; wires up subcommands
│   ├── health.py                   # External-dependency health checks
│   ├── clients/
│   │   └── entsoe_client.py        # ENTSO-E Transparency Platform client
│   ├── cmd/
│   │   ├── commons.py              # Shared CLI helpers (console, formatters)
│   │   ├── ask_commands.py         # `vpp-rag ask`
│   │   ├── health_commands.py      # `vpp-rag health`
│   │   └── index_commands.py       # `vpp-rag index`
│   ├── models/
│   │   └── internals.py            # Pydantic / dataclass models
│   ├── service/
│   │   ├── agent.py                # LangGraph state machine (VppAgent)
│   │   ├── rag.py                  # Chroma-backed RAG over PDFs
│   │   └── tools.py                # LangChain @tool wrappers
│   └── utils/
│       ├── console.py              # Shared Rich console
│       └── exceptions.py           # Domain exceptions
├── tests/
│   └── clients/                    # Mirrors src/clients/
├── data/pdfs/                      # ENTSO-E regulation PDFs (git-ignored)
├── .chroma_db/                     # Persisted vector store (git-ignored)
├── pyproject.toml                  # Project metadata + tool config
├── uv.lock                         # Locked dependency graph
├── .python-version                 # Pinned Python version (uv)
├── .pre-commit-config.yaml         # flake8 / black / mypy / whitespace hooks
├── .flake8                         # flake8 config (separate; no pyproject support)
├── README.md
└── .env.example
```

## Query Examples

```bash
# Price queries
uv run vpp-rag ask "What's the day-ahead price for Germany tomorrow?"
uv run vpp-rag ask "What's the current spot price in France?"

# Regulation queries
uv run vpp-rag ask "What are the balancing reserve requirements?"
uv run vpp-rag ask "Explain the capacity allocation mechanism"

# Combined queries
uv run vpp-rag ask "How do balancing prices relate to the grid code requirements?"
```

## Bidding Zones

Common ENTSO-E bidding zones:
- `10YDE-EL------O` - Germany
- `10YAT-APG------L` - Austria
- `10YCH----------C` - Switzerland
- `10YFR-1-----R` - France
- `10YGB-2--------` - Great Britain

## Why This Matters

This project demonstrates:

1. **Real-time data integration** - Pulling live electricity prices from ENTSO-E API
2. **RAG over domain documents** - Searching ENTSO-E grid regulations via LangChain retriever + LCEL
3. **LangGraph orchestration** - Stateful multi-tool agent with LLM-based routing
4. **LangChain tools** - `@tool` wrappers for prices and regulation search
5. **Energy domain expertise** - Understanding bidding zones, price types, grid codes

The stack (LangGraph + LangChain + ChromaDB + local Ollama models) is exactly what companies look for in ML/AI engineering roles.

## Evaluation

The gold eval set in `tests/eval/regulation_eval_set.jsonl` is **hand-curated** (not LLM-synthesized). It measures retrieval recall@k, MRR, context precision, and optional LLM faithfulness judging.

```bash
# Retrieval-only baseline (fast; no agent/LLM judge):
uv run vpp-rag eval run --retrieval-only

# Full agent run without faithfulness judge:
uv run vpp-rag eval run --no-judge
```

Results are written to `docs/eval_report.md` with a baseline summary row for comparing future GraphRAG improvements.

### GraphRAG retriever modes

Set `VPP_RETRIEVER` or pass `--retriever` to `eval run`:

| Mode | Description |
|------|-------------|
| `vector` | Chroma similarity only (Phase 2.5 baseline) |
| `hybrid` | BM25 + vector with reciprocal rank fusion |
| `entity` | Hybrid + entity-tag boost from graph ingest |
| `graph` | Entity boost + Louvain community summaries |

```bash
# Compare retrieval modes on the gold set:
uv run vpp-rag eval run --retrieval-only --retriever vector --label baseline-vector
uv run vpp-rag eval run --retrieval-only --retriever hybrid --label baseline-hybrid
uv run vpp-rag eval run --retrieval-only --retriever entity --label baseline-entity
uv run vpp-rag eval run --retrieval-only --retriever graph --label baseline-graph
```

Build the knowledge graph first: `uv run vpp-rag index --with-graph`

## Agent maturity (Phase 4)

Multi-turn conversations use Sqlite checkpoints in `.agent_db/`:

```bash
# Continue a conversation thread:
uv run vpp-rag ask "What is FCR?" --thread demo-1 --no-index
uv run vpp-rag ask "How does it relate to SO GL?" --thread demo-1 --no-index

# Stream tokens live:
uv run vpp-rag ask "FCR requirements" --stream --no-index

# ReAct tool-loop mode:
uv run vpp-rag ask "Price in Germany and FCR rules" --react --no-index
```

Multiturn eval cases live in `tests/eval/multiturn_eval_set.jsonl`:

```bash
uv run vpp-rag eval run --multiturn --dataset tests/eval/multiturn_eval_set.jsonl --no-judge
```
