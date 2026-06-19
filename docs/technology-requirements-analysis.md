# Project Analysis: LangChain/LangGraph/LangSmith & Graph Technologies

## Executive Summary

**vpp-rag-agent** is a focused domain CLI that combines a **LangGraph state machine**, **vector RAG over ENTSO-E regulation PDFs**, and **live price data** from the ENTSO-E API. It is a credible portfolio piece for **LangGraph basics** and **vector RAG**, but it only partially covers the two requirement areas listed below.

| Requirement area | Fit | Score (for interview/portfolio signaling) |
|------------------|-----|-------------------------------------------|
| LangChain / LangGraph / LangSmith | Partial | **7/10** |
| Knowledge graphs / GraphRAG / graph DB | Missing | **1/10** |

The strongest signal today is **LangGraph orchestration on a real energy-domain use case**, plus **LangChain tools, retriever/LCEL, and structured LLM routing**. The largest gaps are **LangSmith observability**, **graph-based retrieval**, and **advanced LangGraph patterns** (checkpointing, streaming).

**Graph storage strategy:** Phases 2–3 use **NetworkX + GraphML** by default (local-first, zero extra infra). A thin **`GraphStore` abstraction** keeps an optional **Neo4j backend** (Phase 2b) available without complicating the core project.

---

## 1. How Well Does This Project Meet the Requirements?

### 1.1 LangChain, LangGraph, LangSmith

#### LangGraph — Strong baseline (demonstrated)

The core of the project is a real `StateGraph` with conditional routing in `src/service/agent.py`:

- Custom `AgentState` TypedDict with typed fields (`query_type`, `prices`, `regulation_context`, etc.)
- Conditional edges based on LLM-structured query classification
- Multi-step pipeline: classify → fetch data → generate answer
- Separation of orchestration (`agent.py`) from data access (`rag.py`, `entsoe_client.py`, `tools.py`)

**What is still missing for a senior-level LangGraph story:**

- No **checkpointing** or conversation memory
- No **streaming** (`.stream()` / `.astream()`)
- No **human-in-the-loop** or interrupt nodes
- No **LLM-driven tool selection** — routing is structured classification, tools are invoked by graph nodes
- No use of `langgraph-prebuilt` (e.g. `create_react_agent`)

#### LangChain — Improved (component + idioms)

| LangChain piece | Used? | Where |
|-----------------|-------|-------|
| `ChatOllama` | Yes | `src/service/agent.py` |
| `OllamaEmbeddings` | Yes | `src/service/rag.py` |
| `Chroma` vector store | Yes | `src/service/rag.py` |
| `HumanMessage` / `SystemMessage` | Yes | `src/service/agent.py` |
| `Document` | Yes | `src/service/rag.py` |
| `RecursiveCharacterTextSplitter` | Yes | `src/service/rag.py` |
| `@tool` / `StructuredTool` | Yes | `src/service/tools.py` |
| `as_retriever()` / LCEL chain | Yes | `src/service/rag.py` |
| `with_structured_output()` | Yes | `src/service/agent.py` |
| `llm.bind_tools()` | Yes | `src/service/agent.py` |
| LCEL answer chains | No | Answer generation is direct `invoke()` |
| `create_react_agent` | No | — |
| `langchain_community` loaders | No | Declared but unused |

#### LangSmith — Not used (deferred to Phase 5)

- No `langsmith` direct dependency
- No application imports or tracing configuration
- No run logging, datasets, or evaluations
- `langsmith` may appear only as a **transitive** dependency of `langchain-core`

LangSmith is intentionally out of scope until Phase 5. The project does not claim observability it does not implement.

---

### 1.2 Knowledge Graphs, GraphRAG, Graph Storage

**No graph technology is present anywhere in the codebase.**

Current retrieval is **flat vector search only**:

- PDFs → chunks → embeddings → Chroma
- Query → retriever/LCEL chain → context string

There is no entity/relation extraction, graph construction, graph traversal, GraphRAG community summaries, or hybrid vector + graph retrieval.

**Why this matters for this domain:** ENTSO-E regulations are inherently **relational** — network codes reference articles, zones map to TSOs, balancing products link to reserve types and activation rules. Relational questions benefit from **entity-linked graph traversal**, not just semantic similarity over text chunks.

**Why NetworkX over Neo4j by default:**

| Criterion | NetworkX + GraphML | Neo4j |
|-----------|-------------------|-------|
| Fits local CLI scale (hundreds–low thousands of nodes) | ✅ | ⚠️ Often overkill |
| Matches local-first stack (Ollama + Chroma) | ✅ | ❌ Requires Docker/service |
| GraphRAG patterns (paths, communities) | ✅ Native | ✅ Via Cypher |
| Resume: knowledge graphs / GraphRAG | ✅ | ✅ |
| Resume: industrial graph DB | ❌ | ✅ |
| Zero extra infra for contributors | ✅ | ❌ |

Neo4j remains an **optional upgrade path** (Phase 2b) for roles or deployments that explicitly need a graph database.

---

### 1.3 Overall Assessment Matrix

| Technology | Present? | Depth | Notes |
|------------|----------|-------|-------|
| LangGraph StateGraph | Yes | Medium–High | Core orchestration pattern is solid |
| LangChain integrations | Yes | Medium | Tools, retriever, structured output added |
| LangSmith tracing | No | None | Deferred to Phase 5 |
| Vector RAG | Yes | Medium | Retriever + LCEL; no reranker or hybrid search |
| Knowledge graph | No | None | Planned: NetworkX + GraphML |
| GraphRAG | No | None | Planned: Phase 3 |
| Graph DB (Neo4j) | No | None | Optional Phase 2b only |

**Honest positioning for a job application:**

| You can credibly claim | You cannot yet claim |
|------------------------|----------------------|
| Built a LangGraph agent with conditional routing | Production LangSmith observability |
| Integrated LangChain tools, retriever, and structured output | LangChain ReAct / agent-driven tool selection |
| Implemented a RAG pipeline over domain PDFs | Knowledge graph or GraphRAG experience (until Phase 2–3) |
| Combined live API data with document retrieval | Neo4j production deployment (unless Phase 2b done) |
| Energy-domain understanding (ENTSO-E, bidding zones) | Hybrid graph + vector retrieval |
| Designed swappable graph storage (`GraphStore`) | — |

---

## 2. Enhancement Plan

Phases 1–4 focus on **LangChain, LangGraph, and graph/RAG depth**. LangSmith is isolated in **Phase 5**. Graph storage defaults to **NetworkX**; **Neo4j is optional** and not required for Phases 2–3 to deliver value.

```
Phase 1  → smarter routing + LangChain idioms        ✅ Done
Phase 2  → knowledge graph (NetworkX + GraphStore)
Phase 2b → optional Neo4j backend                  (only if needed)
Phase 3  → GraphRAG hybrid retrieval
Phase 4  → conversational, resilient agent UX
Phase 5  → LangSmith observability & evals
```

---

### Phase 1 — LangChain idioms ✅ Done

**Branch:** `feature/langchain-tools-retriever-structured-routing`

**Goal:** Make LangChain usage idiomatic without adding external observability.

| Enhancement | Status | Details |
|-------------|--------|---------|
| **LangChain `@tool` wrappers** | Done | `get_electricity_prices`, `search_regulations` in `src/service/tools.py` |
| **LangChain Retriever + LCEL** | Done | `get_retriever()`, `get_retrieval_chain()`, `get_context_via_retriever()` in `src/service/rag.py` |
| **Structured LLM classification** | Done | `QueryClassification` + `with_structured_output()` replaces keyword routing |
| **`llm.bind_tools()`** | Done | Tools bound on answer-generation LLM in `src/service/agent.py` |

**Graph flow after Phase 1:**

```
User Query
    │
    ▼
classify (structured LLM output → QueryClassification)
    │
    ├── price      → get_electricity_prices @tool
    ├── regulation → search_regulations @tool (retriever/LCEL)
    ├── both       → both tools
    └── unknown    → general path
    │
    ▼
generate_answer (llm.bind_tools) → AgentOutput
```

#### Contribution to the project

| Area | What improves |
|------|---------------|
| **User experience** | Better query routing — e.g. *"How much does balancing cost?"* reaches the price tool even without exact keywords |
| **Inference / intelligence** | One classification LLM call replaces brittle heuristics; tools return structured context for the answer step |
| **Architecture** | Reusable, testable tools; foundation for ReAct (Phase 4) and tracing (Phase 5) |
| **Optimization** | Slightly higher latency (classification call); cleaner prompts reduce wasted tokens downstream |
| **Not yet** | Multi-turn memory, graph-aware answers, streaming |

---

### Phase 2 — Knowledge graph: NetworkX + GraphStore (1–2 weeks)

**Goal:** Add structured regulation knowledge with **zero extra infrastructure**, behind a swappable abstraction.

#### 2a. `GraphStore` protocol

Add `src/service/graph_store.py` — one interface, one default implementation:

```python
class GraphStore(Protocol):
    def add_triple(self, subject, predicate, object, metadata: dict) -> None: ...
    def neighbors(self, node_id: str, hops: int = 2) -> list[GraphNode]: ...
    def find_path(self, source: str, target: str) -> list[str]: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> GraphStore: ...
```

- **Default:** `NetworkXGraphStore` — in-memory graph, persisted to `.graph_db/regulations.graphml`
- Agent and ingest code depend on `GraphStore`, not NetworkX directly
- Neo4j plugs in later without rewriting the agent (Phase 2b)

#### 2b. Schema design (domain-native)

```
(Document)-[:HAS_SECTION]->(Section)-[:DEFINES]->(Concept)
(Concept)-[:REFERENCES]->(Concept)
(BiddingZone)-[:GOVERNED_BY]->(NetworkCode)
(BalancingProduct)-[:REGULATED_BY]->(Article)
(TSO)-[:OPERATES_IN]->(BiddingZone)
```

Example entities: EB GL, SO GL, NC RfG; FCR, aFRR, mFRR; bidding zones and TSOs.

#### 2c. Ingestion pipeline

Add `src/service/graph_ingest.py`:

1. **Extract** entities/relations from PDF chunks via LLM structured output
2. **Normalize** against ontology in `src/models/graph.py`
3. **Load** into `GraphStore` via `add_triple()`
4. **Persist** to `.graph_db/regulations.graphml` alongside Chroma indexing

```python
class RegulationTriple(BaseModel):
    subject: str      # e.g. "FCR"
    predicate: str    # e.g. "REQUIRES_MIN_CAPACITY"
    object: str       # e.g. "4 hours"
    source_doc: str
    page: int
```

#### 2d. Graph-augmented retrieval node

Add LangGraph node `search_graph`:

1. Extract entities from user query (zone, product, code name)
2. Call `graph_store.neighbors(entity, hops=2)`
3. Merge graph context with vector chunks in `regulation_context`

```python
# NetworkX traversal (via GraphStore)
neighbors = graph_store.neighbors("FCR", hops=2)
path = graph_store.find_path("FCR", "SO GL")
```

#### 2e. Dependencies and CLI

```toml
"networkx>=3.0",
```

**New CLI commands:**

- `vpp-rag index --with-graph` — build vector + graph indexes
- `vpp-rag graph query "FCR Germany"` — show nodes/paths (debug)
- Extend `vpp-rag health` — verify `.graph_db/regulations.graphml` loads

**Files to add:** `src/service/graph_store.py`, `src/service/graph_ingest.py`, `src/models/graph.py`

**Files to touch:** `src/service/agent.py`, `src/service/tools.py`, `src/cmd/index_commands.py`, `src/health.py`

#### Contribution to the project

| Area | What improves |
|------|---------------|
| **User experience** | Relational regulation answers — *"Which SO GL articles govern FCR in Germany?"* follows concept → article → document links, not just similar text |
| **Inference / intelligence** | Entity-aware context expansion; explicit citation paths (concept → section → page) |
| **Optimization** | Targeted 1–2 hop traversal vs broad similarity search → shorter prompts, lower token cost |
| **Architecture** | `GraphStore` protocol; graph persisted like Chroma — no Docker required |
| **Developer UX** | `vpp-rag graph query` for debugging; health check for graph file |
| **Not yet** | Hybrid ranking, community summaries, multi-hop loops (Phase 3) |

---

### Phase 2b — Optional Neo4j backend

**Goal:** Same `GraphStore` interface, industrial graph DB — **only when explicitly needed**.

**When to implement:**

- Job posting requires Neo4j / Cypher experience
- Graph outgrows in-memory scale (~50k+ nodes)
- Multiple services need concurrent graph access

**What to add:**

- `Neo4jGraphStore` implementing `GraphStore`
- `docker-compose.yml` with Neo4j (`docker compose --profile neo4j up`)
- Env vars: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- Factory: `get_graph_store()` returns NetworkX by default, Neo4j when configured

**What stays the same:** ingest pipeline, agent nodes, CLI commands, Phase 3 hybrid retriever.

#### Contribution to the project

| Area | What improves |
|------|---------------|
| **Portfolio** | Demonstrates graph DB ops (Cypher, Docker, production migration path) |
| **Architecture** | Proves `GraphStore` abstraction was worth the thin seam |
| **User experience** | No change for default users — opt-in only |
| **Cost** | Extra infra, tests, and docs — implement only if targeted |

---

### Phase 3 — GraphRAG: Hybrid retrieval (2–3 weeks)

**Goal:** Move from "we have a graph" to "we do GraphRAG" — on top of `GraphStore` (NetworkX or Neo4j).

| Pattern | Implementation | Use case |
|---------|----------------|----------|
| **Entity-linked retrieval** | Query → entities → graph lookup → fetch linked Chroma chunks by `chunk_id` | *"What does SO GL say about aFRR activation?"* |
| **Community summaries** | `networkx.community.louvain_communities()` → LLM summarize → store as nodes | Cross-document overview questions |
| **Hybrid ranker** | Reciprocal rank fusion of vector scores + graph proximity | Relational questions with less noise |
| **Multi-hop reasoning** | LangGraph loop: expand graph → retrieve → LLM decides if more hops needed | Complex cross-code questions |

**Enhanced agent graph:**

```
classify
    │
    ├── regulation → extract_entities
    │                    ├── graph_search (GraphStore)
    │                    └── vector_search (Chroma retriever)
    │                              │
    │                              ▼
    │                         hybrid_merge → generate_answer
    │
    ├── price → get_electricity_prices → generate_answer
    │
    └── both → get_electricity_prices + hybrid_search → generate_answer
```

Add `src/service/hybrid_retriever.py` — LangChain `BaseRetriever` fusing vector + graph sources.

#### Contribution to the project

| Area | What improves |
|------|---------------|
| **User experience** | Hard questions answered well — *"Compare FCR and aFRR across network codes"* uses community summaries + linked chunks |
| **Inference / intelligence** | Better context, not just more context; iterative multi-hop for cross-reference questions |
| **Optimization** | Hybrid ranker improves precision at same `k`; precomputed summaries reduce tokens for broad queries |
| **Architecture** | Single `BaseRetriever` abstraction shared by agent, tools, and tests |
| **Not yet** | Multi-turn, streaming, automated quality measurement (Phases 4–5) |

---

### Phase 4 — LangGraph maturity (ongoing)

**Goal:** Production-grade agent patterns — conversational, responsive, resilient — without observability tooling.

| Enhancement | Benefit |
|-------------|---------|
| **Checkpointing** (`MemorySaver` or Postgres) | Multi-turn: *"What's the German price?"* → *"And France?"* → *"How does that relate to balancing rules?"* |
| **Streaming** (`graph.stream()`) | Token-by-token CLI output — lower perceived latency with Ollama |
| **Subgraphs** | `price_subgraph` + `regulation_subgraph` composed in main graph; parallel where possible |
| **Error recovery node** | Retry ENTSO-E API; fallback to cached prices |
| **LLM tool-calling loop** | Optional ReAct node — e.g. read regulation → notice zone → fetch prices dynamically |
| **LCEL answer chain** | Composable Runnable pipeline for retrieval + generation |
| **Integration tests** | End-to-end tests with mocked LLM, ENTSO-E, Chroma, and GraphStore |

#### Contribution to the project

| Area | What improves |
|------|---------------|
| **User experience** | Conversational CLI; streaming answers; graceful failures instead of opaque errors |
| **Inference / intelligence** | ReAct loop — tools chosen mid-conversation based on intermediate results |
| **Optimization** | Subgraphs enable parallel price + regulation fetch; streaming improves perceived speed |
| **Reliability** | Retries and fallbacks; integration tests catch regressions before release |
| **Architecture** | LCEL chains make model/prompt swaps trivial |
| **Not yet** | Objective quality metrics (Phase 5) |

---

### Phase 5 — LangSmith observability & evaluation (1 week)

**Goal:** Observability and quality measurement as a standalone, opt-in layer.

#### 5a. Tracing setup

| Task | Details |
|------|---------|
| Add `langsmith` dependency | Direct dep in `pyproject.toml` |
| Environment configuration | `LANGCHAIN_TRACING_V2`, `LANGCHAIN_PROJECT`, `LANGSMITH_API_KEY` in `.env.example` |
| README documentation | Setup guide; tracing is opt-in |
| Trace coverage | Verify spans for: classify → tools → retriever → graph → generate_answer |

#### 5b. Run metadata and tagging

Tag runs by query type; attach bidding zone, chunk count, graph node count, error state.

#### 5c. Evaluation datasets

- 20–30 ENTSO-E Q&A pairs with expected routes and sources
- Evaluators: routing accuracy, context relevance, answer faithfulness
- Optional `uv run vpp-rag eval` or CI GitHub Action

#### 5d. Feedback and monitoring

- CLI `--feedback` flag; regression tracking across branches; trace filters for failed tool calls

#### 5e. Deliverables checklist

- [ ] `langsmith` added as direct dependency
- [ ] `.env.example` and README updated
- [ ] Eval dataset uploaded to LangSmith
- [ ] `vpp-rag eval` command or documented script
- [ ] Trace verified end-to-end
- [ ] At least one routing-accuracy evaluator

#### Contribution to the project

| Area | What improves |
|------|---------------|
| **User experience** | Indirect — fewer regressions, optional `--feedback` for quality loop |
| **Developer / operator UX** | Trace any run; debug bad routing or retrieval; compare branches objectively |
| **Inference / quality** | Eval datasets catch routing/retrieval regressions before users do |
| **Optimization** | Latency breakdown per node — target slow steps (classify vs graph vs LLM) |
| **Portfolio** | Satisfies **LangSmith** requirement: tracing, datasets, evaluators, CI |
| **Does not** | Make the agent smarter by itself — it measures and improves Phases 1–4 |

---

## 3. How Phases Stack — End-to-End Example

**Query:** *"How do today's German balancing prices relate to FCR requirements in SO GL?"*

| Phase | What happens | User gets |
|-------|--------------|-----------|
| **Phase 1** | LLM classifies as `both`; tools fetch prices + vector chunks | Price data + similar PDF excerpts |
| **Phase 2** | Graph finds `FCR → REGULATED_BY → SO GL articles` | Explicit article links, not just similar text |
| **Phase 3** | Hybrid ranker merges price context + FCR graph neighborhood + top vector chunks | Focused, cross-linked answer |
| **Phase 4** | Follow-up *"What about France?"* uses checkpoint memory | Conversational flow |
| **Phase 5** | Trace shows 2.1s retrieval, correct route, eval score 0.92 | Consistent quality over time |

---

## 4. Phase Contributions Summary

| Contribution | Phase 1 | Phase 2 | Phase 2b | Phase 3 | Phase 4 | Phase 5 |
|--------------|---------|---------|----------|---------|---------|---------|
| **UX** | Better routing | Relational answers | — (opt-in) | Hard-query quality | Multi-turn, streaming | Optional feedback |
| **Smarter inference** | Structured classify | Entity + graph context | Same via Neo4j | Hybrid rank, multi-hop | ReAct tool loop | Eval-driven tuning |
| **Optimization** | Cleaner tool I/O | Targeted graph queries | — | Better precision/token | Parallel subgraphs | Latency profiling |
| **Reliability** | Testable tools | Graph health check | Neo4j health | Hybrid fallbacks | Retry, checkpoint | Regression detection |
| **Architecture** | LangChain idioms | GraphStore protocol | Neo4j backend | Hybrid retriever | Subgraphs, LCEL | Eval CLI, CI |
| **Infra cost** | None | None (NetworkX) | Docker + Neo4j | None | None | LangSmith account |

---

## 5. Recommended Priority Roadmap

| Priority | Work item | Phase | Requirement addressed | Effort |
|----------|-----------|-------|----------------------|--------|
| — | LangChain tools + retriever + structured routing | 1 | LangChain + LangGraph | **Done** |
| P0 | `GraphStore` + NetworkX ingest + GraphML persistence | 2 | Knowledge graph | Medium |
| P0 | Graph-augmented retrieval node | 2 | Knowledge graph | Medium |
| P1 | Hybrid retriever (vector + graph fusion) | 3 | GraphRAG | Medium |
| P1 | Community summaries (Louvain + LLM) | 3 | GraphRAG | Medium |
| P2 | Multi-hop reasoning node | 3 | GraphRAG advanced | High |
| P2 | Checkpointing + streaming | 4 | LangGraph depth | Medium |
| P2 | LCEL answer chain + integration tests | 4 | LangChain depth | Medium |
| P3 | LangSmith tracing + README | 5 | LangSmith | Low |
| P3 | Eval dataset + evaluators | 5 | LangSmith + quality | Medium |
| P4 | Run metadata, feedback, CI eval | 5 | LangSmith maturity | Medium |
| — | Neo4j `GraphStore` backend + docker profile | 2b | Graph DB (optional) | Medium |

---

## 6. How to Talk About This Project After Each Phase

**After Phase 1 (current):**

> "I built a LangGraph agent that uses LangChain tools and a retriever/LCEL chain for regulation Q&A, with LLM structured output for query routing. Live electricity prices come from the ENTSO-E API."

**After Phase 2–3:**

> "Regulation Q&A combines Chroma vector search with a NetworkX knowledge graph of network codes, concepts, and bidding zones — behind a GraphStore abstraction — with hybrid GraphRAG retrieval and community summaries."

**After Phase 2b (if implemented):**

> "The graph layer supports both NetworkX for local development and Neo4j for graph-database deployments, using the same ingest pipeline and agent nodes."

**After Phase 4:**

> "The agent supports multi-turn sessions via LangGraph checkpointing, streams answers to the CLI, and uses composable LCEL chains for retrieval and generation."

**After Phase 5:**

> "All agent runs are traced in LangSmith with eval datasets measuring routing accuracy and answer quality — giving full observability over the LangGraph pipeline."

---

## 7. What to Keep (Don't Over-Engineer)

- Clear separation: `clients/` → `service/` → `cmd/`
- Pydantic models in `src/models/internals.py`
- Idempotent indexing in `src/service/rag.py`
- LangChain tools isolated in `src/service/tools.py`
- **One graph backend (NetworkX) until Phase 2b is explicitly needed**
- **`GraphStore` protocol** — thin seam, not a plugin framework

Enhancements should **extend** this structure rather than rewrite the agent from scratch.

---

## 8. Key File Reference

| Concern | Path |
|---------|------|
| LangGraph agent | `src/service/agent.py` |
| LangChain tools | `src/service/tools.py` |
| Vector RAG + retriever | `src/service/rag.py` |
| Graph storage (planned) | `src/service/graph_store.py` |
| Graph ingest (planned) | `src/service/graph_ingest.py` |
| Hybrid retriever (planned) | `src/service/hybrid_retriever.py` |
| Graph models (planned) | `src/models/graph.py` |
| State/models | `src/models/internals.py` |
| ENTSO-E client | `src/clients/entsoe_client.py` |
| CLI | `src/cli.py`, `src/cmd/*.py` |
| Dependencies | `pyproject.toml`, `uv.lock` |
| Graph persistence (planned) | `.graph_db/regulations.graphml` |
| Documentation | `README.md`, this file |
| Environment template | `.env.example` |

---

## 9. Known Gaps and Partial Implementations

| Item | Status |
|------|--------|
| `langchain-community` | Declared but never imported |
| Domain exceptions (`src/utils/exceptions.py`) | Defined but unused |
| `general` query path | No-op; unknown queries get LLM answers without RAG or price data |
| Test coverage | ENTSO-E client + service unit tests; no integration tests |
| README "price forecasting" | Code fetches day-ahead prices only |
| LangSmith | Not integrated; all observability in Phase 5 |
| Knowledge graph / GraphRAG | Not started; Phases 2–3 |
| Neo4j | Not planned as default; optional Phase 2b |

---

*Last updated: June 2025 — vpp-rag-agent v0.1.0, branch `feature/langchain-tools-retriever-structured-routing`*
