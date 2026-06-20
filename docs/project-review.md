# VPP RAG Agent — Project Review & Interview Prep

A self-contained portfolio review and interview prep guide for the VPP RAG Agent project. Use the first half for resume framing and the second half for interview practice.

---

## 1. Elevator pitch (30 seconds)

A LangGraph agent that answers questions about the European electricity market by combining live ENTSO-E day-ahead prices with grid regulation PDFs. Built as a fully local system — Ollama for inference, Chroma for vectors, NetworkX for a regulation knowledge graph — with a hand-curated evaluation harness that measures retrieval and answer quality on every change. Demonstrates RAG, GraphRAG, agent orchestration, structured output, and a measurement-first engineering style.

---

## 2. What it does

**Domain.** European electricity markets (ENTSO-E API) + grid regulations (FCR, aFRR, SO GL, EB GL, balancing, settlement, network codes).

**Problem.** Live price data and grid regulations are siloed. A trader, balancing-group manager, or virtual-power-plant operator wanting to reason across both has to context-switch between two very different sources.

**Solution.** A single natural-language interface that classifies the question (price / regulation / both / general), routes it to the right subgraph, and synthesizes a cited answer. The regulation side uses GraphRAG (vector + BM25 + entity-tagged chunks + knowledge-graph community summaries). The price side calls the ENTSO-E REST API and parses the XML payload.

**Why it matters as a portfolio piece.** Real domain, real API, real PDFs, real measurements. Not a toy chatbot.

---

## 3. Architecture at a glance

```
            ┌──────────────────────────────────────────┐
User ──►    │  vpp-rag CLI (Click + Rich)              │
            │  ask | index | eval | graph | health     │
            └────────────────┬─────────────────────────┘
                             │
            ┌────────────────▼─────────────────────────┐
            │  VppAgent (LangGraph)                    │
            │  classify → route → [price | reg | both] │
            │                  ↓                       │
            │       answer (degraded path on failure)  │
            │  Checkpointed via SqliteSaver (.agent_db)│
            └────────┬───────────────────┬─────────────┘
                     │                   │
        ┌────────────▼────┐   ┌──────────▼──────────────┐
        │ ENTSO-E client  │   │ GraphRAG pipeline       │
        │ XML parse + UTC │   │ vector | hybrid (BM25)  │
        │ namespace-safe  │   │ entity-tagged chunks    │
        └─────────────────┘   │ Louvain community       │
                              │ summaries (cached)      │
                              └──────────┬──────────────┘
                                         │
                              ┌──────────▼──────────────┐
                              │ Chroma + NetworkX       │
                              │ (.chroma_db, .graph_db) │
                              └─────────────────────────┘
```

**Stack.** Python 3.10+, uv, LangGraph, LangChain, Ollama (`deepseek-r1:8b` for chat, `nomic-embed-text` for embeddings), ChromaDB, NetworkX, rank-bm25, Pydantic, Click, Rich, pytest, mypy strict, pre-commit.

**Layering.**
- `src/clients/` — external APIs (ENTSO-E)
- `src/service/` — business logic (agent, RAG, retrievers, graph, eval)
- `src/cmd/` — CLI surface
- `src/models/` — Pydantic types
- `src/eval/` — evaluation harness
- `tests/` — unit, service, and E2E layers

---

## 4. Engineering highlights — talking points

| Area | What you can say |
|------|------------------|
| **Agent orchestration** | LangGraph state machine with conditional routing, subgraphs (`price_subgraph`, `regulation_subgraph`), and ReAct mode. SqliteSaver checkpointing for multi-turn threads. |
| **Retrieval (GraphRAG)** | Four selectable modes — `vector`, `hybrid` (BM25 + vector with Reciprocal Rank Fusion), `entity` (graph-tagged chunk boost), `graph` (Louvain community summaries). One env var or `--retriever` flag switches modes; eval measures each. |
| **Knowledge graph** | NetworkX + GraphML, protocol-based `GraphStore` interface (easy to swap to Neo4j), hybrid heuristic + LLM extraction, in-memory by default — zero infra. |
| **Eval harness** | Custom metrics (recall@k, MRR, context precision, faithfulness via local LLM judge, answer-contains, p95 latency). 35 hand-curated ENTSO-E questions. Markdown report regenerated on every run. No RAGAS, no LangSmith — fully local and reproducible. |
| **Streaming & state** | Token streaming via `astream_events`, multi-turn via SqliteSaver, per-thread state replay. ReAct loop available as an alternative orchestration mode. |
| **Robustness** | Namespace-agnostic XML parsing (handles ENTSO-E API drift), stable SHA-256 chunk IDs (idempotent indexing), BM25 corpus rebuilt with vector store, retry-with-backoff on price fetch, degraded-answer fallback when tools fail. |
| **Tooling** | uv-native workflow, pre-commit (flake8/black/mypy strict), pytest with mocking, 100+ tests across unit/service/E2E. |
| **Type safety** | Pydantic v2 for all I/O contracts, mypy strict on `src/`, typed CLI options, structured LLM output (`QueryClassification`, `FaithfulnessJudgment`). |

---

## 5. Tradeoffs you made consciously (honest framing)

These read well in interviews because they show you understood the alternatives, not that you missed them.

- **Local-only (no LangSmith, no RAGAS).** Zero external dependencies, zero per-run cost, fully reproducible on a laptop. Tradeoff: no hosted observability dashboard; would need a tracer module for production.
- **Ollama with `deepseek-r1:8b` by default.** Free, private, runs anywhere. Tradeoff: a reasoning model is slow on commodity hardware (~4 tok/s) — would expose a `--model` flag and default to a faster non-reasoning model (`qwen2.5:7b`) for the demo.
- **NetworkX over Neo4j.** No Docker, no extra service, ships in a single repo. Tradeoff: in-memory; Protocol-based `GraphStore` means a Neo4j backend is a future drop-in if scale demands it.
- **Hand-curated gold set (35 questions).** Higher signal per case than LLM-synthesized. Tradeoff: statistically underpowered; planning to expand to ~100 with bootstrap confidence intervals.
- **Custom faithfulness judge instead of RAGAS.** Lightweight, no OpenAI dependency. Tradeoff: same model judging its own output is biased; would split judge from answerer in a follow-up.
- **Sqlite checkpoints.** Single-file, no infra. Tradeoff: single-writer; concurrent FastAPI demo would need Postgres.

---

## 6. Known caveats (also good interview material)

Naming things you'd fix shows seniority. Pick 2–3 to surface unprompted.

- **`_QueryEntityRetriever` mutates inner retriever state per call.** Fine for single CLI use; would race under concurrent HTTP requests. Fix: pass entities through the `RunnableConfig` instead of attribute assignment.
- **ReAct mode bypasses GraphRAG retrievers.** The `--react` flag uses bound tools whose `search_regulations` is the simple vector path. Would unify retriever invocation across both modes.
- **Faithfulness ≠ correctness.** Judge measures "is the answer supported by retrieved context." A hallucinated-but-consistent answer scores well. Would add an answer-correctness metric against reference answers on a labeled subset.
- **E2E tests mock the LLM.** Verifies orchestration but not prompts. Would add a smoke suite that uses a tiny real model (`gemma3:270m`) on a handful of cases.

---

## 7. Resume bullets — ready to paste

Pick 4–6 depending on space; keep verbs strong and numbers honest.

- Built a **LangGraph agent** combining real-time ENTSO-E electricity prices with regulation PDFs over a **GraphRAG retrieval pipeline** (vector / hybrid BM25 / entity-tagged / Louvain communities), switchable via env var or CLI flag.
- Implemented a **regulation knowledge graph in NetworkX** with a `GraphStore` Protocol allowing pluggable backends (NetworkX today, Neo4j on demand); hybrid heuristic + LLM triple extraction.
- Designed a **fully local evaluation harness** — custom metrics (recall@k, MRR, context precision, faithfulness via local LLM judge), 35 hand-curated ENTSO-E questions, markdown report regenerated on every retriever change — with zero external dependencies (no LangSmith, no RAGAS).
- Added **LangGraph maturity layer**: `SqliteSaver` checkpointing for multi-turn threads, token streaming via `astream_events`, subgraphs for price and regulation flows, optional `create_react_agent` tool loop, retry-with-backoff and degraded-answer paths.
- Built robust **ENTSO-E XML parsing** that is namespace-agnostic, timezone-normalized (UTC), and resilient to API version drift; stable SHA-256 chunk IDs make indexing idempotent.
- Engineered for review: **uv-native** toolchain, pre-commit (flake8/black/mypy strict), 100+ tests across unit/service/E2E layers, Pydantic v2 contracts on every boundary, structured LLM outputs.

---

## 8. Mock interview

Realistic questions a senior AI/ML interviewer would ask. Suggested answers in your voice — read, then rephrase. Time yourself: stay under 90 seconds per answer.

### Q1 — "Walk me through this project in two minutes."

> It's a LangGraph agent over the European electricity market — it answers questions about live prices and grid regulations in a single interface. Prices come from the ENTSO-E REST API; regulations come from PDFs indexed into Chroma and a NetworkX knowledge graph. The agent classifies each query with structured Pydantic output, routes through a price subgraph, a regulation subgraph, or both, and synthesizes a cited answer. The interesting part for me is the retrieval layer: it's GraphRAG with four selectable modes — vector only, hybrid BM25 + vector with RRF, entity-tagged chunk boost using the knowledge graph, and full graph-augmented with Louvain community summaries. Everything is measured by a custom local eval harness so I can show retrieval deltas as a table, not a story.

### Q2 — "Why local-only? Why no LangSmith or RAGAS?"

> Three reasons. First, reproducibility — anyone with a laptop and Ollama can clone and run the exact eval pipeline I designed; nothing is gated by an API key. Second, cost — running RAGAS at scale calls OpenAI by default, and I wanted measurements I could iterate on without thinking about a bill. Third, defensibility in an interview — I wanted to *build* the metrics myself so I understand them. The tradeoff is real: no hosted dashboard, no team-shared experiments. If this graduated to a team project I'd add LangSmith for tracing and keep my custom metric stack as the eval reference.

### Q3 — "Why GraphRAG and not just better embeddings?"

> Two reasons specific to this domain. Regulations are *relational* — FCR is governed by SO GL, which is implemented by TSOs, which operate in bidding zones. A pure semantic retriever pulls clauses by similarity but loses the relationships. Second, regulation queries often need cross-document context — "what changes between FCR and aFRR procurement" pulls chunks from two different network codes that vector similarity won't co-locate. The graph captures both: entities boost relevant chunks at retrieval time, and Louvain community summaries inject relational context the vector alone misses. I measured each layer separately so I can defend that the lift is real and not just complexity for its own sake.

### Q4 — "How would you evaluate this is actually better than vector RAG?"

> The eval harness records four metrics per run — recall@k, MRR, context precision, and faithfulness via a local LLM judge — plus p95 latency, on a 35-question hand-curated gold set. Each retriever mode (`vector`, `hybrid`, `entity`, `graph`) gets a labeled row in `docs/eval_report.md`. The story is the delta between rows. I'll be honest about two limitations: 35 is too few questions to declare wins under 10pp with confidence, and faithfulness scored by the same model that generates answers is biased upward. If this were production I'd expand to ~100 questions with bootstrap CIs and use a different judge model — both items on the roadmap.

### Q5 — "Walk me through the LangGraph state machine."

> The state is a `TypedDict` carrying `messages`, `query_type`, `prices`, `regulation_context`, `final_answer`, `error`, `degraded`, and `bidding_zone`. Nodes are `classify`, `price_subgraph`, `regulation_subgraph`, `get_both`, `general`, and `answer`. The classifier uses `llm.with_structured_output(QueryClassification)` — Pydantic constrains the LLM to a typed enum, so routing logic is just a dict lookup. The conditional edge from `classify` picks the next node by `QueryType`. Both subgraphs converge on `answer`, which builds a context prompt from whatever state is populated. The whole graph compiles with a `SqliteSaver` checkpointer so threads persist across CLI invocations — `vpp-rag ask --thread demo-1` continues the conversation.

### Q6 — "What's the hardest bug you hit?"

> The streaming code path. Token streaming uses LangGraph's `astream_events("v2")`, which is async, but the CLI is sync. I wired it through a manual event loop with `__anext__()` and a try/except fallback to per-node `stream` mode if the token stream fails. It works for the common case, but I learned afterwards that the fallback try/except is too broad — it swallows real errors and silently downgrades to non-streaming. The honest version of the answer is: the bug is still there as a known caveat. The fix is to commit to async end-to-end or use a narrower exception class. I left a comment in the roadmap so I don't forget.

### Q7 — "How would you scale this to serve concurrent users?"

> Three concrete changes. One: replace `SqliteSaver` with `PostgresSaver` because Sqlite is single-writer and concurrent threads would block each other. Two: fix `_QueryEntityRetriever` — it currently mutates `self.inner.query_entities` per call, which is a race condition under concurrency. The right pattern is to pass entities through the `RunnableConfig` and read them in the retriever. Three: move the synchronous `time.sleep` retries to `asyncio.sleep` so a slow ENTSO-E call doesn't block the event loop. Beyond that, I'd front the agent with a FastAPI service that rate-limits per IP and a hosted model backend gated by `VPP_LLM_BACKEND` so the demo doesn't depend on Ollama-on-Fly.

### Q8 — "Why this specific tech stack?"

> LangGraph because the project's value is in the *state machine* — conditional routing, subgraphs, ReAct, checkpointing all came for free. LangChain because of structured output, retrievers, LCEL chains, and the tool decorator — it's the ecosystem with the most idiomatic primitives for what I was building. Ollama because it's local, free, and private. Chroma because it ships in-process — no Docker. NetworkX because the graph is small enough to fit in memory and GraphML serializes cleanly. uv because it's fast and reproducible. Every choice has the same logic: minimize infra, keep the surface explicit, and stay portable.

### Q9 — "What would you have done differently with more time?"

> Five things, ranked. First, expand the gold set to ~100 questions and report deltas with bootstrap CIs — 35 is too few for confident claims. Second, split the faithfulness judge from the answerer; using the same model is biased. Third, expose `--model` and `VPP_MODEL` env var because the default reasoning model is slow on commodity hardware. Fourth, unify retriever invocation between the classify-route and ReAct paths — right now `--react` silently bypasses GraphRAG, which is a bait-and-switch. Fifth, add a CI eval gate so PRs touching retrieval automatically show metric diffs.

### Q10 — "What did you learn?"

> Three things stand out. One — the eval harness is the project. Without it, every architectural change is folklore; with it, I can defend every claim. Two — LangGraph is more than orchestration; checkpointing and `astream_events` are the difference between a script and a system. Three — Protocols are the right abstraction for swappable infrastructure. The `GraphStore` Protocol means NetworkX today, Neo4j tomorrow, and the agent code doesn't move. I underweighted that pattern before this project.

---

## 9. How to use this document

1. **For your resume** — lift 4–6 bullets from section 7. Cut any you can't defend in 90 seconds.
2. **For interview prep** — read each Q&A out loud once, then close the doc and answer cold. Record yourself. If an answer is over 90 seconds, trim it.
3. **For the GitHub README** — pull the elevator pitch (section 1), architecture diagram (section 3), and engineering highlights (section 4). Skip the caveats — those are for interviews, not your repo's front page.
4. **For follow-up questions** — practice the "what would you do differently" question. Hiring managers ask it almost universally; a precise answer is a strong signal.
