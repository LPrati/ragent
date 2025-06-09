<!--
AGENTS.md — Repository-wide convention file
OpenAI Codex reads this file to understand how to build, test, and deploy agents
in this repo. Keep it up to date whenever the codebase evolves.
-->

# Agent Development Guide (Google ADK + OpenAI GPT‑4o‑mini)

## 1. Overview

This repository is a blueprint for building **production‑grade AI agent systems**. It combines:

| Layer        | Technology                             | Rationale                                           |
| ------------ | -------------------------------------- | --------------------------------------------------- |
| LLM provider | **OpenAI GPT‑4o‑mini**                 | state‑of‑the‑art reasoning with low latency         |
| Agent SDK    | **Google Agent Development Kit (ADK)** | composable, model‑agnostic framework                |
| Vector store | **ChromaDB**                           | lightweight, self‑hosted, high‑throughput retrieval |
| UI           | **Gradio**                             | rapid prototyping & shareable demos                 |
| Packaging    | **Docker + GitHub Actions**            | reproducible builds & CI/CD ready                   |

> **Principle:** Code *is* infrastructure. Treat agents like micro‑services. Apply SOLID, Clean Architecture & 12‑Factor App guidelines throughout.

---

## 2. Standard Directory Layout

```text
my_agent_app/
├── agents/               # ADK agent definitions
│   ├── __init__.py
│   ├── base.py           # BaseAgent with shared behaviour
│   ├── rag_agent.py
│   ├── manager.py
│   └── worker.py
├── tools/                # ADK tools (functions, OpenAPI specs, wrappers)
│   ├── __init__.py
│   ├── retrieval.py      # ChromaDB retrieval helpers
│   └── ...
├── adapters/             # Ports in Hexagonal/Clean Architecture
│   ├── litellm_config.py     # Only place that imports `openai`
│   └── vector_chroma.py
├── core/                 # Enterprise logic, domain models, use‑cases
│   └── ...
├── infra/                # Framework glue (FastAPI, DB, etc.)
│   └── ...
├── tests/                # pytest unit & integration tests
├── scripts/              # One‑off CLIs (seed DB, run eval, etc.)
├── docker/               # Dockerfiles & compose
│   └── Dockerfile
├── .github/workflows/    # CI pipelines
│   └── ci.yml
├── docs/                 # MkDocs sources (auto‑deployed)
├── pyproject.toml        # Poetry + Black + Ruff + Mypy
└── AGENTS.md             # ← this file (update me!)
```

### 2.1 Clean Architecture Mapping

* **Domain / Core** – business entities, value objects, use‑cases *(pure Python)*.
* **Adapters** – translate between domain and external services (LLM, vector store).
* **Framework / Drivers** – ADK, Gradio, FastAPI, or GCP Functions.
  Dependencies must always point **inward**.

---

## 3. Environment Setup

```bash
# 1. Install Python 3.11+
curl -sSL https://install.python-poetry.org | python3 -

# 2. Clone and bootstrap
git clone <repo> && cd <repo>
poetry install --with dev

# 3. Run pre‑commit hooks
poetry run pre-commit install
```

Environment variables (see `configs/.env.example`):

```bash
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
CHROMA_HOST=localhost
CHROMA_PORT=8000
```

---

## 4. Configuration Conventions

* **pyproject.toml** centralises linting & formatting.
* **settings.yaml** (pydantic‑based) stores non‑secret config.
* **.env** for secrets, loaded via `python‑dotenv`.

---

## 5. Coding Standards & Architectural Principles

* Follow **SOLID** and **Clean Architecture** – business logic in `core/`, adapters in `adapters/`, frameworks in `infra/`.
* Enforce static typing with `mypy --strict`.
* Style: `black`, `ruff`, `isort`.
* Commit hooks: `pre‑commit`. Merge only when CI is green.
* Inverted dependencies: UI layers never import `openai` directly; they invoke the adapter.

---

## 6. Working With ADK (v1.0)

This section has been refreshed to match **ADK v1.0.0** (Python) released at Google I/O ’25. Major changes:

* New import path `google.adk` (was `adk`).
* Plain Python functions are auto‑wrapped as `FunctionTool` – decorators are optional.
* Lifecycle hooks are now exposed via **callbacks** (`before_agent_callback`, `after_model_callback`, …) instead of ad‑hoc methods like `on_start`.

### 6.1 Minimal LLM Agent

```python
# agents/rag_agent.py
from google.adk.agents import LlmAgent  # alias Agent
from google.adk.models.lite_llm import LiteLlm  # custom BaseLlm impl
from tools.retrieval import retrieve_context

rag_agent = LlmAgent(
    name="doc_qa",
    model=LiteLlm(model="openai/gpt-4o-mini"),  # or a plain model string
    instruction=(
        "You are a helpful assistant that answers questions grounded in private docs. "
        "When unsure, say you don't know. Always cite the 'source' field returned by the retrieval tool."
    ),
    tools=[retrieve_context],  # ADK wraps the fn as FunctionTool automatically
)
```

### 6.2 Defining Tools

ADK auto‑converts any plain function placed in the `tools` list into a **FunctionTool**; a decorator is **no longer required**.

```python
# tools/retrieval.py
from adapters.vector_chroma import chroma_query

def retrieve_context(query: str, top_k: int = 5) -> list[dict]:
    """Search ChromaDB and return the top‑k docs most relevant to *query*.
    Each dict must include a `page_content` and `source` key so the agent can cite it."""
    return chroma_query(query, k=top_k)
```

For long‑running operations wrap explicitly:

```python
from google.adk.tools import FunctionTool

long_task_tool = FunctionTool(
    fn=my_long_task,
    name="long_task",
    description="Performs an asynchronous ETL job and streams progress updates."
)
```

### 6.3 Callbacks for Custom Behaviour

Replace legacy `on_start` / `handle_error` with **callback hooks**:

```python
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.genai import types
from typing import Optional

def log_entry(ctx: CallbackContext) -> Optional[types.Content]:
    ctx.logger.info(f"Invoking {ctx.agent_name}")
    return None  # continue execution

def after_model(ctx: CallbackContext, response: types.Content):
    ctx.logger.info(f"Tokens used → {response.usage.total_tokens}")

logging_agent = LlmAgent(
    name="logging_agent",
    model="gpt-4o-mini",
    instruction="Just echo the user input.",
    before_model_callback=log_entry,
    after_model_callback=after_model,
)
```

| Stage               | Hook                                            | Typical use          |
| ------------------- | ----------------------------------------------- | -------------------- |
| **Agent lifecycle** | `before_agent_callback`, `after_agent_callback` | audit, early exit    |
| **LLM interaction** | `before_model_callback`, `after_model_callback` | prompt observability |
| **Tool execution**  | `before_tool_callback`, `after_tool_callback`   | tracing, retries     |

### 6.4 Orchestration Patterns (unchanged)

Pattern table moved intact from previous revision; updated to clarify that `Agent` is an alias for `LlmAgent` and to reference the Agent‑to‑Agent (A2A) protocol for manager/worker designs.

### 6.5 Running Agents

```python
from google.adk.runners import InMemoryRunner
runner = InMemoryRunner(root_agent=rag_agent)
response = runner.run(user_id="demo", session_id="local", input_content="How many GPUs do we own?")
print(response.text)
```

Edge deployment targets: **Vertex AI Agent Engine**, Cloud Run or bare k8s; see the Deploy section for manifests.

## 7. Memory, Context & ChromaDB

1. **Embedding ingestion**

```bash
poetry run python scripts/seed_chroma.py data/seed_docs
```

2. **Retriever helper**

```python
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

client = PersistentClient("chroma/", settings=...)
embedder = SentenceTransformer("thenlper/gte-small")

def chroma_query(text: str, k: int = 5):
    vec = embedder.encode(text, normalize=True).tolist()
    return client.query(vec, top_k=k)
```

Agents call `retrieve_context` (above); results are appended to chat history via ADK `ContextMemory`.

---

## 8. UI Layer (Gradio)

```python
import gradio as gr
from agents.rag_agent import RagAgent

agent = RagAgent()

with gr.Blocks(title="Visio AI Assistant") as demo:
    chat = gr.Chatbot()
    msg = gr.Textbox(label="Ask me anything")

    def respond(user_msg, chat_history):
        answer = agent(user_msg)
        return answer, chat_history + [(user_msg, answer)]

    msg.submit(respond, [msg, chat], chat)

demo.launch(share=True)
```

Expose `/healthz` for k8s readiness probes.

---

## 9. Testing Strategy

* **Unit** – pytest, 100 ms max per test, mock OpenAI with `respx` or `openai-mock`.
* **Integration** – spin up ChromaDB container via `docker compose` in CI.
* **Contract** – JSONSchema snapshots for tool I/O.
* **Eval** – use `adk.evaluate` to run scenario benchmarks before merge.

Target coverage ≥ 90 %.

---

## 10. Continuous Integration

`.github/workflows/ci.yml` stages:

1. **static‑check** – Ruff, MyPy, Black --check.
2. **test** – pytest -n auto --cov.
3. **build‑image** – docker build, tag `:sha`.
4. **publish‑docs** – mkdocs gh‑pages.

PRs cannot merge unless CI is green.

---

## 11. Dockerfile (slim)

```Dockerfile
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.3

RUN pip install --no-cache-dir poetry

WORKDIR /app
COPY pyproject.toml poetry.lock* /app/
RUN poetry install --without dev --no-root

COPY . /app

CMD ["poetry", "run", "python", "-m", "agents.serve"]
```

Multi‑stage builds with `--platform` (amd64/arm64) recommended for CI.

---

## 12. Deployment

```bash
docker build -t ghcr.io/org/my-agent:$GITHUB_SHA .
docker push ghcr.io/org/my-agent:$GITHUB_SHA
kubectl apply -f k8s/manifest.yaml
```

Use `HorizontalPodAutoscaler` on CPU/GPU metrics for elasticity.

---

## 13. Documentation Workflow

* **MkDocs Material** theme.
* Auto‑generate API docs via `mkdocstrings[pytkdocs]`.
* Each PR must update docs; CI fails if `mkdocs build --strict` reports broken links.

---

## 14. Contribution Checklist

* [ ] Feature has a dedicated agent or tool.
* [ ] OpenAI usage is cost‑capped via rate‑limiter.
* [ ] New dependencies are justified and pinned.
* [ ] Added/updated unit & integration tests.
* [ ] Documentation updated.
* [ ] `pre‑commit run --all-files` passes.

---

### 15. Quick Commands

| Purpose               | Command                                         |
| --------------------- | ----------------------------------------------- |
| Format code           | `poetry run ruff format .`                      |
| Run tests             | `poetry run pytest -q`                          |
| Launch local ChromaDB | `docker compose -f docker/chroma.yml up`        |
| Evaluate agent        | `poetry run adk evaluate ./eval/scenarios.yaml` |

> *Stay modular, stay testable, ship daily.*
