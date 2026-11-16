# MedAgent-Swarm

MedAgent-Swarm is the medical multi-agent inference component of **MedReason Lab**. It uses the `MedReason-VL-2B-RL` model exported by `medvlm-rl`, served through an OpenAI-compatible vLLM endpoint. Its Lead–Worker orchestration supports clinical consultation, symptom assessment, research and tool-based retrieval.

## Architecture

```text
User → SwarmCoordinator → LeadAgent (task decomposition)
                    ├─ ConsultationAgent ─┐
                    ├─ DiagnosticAgent ───┼─ SharedContext → LeadAgent synthesis
                    └─ ResearchAgent ─────┘
Worker → AgentLoop → LLMClient ⇄ SkillRegistry → Medical skills / Milvus / web search
           ↑
      ContextAssembler ← short-term memory + Mem0 + task context
```

The task assignment is centrally planned by LeadAgent; workers execute asynchronously. Each agent shares the same LLM service but uses its own role prompt and execution context. This is orchestrated multi-agent collaboration, not decentralized task bidding.

## Code layout

- `src/medagent_swarm/core/`: LLM client, AgentLoop, skill discovery and execution
- `src/medagent_swarm/agents/`: consultation, diagnostic and research workers
- `src/medagent_swarm/orchestration/`: LeadAgent, coordinator, shared task state, events
- `src/medagent_swarm/resources/skills/`: declarative `SKILL.md` descriptions and Python implementations
- `src/medagent_swarm/retrieval/`: Milvus Lite local medical retrieval and import script
- `src/medagent_swarm/research/`: web search and multi-source research synthesis
- `src/medagent_swarm/memory/`: conversation history, Mem0, session summaries
- `src/medagent_swarm/context/`: token budget, contextual selection, structured memory extraction, assembly
- `src/medagent_swarm/guardrails/` and `fixers/`: tool constraints and output checks
- `tests/`: focused unit tests

## Model dependency

The model is stored outside this Python package:

```text
../medvlm-rl/models/MedReason-VL-2B-RL/
```

Use `medvlm-rl/scripts/export_model.sh` to export the training checkpoint. An export contains the complete model weights, config and tokenizer/processor assets. Model artifacts are local files and not tracked in Git. The agent itself never imports the RL training framework.

## Setup

Use Python 3.10+ in a separate agent environment. For Milvus Lite and vLLM use a supported Linux deployment; model serving requires compatible GPU resources.

```bash
cd medagent-swarm
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
# optional cloud memory: pip install -e '.[memory]'
# load values from .env.example into your shell as appropriate
```

Start vLLM in a **separate environment** with a compatible GPU and the exported model in place:

```bash
bash scripts/serve_model.sh
```

The service exposes `http://127.0.0.1:8000/v1`, model name `medreason-vl-2b-rl`. The serving command enables automatic tool calling using the Hermes parser; confirm compatibility with the selected vLLM version and the model's chat template. An OpenAI-compatible text endpoint alone is not sufficient to support tool calls.

In the agent environment:

```bash
export MEDREASON_BASE_URL=http://127.0.0.1:8000/v1
export MEDREASON_MODEL=medreason-vl-2b-rl
export MEDREASON_API_KEY=EMPTY
medagent-swarm
```

For the medical knowledge corpus, import the bundled `.txt` documents with `python -m medagent_swarm.retrieval.scripts.import_hardcoded_data`; the SQLite-backed Milvus Lite database is stored under `~/.cache/medreason-lab/`. Embedding model `BAAI/bge-small-zh-v1.5` is downloaded by Sentence Transformers on first use. The bundled medical texts are reference material, not a substitute for authoritative clinical sources.

## Context engineering

The runtime assembles agent-specific context from conversation history, retrieved memory and task metadata. `ContextAssembler` uses an explicit input-token budget, accounts for tool definitions, deduplicates exact duplicates, prioritizes task-related and safety-relevant statements, and preserves attribution of patient-reported information. Historical claims remain data rather than verified clinical facts.

Environment controls: `MEDREASON_CONTEXT_TOKENS`, `MEDREASON_CONTEXT_RESERVE`, `MEDREASON_USER_ID`, and optionally `MEM0_API_KEY`. Context token estimates use `tiktoken` as an approximation of the serving model's tokenizer.

## Tests

```bash
pytest -q tests/
python -m compileall -q src/medagent_swarm
```

Agent operation requires the exported model and a compatible model server. Test suites cover deterministic modules separately from model-dependent request execution. For clinical use, professional review, provenance verification and additional safety safeguards are necessary.
