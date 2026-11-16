# MedReason Lab

MedReason Lab is a two-part research codebase for reinforcement-learning-based medical vision-language modeling and multi-agent inference.

| Component | Purpose |
|---|---|
| [`medvlm-rl/`](medvlm-rl/) | Qwen3-VL-2B-Instruct medical RL training, rewards, evaluation, checkpoint export |
| [`medagent-swarm/`](medagent-swarm/) | Medical multi-agent orchestration, Skill execution, retrieval, memory and context engineering |

## Model handoff

```text
medvlm-rl (train)
  → checkpoints/medvlm_2b_dapo/global_step_N/actor (FSDP checkpoint)
  → scripts/export_model.sh (merge + copy)
  → models/MedReason-VL-2B-RL/ (Hugging Face inference model)
  → vLLM OpenAI-compatible server
  → medagent-swarm (LeadAgent and Workers)
```

The agent model is the **locally exported MedReason-VL-2B-RL** checkpoint. Both projects have separate environments; the shared contract is the deployed model's API.

## Getting started

1. Follow [`medvlm-rl/README.md`](medvlm-rl/README.md) to install its pinned training framework, configure the model and run training.
2. Export the selected Actor checkpoint with `bash medvlm-rl/scripts/export_model.sh checkpoints/medvlm_2b_dapo/global_step_N/actor` while inside the RL project directory.
3. Start the model service with `bash medagent-swarm/scripts/serve_model.sh` in a vLLM environment.
4. Install the agent package with `pip install -e './medagent-swarm[test]'`, import the local retrieval corpus, and run `medagent-swarm`.

`config.py` contains shared model endpoint defaults; the agent runtime reads equivalent environment variables through `medagent_swarm.config`. No API credentials or model weights are committed.

## Project organization

```text
medreason-lab/
├── config.py
├── medvlm-rl/
│   ├── configs/ scripts/ rewards/ evaluation/ reports/
│   ├── checkpoints/medvlm_2b_dapo/
│   └── models/MedReason-VL-2B-RL/
└── medagent-swarm/
    ├── pyproject.toml
    ├── src/medagent_swarm/
    ├── scripts/
    └── tests/
```

The two projects maintain independent dependency environments. The multi-agent module uses the model's text and tool-use interface; its tool-calling behavior must be assessed independently of medical VQA model performance.
