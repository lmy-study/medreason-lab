# MedVLM-RL

**Medical vision-language model post-training with composite rewards and DAPO-style policy optimization.**

MedVLM-RL is the medical VLM reinforcement-learning module of **MedReasonLab**. It trains `Qwen/Qwen3-VL-2B-Instruct` for medical visual question answering (VQA), combining answer-quality, semantic-similarity, imaging-modality, and response-format rewards.

## Highlights

- **Medical VLM:** Qwen3-VL-2B-Instruct with image–question prompts and structured reasoning/answer output.
- **Group-based RL:** three responses per question; GRPO group-relative advantage estimation.
- **DAPO-style optimization:** asymmetric clipping (`0.2` / `0.28`), online reward-based group filtering, and token-level policy-loss aggregation.
- **Composite terminal reward:** LLM-based answer assessment, medical-text embedding similarity, imaging-modality correctness, and format compliance.
- **Training and evaluation artifacts:** training curves, an analysis report, and per-question VQA-RAD evaluation outputs.

## Repository structure

```text
medvlm-rl/
├── configs/
│   └── medvlm_2b_dapo.yaml      # Model, data, RL and distributed-training settings
├── prompts/
│   └── medical_format.jinja     # Medical response template
├── rewards/
│   └── medical.py               # Composite reward function
├── scripts/
│   ├── train_2b.sh              # Training entry point
│   ├── merge_checkpoint.sh      # FSDP checkpoint conversion
│   └── model_merger.py
├── evaluation/                  # VQA-RAD evaluation code and results
├── reports/training/            # Training curves and training report
├── tests/                       # Lightweight static/CPU checks
├── requirements-project.txt
└── README.md
```

The parent repository also contains `medix-agent-swarm/` and `config.py`. Their integration is maintained separately.

## Environment setup

Training is organized with **EasyR1** (imported in Python as `verl`), built on the veRL ecosystem. The project specifies EasyR1 v0.3.2 at commit `a23fb5b` as its dependency baseline.

```bash
git clone https://github.com/hiyouga/EasyR1.git
git -C EasyR1 checkout a23fb5b
pip install -e ./EasyR1

cd medreason-lab/medvlm-rl
pip install -r requirements-project.txt
```

Use a CUDA/PyTorch/vLLM environment appropriate to the installed EasyR1 version. The code imports the RL framework as `verl`; do not install a second, competing `verl` distribution in the same environment.

## Data and inputs

The training configuration uses the medical VQA dataset with the fields:

| Field | Purpose |
|---|---|
| `image` | Medical image input |
| `problem` | Question presented to the model |
| `solution` | Reference answer used for scoring |

Configured splits: `train` for training and `test` for validation data. The training configuration is located in `configs/medvlm_2b_dapo.yaml`.

The answer format is defined in `prompts/medical_format.jinja` and includes a modality label, `<thinking>` section, and `<answer>` section.

## Training configuration

The recorded training run used **2 × NVIDIA RTX PRO 6000 GPUs**. Training artifacts include curves through approximately **600 optimizer steps** and evaluation results associated with `global_step_600`.

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen3-VL-2B-Instruct` |
| Algorithm | GRPO advantage + DAPO-style policy optimization |
| Responses per question | 3 |
| Rollout / mini-rollout / actor batch | 16 / 16 / 16 |
| Dynamic group filtering | Enabled; group mean reward in `(0.01, 0.99)` |
| Clip lower / upper | `0.2` / `0.28` |
| Prompt / response maximum length | 4,352 / 4,096 tokens |
| Learning rate | `1e-6` |
| Configured epochs | 2 |
| GPUs | 2 × NVIDIA RTX PRO 6000 |
| KL loss | Disabled |
| Overlong reward shaping | Not enabled |

Start training from `medvlm-rl/`:

```bash
bash scripts/train_2b.sh
```

Training is launched through `python -m verl.trainer.main`, with `configs/medvlm_2b_dapo.yaml` specifying the medical task and DAPO-style hyperparameters. The framework performs rollout generation, reward calculation, group-relative advantage estimation and actor updates.

## Composite medical reward

`rewards/medical.py:compute_score` calculates a terminal reward from four components:

R = 0.5175 R_LLM + 0.3375 R_Embed + 0.045 R_Modality + 0.1 R_Format

| Reward | Description |
|---|---|
| LLM correctness | Reference-based assessment of the predicted answer; exact matches are accepted directly |
| Embedding similarity | Medical answer similarity using `abhinand/MedEmbed-large-v0.1` and a cosine-similarity threshold of `0.8` |
| Modality | Correct medical image modality tag |
| Format | Correct `<thinking>` and `<answer>` output structure |

The LLM judge uses an OpenAI-compatible API. Its configuration can be provided through `MEDVLM_JUDGE_API_KEY`, `MEDVLM_JUDGE_BASE_URL`, and `MEDVLM_JUDGE_MODEL`; the existing repository-root `config.py` remains supported. Keep credentials outside version control.

## Evaluation results

Evaluation scripts and individual prediction records are saved in `evaluation/`. The recorded **Qwen3-VL-2B DAPO checkpoint at step 600** achieved the following result on **VQA-RAD (radiology visual question answering)**:

| Benchmark | Evaluated samples | Correct | Accuracy |
|---|---:|---:|---:|
| VQA-RAD | 451 | 223 | **49.45%** |

The saved result is in `evaluation/results/huggingface/`, including the scored answer records and summary. Evaluation uses the project's configured answer-judging workflow. Results should be compared under a consistent dataset split and judge configuration.

### Checkpoint merging

FSDP checkpoint shards can be converted into a Hugging Face model directory with:

```bash
bash scripts/merge_checkpoint.sh /absolute/path/to/global_step_600/actor
```

The evaluation entry point is `evaluation/eval.sh`; update its model and dataset locations to match your environment. The archive contains the training/evaluation scripts and experiment artifacts, not large model-weight files.

## Training curves

See `reports/training/` for saved plots covering policy-gradient loss, policy entropy, mean response length, aggregate reward, and individual reward components. `training_report.md` discusses learning dynamics observed over the run.

## Third-party software and attribution

The RL training infrastructure uses [EasyR1](https://github.com/hiyouga/EasyR1) and [veRL](https://github.com/volcengine/verl); This repository's organization, configuration and experiment artifacts are maintained as the MedVLM-RL project. Third-party copyright and license obligations remain applicable.
