# [ACL'26 Oral] ILVR: Interleaved Latent Visual Reasoning with Selective Perceptual Modeling

[![arXiv](https://img.shields.io/badge/arXiv-2512.05665-b31b1b.svg)](https://arxiv.org/abs/2512.05665)
[![ACL 2026](https://img.shields.io/badge/ACL%202026-Accepted-success)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Authors:** Shuai Dong, Siyuan Wang, Xingyu Liu, Chenglin Li, Haowen Hou, Zhongyu Wei

**Affiliations:** China University of Geosciences, Wuhan; Shanghai Innovation Institute; University of Southern California; Fudan University; Zhejiang University; Shanghai Jiao Tong University.

**Venue:** ACL 2026 (**Oral Presentation**).

---

## 📖 Abstract

<!-- Please ensure you have created an 'assets' folder and uploaded 'framework.png' -->
![Model Architecture](assets/framework.png)

> **Abstract:** Interleaved reasoning paradigms enhance Multimodal Large Language Models (MLLMs) with visual feedback but are hindered by the prohibitive computational cost of re-encoding pixel-dense images. A promising alternative, latent visual reasoning, circumvents this bottleneck yet faces limitations: methods either fail to capture intermediate state evolution due to single-step, non-interleaved structures, or sacrifice precise perceptual modeling by over-compressing features. We introduce Interleaved Latent Visual Reasoning (ILVR), a framework that unifies dynamic state evolution with precise perceptual modeling. ILVR interleaves textual generation with latent visual representations that act as specific, evolving cues for subsequent reasoning. Specifically, we employ a self-supervision strategy where a momentum teacher model selectively distills relevant features from ground-truth intermediate images into sparse supervision targets. This adaptive selection mechanism guides the model to autonomously generate context-aware visual signals. Extensive experiments on multimodal reasoning benchmarks demonstrate that ILVR outperforms existing approaches, effectively bridging the gap between fine-grained perception and sequential multimodal reasoning.

---

## 📢 News
* **[2026-05-29]** 🎉 **ILVR** has been selected for an **Oral Presentation** at ACL 2026.
* **[2026-04-08]** 🎉 **ILVR** has been accepted to **ACL 2026**.
* **[2026-01-20]** 🚀 We have released the evaluation scripts (`eval.sh` and `eval.py`) and uploaded the model checkpoints to our Hugging Face repository: [shuai22/comt_ckpt](https://huggingface.co/shuai22/comt_ckpt).
* **[2025-12-08]** The code is released.
* **[2025-12-08]** The paper is released on [arXiv](https://arxiv.org/abs/2512.05665).

---

## 🛠️ Installation

### 1. Environment Setup
The code is tested with **Python 3.11**. We recommend using Conda for environment management.

```bash
# 1. Create a conda environment
conda create -n ilvr python=3.11
conda activate ilvr

# 2. Install standard dependencies
pip install -r requirements.txt

# 3. Install custom Transformers library
# ILVR requires modifications to the standard transformers library.
# We provide the modified source code in this repository.
cd transformers
pip install -e .
cd ..
```

### 2. Accelerate Configuration
This project uses HuggingFace `accelerate` for distributed training. Please configure it before running the training script.

```bash
accelerate config
```

---

## 📚 Data Preparation

We utilize the CoMT dataset (Chain of Multi-modal Thought) as an example for constructing training data. For more details about the benchmark, please refer to the [CoMT paper](https://arxiv.org/abs/2412.12932).

### Download Data
We provide the processed data on HuggingFace. Please download it from [shuai22/comt](https://huggingface.co/datasets/shuai22/comt) and organize the directory as follows.

1. Download `TRAIN.jsonl`, `TEST.jsonl`, and `comt.tar.gz`.
2. Extract the images from the tarball.

**Expected Directory Structure:**
```text
ILVR/
├── data/
│   ├── TRAIN.jsonl
│   ├── TEST.jsonl
│   └── images_comt/      <-- Extracted from comt.tar.gz
│       ├── creation/
│       └── ...
├── src/
├── transformers/
├── run_training.sh
└── README.md
```

### Data Format
The dataset follows the JSONL format:
- **text_input**: The question/instruction.
- **image_input**: Initial input images.
- **sequence_plan**: The interleaved chain-of-thought rationale containing "text" and "helper_image" paths.

---

## 🚀 Training

We provide a shell script `run_training.sh` to launch distributed training.

### 1. Configure Script
Open `run_training.sh` and modify the paths to match your local setup:

```bash
# In run_training.sh:

# Path to the directory containing TRAIN.jsonl
DATA_PATH="/path/to/your/data" 

# Directory to save model checkpoints
SAVE_MODEL_PATH="/path/to/save/checkpoints"

# File path for training logs
LOG_FILE="/path/to/save/train.log"

# (Optional) HuggingFace Cache Directory
export HF_HOME="/path/to/cache" 
```

### 2. Run Training
Start training with the following command:

```bash
bash run_training.sh
```

**Default Hyperparameters:**
- Base Model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Epochs: 15
- Gradient Accumulation Steps: 8
- Latent Size: 8

### Evidence-window diagnostics and training objectives

Evidence-window training asks a local question: when one generated visual
evidence span is replaced by a zero latent while the token history stays fixed,
how much does that span change each later reasoning window? The implementation
is optional. With no `--ew_config`, or with
`configs/evidence_window/baseline.json`, the original ILVR path is exact: no
reference model, evidence-window layout, auxiliary forward, or auxiliary metric
buffer is constructed.

#### Native spans and the zero-latent intervention

For an example `b`, visual evidence `V_j` is the interior of a matched
`<|latent_start|> ... <|latent_end|>` range, excluding the delimiters. Only
positions selected by the post-selection `image_out_mask` (`firstK_mask` in the
trainer) are active evidence positions; unused latent-pad template positions do
not participate in an intervention.

`R_0` is the supervised assistant-text run before the first visual span and has
no visual predecessor. For each `V_j`, its native downstream reasoning window
is the supervised text after `V_j` and before the next latent span, assistant
termination, or padding. A reasoning token must have `attention_mask == 1`,
`labels != -100`, lie outside every latent span, and not be a tokenizer special
token. Adjacent latent spans do not create empty windows. EW preference training
uses the adjacent native pair; all causally earlier `V_j -> R_k` pairs remain
available to relation diagnostics and distillation.

The conditions are:

- `full`: all active EMA-selected latent vectors are injected;
- `all_null`: all active latent vectors are replaced by zero; and
- `loo(j)`: only active vectors in `V_j` are replaced by zero.

The intervention copies the compact latent-value tensor. It preserves input
IDs, labels, all text and other visual evidence, sequence length, attention
masks, positions, and RoPE layout. It is therefore a zero-latent,
frozen-token-history LOO dependency measurement. It includes effects propagated
through recomputed hidden representations, but it is not the total causal
effect of removing evidence and regenerating the text trajectory.

#### Implemented objective

For target position `t`, every auxiliary role uses the causal score

```text
log p_t(C) = log_softmax(logits[:, t-1, :])[target_t]
S_k^M(C)   = sum over t in R_k of log p_t^M(C)
```

Preference scores use sums, not length normalization. With policy `pi`, fixed
reference `ref`, and `m_k^all = [S_k^pi(full)-S_k^pi(all_null)] -
[S_k^ref(full)-S_k^ref(all_null)]`, the implemented losses are:

```text
M_global = sum_k m_k^all
L_global = -log sigmoid(beta * M_global)
L_window = mean_k[-log sigmoid(beta * m_k^all)]

m_jk = [S_k^pi(full)-S_k^pi(loo(j))]
       - [S_k^ref(full)-S_k^ref(loo(j))]
L_EW = mean over adjacent native pairs of -log sigmoid(beta * m_jk)
```

Relation distillation uses all selected causal predecessor pairs:

```text
C_jk^M = [S_k^M(full)-S_k^M(loo(j))] / |R_k|
q_M(j|k) = softmax_j(C_jk^M / temperature)
L_rel = mean_k JSD(q_EMA(.|k), q_policy(.|k))
L_mag = mean_(j,k) Huber(C_jk^policy, C_jk^EMA)
```

JSD skips windows with fewer than two valid candidates; magnitude loss still
uses single-candidate windows. Logit-residual KD compares
`W_lm_head * (h_full - h_loo(j))` between policy and EMA at selected causal
predictor positions. It supports Huber or cosine distance and chunks the LM-head
vocabulary, so it never materializes a full `[selected_tokens, vocabulary]`
residual matrix.

The complete implemented objective is:

```text
L = L_ILVR
  + lambda_global * L_global
  + lambda_window * L_window
  + lambda_EW * L_EW
  + lambda_rel * L_rel
  + lambda_mag * L_mag
  + lambda_logit * L_logit_residual
```

`L_ILVR` remains the existing weighted CE plus latent similarity objective, with
its original CE fallback. Coefficients are independent and default to zero.

All policy, fixed-reference, and EMA causal target scores share the same bounded
FP32 reduction. At most a `[B, 32, vocabulary]` token chunk is cast to FP32 for
log-softmax/gather; grad-enabled policy chunks are checkpointed and recomputed
during backward. The retained score tensor is only `[B, T]`. This does not
change the model forward/autocast dtype or the original ILVR CE calculation.

#### Forward schedule and fixed models

The original policy `full` output is reused. On one rank, each role performs at
most `O(1 + J_selected)` full-sequence conditions, never one forward per
evidence-window pair. Conditions shared by objectives are cached. In distributed
training, ranks first pad to the global evidence width and take the union of
selected evidence ordinals. Every rank then executes the same canonical padded
ledger, giving `O(1 + J_global_selected)` conditions per role; missing local
conditions are graph-free dummy forwards with unchanged latents. This keeps
collective-bearing model schedules aligned even when local examples have
different evidence counts.

The EMA teacher selects latent targets and supplies relation/residual targets.
It receives no gradients and updates exactly once after each successful Trainer
optimizer step, not after each gradient-accumulation microbatch. It is never the
DPO reference.

Two fixed-reference modes are available:

- `adapter_disabled` runs the active PEFT wrapper under `disable_adapter()` and
  `torch.no_grad()`. It is valid only when all trainable parameters are adapter
  parameters and the shared base is frozen. It adds no full base-model copy.
- `frozen_copy` snapshots the initialized policy before optimizer updates,
  switches it to evaluation/no-grad mode, prepares it as an evaluation model,
  and excludes it from optimizer groups and policy checkpoints.

Neither ZeRO-1 nor ZeRO-2 partitions frozen model parameters. Consequently, a
frozen 7B-class reference is replicated on every data-parallel rank while the
existing full EMA teacher also remains resident. ZeRO-1 partitions optimizer
state only; ZeRO-2 also partitions policy gradients. Expect roughly another
15–17 GB of BF16 weights per rank before activations, optimizer state, or
gradients. Startup logging reports reference parameter count/bytes/dtypes, the
detected ZeRO stage, replication under ZeRO-1 or ZeRO-2, the stage-specific
partitioning semantics, EMA coexistence, the selected intervention/cap, all
objective coefficients, diagnostics, and any active nonzero dropout. The
program preserves the explicitly selected stage and never silently switches it.

#### Configurations and launch examples

`--ew_config PATH` loads a strict version-1 JSON config; unknown keys, negative
coefficients, invalid caps, unsupported interventions, and preference modes
without a fixed reference fail at startup. `--deepspeed_config PATH` selects the
DeepSpeed file and otherwise remains `configs/config_stage3.json`. Preference
experiments with LoRA should pass both `--use_lora` and `--lora_dropout 0` so
condition contrasts do not use different dropout masks.

The shipped variants are:

| Config | Active behavior |
| --- | --- |
| `baseline.json` | Exact disabled ILVR path |
| `diagnostic_global.json` | Global all-null saturation metrics only |
| `window_preference.json` | Independent all-null window preference |
| `loo_window_preference.json` | Adjacent-pair EW preference |
| `relation_kd.json` | Relation JSD and magnitude KD |
| `logit_residual.json` | Chunked EMA logit-residual KD |
| `loo_relation_combined.json` | EW preference plus both relation losses |
| `frozen_reference_zero1.json` | Frozen-reference EW preference plus global diagnostics for ZeRO-1 |
| `frozen_reference_zero2.json` | Frozen-reference EW preference plus global diagnostics for ZeRO-2 |

For a LoRA EW run, add the usual dataset/output arguments used by
`run_training.sh`:

```bash
source /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/activate
accelerate launch --num_processes="${NUM_PROCESSES}" src/main.py \
  --stage stage1 --task zebra-cot \
  --data_path /path/to/TRAIN.jsonl \
  --save_model_path /path/to/output \
  --log_file /path/to/train.log \
  --cache_dir /path/to/cache \
  --use_lora --lora_dropout 0 \
  --ew_config configs/evidence_window/loo_window_preference.json
```

The explicit frozen-reference ZeRO-1 variant uses optimizer-state partitioning
while keeping policy gradients, policy parameters, EMA, and the frozen
reference replicated:

```bash
source /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/activate
accelerate launch --num_processes="${NUM_PROCESSES}" src/main.py \
  --stage stage1 --task zebra-cot \
  --data_path /path/to/TRAIN.jsonl \
  --save_model_path /path/to/output \
  --log_file /path/to/train.log \
  --cache_dir /path/to/cache \
  --lora_dropout 0 \
  --ew_config configs/evidence_window/frozen_reference_zero1.json \
  --deepspeed_config /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/.worktrees/ours-trial1/configs/config_stage1.json
```

The explicit frozen-reference ZeRO-2 variant uses optimizer-state and policy-
gradient partitioning with the stage-2 configuration:

```bash
source /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/activate
accelerate launch --num_processes="${NUM_PROCESSES}" src/main.py \
  --stage stage1 --task zebra-cot \
  --data_path /path/to/TRAIN.jsonl \
  --save_model_path /path/to/output \
  --log_file /path/to/train.log \
  --cache_dir /path/to/cache \
  --lora_dropout 0 \
  --ew_config configs/evidence_window/frozen_reference_zero2.json \
  --deepspeed_config configs/config_stage2.json
```

`max_spans_per_example=0` and `max_tokens_per_window=0` mean full coverage.
Positive caps select deterministic evenly spaced items and include temporal
endpoints when possible. `vocab_chunk_size` bounds logit-residual vocabulary
work. Global preference is implemented but the shipped global file is
diagnostic-only; set `preference.lambda_global` explicitly in a copied config to
train that ablation. JSD and magnitude coefficients may likewise be enabled
independently.

Training logs use bounded scalar keys for ILVR/auxiliary losses, global/window/
EW margins, the exact global outer coefficient, student/EMA relation summaries,
and selected LOO/residual counts. Numerator/count sufficient statistics are
reduced across ranks before means are emitted. Matrix dumps are off when
`matrix_dump_interval=0`; when enabled, each rank writes a numeric JSONL file in
`matrix_dump_dir` with only opaque rank/step/local-counter identifiers—never raw
text, sample IDs, or image paths.

Current limitations: there is no arbitrary attention blocking, hidden-state
residual KD, group/pair ablation, Shapley attribution, or supervised evidence-
window link selector. Adjacent-pair EW training can miss long-range requirements;
single-evidence LOO cannot identify synergy or redundancy. The helper-image
producer remains per-device batch size one. CPU/synthetic validation is
complete, including previously recorded two-rank schedule/metric tests and real
two-rank ZeRO-1 and ZeRO-2 frozen-reference smokes; the approved 7B/B200
optimizer-smoke matrix is pending completion.

---

## 📝 Citation

If you find this project or the ILVR framework useful, please cite our paper:

```bibtex
@article{dong2025interleaved,
  title={Interleaved latent visual reasoning with selective perceptual modeling},
  author={Dong, Shuai and Wang, Siyuan and Liu, Xingyu and Li, Chenglin and Hou, Haowen and Wei, Zhongyu},
  journal={arXiv preprint arXiv:2512.05665},
  year={2025}
}
```

## 🙏 Acknowledgements
This codebase is built upon [Qwen-VL](https://github.com/QwenLM/Qwen-VL) 、 [Transformers](https://github.com/huggingface/transformers) and [Mirage](https://github.com/UMass-Embodied-AGI/Mirage). We thank the authors for their open-source contributions.

## 📄 License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
