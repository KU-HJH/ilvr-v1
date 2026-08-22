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

### Stage-2 evidence-window credit (version 2)

The new training path gives the two ILVR stages deliberately different jobs:

- Stage 1 remains the original node-grounding path. Its EMA teacher,
  helper-image teacher forcing, CE/similarity objective, masks, and mixed-
  precision behavior are unchanged.
- Stage 2 starts from an explicit Stage-1 checkpoint, generates its own latent
  states, and uses assistant CE plus optional evidence-to-window edge losses.
  It does not reuse Stage-1 helper-image supervision or its EMA teacher.

With no `--ew_config`, or with
`configs/evidence_window/stage2/baseline.json`, Stage 2 performs exactly one
ordinary CE forward and creates no evidence-window engine or reference. All
version-2 coefficients default to zero. Version-1 configs under
`configs/evidence_window/` remain historical Stage-1 ablations and are not
accepted as enabled Stage-2 configs.

#### Exact spans and intervention

For Stage 2, `Z_j` is the complete marker-inclusive token range
`<|latent_start|> <|latent_pad|>... <|latent_end|>`. It is valid evidence only
when at least one latent-pad token lies between the markers. `R_k` is one
assistant-only supervised text window between latent ranges. Padding,
user/system text, latent markers/pads, assistant termination, special tokens,
and every `labels == -100` position are excluded. Preserved `sequence_plan`
types are checked against token parsing.

The likelihood target at token row `t` is predicted by decoder query row
`t-1`. Therefore a selected edge `(Z_j, R_k)` blocks exactly the Cartesian
product

```text
predictor queries {t - 1 : t in R_k}  x  marker-inclusive keys in Z_j
```

and no other query-key entries. It does not delete or zero tokens, change
positions, remove another visual span, or recompute earlier reasoning under a
different intervention. Ordinary CE stays on FlashAttention2. Both the full
and LOO auxiliary branches use the same model weights and the same explicit
SDPA backend, so backend differences cannot masquerade as contribution.

The default window score is mean target log-probability (`sum` is an explicit
config alternative):

```text
ell_M(R_k | H) = mean_{t in R_k} log p_M(y_t | H, y_<t)
C^M_jk = ell_M(R_k | full) - ell_M(R_k | block R_k -> Z_j)
```

This fixed-history LOO value estimates a conditional/direct re-reference
dependency while retaining previous reasoning and every other evidence span.
It is not the total causal effect of removing evidence and regenerating the
downstream trajectory.

#### Stage-2 objective and variants

The implemented objective is

```text
L_stage2 = L_CE
         + lambda_ew(r)  * mean_edges[-log sigmoid(beta * M_jk)]
         + lambda_rel(r) * mean_windows JSD(q_ref(.|k), q_policy(.|k))
         + lambda_res(r) * mean_edges,hidden Huber(delta_h_policy, stop(delta_h_ref))
```

For reference-free EW, `M_jk = C_policy_jk`. With a fixed reference,
`M_jk = C_policy_jk - stop(C_ref_jk)`. Relation distributions normalize only
over valid predecessor evidence for the current window. Hidden residuals are
mean-pooled on the same causal predictor-query rows and use
`delta_h = h_full - h_LOO`; they are effects to match, never representations
to subtract from normal reasoning.

Every loss has an independent configurable CE-only warmup and linear ramp.
Before warmup its effective weight is zero; it grows linearly during the ramp
and stays at the configured coefficient afterward. A zero ramp switches at
the boundary. Zero effective weights cause no auxiliary replay, reference
lookup, or hidden-state collection.

The source-edit-free version-2 experiment matrix is:

| Config | Only enabled objective | Reference |
| --- | --- | --- |
| `configs/evidence_window/stage2/baseline.json` | none | none |
| `configs/evidence_window/stage2/ew_reference_free.json` | EW preference | none |
| `configs/evidence_window/stage2/ew_precomputed_dpo.json` | EW-DPO | precomputed |
| `configs/evidence_window/stage2/relation_kd.json` | relation JSD | precomputed |
| `configs/evidence_window/stage2/hidden_residual_kd.json` | hidden residual Huber | precomputed |
| `configs/evidence_window/stage2/online_frozen_zero1.json` | EW-DPO | online frozen, ZeRO-1 only |
| `configs/evidence_window/stage2/online_frozen_zero2.json` | EW-DPO | online frozen, ZeRO-2 only |

Artifact/checkpoint entries in the template configs are intentionally explicit
absolute-path placeholders. Copy the selected JSON into the run directory and
replace only those path values; switching objectives never requires a source
edit. The online files do not select a DeepSpeed stage: pair them explicitly
with `configs/config_stage1.json` (ZeRO-1) or `configs/config_stage2.json`
(ZeRO-2). `config_stage*.json` names describe DeepSpeed ZeRO, not ILVR Stage 1
or Stage 2. The smoke harness enforces those exact named-case pairings and
rejects cross-pairing. A named case also validates the resolved objective,
reference mode, required outputs, and intervention; `--ew-config` may replace
a path only when its semantic signature still matches the case and is rejected
for `all-single-gpu`.

#### Launch and reference artifacts

All examples below assume this repository root and the required environment:

```bash
cd /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/.worktrees/ours-trial1
source /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/activate
```

Reference-free Stage 2 can run under ZeRO-3 because it owns no second policy:

```bash
accelerate launch src/main.py \
  --stage stage2 \
  --model /absolute/path/to/Qwen2.5-VL-7B-Instruct \
  --load_model_path /absolute/path/to/stage1-checkpoint \
  --task zebra-cot \
  --data_path /absolute/path/to/TRAIN.jsonl \
  --save_model_path /absolute/path/to/stage2-output \
  --log_file /absolute/path/to/stage2.log \
  --cache_dir /absolute/path/to/cache \
  --ew_config configs/evidence_window/stage2/ew_reference_free.json \
  --deepspeed_config configs/config_stage3.json
```

Under Stage 2, `--load_model_path` supplies the model and processor
initialization; `--model` remains a Stage-1 launcher argument. The checkpoint
must exist. Relative image paths in the current dataset loader still resolve
through `task_deepseed.TRAIN_IMAGE_ROOT`; use absolute image paths in data or
set that repository constant for the environment before launching.

Produce immutable reference targets once with Stage-2 self-generated latent
semantics. Add `--include-hidden-residuals` only for the hidden-residual config:

```bash
/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python \
  src/precompute_stage2_reference.py \
  --data-path /absolute/path/to/TRAIN.jsonl \
  --reference-checkpoint /absolute/path/to/stage1-checkpoint \
  --reference-checkpoint-id stage1-approved \
  --reference-checkpoint-witness /absolute/or-authenticated-checkpoint-witness \
  --artifact-dir /absolute/path/to/stage2-reference-artifact \
  --tokenizer-id /absolute/path/to/stage1-checkpoint \
  --model-config-id qwen2.5-vl-stage2-self-latent-v1 \
  --contribution-reduction mean
```

The producer hashes the dataset once unless
`--dataset-sha256-witness` supplies an authenticated digest. It never hashes
checkpoint weights. Training reconstructs and verifies the manifest identity,
so precomputed runs must also pass matching
`--reference_dataset_sha256`, `--reference_checkpoint_id`,
`--reference_checkpoint_witness`, `--reference_tokenizer_id`,
`--reference_parser_version`, and `--reference_model_config_id` arguments.
Artifacts are keyed by deterministic source-row index, exact special tokens,
parser/reduction/intervention/backend semantics, edge cap, and model identity.

Online frozen EW loads the exact same Stage-1 checkpoint independently and
runs it under Stage-2 self-generated-latent semantics with `eval()` and
`torch.no_grad()`. It is not the Stage-1 EMA teacher and is never optimized.
ZeRO-1 and ZeRO-2 replicate this full frozen model on every rank; they do not
partition its parameters. Expect approximately one extra full parameter copy
per rank plus transient reference activations. ZeRO-3 is rejected before
reference construction; use reference-free or precomputed mode instead.

#### Cost, smoke harness, and interpretation limits

For `E` selected local evidence-window edges, a policy step performs one normal
CE forward plus `1 + E` matched auxiliary forwards. This is
`O(1 + E_local)` auxiliary work. Distributed ranks pad to the global maximum
edge count with all-false same-policy slots so collective-bearing schedules
remain identical; the effective schedule is `O(1 + E_global_max)` per role.
Online fixed reference repeats that schedule under no-grad. Residual mode alone
requests final hidden states.

The CPU config/optimizer smoke is:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:transformers/src \
  /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python \
  -m pytest \
  tests/evidence_window/test_stage2_experiment_configs.py \
  tests/evidence_window/test_stage2_smoke.py -q
```

The callable B200 harness performs exactly one optimizer step per requested
case and writes small JSON witnesses for finite losses, edges/forwards,
policy gradients, fixed-model gradients, dtype, and device. `--output-dir`
must name a nonexistent attempt directory. An ordinary single-process call may
omit launcher variables. If any distributed launcher variable is present, the
harness requires valid `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, and
`MASTER_PORT` together; it never invents missing rank state. It validates
optional `LOCAL_WORLD_SIZE`, the master-port range, and the local rank against
visible CUDA devices before device selection or rendezvous. It then initializes
the launcher-provided `env://` process group with NCCL on the CUDA target or
Gloo on CPU before ownership or preflight. A preinitialized group is reused
only when its rank, world size, backend, and selected local device match that
same contract. Initialization or validation failure stops before processor/model
loading. Rank 0 then exclusively claims the directory, all ranks collectively
agree on claim success, and only then does every rank validate the claim and run
preflight. A concurrent launcher or retry using the same path therefore loses
collectively without loading a model. `--attempt-id` is a path-safe identifier
bound into the claim, every rank filename, rank JSON, and completion manifest:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:transformers/src \
  /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python \
  .superpowers/sdd/2026-08-22-ilvr-stage2-evidence-window-credit/stage2_large_model_smoke.py \
  --case all-single-gpu \
  --model /absolute/path/to/Qwen2.5-VL-7B-Instruct \
  --stage1-checkpoint /absolute/path/to/stage1-checkpoint \
  --data /absolute/path/to/TRAIN.jsonl \
  --image-root /absolute/path/to/image-root \
  --sample-index 920 \
  --attempt-id ew-v2-smoke-001 \
  --deepspeed-config configs/config_stage3.json \
  --reference-artifact-dir /absolute/path/to/stage2-reference-artifact-with-hidden-residuals \
  --reference-dataset-sha256 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --reference-checkpoint-id stage1-approved \
  --reference-checkpoint-witness /absolute/or-authenticated-checkpoint-witness \
  --reference-tokenizer-id /absolute/path/to/stage1-checkpoint \
  --reference-parser-version stage2-marker-v1 \
  --reference-model-config-id qwen2.5-vl-stage2-self-latent-v1 \
  --output-dir /absolute/path/to/stage2-smoke-results
```

The harness reconstructs the expected artifact identity from these explicit
consumer-side witnesses. It never accepts the producer manifest itself as the
expected identity and does not hash checkpoint weights. Before atomically
starting model work, it resolves every selected case (including the complete
`all-single-gpu` matrix), validates config signatures and DeepSpeed pairings,
checks obvious paths, parses the reference manifest, and compares every
independent identity field. Exact tokenizer special-token identity is checked
after processor construction.

After training, every rank first validates all local named losses, finite
totals, eligible edges/windows, exact padded forward ledgers, nonzero policy
learning witnesses, and absence of reference/EMA gradients. Validation errors
are aggregated collectively before any rank JSON is published. Rank JSON files
are serialized to same-directory temporary files, flushed and file-`fsync`ed,
then atomically exposed with no-overwrite hard links; pre-link/link publication
errors are collectively checked. Every rank next validates the complete
expected file set and attempt identity in a final precommit collective. Only
after that agreement does rank 0 expose `completion.json` with the same
protocol. The successful hard link is the logical terminal commit point.
Temporary cleanup and directory `fsync` happen afterward on a best-effort,
non-throwing basis: failure there does not retract a visible completion, though
it means crash durability was not additionally confirmed by that directory
`fsync`. Hard-link failure leaves no destination. No collective or other
required success operation occurs after the terminal link.
Consumers must treat `completion.json`, not an individual rank JSON, as the
success witness. A rank-0 failure before or at the completion link leaves no
completion and fails that rank; once linked, completion remains logically
committed even if best-effort cleanup/directory sync fails or a process later
exits unexpectedly. Failed or partial attempts are immutable diagnostic state;
retry with a new attempt ID and output path.

Do not treat a high relation entropy, dense attention, or positive LOO score as
proof of correct evidence use. Sparse dependency can be correct. The local
loss intentionally differs from a global preference loss to prevent one edge
from saturating every other edge's outer gradient. Single-evidence LOO cannot
resolve synergy or redundancy, and no group/Shapley attribution or automatic
task-required edge labeler is implemented.

### Historical Stage-1 zero-latent evidence-window ablations (version 1)

The following section documents the retained version-1 research branch. It
uses Stage-1 EMA/helper-image semantics and zero-latent interventions. It is
not the exact Stage-2 `window_attention_block` implementation described above,
and its configs must not be used to claim Stage-2 results.

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
  --deepspeed_config configs/config_stage1.json
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

#### 7B/B200 optimizer-smoke results

The production Qwen2.5-VL 7B path, real interleaved sample, LoRA (`r=4`,
`alpha=8`, dropout zero), compact latent producer, condition forwards, Trainer
optimizer/callback lifecycle, and EMA update were validated for one optimizer
step per case on NVIDIA B200 GPUs. Successful cases were not repeated.

| Case | Ranks / ZeRO | Active auxiliary loss | Canonical evidence conditions per rank (`P/R/E`) | Peak allocated / reserved | Wall time |
| --- | --- | --- | --- | --- | --- |
| Baseline disabled | 1 / none | none | `F1/F0/F0` | 33.50 / 34.74 GiB | 27.65 s |
| Global diagnostic | 1 / none | metrics only | `F1+A1/F1+A1/F0` | 33.88 / 34.74 GiB | 22.99 s |
| Window preference | 1 / none | `0.69314718` | `F1+A1/F1+A1/F0` | 35.86 / 37.01 GiB | 22.65 s |
| EW-DPO, adapter-disabled | 1 / none | `0.69314718` | `F1+L3/F1+L3/F0` | 38.12 / 39.05 GiB | 23.70 s |
| EW-DPO, frozen copy | 2 / ZeRO-1 | `0.67923099` | `F1+A1+L3/F1+A1+L3/F0` | 51.81 / 56.59–56.64 GiB | 57.30 s |
| EW-DPO, frozen copy | 2 / ZeRO-2 | `0.67923099` | `F1+A1+L3/F1+A1+L3/F0` | 51.81 / 54.70 GiB | 80.02 s |
| Relation JSD | 1 / none | `0.0` at identical initialization | `F1+L3/F0/F1+L3` | 38.12 / 39.29 GiB | 23.55 s |
| Relation magnitude | 1 / none | `0.0` at identical initialization | `F1+L3/F0/F1+L3` | 38.12 / 39.29 GiB | 23.25 s |
| Logit residual | 1 / none | `3.784389e-6` | `F1+L3/F0/F1+L3` | 38.63 / 39.25 GiB | 28.10 s |
| Explicit combined | 1 / none | EW `0.69314718`; JSD/magnitude `0.0` | `F1+L3/F1+L3/F1+L3` | 38.12 / 39.05 GiB | 23.92 s |

`F`, `A`, and `L` denote full, all-null, and LOO conditions; `P`, `R`, and
`E` denote policy, fixed reference, and EMA. Every case had finite ILVR and
total loss, a student update, no fixed-reference or EMA gradients, and exactly
one EMA update. Both frozen-copy runs preserved the requested stage and showed
identical replicated-reference structural/scalar witnesses on both ranks.

Current limitations: there is no arbitrary attention blocking, hidden-state
residual KD, group/pair ablation, Shapley attribution, or supervised evidence-
window link selector. Adjacent-pair EW training can miss long-range requirements;
single-evidence LOO cannot identify synergy or redundancy. The helper-image
producer remains per-device batch size one. CPU/synthetic validation is
complete, including previously recorded two-rank schedule/metric tests and real
two-rank ZeRO-1 and ZeRO-2 frozen-reference smokes; the approved 7B/B200
optimizer-smoke matrix above is complete.

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
