# [ACL'26 Oral] ILVR: Interleaved Latent Visual Reasoning with Selective Perceptual Modeling

## Final Stage 1/2 training

For the consolidated Stage 1 → Stage 2 training and evaluation workflow, start
with [the final training runbook](docs/final-stage1-stage2/README.md). Its
[provenance manifest](docs/final-stage1-stage2/PROVENANCE.md) records the exact
source and evidence boundaries.

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

For canonical ZeRO-1/2/3 configuration selection, artifact ownership, and
same-identity resume commands, see [the ZeRO stage training guide](docs/zero-stage-training.md).
For the fixed `korea@172.16.11.41` / `llm_kd` two-node four-GPU topology,
including exact preflight, training, resume, evaluation, capacity, lifecycle,
validation, and exact-owned cleanup commands, see
[the four-GPU multi-node operator guide](docs/multinode-training.md).

Canonical ZeRO compatibility is:

| Stage | Supported methods | Canonical DeepSpeed stages |
|---|---|---|
| Stage 1 | Full / LoRA / DoRA | ZeRO-1 / ZeRO-2 / ZeRO-3 |
| Stage 2 | Full / LoRA / DoRA | ZeRO-1 / ZeRO-2 / ZeRO-3 |

Optimizer resume requires the same ZeRO stage, configuration digest, and world
size as the checkpoint. Optimizer state cannot be shared across different ZeRO
identities.

The supported new-run operator commands live in
[the ZeRO stage training guide](docs/zero-stage-training.md). `run_training.sh`
is a Stage 1 compatibility wrapper; use it only with the guide's canonical
checkpoint, result, and log roots rather than editing generic output paths.

### Stage 1-v5 history queries

Stage1-v5 defaults to `--query_mode legacy`; existing `textual_states` behavior
is preserved. Adaptive queries are opt-in with `--query_mode adaptive`,
`--history_text_gamma 0.9` and `--history_attention_layers last`. New adaptive
training uses the final-normalized Teacher hidden states and accepts `last` only;
`last4` and `all` remain readable for historical artifact inspection.
Adaptive mode rejects an explicitly supplied
`--textual_states`, including the same value as the legacy default.

Use the existing v5 recipe to preview the coordinator command, inference
preparation and TEST400 evaluation commands. Set `STAGE1_MAX_STEPS` from the
retained-data/prepared-loader audit, and set the two remote roots to the
compatible reviewed checkout; the example only prints commands:

```bash
bash scripts/stage1_v5_train_eval.sh \
  --run-id stage1-v5-adaptive-example \
  --main-process-port 29683 \
  --remote-repo-root /root/workspace/ILVR \
  --remote-host-repo-root /home/korea/Workspace/hjh/workspace/ILVR \
  --max-steps "${STAGE1_MAX_STEPS:?set from the prepared-loader audit}" \
  --query_mode adaptive --history_text_gamma 0.9 \
  --history_attention_layers last --dry-run
```

The first printed command invokes `run_multinode_stage1_training.sh`; append
`--dry-run` to that command to inspect the coordinator's mapped node commands.
A preview establishes arguments only. Actual execution requires the existing
two-node/four-rank preflight and separately authorized launch, with compatible
code/environments and the resolved profile recorded on both nodes. The recipe
enforces its selected fresh-run profile after optional `--settings` sourcing:
Qwen2.5-VL/DoRA r32/alpha64/dropout 0.05, latent size 8, per-device batch 1,
GAS 4/global batch 16, frozen initial effective reference/tau 1, BF16,
decoder SDPA, vision FlashAttention2, existing gradient checkpointing and
ZeRO-1. It uses CE 0.1/alignment 0.1, v5 DIL beta 1/margin 0/preference weight 1,
top-k entropy 0.1 with K 50, Text-off KL 0.01, LR 5e-5, seed 42 and warmup 10.
The original base is the initializer; previous Stage2 weights are not inputs.

Stage1-v5 DIL averages eligible token losses across the actual optimizer window
and all four ranks. Entropy/mass still average original-block token means,
including empty blocks in their denominator; CE/alignment/Text-off KL retain
configured-GAS scaling. New token-mean training requires `selection=all` in both
query modes. Historical block-mean sidecars remain readable for inspection;
resuming those runs requires their original reviewed code. Changing to token
mean requires a fresh run, which saves a distinct schema-2 reduction contract.

Adaptive target selection observes the Teacher with the literal ILVR/legacy
wrapper call: full sample token IDs, original images, and actual latent-pad token
embeddings. It makes one full-sequence language observation; selected targets
are never injected into that observation. The native causal mask, sliding-window
configuration and MRoPE preparation stay with the existing wrapper. Helper images
are encoded separately and supply candidates only.

History relevance is external FP32 cosine softmax at fixed temperature 1. For
each current assistant-text row before the current START, it scores all earlier
selected **raw visual target rows**; it then averages the text-row probabilities
and sums the raw values with those weights. This is relative content relevance,
not native attention mass. Query/key chunks 64/2048 bound metric workspace.
Nonempty segment means receive one discounted vote per segment (gamma 0.9 by
default). Intent, available discounted text and available history have equal
coefficients. Empty current text or empty prior history omits the history term;
previous discounted text may still contribute. No current/future helper or
selected target changes the Teacher observation.

`stage1_v5.json` records query identity separately from the loss. Legacy remains
schema 1 with `stage1_legacy_query_v1`; current adaptive uses query schema 2 and
`stage1_placeholder_discounted_segments_cosine_relative_v2`, including its
placeholder Teacher context, native mask, final-normalized hidden states,
gamma, query/key chunks and fixed temperature. Historical native-adaptive query
schema 1 (`stage1_injected_prefix_discounted_segments_attention_mass_v1`) remains
readable, but training requires its original reviewed code; switching to external
content adaptive requires a fresh run. Resume inherits supported saved settings
and rejects semantic conflicts. Historical query-less checkpoints mean legacy
only. The recipe above is for fresh runs; resume through the existing coordinator
with saved identity and a fresh attempt ID. Base/reference/source checks continue
to apply.

The target-selection Teacher remains the frozen initial effective DoRA model
(EMA tau 1). DIL reference scoring retains its existing rules and weights and
consumes the shared detached selected targets. Losses, global-token DIL reduction,
auxiliary scaling, workers, sampler and optimizer lifecycle are unchanged. The
new metric's usefulness, runtime cost and accuracy are **unmeasured**.

The CPU integration witnesses use tiny FP32 Qwen/DoRA models and the actual
v5 parent-Trainer backward path. They check shared selected targets across
F/reference/packed OFF, student gradients with a frozen reference, configured
GAS denominators on partial windows and empty auxiliaries, decoder call counts,
repeated-build RNG/hook cleanup, and saved-config next-target reproducibility.
These tests do not run the production composition/ZeRO lifecycle, BF16/GPU or
FlashAttention2 execution, four-rank training, long real inputs, performance or
benchmark evaluation. Those runtime results remain unverified by this CPU
integration check.

<!-- BEGIN HISTORICAL NONCANONICAL PERFORMANCE WORKFLOW -->
### Historical Full and DoRA performance-workflow provenance — do not run

The former pre-policy `outputs/` workflow is retained only in repository
history and existing read-only evidence. It is noncanonical provenance, not an
operator procedure: do not execute, revalidate, move, rename, overwrite, or
add files to that tree. The active operator path is
[the ZeRO stage training guide](docs/zero-stage-training.md), which enforces
the current canonical artifact topology and DeepSpeed configurations.

This historical workflow's performance scope was Full versus DoRA only; LoRA
was never part of that performance comparison. No historical command is
published here so it cannot be mistaken for a supported launch instruction.
<!-- END HISTORICAL NONCANONICAL PERFORMANCE WORKFLOW -->

### Stage 2 training

Stage 2 starts from a Stage 1 checkpoint and replaces CoMT helper-image steps
with differentiable continuous-latent blocks. The original/user image remains
a model input, but helper images are never loaded as Stage 2 visual inputs. The
entire visual tower is frozen and is encoded once per rank-local batch.

#### Stage 2B context preference

Context preference is the default Stage 2B objective. With real Stage 1 and
Stage 2A checkpoints and dataset paths, launch it in this form:

```bash
STEP_PREFERENCE_ENABLED=true \
STEP_PREFERENCE_MODE=stage2b \
STEP_PREFERENCE_OBJECTIVE=context_preference \
CE_WEIGHT=0.1 \
STEP_PREFERENCE_BETA=10.0 \
STEP_PREFERENCE_WEIGHT=1.0 \
STEP_PREFERENCE_MARGIN=0.5 \
STEP_ENTROPY_WEIGHT=0.1 \
STEP_ENTROPY_MARGIN=0.1 \
STAGE1_CHECKPOINT="$STAGE1_CHECKPOINT" \
STAGE2A_CHECKPOINT_PATH="$STAGE2A_CHECKPOINT_PATH" \
DATA_PATH="$DATA_PATH" IMAGE_ROOT="$IMAGE_ROOT" \
OUTPUT_DIR="$OUTPUT_DIR" \
bash run_stage2_training.sh
```

Set `OUTPUT_DIR` to a distinct run directory below the current worktree's
`checkpoints/` symlink. The context objective shares the clean prefix. Its
default V-only route scores the clean and visual-off (V) contexts; the V+T
opt-out additionally scores the textual-off (T) context, for at most three
scoring decoder calls per local microbatch. Existing clean latent decoder
calls and backward checkpoint recomputation are counted and reported
separately. The default context route performs zero reference forwards,
negative generations, rejected-branch scorings, or preference rollouts.

Context-preference Stage 2B defaults to the selected V-only configuration: the
preference target is `immediate_text`, and the visual-off comparison suppresses
the current latent control boundaries. No extra settings are required.

To restore the former V+T comparison, or to keep the latent boundaries visible
in a V-only run, use the corresponding explicit opt-out:

```bash
STEP_VISUAL_ONLY_CONTEXT_PREFERENCE=false \
bash run_stage2_training.sh

STEP_VISUAL_OFF_SUPPRESS_LATENT_BOUNDARIES=false \
bash run_stage2_training.sh
```

Boundary suppression is valid only with V-only context preference. Its V-off
sidecar blocks root attention to `<|latent_start|>`, every current latent
state, and `<|latent_end|>`, and replaces the replayed `<|latent_end|>` query
embedding with a fixed zero vector. Clean scoring, including ordinary latent
boundary supervision, remains unchanged. This is a content intervention that
preserves the original mRoPE positions and latent-block geometry; it is not
equivalent to deleting the latent block from the sequence.

Each target block score is its mean target-token log probability. Preference
uses `-logsigmoid(beta * delta - margin)`, subtracting the tunable margin
outside beta. Fresh training defaults to CE weight 0.1, ordinary preference
beta 10 / margin 0.5, and preference weight 1. All [DIL-BCE variants](docs/dil-bce.md)
instead use beta 1 / margin 0. Keep existing runs and saved resume coefficients
and loss semantics; the historical 54% result used CE1 / beta0.5 / margin0.
Preference and entropy losses are reduced within each step first, then all V/T steps have
equal weight, so longer text steps do not receive more weight. Every valid
step is scored. If the input does not fit memory, report the input size as a
blocker; do not split sidecar groups into chunks or silently drop steps to fit.
These call-count and numerical checks establish implementation behavior, not
training quality, long real-input memory suitability, or paper-level results.

Ordinary Stage 2 remains available with `STEP_PREFERENCE_ENABLED=false`.
Historical DPO experiments require the explicit
`STEP_PREFERENCE_OBJECTIVE=legacy_dpo` selection and their separate legacy
negative data and manifest files; those artifacts are not inputs to context
preference.

`run_stage2_training.sh` launches the Stage 2 path with this checkout's
`./venv/ilvr-venv` Python and local `transformers/src` and `src` trees. It is
safe to invoke from outside the repository because all runtime paths are
resolved relative to the script. The reproduction-oriented defaults are:

- `LATENT_SIZE=8`;
- `SUPERVISE_LATENT_BOUNDARIES=true` (start and end are always paired);
- `LEARNING_RATE=1e-5`;
- `GRADIENT_ACCUMULATION_STEPS=8`;
- `ATTENTION_IMPLEMENTATION=sdpa`;
- `GRADIENT_CHECKPOINTING=false`;
- `MAX_SEQUENCE_LENGTH=8192`;
- `TUNING_METHOD=full`;
- `LORA_R=32`, `LORA_ALPHA=64`, and `LORA_DROPOUT=0.05`;
- `LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`;
  and
- rank-local/per-device batch size fixed at `1`.

For fresh full tuning, provide a complete Stage 1 checkpoint and do not set
`RESUME_CHECKPOINT`:

```bash
zero_checkpoint_root=/root/workspace/ILVR/checkpoints
zero_result_root=/root/workspace/ILVR/results
zero_log_root=/root/workspace/ILVR/logs/zero-stage-compatibility
run_suffix="$(date -u +%Y%m%dt%H%M%S)-$$-${RANDOM}"
experiment_id="stage2-zero3-full-fresh-example-$run_suffix"
STAGE1_CHECKPOINT=/path/to/stage1/checkpoint \
DATA_PATH=/path/to/TRAIN.jsonl \
IMAGE_ROOT=/path/to/data \
OUTPUT_DIR="$zero_checkpoint_root/$experiment_id" \
LOG_FILE="$zero_log_root/$experiment_id/train.log" \
TUNING_METHOD=full \
LATENT_SIZE=8 \
SUPERVISE_LATENT_BOUNDARIES=true \
LEARNING_RATE=1e-5 \
GRADIENT_ACCUMULATION_STEPS=8 \
ATTENTION_IMPLEMENTATION=flash_attention_2 \
GRADIENT_CHECKPOINTING=true \
MAX_SEQUENCE_LENGTH=8192 \
DEEPSPEED_CONFIG="$PWD/configs/deepspeed_zero3.json" \
./run_stage2_training.sh
```

`zero_result_root/$experiment_id` is the reserved canonical result directory
for any separately generated evidence from this run; this launcher produces
the checkpoint and log paths above.

Fresh LoRA training uses the same Stage 1 base but trains adapters only:

```bash
zero_checkpoint_root=/root/workspace/ILVR/checkpoints
zero_result_root=/root/workspace/ILVR/results
zero_log_root=/root/workspace/ILVR/logs/zero-stage-compatibility
run_suffix="$(date -u +%Y%m%dt%H%M%S)-$$-${RANDOM}"
experiment_id="stage2-zero3-lora-fresh-example-$run_suffix"
STAGE1_CHECKPOINT=/path/to/stage1/checkpoint \
DATA_PATH=/path/to/TRAIN.jsonl \
IMAGE_ROOT=/path/to/data \
OUTPUT_DIR="$zero_checkpoint_root/$experiment_id" \
LOG_FILE="$zero_log_root/$experiment_id/train.log" \
TUNING_METHOD=lora \
LORA_R=32 \
LORA_ALPHA=64 \
LORA_DROPOUT=0.05 \
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
DEEPSPEED_CONFIG="$PWD/configs/deepspeed_zero3.json" \
./run_stage2_training.sh
```

DoRA has the same target and hyperparameter contract, with
`TUNING_METHOD=dora`:

```bash
zero_checkpoint_root=/root/workspace/ILVR/checkpoints
zero_result_root=/root/workspace/ILVR/results
zero_log_root=/root/workspace/ILVR/logs/zero-stage-compatibility
run_suffix="$(date -u +%Y%m%dt%H%M%S)-$$-${RANDOM}"
experiment_id="stage2-zero3-dora-fresh-example-$run_suffix"
STAGE1_CHECKPOINT=/path/to/stage1/checkpoint \
DATA_PATH=/path/to/TRAIN.jsonl \
IMAGE_ROOT=/path/to/data \
OUTPUT_DIR="$zero_checkpoint_root/$experiment_id" \
LOG_FILE="$zero_log_root/$experiment_id/train.log" \
TUNING_METHOD=dora \
LORA_R=32 \
LORA_ALPHA=64 \
LORA_DROPOUT=0.05 \
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
DEEPSPEED_CONFIG="$PWD/configs/deepspeed_zero3.json" \
./run_stage2_training.sh
```

Fresh full/LoRA/DoRA modes can load model and processor state from a complete
full Stage 1 checkpoint, then create new optimizer and scheduler state. A
Stage 1 DoRA artifact instead uses the explicitly separated original base and
`--stage1_adapter_path` flow documented above. Stage 1 DoRA to Stage 2 full or
LoRA, adapter stacking, and implicit merging are rejected.

To resume PEFT training, keep the compatible Stage 1 base explicit and name
the exact Stage 2 adapter checkpoint independently. Supplying all tuning
settings makes the requested topology explicit so compatibility validation can
reject a mismatch:

```bash
# Continue in the same shell as the fresh LoRA command; retain its roots and
# exact experiment_id rather than creating a new ID for this lifecycle.
: "${zero_checkpoint_root:?run the fresh LoRA setup first}"
: "${zero_log_root:?run the fresh LoRA setup first}"
: "${experiment_id:?reuse the fresh LoRA experiment ID}"
STAGE1_CHECKPOINT=/path/to/stage1/checkpoint \
RESUME_CHECKPOINT="$zero_checkpoint_root/$experiment_id/checkpoint-100" \
DATA_PATH=/path/to/TRAIN.jsonl \
IMAGE_ROOT=/path/to/data \
OUTPUT_DIR="$zero_checkpoint_root/$experiment_id" \
LOG_FILE="$zero_log_root/$experiment_id/resume.log" \
TUNING_METHOD=lora \
LORA_R=32 \
LORA_ALPHA=64 \
LORA_DROPOUT=0.05 \
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
DEEPSPEED_CONFIG="$PWD/configs/deepspeed_zero3.json" \
./run_stage2_training.sh
```

Resume with the same training backend and configuration used to create the
checkpoint. In particular, a run started with the ZeRO-3 config
above must keep `DEEPSPEED_CONFIG="$PWD/configs/deepspeed_zero3.json"`; removing or
changing the backend cannot restore its optimizer and scheduler state.

Alternatively, omit `TUNING_METHOD` on resume to inherit the method and all
four adapter settings from `stage2_peft_metadata.json`. In that mode the
launcher deliberately forwards none of the five tuning arguments, even if
valid `LORA_*` variables happen to be present. An explicit `TUNING_METHOD`
always forwards the complete setting set for compatibility validation.

Every PEFT checkpoint and final adapter artifact contains:

```text
adapter_config.json
adapter_model.safetensors
stage2_peft_metadata.json
```

The sidecar uses schema v4 and records a compact Qwen2.5-VL text/vision
architecture snapshot in addition to the adapter and latent-token contract.
Its vocabulary provenance keeps three distinct values: the tokenizer vocabulary
before latent-token insertion, the pretrained model's possibly padded embedding
vocabulary, and the final resized vocabulary after the three latent tokens are
added.

Strict complete v2 artifacts migrate according to their original base-row-only
semantics. Strict complete v3 artifacts migrate only when
`token_row_source=base`, where the legacy unpadded interpretation is
unambiguous. A v3 `sparse_adapter` artifact is rejected even if its original
model happened to be unpadded: v3 did not record the pretrained padded
vocabulary, and the loader does not invent it from a path or from
`resized_vocab_size - 3`. This repository has no state-preserving v3 converter,
so a sparse v3 artifact cannot be loaded or automatically migrated. Restart a
fresh Stage 2 run from its authenticated Stage 1 schema-v2 source checkpoint to
produce a new schema-v4 artifact. This does not preserve the v3 optimizer,
scheduler, global step, or trained Stage 2 weights. Preserving that state would
require a separate future audited offline converter, which is not implemented.
Do not manually add an inferred value. Schema v1 and partial or unknown
v2/v3/v4 payloads are rejected.

A numbered checkpoint also contains `trainer_state.json`, RNG state, and the
optimizer/scheduler state required by its training backend. The concrete
layout depends on both the backend and who owns the scheduler:

| Backend/configuration | Model and optimizer state | Scheduler state |
| --- | --- | --- |
| Ordinary non-DeepSpeed Trainer | root `optimizer.pt` | root `scheduler.pt` |
| Current `configs/deepspeed_zero3.json` | DeepSpeed `global_step*/` rank model and optimizer state | root `scheduler.pt` (HF-owned) |
| DeepSpeed with a configured DeepSpeed scheduler | DeepSpeed `global_step*/` rank model and optimizer state | backend-owned |

The current ZeRO-3 config has no DeepSpeed `scheduler` section, so only its
optimizer/model state is backend-owned; the vendored Trainer saves the
HF-owned scheduler at the numbered checkpoint root. Always resume the complete
numbered checkpoint with the same backend and configuration. Individual
backend-owned Trainer-state filenames are not the compatibility contract and
may vary with configuration.

It does not contain a merged copy of the Stage 1 base. Resume restores the
adapter plus Trainer progress from `RESUME_CHECKPOINT`; `STAGE1_CHECKPOINT`
remains the separate base source. Moving the base and adapter directories is
supported: compatibility is checked structurally through latent-token IDs,
model configuration, target modules, and parameter shapes rather than by
requiring the original absolute path or hashing the full model weights.

For local evaluation/resume directories, the three PEFT files above must be
present together and must not coexist with full-model weight files. Incomplete,
legacy binary-adapter, and mixed full-plus-PEFT layouts fail before any model
weights are loaded. This repository publishes and accepts PEFT weights as the
single unsharded `adapter_model.safetensors` file.

Evaluate a full Stage 2 artifact with `--model_dir` alone. Evaluate a PEFT
artifact by supplying the adapter as `--model_dir` and its compatible Stage 1
base separately:

```bash
zero_result_root=/root/workspace/ILVR/results
run_suffix="$(date -u +%Y%m%dt%H%M%S)-$$-${RANDOM}"
experiment_id="stage2-lora-evaluation-example-$run_suffix"
mkdir -p -- "$zero_result_root/$experiment_id"
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/python eval.py \
  --model_dir /path/to/stage2-lora/output/checkpoint-100 \
  --base_model_dir /path/to/stage1/checkpoint \
  --test_data_path /path/to/TEST.jsonl \
  --task_name comt \
  --output_json_path "$zero_result_root/$experiment_id/lora-evaluation.jsonl"
```

Adapter evaluation loads the base plus adapter without merging them. Adapter
merge/export, adapter stacking/fusion, QLoRA, and alternate PEFT methods are
not implemented.

Stage 2 currently requires a rank-local batch size of exactly `B=1`; arbitrary
local `B>1` latent scheduling is not implemented. Data-parallel ranks and
gradient accumulation are supported, so the effective global batch is
`world_size × 1 × GRADIENT_ACCUMULATION_STEPS`. Set
`SUPERVISE_LATENT_BOUNDARIES=false` to mask both latent boundary tokens from CE;
the latent slots themselves are always masked. Start-only or end-only boundary
supervision is not a supported state.

In all three modes the visual encoder remains frozen. In LoRA and DoRA modes,
the base decoder, token embeddings, and LM head are also frozen; only adapters
on the exact decoder projection allowlist above are trainable. Visual or
out-of-scope target names are rejected before model loading.

Stage 2 uses each sample's original input images only. `helper_image` entries
mark latent reasoning steps in the data, but helper-image pixels/tensors are
not collated or encoded. The recurrent latent update runs with
`use_cache=False`; no training or inference KV-cache optimization is
implemented. Use
`STAGE2_DRY_RUN=true` to print the shell-escaped environment and command without
loading a model or data. Set the reproduction-critical options through the
documented environment variables; passing a launcher-owned option such as
`--stage`, `--latent_size`, or `--per_device_train_batch_size` directly is
rejected. Unmanaged training options such as `--max_steps` remain available as
trailing command-line arguments.

#### CoMT length audit

The audit renders every complete Stage 2 sample through the Stage 1 processor
without destructive truncation, then filters only whole examples above the
configured maximum:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/python -m stage2.data \
  --data-path ../../data/TRAIN.jsonl \
  --image-root ../../data \
  --processor-path ../../checkpoints/comt_ckpt \
  --latent-size 8 \
  --max-sequence-length 8192
```

The available CoMT training data was audited once with that command: 3,450
total samples, 3,379 retained, and 71 filtered (2.057971%). Lengths had minimum
131, median 724, p90 1,557.1, p95 2,691.6, p99 15,769.73, and maximum 16,698
tokens. The latent-block distribution was 924/239/561/1,227/286/213 samples
for one through six blocks, respectively.

#### Verification status

- **Implementation and smoke:** one real Stage 1 checkpoint/CoMT sample
  completed one Stage 2 optimizer step with finite loss
  (`0.00010741489677457139`).
- **FlashAttention:** a tiny one-layer Stage 2 model passed eager versus
  FlashAttention 2 latent/logit parity once on an NVIDIA B200 using bfloat16,
  PyTorch 2.7.1+cu128, and FlashAttention 2.8.3.
- **DeepSpeed:** a reviewed tiny-model two-B200 ZeRO-3 run passed consolidated
  save/load and optimizer/scheduler resume from global step 1 to 2, with every
  observed rank-local batch equal to 1 for the pre-PEFT full-tuning path.
- **LoRA/DoRA CPU:** the production-based tiny harness passed adapter-only
  save, fresh-base trainable reload, and Trainer resume from global step 1 to 2
  for both methods, including optimizer/scheduler and frozen-boundary checks.
- **LoRA ZeRO-3:** the dedicated two-GPU tiny harness passed adapter-only
  checkpoint/sidecar/processor saving and resume from global step 1 to 2, with
  adapter updates, restored optimizer/scheduler state, frozen visual/base
  parameters, and every observed rank-local batch equal to 1.
- **DoRA ZeRO-3:** the same dedicated two-GPU ZeRO-3 save/resume contract
  passed for DoRA, including adapter updates, restored optimizer/scheduler
  state, frozen visual/base parameters, and rank-local batch size 1.
- **Paper-level reproduction:** full CoMT Stage 2 training, benchmark
  evaluation, and reproduction of the paper's reported metrics were **not**
  run. The evidence above establishes implementation, smoke, backend, and data
  readiness only.

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
