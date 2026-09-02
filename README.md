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

For canonical ZeRO-1/2/3 configuration selection, artifact ownership, and
same-identity resume commands, see [the ZeRO stage training guide](docs/zero-stage-training.md).

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

### Stage 1 full and DoRA training

The Qwen2.5-VL-3B full/DoRA comparison is a sealed four-stage workflow. Run
the commands below in order; do not launch a three-epoch run with a batch size
copied by hand. On this host the complete CoMT inputs are:

```text
/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data/TRAIN.jsonl
/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data
```

First create or reuse the Task 15 two-rank, `B=2`, step-3-to-step-4
save/resume witnesses. The checks read only the small evidence envelopes:

```bash
./venv/ilvr-venv/bin/python -m src.stage1.benchmark check-evidence \
  --evidence-dir outputs/stage1-smoke/full \
  --method full --micro-batch 2 --world-size 2
./venv/ilvr-venv/bin/python -m src.stage1.benchmark check-evidence \
  --evidence-dir outputs/stage1-smoke/dora \
  --method dora --micro-batch 2 --world-size 2
```

Run only a case whose check printed `missing`; reuse `reusable-success`:

Each invocation allocates a new append-only
`outputs/stage1-smoke/<method>/attempts/attempt-N/` directory. A failed attempt
is retained in place and the next invocation uses the next monotonic number;
do not delete, move, or overwrite its logs. Legacy flat failure files under the
method directory are also retained and reserve attempt 1. A persistent
mode-`0600` `.execution.lock` uses a kernel `flock` lease across allocation,
both launches, checkpoint sealing, and success publication. A concurrent
invocation fails with an explicit in-progress/retry message and cannot allocate
or launch; process exit releases the lease without deleting the lock inode.
The lease also pins the method parent/name/inode. Failures are immutable
append-only registry history: never archive, move, delete, or overwrite them.
On success, the wrapper publishes exactly one small canonical
`outputs/stage1-smoke/<method>/evidence.json` with exclusive-create semantics.
That schema-v3 evidence embeds both the authenticated producer envelope and the
sealed checkpoint marker while retaining their contained relative provenance
paths. It also binds the lexical absolute requested data, image, DeepSpeed, and
launcher paths to the sealed DeepSpeed/run-config digests. Reuse requires exact
request-path and experiment-setting equality without resolving or statting
those paths.
Reuse reads only this canonical regular, non-aliased evidence inode; it does not
require the source data, images, attempt directory, checkpoint tree, or
attempt-local evidence file. The legacy `archive-failed-attempt` command is
deprecated and always exits nonzero without inspecting or mutating its target;
the next retry allocates a new `attempt-N` in place.

```bash
CUDA_VISIBLE_DEVICES=0,1 ILVR_REAL_GPU_TESTS=1 ILVR_DISTRIBUTED_METHOD=full \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_distributed_smoke.py::test_two_rank_zero3_b2_save_resume_witness \
  -k full -q -s

CUDA_VISIBLE_DEVICES=0,1 ILVR_REAL_GPU_TESTS=1 ILVR_DISTRIBUTED_METHOD=dora \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_distributed_smoke.py::test_two_rank_zero3_b2_save_resume_witness \
  -k dora -q -s

./venv/ilvr-venv/bin/python -m src.stage1.benchmark summarize \
  outputs/stage1-smoke/full/evidence.json \
  outputs/stage1-smoke/dora/evidence.json
```

After both smoke witnesses succeed, seal their common prerequisite chain and
then create or reuse the Task 16 high-cost-row audit. `require-evidence` writes
`outputs/stage1-smoke/prerequisite-chain.json`:

```bash
./venv/ilvr-venv/bin/python -m src.stage1.benchmark require-evidence \
  outputs/stage1-smoke/full/evidence.json \
  outputs/stage1-smoke/dora/evidence.json

./venv/ilvr-venv/bin/python -m src.stage1.benchmark audit \
  --data-path /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data/TRAIN.jsonl \
  --image-root /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data \
  --output outputs/stage1-probes/high-cost-rows.json --count 64

CUDA_VISIBLE_DEVICES=0,1 ./run_stage1_batch_probe.sh \
  --method full --model Qwen/Qwen2.5-VL-3B-Instruct \
  --data-path /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data/TRAIN.jsonl \
  --image-root /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data \
  --output-root outputs/stage1-probes/full \
  --probe-rows outputs/stage1-probes/high-cost-rows.json \
  --prerequisite-chain outputs/stage1-smoke/prerequisite-chain.json \
  --start 8 --gradient-accumulation 2 --learning-rate 1e-5 --seed 42

CUDA_VISIBLE_DEVICES=0,1 ./run_stage1_batch_probe.sh \
  --method dora --model Qwen/Qwen2.5-VL-3B-Instruct \
  --data-path /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data/TRAIN.jsonl \
  --image-root /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data \
  --output-root outputs/stage1-probes/dora \
  --probe-rows outputs/stage1-probes/high-cost-rows.json \
  --prerequisite-chain outputs/stage1-smoke/prerequisite-chain.json \
  --start 8 --gradient-accumulation 2 --learning-rate 1e-5 --seed 42 \
  --dora-rank 32 --dora-alpha 64 --dora-dropout 0.05
```

Each primary probe stops at its first failure and prints the exact
`attempt=...` directory. Diagnose that attempt before recording a failure
class (set each variable to the literal path printed by its launcher).
Record `capacity_oom` only for a confirmed capacity failure:

```bash
./venv/ilvr-venv/bin/python -m src.stage1.benchmark classify-failure \
  --attempt "$FAILED_FULL_ATTEMPT" --classification capacity_oom
./venv/ilvr-venv/bin/python -m src.stage1.benchmark classify-failure \
  --attempt "$FAILED_DORA_ATTEMPT" --classification capacity_oom
```

For a non-capacity launcher or integration failure, retain the failed attempt
and classify it explicitly instead:

```bash
./venv/ilvr-venv/bin/python -m src.stage1.benchmark classify-failure \
  --attempt "$FAILED_ATTEMPT" --classification integration_failure
```

Then rerun the same primary probe command. The controller skips its already
successful prefix, appends a new attempt for the same failed batch, and only
then continues the ordered search. An `integration_failure` never authorizes a
lower fallback and is never reported as the first capacity failure.

If the initial batch 8 is classified `capacity_oom`, that classification must
precede an explicit fallback. Run 4, then—only after another recorded
`capacity_oom`—2, then 1.
The full fallback command is:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./run_stage1_batch_probe.sh \
  --method full --model Qwen/Qwen2.5-VL-3B-Instruct \
  --data-path /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data/TRAIN.jsonl \
  --image-root /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data \
  --output-root outputs/stage1-probes/full \
  --probe-rows outputs/stage1-probes/high-cost-rows.json \
  --prerequisite-chain outputs/stage1-smoke/prerequisite-chain.json \
  --start 4 --descending \
  --gradient-accumulation 2 --learning-rate 1e-5 --seed 42
```

The DoRA fallback uses its independent output and fixed topology:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./run_stage1_batch_probe.sh \
  --method dora --model Qwen/Qwen2.5-VL-3B-Instruct \
  --data-path /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data/TRAIN.jsonl \
  --image-root /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data \
  --output-root outputs/stage1-probes/dora \
  --probe-rows outputs/stage1-probes/high-cost-rows.json \
  --prerequisite-chain outputs/stage1-smoke/prerequisite-chain.json \
  --start 4 --descending \
  --gradient-accumulation 2 --learning-rate 1e-5 --seed 42 \
  --dora-rank 32 --dora-alpha 64 --dora-dropout 0.05
```

For later fallbacks change `--start 4` to `--start 2`, then `--start 1`.
The launcher verifies the already sealed attempt chain and never lowers a
batch automatically. For every run:

```text
effective global batch = world size × per-GPU micro-batch × accumulation
```

Seal the preceding successful candidate with a fresh ten-step smoke, then
compare the two small maximum manifests:

```bash
./venv/ilvr-venv/bin/python -m src.stage1.benchmark validate-maximum \
  --probe-dir outputs/stage1-probes/full --long-smoke-steps 10
./venv/ilvr-venv/bin/python -m src.stage1.benchmark validate-maximum \
  --probe-dir outputs/stage1-probes/dora --long-smoke-steps 10
./venv/ilvr-venv/bin/python -m src.stage1.benchmark compare-maxima \
  outputs/stage1-probes/full/maximum.json \
  outputs/stage1-probes/dora/maximum.json
```

Task 17 consumes those sealed maxima. `prepare-timed-run` resolves the exact
batch and writes an immutable manifest; the timed launcher accepts only that
manifest, so no batch placeholder is resolved by hand:

```bash
./venv/ilvr-venv/bin/python -m src.stage1.benchmark prepare-timed-run \
  --method full --maximum outputs/stage1-probes/full/maximum.json \
  --output-dir outputs/stage1-timed/full-3epoch \
  --epochs 3 --learning-rate 1e-5 --gradient-accumulation 2 --world-size 2
./venv/ilvr-venv/bin/python -m src.stage1.benchmark prepare-timed-run \
  --method dora --maximum outputs/stage1-probes/dora/maximum.json \
  --output-dir outputs/stage1-timed/dora-3epoch \
  --epochs 3 --learning-rate 1e-5 --gradient-accumulation 2 --world-size 2

CUDA_VISIBLE_DEVICES=0,1 ./run_stage1_training.sh \
  --manifest outputs/stage1-timed/full-3epoch/launch-manifest.json
./venv/ilvr-venv/bin/python -m src.stage1.benchmark validate-timed-run \
  --run-dir outputs/stage1-timed/full-3epoch

CUDA_VISIBLE_DEVICES=0,1 ./run_stage1_training.sh \
  --manifest outputs/stage1-timed/dora-3epoch/launch-manifest.json
./venv/ilvr-venv/bin/python -m src.stage1.benchmark validate-timed-run \
  --run-dir outputs/stage1-timed/dora-3epoch

./venv/ilvr-venv/bin/python -m src.stage1.benchmark compare-runs \
  outputs/stage1-timed/full-3epoch/evidence.json \
  outputs/stage1-timed/dora-3epoch/evidence.json \
  --output outputs/stage1-timed/comparison.json
```

If `prepare-timed-run` or the evidence check reports a reusable success, skip
that training command. Each successful timed run is launched and validated
once.

A full final artifact contains model/config weights plus tokenizer and
processor files. A DoRA final artifact contains
`adapter_config.json`, `adapter_model.safetensors`,
`ilvr_token_rows.safetensors`, `stage1_peft_metadata.json`, and tokenizer/
processor files, but no merged full-model weights. Numbered checkpoints add
Trainer optimizer, scheduler, RNG, global-step state, and the versioned
`stage1_ema_teacher.pt` sidecar. The EMA sidecar contains only the exact
EMA-selected parameter snapshots plus method, decay, name, and shape metadata;
it is required when resuming a numbered Stage 1 checkpoint.

Resume an interrupted, separately checkpointed Stage 1 run by repeating its
configuration and naming the exact numbered checkpoint. This `B=2` example
uses a namespace distinct from smoke, probe, and timed evidence:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./run_stage1_training.sh \
  --method dora --model Qwen/Qwen2.5-VL-3B-Instruct \
  --data-path /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data/TRAIN.jsonl \
  --image-root /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data \
  --output-dir outputs/stage1-resume-example/dora \
  --resume outputs/stage1-resume-example/dora/checkpoint-100 \
  --micro-batch 2 --epochs 3 --learning-rate 1e-5 \
  --gradient-accumulation 2 --seed 42 \
  --deepspeed-config configs/config_stage3.json \
  --attention-implementation flash_attention_2 --save-strategy steps --final-save
```

That restores Stage 1 optimizer/scheduler/global-step state. A fresh Stage 2
DoRA continuation is different: `--load_model_path` identifies the original
full Qwen base while `--stage1_adapter_path` identifies the Stage 1 DoRA
artifact. It copies the learned adapter and sparse latent-token rows but starts
a new Stage 2 optimizer, scheduler, and global step:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/python src/main.py \
  --stage stage2 \
  --load_model_path /path/to/original-qwen-base \
  --stage1_adapter_path /path/to/stage1-dora-artifact \
  --tuning_method dora \
  --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --data_path /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data/TRAIN.jsonl \
  --image_root /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data \
  --save_model_path /path/to/stage2-dora-output \
  --per_device_train_batch_size 1 --learning_rate 1e-5
```

Adapter merging, stacking/fusion, and QLoRA are not supported. Stage 2 remains
rank-local `B=1`, uses the correctness-first recurrence without a
differentiable KV-cache optimization, and has not been run as a paper-level
reproduction in this workflow.

Small authenticated evidence is written at:

```text
outputs/stage1-smoke/{full,dora}/evidence.json
outputs/stage1-probes/{full,dora}/maximum.json
outputs/stage1-timed/{full-3epoch,dora-3epoch}/evidence.json
outputs/stage1-timed/comparison.json
outputs/stage1-timed/final-report.json
```

The `stage1-smoke`, `stage1-probes`, and `stage1-timed` trees have distinct
owners and must never be reused as one another's output directory.

Task 18 validates and combines those six sealed inputs without rerunning a GPU
job or traversing checkpoint trees:

```bash
./venv/ilvr-venv/bin/python -m src.stage1.benchmark final-report \
  --smoke-full outputs/stage1-smoke/full/evidence.json \
  --smoke-dora outputs/stage1-smoke/dora/evidence.json \
  --maximum-full outputs/stage1-probes/full/maximum.json \
  --maximum-dora outputs/stage1-probes/dora/maximum.json \
  --timed-full outputs/stage1-timed/full-3epoch/evidence.json \
  --timed-dora outputs/stage1-timed/dora-3epoch/evidence.json \
  --output outputs/stage1-timed/final-report.json
```

Before any GPU smoke, probe, timed run, or report generation, load these small
manifests and reuse a matching successful witness. Do not rerun an expensive
successful job or reread/hash unchanged model/checkpoint trees merely to
recreate evidence.

### Stage 2 training

Stage 2 starts from a Stage 1 checkpoint and replaces CoMT helper-image steps
with differentiable continuous-latent blocks. The original/user image remains
a model input, but helper images are never loaded as Stage 2 visual inputs. The
entire visual tower is frozen and is encoded once per rank-local batch.

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
STAGE1_CHECKPOINT=/path/to/stage1/checkpoint \
DATA_PATH=/path/to/TRAIN.jsonl \
IMAGE_ROOT=/path/to/data \
OUTPUT_DIR=/path/to/stage2/output \
LOG_FILE=/path/to/stage2.log \
TUNING_METHOD=full \
LATENT_SIZE=8 \
SUPERVISE_LATENT_BOUNDARIES=true \
LEARNING_RATE=1e-5 \
GRADIENT_ACCUMULATION_STEPS=8 \
ATTENTION_IMPLEMENTATION=flash_attention_2 \
GRADIENT_CHECKPOINTING=true \
MAX_SEQUENCE_LENGTH=8192 \
DEEPSPEED_CONFIG=configs/config_stage3.json \
./run_stage2_training.sh
```

Fresh LoRA training uses the same Stage 1 base but trains adapters only:

```bash
STAGE1_CHECKPOINT=/path/to/stage1/checkpoint \
DATA_PATH=/path/to/TRAIN.jsonl \
IMAGE_ROOT=/path/to/data \
OUTPUT_DIR=/path/to/stage2-lora/output \
LOG_FILE=/path/to/stage2-lora.log \
TUNING_METHOD=lora \
LORA_R=32 \
LORA_ALPHA=64 \
LORA_DROPOUT=0.05 \
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
DEEPSPEED_CONFIG=configs/config_stage3.json \
./run_stage2_training.sh
```

DoRA has the same target and hyperparameter contract, with
`TUNING_METHOD=dora`:

```bash
STAGE1_CHECKPOINT=/path/to/stage1/checkpoint \
DATA_PATH=/path/to/TRAIN.jsonl \
IMAGE_ROOT=/path/to/data \
OUTPUT_DIR=/path/to/stage2-dora/output \
LOG_FILE=/path/to/stage2-dora.log \
TUNING_METHOD=dora \
LORA_R=32 \
LORA_ALPHA=64 \
LORA_DROPOUT=0.05 \
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
DEEPSPEED_CONFIG=configs/config_stage3.json \
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
STAGE1_CHECKPOINT=/path/to/stage1/checkpoint \
RESUME_CHECKPOINT=/path/to/stage2-lora/output/checkpoint-100 \
DATA_PATH=/path/to/TRAIN.jsonl \
IMAGE_ROOT=/path/to/data \
OUTPUT_DIR=/path/to/stage2-lora/output \
LOG_FILE=/path/to/stage2-lora.log \
TUNING_METHOD=lora \
LORA_R=32 \
LORA_ALPHA=64 \
LORA_DROPOUT=0.05 \
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
DEEPSPEED_CONFIG=configs/config_stage3.json \
./run_stage2_training.sh
```

Resume with the same training backend and configuration used to create the
checkpoint. In particular, a run started with the ZeRO-3 config
above must keep `DEEPSPEED_CONFIG=configs/config_stage3.json`; removing or
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
| Current `configs/config_stage3.json` | DeepSpeed `global_step*/` rank model and optimizer state | root `scheduler.pt` (HF-owned) |
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
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/python eval.py \
  --model_dir /path/to/stage2-lora/output/checkpoint-100 \
  --base_model_dir /path/to/stage1/checkpoint \
  --test_data_path /path/to/TEST.jsonl \
  --task_name comt \
  --output_json_path /path/to/lora-evaluation.jsonl
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

#### Stage 2 no-cache efficiency gate

The focused probe measures the existing no-cache recurrence and mandatory final
full forward; it does not implement or prototype KV caching. Validate the CUDA
measurement path with an immutable tiny smoke result and matching log:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=transformers/src:src \
  ./venv/ilvr-venv/bin/python -m stage2.efficiency_probe \
  --self-test --experiment-id stage2-efficiency-self-test
```

A representative run requires the already sealed Stage 2 length manifest and
its SHA-256, plus the exact model/tuning configuration. The probe selects the
retained examples closest to the joint p50 and p95 length/block-count targets,
benchmarks DataLoader worker counts 0, 2, and 4, then retains three warm-up and
ten measured optimizer steps per representative:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=transformers/src:src \
  ./venv/ilvr-venv/bin/python -m stage2.efficiency_probe \
  --experiment-id <unique-run-id> \
  --model-source <stage1-base-model-or-full-checkpoint> \
  --stage1-adapter <stage1-dora-checkpoint> \
  --data-path /root/workspace/ILVR/data/TRAIN.jsonl \
  --image-root /root/workspace/ILVR/data \
  --stage2-length-manifest results/zero-stage-compatibility/<manifest-run>/stage2-length-manifest.json \
  --stage2-length-manifest-sha256 <sha256> \
  --latent-size 8 --tuning-method dora \
  --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 \
  --attention-implementation flash_attention_2 \
  --gradient-checkpointing
```

Results are published exclusively to
`results/zero-stage-compatibility/<unique-run-id>/stage2-efficiency.json` and
logs to the matching directory under `logs/`. The strict schema-v1 field
`segmented_kv_authorized` is true only when the median recurrence-prefix decoder
time divided by the median total optimizer-step time is at least `0.30`.
Only a representative B200 run authorizes segmented-KV follow-up work; a tiny
self-test or a run on different hardware validates instrumentation but does not
satisfy that premise.

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
