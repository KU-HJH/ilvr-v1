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

### Download All Required Assets

Run the unified downloader from the repository root. It uses the mandated
virtual environment, resumes interrupted Hugging Face transfers, and validates
each artifact before reporting it ready.

```bash
./venv/ilvr-venv/bin/python -m scripts.download_assets
```

| Artifact | Local path | Approximate size |
| --- | --- | ---: |
| Qwen2.5-VL 3B experiment base | `models/Qwen2.5-VL-3B-Instruct/` | 7.52 GB |
| Complete prepared CoMT dataset | `data/` | 11.5 GB |
| Published ILVR comparison checkpoint | `models/comt_ckpt/` | 31.8 GB |

Download or retry one asset without touching completed assets:

```bash
./venv/ilvr-venv/bin/python -m scripts.download_assets --only model
./venv/ilvr-venv/bin/python -m scripts.download_assets --only dataset
./venv/ilvr-venv/bin/python -m scripts.download_assets --only checkpoint
```

Use `--only all` to request the same complete asset set as the default command.

Prepare or validate only the CoMT dataset with the independent commands:

```bash
./venv/ilvr-venv/bin/python -m scripts.prepare_comt prepare --data-dir data
./venv/ilvr-venv/bin/python -m scripts.prepare_comt validate --data-dir data
```

Allow at least 51 GB for final payloads plus additional temporary/resume
metadata. `models/comt_ckpt` is the project's published 7B-derived comparison
checkpoint. It is not an adapter and must not be combined with the 3B base
model.

The downloader removes `HF_HUB_OFFLINE` only inside its own process. It honors
caller-provided `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`, `HF_TOKEN`, and
`HF_ENDPOINT` values; without `HF_ENDPOINT`, it uses the official Hugging Face
service. It never selects a third-party mirror automatically. Partial Hub
downloads and `data/.staging/` are retained so the same command can resume.

For its `dataset` asset, the unified downloader prepares `TRAIN.jsonl`,
`TEST.jsonl`, and `comt.tar.gz`, validates every referenced image, and promotes
the finished files only after validation succeeds. It records the resolved
official revision, checksums, sizes, and validation counts in
`data/comt_manifest.json`. Dataset preparation only constructs and validates
local data; it does not start model training.

**Expected Directory Structure:**
```text
ILVR/
├── data/
│   ├── TRAIN.jsonl
│   ├── TEST.jsonl
│   ├── images_comt/      <-- Extracted from comt.tar.gz
│       ├── creation/
│       └── ...
│   └── comt_manifest.json <-- Source revision, file hashes, and validation metrics
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

### Run the 3B Experiment

Training defaults to the downloaded `models/Qwen2.5-VL-3B-Instruct` base,
`data/TRAIN.jsonl`, and `data/images_comt`:

```bash
bash run_training.sh
```

Override the base model without editing the launcher:

```bash
MODEL_NAME=/path/to/another-base-model bash run_training.sh
```

Override the training data and image paths together when using another prepared
dataset:

```bash
DATA_PATH=/path/to/TRAIN.jsonl IMAGE_ROOT=/path/to/images_comt bash run_training.sh
```

### Evaluate

Evaluation defaults to the published `models/comt_ckpt` comparison checkpoint
and `data/TEST.jsonl`:

```bash
bash eval.sh
```

Evaluate a newly trained 3B checkpoint with:

```bash
MODEL_DIR=/path/to/trained-3b-checkpoint bash eval.sh
```

Override the evaluation checkpoint, test split, and image root together:

```bash
MODEL_DIR=/path/to/checkpoint TEST_DATA_PATH=/path/to/TEST.jsonl IMAGE_ROOT=/path/to/images_comt bash eval.sh
```

**Default Hyperparameters:**
- Epochs: 15
- Gradient Accumulation Steps: 8
- Latent Size: 8

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
