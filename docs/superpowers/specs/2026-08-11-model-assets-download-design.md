# ILVR Model and Dataset Asset Downloader Design

## Goal

Provide one resumable command that downloads every public artifact needed for
the planned Qwen2.5-VL 3B experiment and for comparison with the published ILVR
result. The command must leave each artifact in an explicit repository-local
location and verify that the expected files are present before reporting
success.

## Scope

The downloader will acquire these three assets from their official Hugging
Face repositories:

- the `Qwen/Qwen2.5-VL-3B-Instruct` base model, approximately 7.52 GB;
- the complete `shuai22/comt` dataset used by this repository, approximately
  11.5 GB after download; and
- the published `shuai22/comt_ckpt` evaluation checkpoint, approximately
  31.8 GB.

The published checkpoint was trained from the repository's original 7B base
configuration. It is downloaded as a separate comparison artifact and will
not be combined with the 3B base model.

The downloader will not start training or evaluation, modify model weights,
choose an unofficial mirror automatically, or delete an existing completed
asset.

## Interface and Layout

The primary command will run through the repository virtual environment:

```bash
./venv/ilvr-venv/bin/python -m scripts.download_assets
```

It will support selecting a subset without changing the default:

```bash
./venv/ilvr-venv/bin/python -m scripts.download_assets --only model
./venv/ilvr-venv/bin/python -m scripts.download_assets --only dataset
./venv/ilvr-venv/bin/python -m scripts.download_assets --only checkpoint
./venv/ilvr-venv/bin/python -m scripts.download_assets --only all
```

The final repository-local layout will be:

```text
models/
├── Qwen2.5-VL-3B-Instruct/
└── comt_ckpt/
data/
├── TRAIN.jsonl
├── TEST.jsonl
├── images_comt/
└── comt_manifest.json
```

Downloaded model directories, Hub metadata, dataset files, and staging files
will remain local artifacts excluded from Git.

## Components and Data Flow

### Unified command

`scripts/download_assets.py` will own argument parsing and orchestration. By
default it will process the 3B model, CoMT dataset, and published checkpoint
sequentially. Sequential processing keeps progress and failure messages clear
and avoids competing large transfers.

For model repositories, the command will use `huggingface_hub.snapshot_download`
with an explicit `local_dir`. Hugging Face's cache metadata will allow an
interrupted transfer to continue when the command is rerun. The command will
not download repository history or create Git repositories.

For the dataset, the command will call the existing CoMT preparation API. That
API downloads only `TRAIN.jsonl`, `TEST.jsonl`, and `comt.tar.gz`, safely
extracts the archive in `data/.staging/`, validates every referenced image,
and promotes the finished layout only after validation succeeds.

### Connectivity and environment

The command will remove `HF_HUB_OFFLINE` from its own process before any Hub
operation because its purpose is explicitly online download. It will preserve
the caller's standard `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`, `HF_TOKEN`, and
`HF_ENDPOINT` settings. Without a caller-supplied endpoint it will use the
official Hugging Face service; it will never select a mirror by itself.

### Verification

After each model snapshot completes, the command will check repository-specific
invariants:

- the 3B base directory contains `config.json`, processor/tokenizer metadata,
  `model.safetensors.index.json`, and the two indexed safetensor shards;
- the published checkpoint contains `config.json`, processor/tokenizer
  metadata, `pytorch_model.bin.index.json`, and the seven indexed PyTorch
  shards; and
- each index references files that exist as regular, non-empty files below the
  selected artifact directory.

Index parsing, path containment, and file checks will be implemented as a
small reusable verifier. This prevents a partial directory from being reported
as complete while avoiding a second full hash pass over roughly 40 GB of model
weights; the Hub client already verifies downloaded objects against repository
metadata during transfer.

Dataset success will use the existing complete CoMT validator and manifest
rather than duplicating dataset rules in the new command.

## Reruns and Failure Handling

- An interrupted Hub transfer will leave resumable metadata in place; rerunning
  the same command will continue or reuse completed files.
- A completed and verified model directory will be checked and reused instead
  of downloaded again.
- A completed and validated dataset will be reused by the existing preparation
  command.
- Network, authentication, disk, or verification failures will identify the
  failed asset and print the exact command needed to retry only that asset.
- The command will return a nonzero status as soon as an asset fails. Assets
  completed earlier in the sequence will remain valid and reusable.
- Existing conflicting or incomplete final data will not be overwritten
  implicitly. Destructive rebuild behavior remains available only through the
  existing dataset preparation command's explicit force option.

## Training and Evaluation Integration

The training launcher will use the local
`models/Qwen2.5-VL-3B-Instruct` directory as its default model and retain an
environment-variable override for another model path. Its offline setting is
appropriate after the download has completed.

The published `models/comt_ckpt` directory will be the documented default for
reproducing the released evaluation result. A newly trained 3B checkpoint can
still be supplied through the existing evaluation override.

The README will distinguish the 3B experiment base model from the original
published 7B-derived checkpoint and document the unified command, subset
commands, final paths, proxy/token environment variables, expected storage,
and retry behavior.

## Testing and Acceptance

Tests will avoid real multi-gigabyte downloads. Temporary directories and an
injected download boundary will cover:

- the default order and repository identifiers for all three assets;
- `--only` selection for each asset;
- removal of `HF_HUB_OFFLINE` while preserving user networking variables;
- acceptance of complete safetensor and PyTorch shard indexes;
- rejection of missing, empty, absolute, parent-traversing, and unindexed
  weight files;
- reuse of an already verified snapshot; and
- a failed asset's nonzero exit status and single-asset retry message.

The implementation is accepted when all automated tests pass in
`./venv/ilvr-venv`, Bash launchers pass syntax checks, the README and launcher
defaults agree with the final paths, and a dry run can exercise selection and
validation without accessing the network. A live full download is a separate
runtime operation whose success still depends on this host being able to reach
the configured Hugging Face endpoint.
