# CoMT Dataset Construction Design

## Goal

Build the complete public CoMT dataset described by the ILVR README and make
it directly usable by both training and evaluation. The build must be
repeatable, resumable, safe to rerun, and verifiable before the data is used.

## Scope

The implementation will:

- download `TRAIN.jsonl`, `TEST.jsonl`, and `comt.tar.gz` from the official
  `shuai22/comt` Hugging Face dataset repository;
- construct the README-compatible layout under `data/`;
- validate every JSONL record and every referenced image;
- record source and validation metadata;
- remove developer-machine image path assumptions from training and
  evaluation; and
- configure the provided training and evaluation launchers to use the local
  dataset by default.

The implementation will not transform examples, generate new rationales,
change train/test membership, or use an unofficial dataset mirror
automatically.

## Final Layout

```text
data/
├── TRAIN.jsonl
├── TEST.jsonl
├── images_comt/
│   ├── creation/
│   └── ...
└── comt_manifest.json
```

Transient downloads and extraction output will live under `data/.staging/`.
The archive, staging content, JSONL files, images, and generated manifest are
local artifacts and will be excluded from Git.

## Components

### Dataset preparation command

A repository script will provide one command that:

1. queries the official dataset repository and resolves its current immutable
   commit revision;
2. downloads the three named source files with Hugging Face Hub resume support
   into `data/.staging/`;
3. computes SHA-256 hashes for the downloaded files;
4. inspects every tar member before extraction and rejects absolute paths,
   parent-directory traversal, device entries, and links that escape the
   extraction root;
5. extracts into a separate staging directory and normalizes only the expected
   image directory to `data/images_comt/`;
6. runs the complete validation described below; and
7. promotes staged output to the final layout only after validation succeeds.

Rerunning the command will reuse resumable downloads. A complete, validated
dataset will be detected and left unchanged unless the user explicitly asks
for a rebuild.

### Dataset validator

The validator will be callable independently and will fail with a nonzero exit
status when any invariant is violated. It will:

- parse every non-empty line in both JSONL files;
- require `text_input` and `image_input` on all examples;
- require training examples to contain a list-valued `sequence_plan`;
- accept only `text` and `latent` sequence step types;
- require `content` for text steps and `helper_image` for latent steps;
- resolve relative image references under `data/images_comt/` while allowing
  already-absolute references;
- require every `image_input` and training `helper_image` reference to exist;
- open and verify every unique referenced image with Pillow; and
- reject empty training or evaluation splits.

After a successful validation, `data/comt_manifest.json` will contain the
source repository, resolved revision, source-file hashes and sizes, train/test
record counts, unique referenced-image count, missing/corrupt-image counts
(both zero), and validation time in UTC.

### Runtime path configuration

The current fixed training path
`/mnt/public/users/dongshuai/LVR/ILVR_PUB/COMT` and empty evaluation roots will
be removed. Training and evaluation will receive a common `--image-root`
argument, defaulting to `data/images_comt` when launched from the repository.
Path resolution will happen at runtime rather than import time.

`run_training.sh` will default to `data/TRAIN.jsonl` and the local image root.
`eval.sh` will default to `data/TEST.jsonl` and the same image root. Existing
command-line overrides will remain possible so the code can still use data
stored elsewhere.

## Failure Handling

- Network failures leave final data untouched and retain resumable staging
  downloads.
- Hugging Face connectivity uses the official endpoint by default. An existing
  user-provided `HF_ENDPOINT` may be honored, but the setup will not silently
  select a third-party mirror.
- Invalid archives are rejected before extraction.
- Invalid JSONL, missing images, corrupt images, or unexpected extracted
  layouts prevent promotion to `data/` and produce a concise error summary.
- Existing complete data is never overwritten implicitly.

## Testing and Acceptance

Automated tests will use small synthetic archives and JSONL fixtures to cover:

- valid train/test records;
- malformed JSON;
- missing required fields;
- invalid sequence step types;
- missing and corrupt images;
- tar path traversal and unsafe links; and
- rerunning against an already valid dataset.

The full dataset build is accepted only when:

1. all three official source files have been downloaded;
2. the final README-compatible directory layout exists;
3. the validator completes successfully across both full splits;
4. the manifest reports zero missing and zero corrupt referenced images;
5. one training preprocessing sample resolves all of its images; and
6. one evaluation preprocessing sample resolves all of its images using the
   new `--image-root` path.

The full model training and benchmark run are separate follow-up work after
this dataset milestone.
