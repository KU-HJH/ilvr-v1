# Evaluation protocols

The offline scorer records the protocol identity in every summary. The P1
fixture configuration uses `protocol_kind: modified` and is a synthetic smoke
fixture, not a paper result or a replacement for the real VSP dataset.

Use an explicit benchmark/profile pair. The current offline implementation
supports `vsp` with `vsp_validated_v1` and `comt` with
`comt_reference_rules_v1`, and `visulogic` with `visulogic_vlmeval_rules_v1`;
unknown values fail preflight rather than falling
back to another scorer. A saved response run must use the recorded model
identity, prompt template identity, generation configuration, manifest hash,
and run fingerprint from that run.

The validated parser accepts only `U/D/L/R` (case-insensitive), ASCII space,
tab, CR, and LF, and commas between nonempty action groups. It rejects quotes,
periods, unknown characters, and leading/trailing/duplicate commas without
silently deleting characters. It prefers the last balanced `boxed` answer,
then the last final-answer/action-plan marker, then the full response; an
incomplete final boxed expression is invalid and does not fall back. In the
legacy comparison, the original parser was observed to accept `XR` and `RX`
by deleting `X`, while the validated parser returns `invalid_format` for both.
The legacy behavior is audit-only and is not registered as a runtime scorer.
The legacy and validated outcomes are separate and must be reported separately:

| Input | Legacy audit outcome | `vsp_validated_v1` outcome |
| --- | --- | --- |
| `R` on `[[1,2],[0,0]]` | success | success at step 1 |
| `RD` on the same map | failure after leaving the goal | success at step 1 when the episode ends on goal |
| `XR` on the same map | success (`X` deleted) | `invalid_format` |
| `RX` on the same map | success (`X` deleted) | `invalid_format` |

VSP simulation is deterministic: boundary moves consume a step and stay put,
holes terminate with failure, goals terminate with reward 1, and a step limit
sets truncation after the transition. Goal and limit on the same transition
retain both `terminated` and `truncated`. No shortest-path requirement is
added. Scores use manifest size `N_expected` as the denominator; missing or
infrastructure-error predictions make the run incomplete and accuracy null.
Coverage counts usable model responses divided by `N_expected`. Categories are
the manifest subtask values and must be listed in config; unknown categories
fail preflight, while an empty configured category has count zero and null
accuracy. Model identity, prompt template identity, generation configuration,
manifest SHA-256, and common run fingerprint are run-level metadata; optional
per-sample prompt hashes do not change that fingerprint.

The command is run from the repository root (or receives an explicit root in
Python tests):

```text
PYTHONPATH=src ./venv/ilvr-venv/bin/python -m scripts.score_eval \
  --manifest data/evaluation/<dataset>/manifest.jsonl \
  --predictions results/evaluation/<generation-run>/predictions.jsonl \
  --config configs/eval/vsp_validated_v1.yaml \
  --output-dir results/evaluation/<unique-scoring-run>
```

Manifests and assets belong below `data/evaluation/`. Predictions and score
reports belong below `results/evaluation/<run-id>/`, with execution logs below
`logs/evaluation/<run-id>/`. Output and log directories must be fresh direct
children named by the same run ID. The CLI never downloads data, loads a
model, invokes a judge, or imports the legacy inference runtime.

The synthetic smoke recipe is useful for development only. From the worktree
root, this copy-pastable command creates a fixture config and invokes the real
CLI with the fixture's actual manifest, prediction, config, and output paths:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. \
  ./venv/ilvr-venv/bin/python - <<'PY'
import os, subprocess, sys, uuid
from offline_fixtures import make_case
import yaml

case = make_case(2)
config = case["data_root"] / "config.yaml"
config.write_text(yaml.safe_dump(case["config"]))
run = case["root"] / "results/evaluation" / ("p1-quickstart-score-" + uuid.uuid4().hex)
env = dict(os.environ, PYTHONPATH="src:tests/evaluation:.", PYTHONDONTWRITEBYTECODE="1")
subprocess.run([
    sys.executable, "-m", "scripts.score_eval",
    "--manifest", str(case["manifest"]),
    "--predictions", str(case["prediction_path"]),
    "--config", str(config),
    "--output-dir", str(run),
], cwd=case["root"], env=env, check=True)
print(run / "summary.json")
PY
```

This fixture does not download a dataset and its score must not be reported as
a real benchmark result. Each invocation chooses a unique output name.

Exit status is `0` for complete scoring, `1` for an incomplete evaluation or
an unexpected scoring/output infrastructure failure, and `2` for argument,
config, manifest, prediction, benchmark/profile, or path preflight errors.

Pinned source provenance is recorded in
[references.lock.json](../../references.lock.json): Visual-Spatial-Planning
(`6d74d7bb46ae71bf41aa4d3d874a4f31fc5f0366`), including
`VSP-main/maze/task-main/eval.py` and its `gemini.yml`. The repository/file
inventory, source hashes, and observed legacy `R`, `RD`, `XR`, `RX` and
`"None"` probe are documented in [audit.md](audit.md) and
`results/evaluation-p0-20260910-01/local-vsp-probe.json`; those are source and
isolated-function evidence, not a Gym runtime execution. P1's new modules are
independent implementations; no upstream source code was copied or adapted.
If a later task imports upstream code, it must add the exact repository, commit,
file/symbol, and license provenance before doing so.

Comparison with the original Gym runtime, upstream environment, GPU smoke,
full benchmark execution, and paper-protocol verification are separate
activities. They are not established by the offline scorer.

## P2 prepared VSP generation recipe

P2 adds a Reference-free path from prepared VSP inputs through the existing
ILVR generation function and the P1 scorer. The checked-in recipe is
`configs/eval/vsp_ilvr_three_images_v1.yaml`. It fixes the dataset root at
`data/evaluation/vsp-pinned-6d74d7bb46ae-v1`, categories `level3` through
`level8`, `max_steps: 100`, model identity `stage1-dora-continuous-v29`, and
prompt identity
`vsp-ilvr-three-images-v1:8034d008ea5ca5e693c73b5bc8a079e2aa55a1270bbe7e5a4e78c706037af03d`.
The default generation settings are `max_new_tokens: 4096`,
`max_sequence_length: 32768`, `repetition_penalty: 1.05`, `temperature: 0.0`,
`top_p: 1.0`, and `do_sample: false`.

The source preparation command is:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. \
  ./venv/ilvr-venv/bin/python -m scripts.prepare_eval \
  --benchmark vsp --profile vsp_validated_v1 \
  --source-root Reference/Visual-Spatial-Planning \
  --output-dir data/evaluation/vsp-pinned-6d74d7bb46ae-v1
```

Preparation is the only command that reads the pinned source checkout. It
copies the needed maps, prompt assets, prompt source, and license into `data/`.
Once preparation succeeds, generation and scoring require only that prepared
dataset. Do not rerun preparation against the existing destination; the CLI
requires fresh data, result, and log destinations.

The prepared dataset contains `manifest.jsonl` with 600 examples, 100 in each
level, and `smoke-manifest.jsonl` with the ordered source ID `0` from each of
the six levels. Both manifests use the same dataset assets. Each model input is
exactly the processed text and three ordered images: legend, worked example,
then target maze. Target maps remain outside the model payload. Preparation
records are in
`results/evaluation/vsp-pinned-6d74d7bb46ae-v1/prepare.json`,
`results/evaluation/vsp-pinned-6d74d7bb46ae-v1/prepared-validation.json`, and
`logs/evaluation/vsp-pinned-6d74d7bb46ae-v1/prepare.log`. The additional
`full-image-preflight.json` records successful decoding of all 602 distinct
PNGs referenced by 600 inputs. That check covers decoding, paths, order, and
hashes; it does not prove the semantic correspondence of every image and map.

Use a fresh generation run ID and a different fresh scoring run ID:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. \
  ./venv/ilvr-venv/bin/python -m scripts.run_eval \
  --manifest data/evaluation/vsp-pinned-6d74d7bb46ae-v1/smoke-manifest.jsonl \
  --config configs/eval/vsp_ilvr_three_images_v1.yaml \
  --model-dir checkpoints/evaluation-staging/eval-stage1-dora-b1-e1-lr2p5e5-egb8-continuous-v29-20260902-122700/local-model \
  --base-model-dir /root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct \
  --cache-dir data/cache/evaluation \
  --model-identity stage1-dora-continuous-v29 \
  --output-dir results/evaluation/<fresh-generation-run-id>

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. \
  ./venv/ilvr-venv/bin/python -m scripts.score_eval \
  --manifest data/evaluation/vsp-pinned-6d74d7bb46ae-v1/smoke-manifest.jsonl \
  --predictions results/evaluation/<fresh-generation-run-id>/predictions.jsonl \
  --config configs/eval/vsp_ilvr_three_images_v1.yaml \
  --output-dir results/evaluation/<fresh-scoring-run-id>
```

The adapter and base paths above were the approved local GPU candidates. The
Stage1 loader compares its stored base reference as an exact string, so the
`--base-model-dir` argument must retain the literal absolute spelling shown
above. Preflight still resolves it canonically to enforce checkpoint-tree
containment and records the canonical path. Candidate metadata alone does not
prove loadability; the P2 six-example run cited in the validation document is
the direct observation for these exact artifacts.

The checked-in recipe is also the full-run contract: using `manifest.jsonl`
instead of `smoke-manifest.jsonl` requests all 600 examples. P2 did not execute
that full generation. The six-example GPU integration used a separate data
config with `max_new_tokens: 128`, so its output is not evidence for the
checked-in 4096-token recipe or a full benchmark result.

The declared `model_identity` and P1 run fingerprint identify the selected
artifact, prompt, and generation configuration. They are not hashes of model
weight contents. The generation run separately records canonical model/base
paths and hashes of small config, tokenizer, processor, adapter, and Stage1
metadata files. It assumes model files are not modified during a run.

The P2 protocol remains `modified`: it replaces source placeholders with an
explicit three-image order and uses the stricter P1 parser described above. It
does not claim the paper or Gemini protocol. A generation input error exits 2;
a model load, sample generation, or output failure exits 1 and preserves a
failed run record plus any safe partial records. Complete generation exits 0
regardless of answer correctness. Scoring exits 0 for a complete population,
1 for incomplete coverage or scoring/output failure, and 2 for input
preflight failure. Blank, invalid, and token-limit model responses are observed
predictions rather than infrastructure failures.

## P3 CoMT offline recipe

CoMT uses two separately named contracts. `comt_reference_rules_v1` is the
answer-extraction and scoring profile. `ilvr-test400-corrected-v1` is the
prepared dataset profile: the existing TEST selection of 400 rows, with 100
creation, deletion, update, and selection rows each. The word `reference` in
the scorer profile describes the observed source extraction rules; it does not
mean that the corrected data, denominator policy, CLI, or complete upstream
environment is reproduced.

Prepare the real data once with a fresh destination:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. \
  ./venv/ilvr-venv/bin/python -m scripts.prepare_eval \
  --benchmark comt --profile comt_reference_rules_v1 \
  --source-root data/evaluation/comt-source-discovery-20260911-01 \
  --selection-manifest data/TEST.jsonl \
  --image-root data/images_comt \
  --output-dir data/evaluation/comt-ilvr-test400-corrected-v1
```

The source root and selection file are pinned by revision and SHA-256. The
preparer joins TEST rows to source IDs by category and `IMAGE0`, validates the
observed question/options text, keeps the TEST order, and references exactly
one data-root-relative asset as `images_comt/<category>/<file>`. It does not
copy images. Rationale, annotations, `IMAGE1`, helper images, and sequence-plan
content never enter `ModelInput`.

Two source-grounded corrections are applied only in the new prepared data:
`creation-12125` changes
`['17','','18','','22','','34','']` to `['17','18','22','34']` with B=`18`,
and `creation-12164` changes
`['10cm','12cm','16','','24','']` to `['10cm','12cm','16','24']` with D=`24`.
The source JSONL, `data/TEST.jsonl`, and TRAIN data remain unchanged. The
correction rationale, exact arrays, pinned source hash, source-to-selection
links, and preparation metadata are written to `corrections.json`,
`source-selection.json`, and `SOURCE.json` beside `manifest.jsonl`. The source
and image discovery evidence is documented in
[p3-comt-discovery.md](p3-comt-discovery.md).

`configs/eval/comt_reference_rules_v1.yaml` is an offline-scoring template. Its
`model_identity: external:model`,
`prompt_template_identity: ilvr-test400-corrected-v1`, and empty
`generation_config` are labels, not observed generation provenance. Before a
real score, replace all three run-metadata values with the actual model,
prompt, and generation settings, and ensure every saved Prediction has the
matching P1 `id`, `manifest_sha256`, and `run_fingerprint`. Existing responses
must not be reattached by row order or relabeled with a new manifest hash.

Score those externally generated responses with a separate fresh scoring ID:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. \
  ./venv/ilvr-venv/bin/python -m scripts.score_eval \
  --manifest data/evaluation/comt-ilvr-test400-corrected-v1/manifest.jsonl \
  --predictions results/evaluation/<fresh-response-run-id>/predictions.jsonl \
  --config configs/eval/comt_reference_rules_v1.yaml \
  --output-dir results/evaluation/<fresh-scoring-run-id>
```

This recipe does not enable `scripts.run_eval` for CoMT. That command remains
the VSP generation bridge and rejects a valid CoMT config before loading a
model. Obtain responses through an external generation path that emits the P1
Prediction schema, then score the saved file.

Whitespace-only responses are observed `empty` answers; other responses that
do not yield A-D are observed `invalid_format` answers. Both are complete model
observations and count as incorrect. A missing Prediction or an
`infrastructure_error` keeps the 400-row denominator but makes the run
incomplete and its overall accuracy null. Malformed JSONL, missing or extra
schema fields, unknown benchmark/profile, duplicate or unknown IDs, and
manifest/run-fingerprint mismatches are input errors and exit 2. A complete
score exits 0; incomplete coverage or scoring/output infrastructure failure
exits 1. Preparation likewise exits 0 on success, 2 for input or contract
errors, and 1 for infrastructure failure. Preparation and scoring require
fresh data/result/log names and never overwrite an existing destination.

## P4 VisuLogic rules-only recipe

Use `visulogic` with `visulogic_vlmeval_rules_v1`. The pinned dataset is
`VisuLogic/VisuLogic` revision `3e483f8cca0bc766ecc5f70299f2b8a34b5035ca`.
The `visulogic-pqs579-v1` dataset profile selects every row in these categories,
in source order: Positional Reasoning → Position (136), Quantitative Reasoning
→ Quantity (353), Stylistic Reasoning → Style (90). The other 421 rows belong
to the three categories outside this selected scope. No selected row is dropped
and no gold answer is repaired. This subset is not asserted to be the paper split.

Preparation reads pinned `dataset-info.json`, `data.jsonl`, and `images.zip` from
`data/evaluation/visulogic-source-discovery-20260911-01`. JSONL SHA-256 is
`969ff4407a433f7282d52635dc4f30bdeacd6ac0e4ea31cf604b85313c8562c0`; archive SHA-256 is
`86157164e4ee61296ed923dca61c115f08ba0633ca7c973029dbec3e0edf555b`.
Each source question remains unchanged. Its ordered line-start A/B/C/D option
blocks become four nonempty trimmed choices; the source label selects the MCQ
target index/content. Inputs contain exactly the question, choices and one
source image, with no appended gold or reasoning trace. Images are decoded for
validation and their original bytes are retained, including JPEG-encoded members
whose filenames end in `.png`.

The completed preparation uses the fresh retry directory below. The first run
stopped before writing a dataset because it incorrectly required PNG encoding;
its failure log is retained under `logs/evaluation/visulogic-pqs579-v1/`.
The `-r02` suffix identifies the execution attempt, not a new scoring protocol.
For another preparation, choose a new output directory and adjust the recipe's
`dataset_root` to match. Existing data/result/log directories cannot be overwritten.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  ./venv/ilvr-venv/bin/python -m scripts.prepare_eval \
  --benchmark visulogic --profile visulogic_vlmeval_rules_v1 \
  --source-root data/evaluation/visulogic-source-discovery-20260911-01 \
  --output-dir data/evaluation/visulogic-pqs579-v1-r02
```

The output contains `manifest.jsonl`, `SOURCE.json`, `source-selection.json`, and
579 images. Preparation metadata and logs use the same output basename beneath
`results/evaluation/` and `logs/evaluation/`. CoMT-only preparation arguments are
rejected for VisuLogic.

The scorer independently implements the rules in pinned VLMEvalKit
`b06e0716f22127c1920364c6f4d80b9c32e42c09`,
`vlmeval/dataset/utils/visulogic.py`: box-candidate extraction followed by the
source language-marker rules, then exact uppercase A–D label comparison. It
does not use choice-content matching, VSP's parser, `<answer>` extraction, or
GPT fallback. The source has deliberate compatibility quirks: bare `A` and
padded ` A ` have method `box`; bare `a` fails but `The answer is b.` yields B;
an unfinished final box can leave a previous completed box selected. Nested
LaTeX is processed with the source token rules, not a general balanced parser.
The official VisuLogic evaluator's hybrid GPT path differs, including padded
letters, so this profile does not claim equivalence to that evaluator.

Empty/whitespace responses have status `empty`; rule failures have
`invalid_format`, answer `Z`, and method `raw`. Both count as observed wrong
answers. Valid answers retain source methods `box`/`lang`. Extraction never
receives a gold target. Missing/infrastructure rows retain the manifest
population and make accuracy null. Configured empty categories have count zero
and null accuracy. Only a nonempty unique subset of Position/Quantity/Style is
accepted in scoring config; `max_steps` must be omitted or null. LLM profiles
remain unregistered and fail explicitly.

Before scoring real responses, replace the template run metadata in
`configs/eval/visulogic_vlmeval_rules_v1.yaml` with the actual model identity,
prompt identity and generation settings. Predictions must match the manifest
IDs/hash and run fingerprint; do not relabel unrelated existing responses.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  ./venv/ilvr-venv/bin/python -m scripts.score_eval \
  --manifest data/evaluation/visulogic-pqs579-v1-r02/manifest.jsonl \
  --predictions results/evaluation/<response-run-id>/predictions.jsonl \
  --config configs/eval/visulogic_vlmeval_rules_v1.yaml \
  --output-dir results/evaluation/<fresh-scoring-run-id>
```

`scripts.run_eval` remains VSP-only and rejects VisuLogic before model loading.
P4 has no model-generation or external-judge connection. Summaries keep
`protocol_kind=modified` and `verification.paper_protocol_verified=false`;
preparation records `ood_verified=unknown`. Software contract tests and
synthetic fixture accuracy are not model performance or independent semantic
verification of every gold answer.

## P5 EMMA-mini offline recipe

EMMA uses `benchmark=emma` and `profile=emma_safe_rules_v1`. The prepared
dataset is the pinned `luckychao/EMMA-mini` test split at revision
`6b9ae9a74733bb57f0b741213f8d0ec9ebae067a`. The completed retry is
`data/evaluation/emma-mini400-v1-r02`: 400 rows, Chemistry/Coding/Math/Physics
100 each, 341 multiple-choice and 59 open-ended questions, and 614 images. Its
manifest SHA-256 is
`b41e7b210898417dfea6d48280737bc3e35c2133de183937fae1dc2ca805f961`.
No rows were excluded. The `-r02` suffix records the retry after the first
preflight rejected 50 valid null, audit-only `solution` fields; it does not
name a different protocol.

Preparation is reproducible only into a fresh destination. The completed
command was:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  ./venv/ilvr-venv/bin/python scripts/prepare_eval.py \
  --benchmark emma --profile emma_safe_rules_v1 \
  --source-root data/evaluation/emma-source-discovery-20260913-01 \
  --output-dir data/evaluation/emma-mini400-v1-r02
```

The manifest preserves each raw pid and source answer, including lowercase
labels. A model input contains the source context and question (joined by two
newlines when context exists), the choices, and ordered images. Gold answer,
target content/index, and source solution are excluded from `ModelInput`.
Open-ended targets use canonical `open_ended`; the audit copy alone retains the
source spelling `open-ended`. Preparation accepts source `solution` only as a
string or null and never exposes it to the model.

Extraction reads only the response and `ModelInput`. After trimming, a whole
in-range one-letter label is accepted case-insensitively; an uppercase label
prefix requires an immediate `:` or `.`, so prose such as `A reasonable...`
stays whole raw text. A whole decimal/scientific number stays intact. Otherwise
the last complete balanced `\boxed{...}` wins (and a later incomplete box does
not erase it), then the first marker kind in the fixed priority `final answer
is`, `correct answer is`, `answer should be`, `answer is`, `answer:` uses its
last occurrence and first following line. A single terminal prose period is
removed without truncating a decimal. One enclosing `\text{...}` wrapper is
removed only from content selected from a box; raw and marker candidates retain
it. One whole `$...$` or `\(...\)` math wrapper is removed only from the
comparison representation; the recorded extracted candidate retains that wrapper.
Each comparison operand is normalized once, so `$$0.5$$` does not match `0.5`.
Empty responses are `empty`; an empty selected box/marker or a candidate whose
single math wrapper encloses only whitespace is
`invalid_format/empty_extraction`; standalone signed
`nan`/`inf`/`infinity`/`∞`/`\infty` variants are
`invalid_format/non_finite_value`.

For multiple choice, a valid label after one math-wrapper normalization is decisive.
A wrong label (including `$B$` when A is correct) is
wrong even when later response text contains the correct choice content. When
the extracted value is not a valid label, comparison uses exact choice content
and then the bounded value rules. Duplicate choice content is therefore an
inherent limitation: content-only output can match the target content without
identifying which duplicate option the model intended.

Value comparison first removes one whole math wrapper, rejects standalone
nonfinite literals, and accepts nonempty exact string equality. This ordering
keeps exact code or choice content correct even above the 512-character math
input limit; the pinned source includes 178 choices longer than 512 characters,
with a maximum of 3,504. Remaining expressions use a closed grammar of numeric
literals, single-letter variables, `pi`/`\pi`, `sqrt`/`\sqrt`, `\frac`, explicit
arithmetic, grouping, and integer powers. Implicit multiplication, functions,
arrays, attributes, calls, chained powers, negative variable powers, and other
syntax are unsupported. Number words such as `twenty one` have no numeric
interpretation and succeed only through exact string equality.

The parser limits input to 512 characters, 64 operation nodes, grouping/unary
depth 16, 64 mantissa digits, scientific exponent magnitude 100, and integer
power magnitude 8. Every retained rational numerator and denominator is limited
to 4,096 bits; finite real intermediate/final magnitudes are limited to `1e100`.
Unsupported grammar, nonfinite values, and resource bounds yield
`unsupported_expression`, `non_finite_value`, and `math_limit_exceeded` rather
than executing input. Unexpected internal errors fail the run. Symbolic
polynomials compare by exact simplified difference within the grammar. Numeric
expressions compare `round(float(value), 2)` exactly; this deliberately inherits
Python binary-float and tie-rounding behavior and adds no relative or absolute
tolerance.

Before scoring real responses, replace all template values under `run_metadata`
in `configs/eval/emma_safe_rules_v1.yaml` with the actual model identity, prompt
identity, and generation settings. Every Prediction must carry the matching
manifest hash and run fingerprint; existing responses cannot be relabeled or
joined by row order. Score into a fresh directory:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  ./venv/ilvr-venv/bin/python -m scripts.score_eval \
  --manifest data/evaluation/emma-mini400-v1-r02/manifest.jsonl \
  --predictions results/evaluation/<response-run-id>/predictions.jsonl \
  --config configs/eval/emma_safe_rules_v1.yaml \
  --output-dir results/evaluation/<fresh-scoring-run-id>
```

Empty, invalid, unsupported, and limit-exceeded responses are observed wrong
answers. Missing or infrastructure-error predictions keep the 400-row
denominator, make the evaluation incomplete, and leave accuracy null. EMMA
summaries additionally group by question type, subject, and subject/task and
report only the three comparison diagnostics above; `empty_extraction` remains
visible in each affected row's `error`, while that row contributes only to the
aggregate `invalid_format` entry in `status_counts`.

The implementation is a modified offline protocol:
`protocol_kind=modified` and `verification.paper_protocol_verified=false`.
The source comparison isolates only three pure extraction helpers and shows 12
of 23 cases with the same selected value and 11 intended differences. The
same-value `nan` and `inf` cases still differ in local status because this
profile rejects nonfinite values. That comparison is not full upstream-runtime
equivalence. No model was run, so the per-model extraction first-nail is
`not_run`; GPU inference, a 400-response model evaluation, the complete upstream
environment, and paper-protocol reproduction are also separate, unverified
activities.

## Zebra-CoT 2D 고정 400개 오프라인 프로토콜

고정 원본 투영은 `multimodal-reasoning-lab/Zebra-CoT` revision
`f6d1defc169d19180a69f517925ed6cbb06a1c97`의 train split이다. Jigsaw와
Search를 각각 200개 사용한다. 다음 명령은 로컬 투영의 두 digest, 400행 대응과
JPEG를 모두 검사한 뒤 새 목적지에 manifest와 감사 자료를 만든다.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  ./venv/ilvr-venv/bin/python -m scripts.prepare_eval \
  --benchmark zebra_2d --profile zebra_2d_rules_v1 \
  --source-root data/evaluation/zebra-source400-20260914-02 \
  --output-dir data/evaluation/zebra-2d400-v1
```

실제 준비 manifest SHA-256은
`966453f6973e68a6a426f00c0fd83bd495a7390f9a9da01e9413e866646f4d5b`다.
`ModelInput`에는 원문 Question, 문제 이미지 하나, 빈 choices만 들어간다. Jigsaw
응답은 A–D 한 글자 문법을 사용하고 Search는 casefold, 공백 축약, 단일 문장 끝
마침표 제거 뒤 보수적으로 비교한다. box가 있으면 마지막 top-level box를 사용하며,
마지막 top-level box가 닫히지 않았으면 앞의 완성 box나 answer marker로 돌아가지 않고
`invalid_format`으로 처리한다. box가 없으면 마지막 answer marker를, 둘 다 없으면
전체 응답을 사용한다.

실제 prediction과 실제 실행 설정을 담은 config를 준비한 뒤 다음처럼 채점한다.
`score_eval --help`에서 확인한 필수 인자는 manifest, predictions, config,
output-dir 네 개다.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  ./venv/ilvr-venv/bin/python -m scripts.score_eval \
  --manifest data/evaluation/zebra-2d400-v1/manifest.jsonl \
  --predictions results/evaluation/<real-prediction-run>/predictions.jsonl \
  --config <real-zebra-config.yaml> \
  --output-dir results/evaluation/<fresh-scoring-run>
```

이 profile은 `protocol_kind=modified`, `ood_verified=unknown`,
`paper_protocol_verified=false`다. 고정 digest는 수집한 source projection의
identity를 고정하며 원본 parquet 전체 bytes를 검증하지 않는다. 논문 sample
identity와 judge 동등성도 별도의 미확보 근거다.
