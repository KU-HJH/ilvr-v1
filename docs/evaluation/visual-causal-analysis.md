# CoMT visual causal analysis

`scripts/analyze_comt_visual.py` runs one condition at a time so every condition starts
with a fresh prompt, embeddings, continuous-latent controller, and intervention state.
Run it from the repository root with the repository environment:

```bash
source ./venv/ilvr-venv/bin/activate
export PYTHONPATH=transformers/src:src:.
```

The approved prepared checkpoints are:

```text
checkpoints/evaluation-staging/stage1-dora-z1-ema-frozen-b1-ga4-e1-lr5e5-egb16-20260902-024225/prepared-checkpoint-215
checkpoints/evaluation-staging/ours1-stage2a-s1matched-egb16-lr5e5-dora-z2-20260908t022000z-r3/prepared-checkpoint-212
checkpoints/evaluation-staging/ours1-ctx-pref-vonly-direct-s1-dora-egb16-lr5e5-default-s212-r2-20260910t022900z/prepared-checkpoint-212
```

All three are PEFT artifacts and need the base model:

```text
/root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct
```

## Select the fixed panel

Indices always refer to original rows in `data/TEST.jsonl`. The checked-in main-panel
manifest can be converted to the CLI's comma-separated form without renumbering rows:

```bash
MAIN_INDICES=$(jq -r 'join(",")' data/evaluation/visual-causal-20260910/main-indices.json)
```

When `--indices` is omitted, rows from `--test-data-path` are matched exactly to the
source rows. Supplying original indices directly is clearer and avoids the legacy
evaluator convention in which a 100-row subset was numbered 0 through 99. Before
comparing a legacy native file, rewrite its `index` values with that run's authenticated
`selection.original_indices`; `compare` deliberately rejects unequal index sets.

## E0 backend bridge

Stage 1 native evaluation already uses the repository's SDPA decoder and FlashAttention
2 vision loader. This command records a one-row native baseline:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B scripts/analyze_comt_visual.py run \
  --model-dir checkpoints/evaluation-staging/stage1-dora-z1-ema-frozen-b1-ga4-e1-lr5e5-egb16-20260902-024225/prepared-checkpoint-215 \
  --base-model-dir /root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct \
  --test-data-path data/TEST.jsonl --source-data-path data/TEST.jsonl --indices 3 \
  --output-dir results/visual-causal-stage1-e0-native-20260910 \
  --visual-ablation-mode baseline --attention-backend native \
  --max-new-tokens 4096 --max-sequence-length 32768 \
  --repetition-penalty 1.05 --temperature 0 --top-p 1
```

Stage 2 native loading uses `src/evaluate_deepseed.py`: it requests FlashAttention 2
and uses that loader's classified eager fallback if initialization fails. The analysis
SDPA bridge uses `eval.py` for either stage. Run both before interpreting an intervention:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B scripts/analyze_comt_visual.py run \
  --model-dir checkpoints/evaluation-staging/ours1-stage2a-s1matched-egb16-lr5e5-dora-z2-20260908t022000z-r3/prepared-checkpoint-212 \
  --base-model-dir /root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct \
  --test-data-path data/TEST.jsonl --source-data-path data/TEST.jsonl --indices 3 \
  --output-dir results/visual-causal-stage2a-e0-native-20260910 \
  --visual-ablation-mode baseline --attention-backend native

PYTHONDONTWRITEBYTECODE=1 python -B scripts/analyze_comt_visual.py run \
  --model-dir checkpoints/evaluation-staging/ours1-stage2a-s1matched-egb16-lr5e5-dora-z2-20260908t022000z-r3/prepared-checkpoint-212 \
  --base-model-dir /root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct \
  --test-data-path data/TEST.jsonl --source-data-path data/TEST.jsonl --indices 3 \
  --output-dir results/visual-causal-stage2a-e0-sdpa-20260910 \
  --visual-ablation-mode baseline --attention-backend sdpa
```

The second Stage 2 checkpoint uses the same command shape:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B scripts/analyze_comt_visual.py run \
  --model-dir checkpoints/evaluation-staging/ours1-ctx-pref-vonly-direct-s1-dora-egb16-lr5e5-default-s212-r2-20260910t022900z/prepared-checkpoint-212 \
  --base-model-dir /root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct \
  --test-data-path data/TEST.jsonl --source-data-path data/TEST.jsonl \
  --indices "$MAIN_INDICES" \
  --output-dir results/visual-causal-context-e0-sdpa-20260910 \
  --visual-ablation-mode baseline --attention-backend sdpa
```

Compare same-index native and SDPA E0 files to measure the backend bridge:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B scripts/analyze_comt_visual.py compare \
  --baseline-jsonl results/visual-causal-stage2a-e0-native-20260910/predictions.jsonl \
  --condition-jsonl results/visual-causal-stage2a-e0-sdpa-20260910/predictions.jsonl \
  --output-dir results/visual-causal-stage2a-e0-bridge-20260910
```

`summary.json` separates exact raw-token, prompt-identity, and scoring agreement from
paired accuracy. This native-versus-SDPA comparison is not the observer no-op gate. A
backend mismatch is a backend result and must not be attributed to the intervention.

Use two distinct same-backend checks before the causal comparisons. First, compare the
original evaluator's no-policy native output with the new runner's native baseline. The
authenticated original indices must already be restored in both files. For example, the
normalized original-evaluator CE baseline is:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B scripts/analyze_comt_visual.py compare \
  --baseline-jsonl results/visual-causal-native-reproduction-20260910/ce-predictions.jsonl \
  --condition-jsonl results/visual-causal-ce-e0-native-runner-20260910/predictions.jsonl \
  --output-dir results/visual-causal-ce-native-reproduction-gate-20260910
```

Second, compare an existing-evaluator SDPA run with no policy against an SDPA baseline
run with the `VisualIntervention("baseline", ...)` observer. Remove the existing
evaluator's final summary row and replace its subset-local indices with the authenticated
original indices before comparison:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B scripts/analyze_comt_visual.py compare \
  --baseline-jsonl results/visual-causal-stage2a-e0-sdpa-no-policy-20260910/predictions-normalized.jsonl \
  --condition-jsonl results/visual-causal-stage2a-e0-sdpa-20260910/predictions.jsonl \
  --output-dir results/visual-causal-stage2a-e0-sdpa-observer-noop-20260910
```

Only this second, same-backend no-policy-versus-observer comparison is the observer
no-op gate. It must match raw suffix IDs, prompt identity, and scorer fields exactly for
every available row. Successful rows with an invalid gold remain part of the raw no-op
comparison. Inference errors or missing raw/prompt/scoring evidence are reported under
`equivalence.unavailable_evidence` and make `equivalence.exact` false.

## E1 through E4

Use the SDPA bridge for the intervention panel and change only the mode and output path:

```bash
for MODE in no_input_image no_latent_execution latent_image_attention_block global_image_attention_block; do
  PYTHONDONTWRITEBYTECODE=1 python -B scripts/analyze_comt_visual.py run \
    --model-dir checkpoints/evaluation-staging/ours1-stage2a-s1matched-egb16-lr5e5-dora-z2-20260908t022000z-r3/prepared-checkpoint-212 \
    --base-model-dir /root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct \
    --test-data-path data/TEST.jsonl --source-data-path data/TEST.jsonl \
    --indices "$MAIN_INDICES" \
    --output-dir "results/visual-causal-stage2a-${MODE}-20260910" \
    --visual-ablation-mode "$MODE" --attention-backend sdpa \
    --max-new-tokens 4096 --max-sequence-length 32768 \
    --repetition-penalty 1.05 --temperature 0 --top-p 1
done
```

E1 removes images before prompt construction. Its prompt has no image entries,
placeholders, pixels, or image grid, while START selection and continuous-latent
execution remain enabled.

E2 suppresses START only at each ordinary free-token decision, after repetition
penalty and the standard PAD/END exclusions. It keeps the same total token budget; it
does not provide matched computation and does not estimate a probability of entering
continuous reasoning. `start_probability` is a softmax diagnostic over decision logits,
not a temperature-zero sampling probability. Nonfinite diagnostics are written as JSON
`null`.

E3 blocks direct attention from the original image keys for START and every PAD query
that produces a latent state. START is included because the hidden state at START is the
producer for the first PAD latent. END itself is not an E3 query. Image information can
still arrive indirectly through earlier text states or other unblocked states, so E3 is
evidence about direct latent-to-image edges rather than complete image independence.

E4 blocks original image keys for every non-image decoder query from the first forward
call. The image tokens, pixels, grids, token count, and mRoPE geometry remain present;
therefore E4 removes direct content access while retaining image-layout geometry. Both
masked modes require every instantiated decoder layer to be
`Qwen2_5_VLSdpaAttention`; the runner fails rather than falling back.

`intervention_forward_records` records the 2-D validity-mask shape received at the
intervention boundary, plus prefix roles and positions. It does not claim to be the
post-conversion 4-D mask delivered to the decoder. Runtime module-class metadata and a
separate decoder pre-hook probe are the appropriate witnesses for that claim.

## E5 matched-image replacement

Generate the independent matched-question groups under `data/`:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B scripts/analyze_comt_visual.py groups \
  --source-data-path data/TEST.jsonl \
  --output-json data/evaluation/visual-causal-20260910/matched-groups.json
```

The manifest contains `groups` and their flattened `indices`. Membership uses only the
whitespace-normalized complete question text and valid, differing gold labels; it does
not use predictions. A directed two-row replacement example is:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B scripts/analyze_comt_visual.py run \
  --model-dir checkpoints/evaluation-staging/ours1-stage2a-s1matched-egb16-lr5e5-dora-z2-20260908t022000z-r3/prepared-checkpoint-212 \
  --base-model-dir /root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct \
  --test-data-path data/TEST.jsonl --source-data-path data/TEST.jsonl \
  --indices 100,137 --replacement-map '{"100":137,"137":100}' \
  --output-dir results/visual-causal-stage2a-e5-100-137-20260910 \
  --visual-ablation-mode image_replacement --attention-backend sdpa
```

For every accepted mapping, the recipient's exact question text is retained while the
donor image and donor gold are used. Both labels must parse, the normalized full
questions must match, and the labels must differ. A three-member group needs separate
runs to cover both possible donors for a recipient. Repeated directed rows in one group
are dependent; report uncertainty at the group level. Donor-gold correctness is a
matched-image discrimination measure, not ordinary accuracy on a newly annotated
question/image pair.

## Outputs and failures

Each `run` directory contains `predictions.jsonl` and `metadata.json`. A prediction row
contains the stable original index and sample ID, checkpoint and condition, category
and its source, text/image identities, replacement lineage, prompt identity and length,
raw output and raw generated suffix IDs, extracted and normalized prediction/gold,
parse status, correctness, START/END suffix and absolute positions, completed and
truncated latent-block evidence, intervention diagnostics, generation-only timing,
wall timing, stop reason, and token counts.

Generation timing starts after image opening and processor preparation, synchronizes a
CUDA device around generation, and excludes model loading. Wall timing covers the whole
sample attempt. Old Stage 2 evaluator rows have no per-row timing; absence remains
`null` and must never be interpreted as measured zero runtime.

Every requested missing or failed row remains in `predictions.jsonl` with
`prediction_parse_status: "inference_error"` and an `error` message. Any such row makes
the run metadata status `failed` and the CLI exit nonzero. Invalid or blank gold labels
remain explicit through `gold_parse_status` and `invalid_gold_reason`; coverage accuracy
keeps them in its denominator while valid-gold accuracy does not.

`metadata.json` records dataset hashes, selected original indices, loader, all
instantiated decoder and vision attention classes, parameter dtype set, configured
dtype/backend values, observed input tensor dtypes, latent size, token IDs, decoding
configuration, counts, aggregate metrics, and total wall time. Model loading reads the
checkpoint and base-model locations; model-bearing state is never staged under
`results/`.

`compare` writes `summary.json`, a flattened `summary.csv`, and `analysis.md`. The
summary includes one-condition aggregates, paired accuracy delta and bootstrap interval,
McNemar result, all four transitions, normalized-answer and raw-token change rates,
category summaries, exclusions, and exact baseline-equivalence evidence.

## CPU verification

These checks do not load checkpoints or use a GPU:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transformers/src:src:. \
  ./venv/ilvr-venv/bin/python -B -m pytest -q \
  tests/evaluation/test_visual_interventions.py \
  tests/evaluation/test_visual_metrics.py \
  tests/evaluation/test_visual_analysis_runner.py \
  tests/evaluation/test_continuous_latent.py \
  tests/stage2/test_eval_integration.py \
  -k 'not evaluation_runtime_sidecar_counts_concrete_blocks_exclusively'

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transformers/src:src:. \
  ./venv/ilvr-venv/bin/python -B scripts/analyze_comt_visual.py --help
```

The excluded existing integration test deliberately constructs a FlashAttention 2
model. Run it separately when one GPU is free:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transformers/src:src:. \
  ./venv/ilvr-venv/bin/python -B -m pytest -q \
  tests/stage2/test_eval_integration.py::test_evaluation_runtime_sidecar_counts_concrete_blocks_exclusively
```

After an E0 same-backend comparison, this numeric no-op gate verifies exact agreement:

```bash
./venv/ilvr-venv/bin/python - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("results/visual-causal-stage2a-e0-sdpa-observer-noop-20260910/summary.json").read_text())
equivalence = summary["equivalence"]
assert equivalence["paired_count"] > 0
assert equivalence["comparable_count"] == equivalence["paired_count"]
assert equivalence["unavailable_count"] == 0
assert equivalence["exact"] is True
assert equivalence["raw_token_id_match_count"] == equivalence["paired_count"]
assert equivalence["prompt_identity_match_count"] == equivalence["paired_count"]
assert equivalence["scoring_match_count"] == equivalence["paired_count"]
print("PASS: same-backend E0 is an exact no-op")
PY
```

The first numeric intervention check should inspect `summary.json` rather than infer from
aggregate accuracy alone: verify identical paired counts, review exclusions and all four
transitions, report the paired interval, and keep normalized-answer change separate from
raw-token change. Activation patching remains deferred. It needs a later design with
alignment, identity-patch controls, and a separately validated causal target.
