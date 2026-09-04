# Mixed Vision FlashAttention2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the observed row-1078 vision-SDPA OOM while preserving the Stage 2B language-mask and packed-FlexAttention semantics.

**Architecture:** Load the same Qwen2.5-VL checkpoint with an explicitly prepared composite config: ordinary language decoder layers use SDPA, vision blocks use FlashAttention2, and the existing packed final-decoder path continues to dispatch its `BlockMask` through FlexAttention. One shared loader validates the actual instantiated attention module classes and fails closed. Negative manifests record both public backend identities, while the Stage 2B consumer derives the full expected settings from its own fixed runtime contract rather than trusting the manifest.

**Tech Stack:** Python 3.11, PyTorch 2.7.1+cu128, vendored Transformers/Qwen2.5-VL, FlashAttention 2.8.3, FlexAttention, pytest.

**Parent spec:** `docs/superpowers/specs/2026-09-04-step-preference-dora-ablation-readiness-design.md`

## Constraints

- Use `./venv/ilvr-venv` and `PYTHONPATH="$PWD/transformers/src:$PWD/src"`.
- Keep ordinary/reference language attention as `sdpa`, vision attention as `flash_attention_2`, and packed final-decoder attention as the existing FlexAttention path.
- Do not pass a public attention dictionary; prepare the composite config explicitly before weight loading.
- Validate every instantiated production model: every vision block must contain `Qwen2_5_VLVisionFlashAttention2`, and every ordinary language layer must contain `Qwen2_5_VLSdpaAttention`.
- Fail on missing FlashAttention2 support or a resolved-class mismatch. Do not fall back to eager or SDPA vision attention.
- Preserve processor inputs, pixels, retained rows, sequence limits, objectives, and optimization settings.
- Preserve ordinary Stage 2 behavior when no distinct vision backend is requested.
- Do not resume capacity or training work until the focused tests and the row-1078 negative-generation first nail pass.

---

### Task 1: Implement and authenticate the mixed Qwen backend

**Files:**
- Create: `src/stage2/attention_backend.py`
- Modify: `src/utils_deepseed.py`
- Modify: `src/main.py`
- Modify: `run_stage2_training.sh`
- Modify: `scripts/generate_step_negatives.py`
- Modify: `src/step_preference/equivalence.py`
- Modify: `scripts/probe_packed_stage2b_capacity.py`
- Modify: `scripts/audit_packed_stage2b_lengths.py`
- Modify: `eval.py`
- Create: `tests/stage2/test_attention_backend.py`
- Modify: `tests/step_preference/test_negative_generation.py`
- Modify: `tests/stage2/test_routing.py`
- Modify: `tests/step_preference/test_packed_capacity_probe.py`
- Modify: `tests/stage2/test_launcher.py`
- Modify: `tests/stage2/test_comt_readiness.py`
- Modify: `tests/stage2/test_eval_integration.py`

**Interfaces:**
- Produces: `load_qwen2_5_vl_with_attention(source, *, attention_implementation, vision_attention_implementation, config_load_kwargs=None, model_load_kwargs=None, load_config=None, load_model=None) -> object`. `None` loaders bind the production `AutoConfig.from_pretrained` and `Qwen2_5_VLForConditionalGeneration.from_pretrained` at call time; injected loaders receive the same separated config/model keyword maps.
- Produces: `validate_qwen2_5_vl_attention_backends(model, *, attention_implementation, vision_attention_implementation) -> None` over an already resolved Qwen model. It checks nonempty `model.visual.blocks[*].attn` and `model.model.layers[*].self_attn` collections and requires exact types `Qwen2_5_VLVisionFlashAttention2` and `Qwen2_5_VLSdpaAttention` for the mixed contract.
- Produces: `--vision_attention_implementation` and `VISION_ATTENTION_IMPLEMENTATION`, defaulting to the language backend for ordinary Stage 2 compatibility.
- Produces: negative-generation settings containing `attention_implementation` and `vision_attention_implementation`.
- Consumes: Stage 2B preflight's own language backend, vision backend, fixed `cuda:0`/BF16 runtime, and configured min/max token values to form an independent complete expected settings map.

The defaults are entry-point-specific and intentional:

| Entry point | Language default | Vision default | Validation |
| --- | --- | --- | --- |
| ordinary Stage 2 launcher/parser | `sdpa` | inherit language backend | existing ordinary path; no mixed-only class requirement |
| Stage 2B canonical launcher | `sdpa` | explicitly `flash_attention_2` | strict mixed-class validation |
| negative generator | `sdpa` | `flash_attention_2` | strict mixed-class validation |
| capacity probe/audit | `sdpa` | explicit `flash_attention_2` in canonical command | strict mixed validation wherever a model is constructed; audit authenticates the manifest identity |
| evaluation | fixed `sdpa` | fixed `flash_attention_2` | strict mixed-class validation, no fallback |

The canonical manifest `generation_settings` JSON object is exactly:

```json
{
  "attention_implementation": "sdpa",
  "vision_attention_implementation": "flash_attention_2",
  "device": "cuda:0",
  "max_new_tokens": 32,
  "min_tokens": 2,
  "torch_dtype": "bfloat16"
}
```

The producer may use its observed `resolved_device` and dtype string so CPU unit tests remain injectable; the production Stage 2B consumer independently constructs the exact CUDA/BF16 object above from its own backend/token arguments and fixed negative-generation contract. A missing, extra, or changed field is an authentication failure.

- [ ] **Step 1: Add failing loader and authentication tests**

Add a tiny real Qwen config test which loads/constructs through the new shared helper and asserts all actual vision attention objects are `Qwen2_5_VLVisionFlashAttention2` while all language self-attention objects are `Qwen2_5_VLSdpaAttention`. Add causal failures for a wrong vision class, wrong language class, and missing blocks/layers. Assert the helper supplies a prepared `config` to the injected model loader and does not pass an `attn_implementation` dictionary.

Extend negative-generation tests so the parser defaults to language SDPA and vision FlashAttention2, the loader receives both values, and the written settings contain both. Add a routing test whose otherwise valid schema-3 manifest has only the vision backend changed and assert Stage 2B preflight rejects it before any model is loaded. Also assert the consumer's expected map includes language backend, vision backend, device, dtype, `min_tokens`, and `max_new_tokens` without reading those expected values back from the manifest.

Extend launcher, capacity-probe, audit, and evaluation tests to require the separate vision backend at every production model-loading entry point. Evaluation must fail closed on FlashAttention2 initialization or class-resolution errors and must not retry eager attention.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_attention_backend.py \
  tests/step_preference/test_negative_generation.py \
  tests/stage2/test_routing.py \
  tests/step_preference/test_packed_capacity_probe.py \
  tests/stage2/test_launcher.py \
  tests/stage2/test_comt_readiness.py \
  tests/stage2/test_eval_integration.py
```

Expected: FAIL only because the mixed backend and its arguments/settings are absent.

- [ ] **Step 3: Implement the minimal shared loader**

In `src/stage2/attention_backend.py`, validate the exact supported mixed contract (`sdpa` language, `flash_attention_2` vision), load the Qwen composite config, call the vendored Qwen FlashAttention2 hard availability check with the requested model dtype/device map, and set these fields before construction:

```python
config._attn_implementation_internal = "sdpa"
config._attn_implementation = "sdpa"
config._attn_implementation_autoset = True
config.vision_config._attn_implementation_internal = "flash_attention_2"
config.vision_config._attn_implementation = "flash_attention_2"
config.vision_config._attn_implementation_autoset = True
```

Call the injected/default model loader with `config=config` and without any `attn_implementation` keyword. Let availability/import/initialization errors propagate without retry. Callers that load a PEFT wrapper resolve it with the existing Stage 1/Stage 2 resolver before invoking the validator. Inspect the exact runtime objects at `model.visual.blocks[*].attn` and `model.model.layers[*].self_attn`; config strings or class-name strings are insufficient.

Keep the current direct loader path and its existing keyword behavior unchanged when the vision backend is omitted or equals the public language backend. The shared helper's mixed branch supports exactly SDPA/FlashAttention2; reject other distinct pairs. Do not alter the packed decoder dispatch: its existing `stage2_final_attention_mask`/`BlockMask` path must continue to call `forward_flex`, while ordinary and reference-scoring tensor/additive masks continue through SDPA.

- [ ] **Step 4: Thread the backend through production entry points**

Add the vision option to argument parsing and launcher ownership/forwarding. Use the shared loader in Stage 2B training, frozen Stage 2A negative generation, the capacity probe through its production component builder, and evaluation for full and PEFT artifacts. Preserve dependency injection used by tests and checkpoint provenance logic.

The generator records the complete settings map. Stage 2B preflight constructs its complete expected map from its own arguments and the fixed BF16 `cuda:0` generation contract, then passes that map to `load_negative_store`; it must never set `expected_generation_settings = manifest.generation_settings_dict`. The capacity and audit identities include both attention values.

- [ ] **Step 5: Run focused verification**

Run the Step 2 command and require PASS. Then run the smallest packed-mask regression proving the decoder split was preserved:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_qwen_flex_attention.py \
  tests/step_preference/test_packed_stage2_model.py
```

Expected: PASS, including CUDA tests rather than skips where marked CUDA-required.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/stage2/attention_backend.py src/utils_deepseed.py src/main.py \
  run_stage2_training.sh scripts/generate_step_negatives.py \
  src/step_preference/equivalence.py scripts/probe_packed_stage2b_capacity.py \
  scripts/audit_packed_stage2b_lengths.py eval.py \
  tests/stage2/test_attention_backend.py \
  tests/step_preference/test_negative_generation.py tests/stage2/test_routing.py \
  tests/step_preference/test_packed_capacity_probe.py \
  tests/stage2/test_launcher.py tests/stage2/test_comt_readiness.py \
  tests/stage2/test_eval_integration.py
git commit -m "fix: split qwen vision attention backend"
```

---

### Task 2: Falsify the row-1078 OOM before resuming capacity work

**Files:**
- Write generated rows/manifest: `results/step-pref-capacity-<fresh-UTC>/`
- Write execution log: `logs/step-pref-capacity-<fresh-UTC>/row-1078-generate.log`
- Update evidence: `.superpowers/sdd/2026-09-04-step-preference-dora-ablation-readiness/task-6-report.md`

- [ ] **Step 1: Allocate a fresh run ID and validate fixed inputs**

Create only the fresh result/log directories. Re-run the parent plan's fixed input SHA-256 checks and select source row 1078 into the result directory. Do not reuse or overwrite any failed run directory.

```bash
capacity_run_id="step-pref-capacity-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "results/$capacity_run_id"
test ! -e "logs/$capacity_run_id"
test ! -e "checkpoints/$capacity_run_id-zero2"
mkdir -p "results/$capacity_run_id" "logs/$capacity_run_id"
test "$(sha256sum data/TRAIN.jsonl | cut -d' ' -f1)" = 5bc1313a594f5713975b67388c745c6632275c37fd260055b27230c9cfedbed8
test "$(sha256sum data/TEST.jsonl | cut -d' ' -f1)" = 6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618
test "$(sha256sum configs/deepspeed_zero2.json | cut -d' ' -f1)" = 20615765b665d38647ee0447ca984ea28d16a1b8738363b505b054871bfc3f85
test "$(sha256sum checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827/config.json | cut -d' ' -f1)" = cbec86a55c91603e7f91d506d94c975113de3af4ef6e9cce6e124e6825bc7b44
test "$(sha256sum checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827/model.safetensors.index.json | cut -d' ' -f1)" = d28f59cc1060d05418f59a2320b4f53f83ba50374d7283d59ee559aefe35864b
./venv/ilvr-venv/bin/python scripts/select_jsonl_rows.py \
  --input-path data/TRAIN.jsonl \
  --output-path "results/$capacity_run_id/row-1078.jsonl" --index 1078
```

- [ ] **Step 2: Run the existing full one-row generator as the first nail**

Use the parent plan's generator command with language `sdpa`, vision `flash_attention_2`, BF16, `cuda:0`, `min_tokens=2`, and `max_new_tokens=32`. Tee the log. This deliberately reuses the real retained execution path instead of adding a separate probe script.

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/python scripts/generate_step_negatives.py \
  --data_path "results/$capacity_run_id/row-1078.jsonl" --image_root data \
  --stage2a_checkpoint_path checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827 \
  --output_path "results/$capacity_run_id/row-1078-negatives.jsonl" \
  --manifest_path "results/$capacity_run_id/row-1078-manifest.json" \
  --latent_size 8 --min_tokens 2 --max_new_tokens 32 \
  --attention_implementation sdpa \
  --vision_attention_implementation flash_attention_2 --device cuda:0 \
  2>&1 | tee "logs/$capacity_run_id/row-1078-generate.log"
```

Require exit zero, an authenticated schema-3 manifest, at least one valid pair, and a recorded mixed-backend settings identity. An OOM, backend mismatch, non-finite result, or authentication error stops the parent plan and returns to design.

- [ ] **Step 3: Resume parent Task 6 only after the first nail passes**

Generate row 2102 under the same fresh run ID, then continue Steps 3-8 of parent Task 6 using the fresh artifacts. Do not launch the full negative generation, ZeRO-2 gate, or any DoRA ablation arm before every applicable capacity gate passes.

- [ ] **Step 4: Record the runtime decision**

Append the fresh run ID, exact commands, exit codes, actual backend classes, generated-pair count, and elapsed generation time to the Task 6 report. Generation itself need not report peak memory. When parent Task 6 capacity probes run, append their peak-memory and finite-loss/gradient witnesses separately. Runtime evidence is the authority; the static class tests alone do not establish OOM readiness.
