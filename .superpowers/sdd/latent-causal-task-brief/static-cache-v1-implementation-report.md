# Task 3 implementation report: static-cache-v1

Status: implementation ready for independent review. Scope and approval are bound to `static-cache-v1-task-brief.md` (root records approval “진행하자”); source baseline is `e0def77dc556946419406cc9719fb7e6c1d85e5a`. Implementer agent: `/root/static_cache_impl`. This report does not claim completion of root-owned independent review, full-suite verification, or GPU comparison.

## Implementation

- Added opt-in `--static-inference-cache`, strict boolean controller/eval boundaries, default false. Existing Stage 1 OFF + KV-cache gates remain prerequisites. Ordinary default rows/manifests omit the new field.
- Added repository-owned `evaluation.static_inference` context. It validates eval/no-grad once, temporarily overrides supported PEFT instances, and restores exact prior instance attribute presence/callables in `finally`, clearing constant dictionaries on all exits. No weights are merged or written.
- Standard installed linear DoRA computes `eye -> A(eye) -> B(...).T -> original get_weight_norm -> magnitude/norm` once on first use per input dtype. The original per-input LoRA, bias, base-result, multiplication order and dtype behavior remain intact. Only supported standard layers are selected at context entry; custom DoRA forwards and unsupported linear classes retain original behavior.
- Trainable-token embeddings reuse the original `get_merged_weights(active_adapters)` result, including prompt setup. Learned rows remain replacements. Non-renormalizing embedding layers are supported; sparse output-head arithmetic remains untouched.
- Original Qwen prompt MRoPE remains authoritative. Ordinary generated text uses integer offsets; physical cache offsets stay unchanged. Text coordinates derive valid-token count rather than blindly applying the pathological masked-position delta. Generated image/video/vision-start markers permanently select original full-prefix MRoPE for that sample, preserving original errors. Zero-budget generation never initializes MRoPE.
- Wired optional execution metadata through main-loop rows, shard manifests, merge validation and paired causal summaries. Absent means false; mixed modes and invalid booleans fail. Historical absence and explicit false compare as the same static execution mode. Execution metadata stays outside decoding/scoring profiles.

## Focused evidence and commands

All commands ran in `/root/.codex/worktrees/6b9d/ILVR` using repository Python, CPU environment `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=transformers/src:src:.`. No GPU/model-loading experiment, site-packages edit, subagent, training launch, or full project suite was run by this implementer. Root explicitly owns the latter suite and GPU evidence.

TDD skill and `writing-good-tests.md` were read before writing tests. The SDD implementer prompt was followed; root retains lifecycle control. Tests exercise real tiny Qwen, installed PEFT DoRA, trained token wrappers and repository sparse output rows with nonzero adapter/token/output deltas. Observers delegate to actual computations and restore original attribute presence.

1. Initial RED: `./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_static_inference.py -q` (CPU environment above), `red.log`: **34 failed**, expected missing controller/eval option, missing helper, metadata field rejection. No production code existed before this RED.
2. Initial focused GREEN: same environment, `-m pytest tests/evaluation/test_static_inference.py tests/evaluation/test_off_kv_cache.py -q`, `green.log`: **67 passed**. Earlier `green-attempt1.log` records fixtures attaching heads after `.eval()`; corrected test fixture mode rather than weakening production eval validation.
3. Edge RED: `-m pytest tests/evaluation/test_static_inference.py -q`, `red-edges.log`: **2 failed, 43 passed**. Original path accepts a zero-budget malformed unused vision prompt, while eager optimized MRoPE raised; absent=false mixed shard serialization was also incorrectly rejected. Both received targeted fixes.
4. Custom-forward RED: `-m pytest tests/evaluation/test_static_inference.py -k custom_dora_forward -q`, `red-custom-forward.log`: **1 failed, 47 deselected**, actual output differed by the dropped instance wrapper contribution. Unsupported custom forwards now retain original behavior.
5. Final focused GREEN: `./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_static_inference.py tests/evaluation/test_off_kv_cache.py tests/evaluation/test_stage1_v5_causal.py tests/multinode/test_evaluation.py -q` (same CPU environment), `green-focused.log`: **154 passed in 11.34s**, no warnings. `git diff --check` also passed. Intermediate `green-attempt2.log` retains a test observer restoration bug discovered by the exception witness; the test utility now preserves instance attribute presence correctly.

Logs are under `logs/stage1-v5-static-cache-20261015-r1/`. Hashes:

| Log | SHA256 |
| --- | --- |
| red.log | cc850504b99bb18e84a898811dcf3c13ca0f5ce92dc038eba343cb4565a4d4b5 |
| red-edges.log | 191a45bbd7fda01fe836b93a5fa412c1faa6648bb63f7bdaf5f9f6a36e0b9a6e |
| red-custom-forward.log | 95b5f849b6525ebe6a10851868c2dcfbbce77ab3ae6f4fd18a15d7edfe6d1fea |
| green-focused.log | 3f027640cf3f3b00b1ddab306874b76f67d827c42e4c434ddf6e55c7c8512f2e |

For four generated tokens/two DoRA modules, direct runtime observers witness original versus optimized: `torch.eye` 8 vs 2, norm preparation 8 vs 2, effective embedding materialization 5 vs 1, original MRoPE calls 4 vs 1. Tests compare exact FP32 logits, sampled suffix IDs, stop reasons, MRoPE positions and physical cache offsets for plain/adapted, left/right/unpadded text/image inputs, plus the one-valid-token text case. Additional witnesses cover exception restoration after populated DoRA/embedding caches, fresh generation after weights/prompt changes, unsupported custom linear/forward fallback, generated structural marker behavior, CLI/main forwarding and default metadata shape, strict booleans and mixed row/manifest/pair modes.

## Changed files and self-review

Requested changes are limited to `eval.py`, `scripts/analyze_stage1_v5_causal.py`, `src/evaluation/continuous_latent.py`, new `src/evaluation/static_inference.py`, `src/evaluation/stage1_v5_causal.py`, `src/multinode_evaluation.py`, new `tests/evaluation/test_static_inference.py`, and this report. Existing unrelated `__pycache__` directories are preserved. The original controller loop moved into a private worker solely so one context spans prompt setup through completion/exception; replay/training/latent-ON behavior was not intentionally changed.

Self-review found and fixed zero-budget eager MRoPE, historical absence/false comparison and custom-forward replacement issues, each with causal RED evidence. No unresolved implementation finding remains. Remaining evidence limits: CPU fixtures establish FP32 behavior on their covered shapes; production BF16 kernels, memory use, real checkpoint output identity and performance require root's planned source12 max32 GPU comparison after independent review. Fixed parameters, active adapters, devices and autocast configuration within a generation are scope assumptions. No accuracy claim or TEST100/TRAIN64 run is made.

Commit identity is the commit containing this report (obtain with `git log -1 --format=%H -- .superpowers/sdd/latent-causal-task-brief/static-cache-v1-implementation-report.md`). Root's subsequent record should bind review and fresh verification to that revision or any reviewed fixes.

## Review fix round 1 — Important I1

Review `/root/static_cache_review` at `4cc3539` identified a material scope deviation: the per-dtype DoRA cache retained the full converted base-weight matrix together with its factor, although the approved brief permits reusing only the factor. The prior self-review's clean statement did not detect this issue; the independent review supersedes it. Root accepted I1 and authorized only this targeted fix, regression and covering verification before scoped re-review.

Read and applied `superpowers:receiving-code-review`; checked the actual cache and installed eval path against the review. The normal eval path already supplies `base_result`, so its converted base weight is unnecessary after norm preparation. The cache now stores only `mag_norm_scale`. Norm preparation retains the original temporary `.to(x.dtype)` conversion and original arithmetic. The `base_result=None` branch obtains its own temporary converted weight before the original `F.linear` expression. No matrix is retained in the cache.

Added one causal mixed-dtype regression, `test_mixed_dtype_dora_releases_converted_weight_inside_active_cache_scope`: an actual PEFT DoRA module has FP32 base weights and FP64 adapters/input/magnitude; an observer of the real norm calculation records only a weak reference to its FP64 converted matrix. Inside the still-active static context, the test requires that weak reference to be dead after forward, checks exact outputs against the installed original for both supplied and absent `base_result`, then verifies factor preparation ran only once across repeated forwards. This establishes lifecycle behavior in the CPU fixture, not GPU allocation behavior or production BF16 equivalence.

Commands executed from the same worktree, with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=transformers/src:src:.`:

- RED before production edit: `./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_static_inference.py -k mixed_dtype_dora_releases -q` → **1 failed, 48 deselected in 9.24s**. Expected failing assertion: `converted base matrix remains resident during generation`, weak reference still pointing to an FP64 tensor. Log: `logs/stage1-v5-static-cache-20261015-r1/fix1-red.log`, SHA256 `3e00b729e3a58606dc9f45fc4fbc7ab2be21da419b4344343dd8ddaafd6a6a8f`.
- GREEN after targeted edit: `./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_static_inference.py tests/evaluation/test_off_kv_cache.py -q` → **82 passed in 9.61s**, no warnings. Log: `logs/stage1-v5-static-cache-20261015-r1/fix1-green.log`, SHA256 `b7ba3f340c358fa47e51c06028f3e04801506f3888493b052ae9f3a131f47370`.
- `git diff --check` passed. Self-review of the diff confirms only the DoRA cache value/fallback conversion, one regression, and this report changed from `4cc3539`.

Implementer `/root/static_cache_impl` ran no broad suite, GPU experiments or subagents. I1 is fixed in the commit containing this appended report; root owns scoped re-review and subsequent full-suite/GPU evidence. No remaining fix-round finding is claimed resolved beyond I1.
