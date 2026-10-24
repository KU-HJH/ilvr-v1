# Stage1 OFF Numerical Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit numerical policy for genuine Stage1 latent-OFF cache/no-cache evaluation and adopt a candidate only after the approved tensor-level GPU checks.

**Architecture:** One shared context applies/restores actual PyTorch precision controls after artifact loading; the evaluator and the existing paired probe use that same context. The standalone evaluator's one/four-shard paths record the numerical identity and reject mixed policies on merge. Historical behavior is the default; numerical candidates are not advertised as proven fixes before empirical acceptance.

**Tech Stack:** Repository Python3.11 venv, PyTorch2.7.1+cu128, PEFT0.17.1, patched local Transformers, pytest. No dependency or weight changes.

**Spec:** `results/kv-numerical-alignment-20261024T070941Z/design-commitment-v2.json` (SHA in sibling `.sha256`); approval `approval-v2.json`. The frozen proposal and critique beside it provide context, not authority overriding the final commitment.

## Global Constraints

- Keep actual KV use enabled in ON and disabled in OFF; no reference-per-token substitution, hidden fallback or token selection manipulation.
- Preserve all existing checkpoints/results and historical execution semantics; a numerical policy changing future values must be explicit and labelled.
- Scope: explicit Stage1 `continuous`, `disable_latent_generation=True` only. Original/native, Stage2, training and historical defaults remain unchanged. The existing coordinator has no latent-OFF interface: do not expand it in this task. Cover the existing standalone evaluator's actual one/four-shard and merge paths.
- Exact policy IDs: `historical`, `tf32_off`, `full_accumulation`, `math_sdpa`. Historical is an actual no-op. All new modes disable TF32 through PyTorch APIs; full_accumulation additionally disables BF16 reduced-precision reduction; math_sdpa additionally allows only SDPA MATH. Preserve model/adapter dtypes; no merge or full-model cast.
- Apply identical policy code after loading in both evaluator/probe. Preserve and restore incoming float32 matmul precision (including `medium`), BF16-reduction and SDPA states on success/exception. Never flip NVIDIA environment variables in-process or pretend environment restoration proves kernel restoration. Reject a forced external TF32 override that prevents the requested effective API state; record relevant controls without claiming a CUDA kernel identity.
- Root alone runs GPU, at most300 aggregate process seconds across all new trials/confirmations. Implementer/reviewers CPU only. Start source12/max_new_tokens32; TF32 then accumulation; SDPA only after Q difference is zero and the first remaining issue is downstream attention. Required extra sources0/3 also cap32. A failed candidate, early/incomplete required witness, or exhausted budget is not success.
- Acceptance for source12: complete32 suffix agreement, identical logical history, finite layer0 Q and final raw/post-penalty logits, zero Q and logit absolute differences at every compared prediction, actual expected cache geometry, and fresh policy-specific unobserved/observed fidelity. Never alter EOS or force extra tokens to fill the cap; insufficient source12 coverage is inconclusive. Additional sources retain normal EOS and report exact witnessed lengths.
- Always repository `venv/ilvr-venv`; data/cache under `data/`; results under `results/`; logs under `logs/`. Model-bearing pytest fixtures use an explicit unique basetemp under canonical `checkpoints/` and its parent must exist. No training, Reference imports, dependency installs or host/container changes.
- TDD has no exception. Every added behavior needs its causal RED before implementation. Root/agents must report any procedure deviation before progression, never retroactively claim compliance. One final full project CPU invocation after implementation/review-required fixes; no duplicate broad run for identical content. Independent task and whole-branch review remain required.

## Review Focus

1. State restoration from nondefault precision and exceptions: exact incoming settings return; legacy execution incurs no changed arithmetic.
2. A nonlegacy policy on original, Stage2 or latent-enabled input: fail globally before sample-loop output rather than publishing a spurious zero-accuracy run.
3. Historical absent/explicit numerical labels and mixed four-shard metadata: old files still merge; incompatible modes cannot silently combine.
4. Incomplete/missing/nonfinite traces or a near-tie token-only pass: require tensor-level evidence and honest inconclusive/fail status.
5. Runtime dtypes/controls and observer fidelity: actual module/tensor observations, not saved config or class names, supply evidence.

## File Map

- Create `src/evaluation/numerics.py`: strict policy validation, scoped controls, active-state/actual-parameter dtype observations.
- Modify `eval.py` and `src/evaluation/protocol.py`: explicit `--numerical-policy`, global supported-mode validation, shared execution context and results/runtime identity propagation.
- Modify `src/multinode_evaluation.py`: optional numerical field in existing generation-config validation, historical normalization and row/manifest/mixed-shard checks. Do not alter coordinator command construction or training.
- Modify `scripts/probe_stage1_off_kv.py`: same policy helper, fresh controls, source selection, actual operand dtype/layout observations where already available, and exact numerical acceptance.
- Create `tests/evaluation/test_numerics.py`; extend `tests/evaluation/test_kv_cache_probe.py`, `tests/evaluation/test_off_kv_cache.py` and `tests/multinode/test_evaluation.py` for owned behavior. Extend existing evaluator entrypoint tests only where the actual CLI/global validation seam requires it. No unrelated refactor.

---

### Task 1: Integrate one numerical policy and its causal witness

**Files:** All files in File Map, as a single coupled policy/evidence feature. Existing default results and runtime wire payloads must remain byte-compatible where they were absent/historical.

**Interfaces:**
- `evaluation.numerics.validate_numerical_policy(policy: str, *, protocol: str, stage: str, disable_latent_generation: bool) -> str` validates exact IDs and applicability.
- `evaluation.numerics.numerical_policy_scope(policy: str)` is the shared context manager; historical does not mutate state. Unknown strings/types raise; capture and restore the full incoming precision state, not just allow_tf32.
- `evaluation.numerics.observe_numerical_policy(policy: str, model=None) -> dict` reports active PyTorch controls and actual loaded parameter dtypes, with names explicitly distinguishing parameters from observed operands/kernel identity.
- `eval.run_one_example(..., numerical_policy: str = "historical")` consumes the policy separately from model generation kwargs; caller/result metadata include it only when nonhistorical.
- Standalone CLI `--numerical-policy` uses the exact IDs above. Nonhistorical runtime output has an explicit numerical identity and active observations; do not silently append fields to the strict historical `EvaluationRuntimeIdentity` dataclass. Reuse current execution-identity validation where possible.
- Probe CLI adds `--numerical-policy`, `--source-index` restricted to0/3/12 (default12), and explicit `--fresh-controls`. Existing frozen OFF/ON CLI remains valid by default. Fresh controls run the unchanged existing evaluator with no observer before the observed pair, using the same loaded model and same policy; retain both outputs and require each observed pair member to match its own control.
- Probe `alignment_verdict(off_trace: PassiveTrace, on_trace: PassiveTrace, off_result: dict, on_result: dict, *, expected_steps: int) -> dict` consumes paired traces/results and returns distinct finite Q-equality, logits-equality, token/history/coverage/cache evidence plus pass/fail/inconclusive; do not call every candidate a fix. Keep report-only observation equality separate from numerical acceptance.

**Exact evidence fields:** Active policy observations use `policy`, `float32_matmul_precision`, `matmul_allow_tf32`, `bf16_reduced_precision_reduction`, `sdpa_enabled` (math/flash/efficient/cudnn booleans), and `parameter_dtypes` (actual first Q base/active adapter tensors; empty when no model is supplied). Probe verdict fields are `status` (pass/fail/inconclusive), `q_equal`, `logits_equal`, `tokens_equal`, `coverage_complete`, and `finite`. Missing required data is inconclusive; nonfinite data or a witnessed tensor/token difference is fail. Source12 expected_steps32; extra sources preserve EOS and record their actual completed length/cap.

**Concrete test assertions owned by Steps1/3/5:** These are contract snippets; use existing real tiny-model/merger fixtures for the actual calls, without reimplementing generation.

```python
def test_full_accumulation_controls_and_medium_restoration():
    torch.set_float32_matmul_precision("medium")
    original_bf16 = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    with numerical_policy_scope("full_accumulation"):
        state = observe_numerical_policy("full_accumulation")
        assert state["float32_matmul_precision"] == "highest"
        assert state["matmul_allow_tf32"] is False
        assert state["bf16_reduced_precision_reduction"] is False
    assert torch.get_float32_matmul_precision() == "medium"
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction == original_bf16

def test_nonlegacy_policy_rejects_original_before_any_sample():
    with pytest.raises(ValueError):
        validate_numerical_policy("tf32_off", protocol="original", stage="stage1", disable_latent_generation=True)

def test_invalid_policy_type_is_not_coerced():
    with pytest.raises((ValueError, TypeError)):
        validate_numerical_policy(True, protocol="continuous", stage="stage1", disable_latent_generation=True)
```

The state-test fixture must restore the complete pre-test state even after assertion failure; add the same assertions around a deliberate body exception and math_sdpa context. Actual merger tests use valid matching fixtures then change only row/manifest numerical labels; expect rejection for tf32_off/full_accumulation mismatch, compatibility for missing/historical. Probe fixtures use real tensor events: equal token suffixes with one logit changed must yield `logits_equal=False,status=fail`; a missing event yields inconclusive; NaN/Inf yields `finite=False,status=fail`; an unequal-history child is excluded and cannot satisfy required coverage. A complete finite equal trace with actual expected cache geometry yields pass. These verdict assertions are independent examples, not assertions on mocked return values.

- [ ] **Step1 — Write causal RED for scoped controls.** Tests name `test_historical_policy_leaves_precision_unchanged`, `test_policy_applies_expected_controls_and_restores_medium`, `test_policy_restores_after_exception`, and `test_policy_rejects_unsupported_execution_and_conflicting_override`. Use real installed PyTorch settings/context and tiny modules; assert exact incoming/outgoing state and invalid combinations. Verify each missing behavior fails before adding it.
- [ ] **Step2 — Implement shared policy and state evidence.** Add only the interfaces above using installed PyTorch2.7 APIs. Snapshot actual parameter dtypes with no model loading/casts/saves. Scoped TF32 control occurs after the loader's existing True assignment; restore original precision exactly, including `medium`. Run the focused contracts GREEN.
- [ ] **Step3 — Write evaluator/metadata RED.** Exercise the actual argument parser, run_one_example and main-loop seams with existing tiny model/processor helpers. Required cases: default/historical compatibility, nonlegacy genuine OFF and ON propagation, global rejection before sample loop for original/Stage2/latent-enabled, and no policy leak into later historical generation. Test result/runtime active controls from real context. Do not break the previously repaired continuous error boundary by unconditional stage resolution in legacy main.
- [ ] **Step4 — Integrate explicit evaluator identity and merge validation.** Pass the keyword separately from generation kwargs; apply scope inside actual inference after model loading and record active controls before restoration. Existing historical payloads stay unchanged. Add optional `numerical_policy` validation/normalization to existing shard generation config and rows; reject unknown values and mixed numerical policies, including mismatched row/manifest labels. Missing and explicit historical are compatible. CPU four-shard tests use the real existing merger, not a new orchestrator.
- [ ] **Step5 — Write probe RED before extending it.** Tests: direct unequal-history truncation; finite/complete versus missing/NaN/Inf and token-only coincidences; fresh control fidelity; source12 default plus0/3 image resolution; exact policy use/restoration in a real tiny native cached/noncached trace. Existing reviewed observer behavior and default frozen CLI remain regression contracts. Carry prior unresolved stride/layout limitation honestly; no claim that class names reveal kernels.
- [ ] **Step6 — Extend the existing probe using the shared policy.** Keep original call/data/state paths and hook cleanup. Add only policy/source/fresh controls, actual operand dtype evidence and explicit numerical verdict. Record physical geometry separately from logical comparison. Require source12 complete32-token witness; do not force EOS. Candidate trial requires Q0 and exact logits0 plus matching complete suffix/history, not an arbitrary relaxed threshold. If Q still differs, SDPA is not a candidate repair. No GPU by implementer.
- [ ] **Step7 — Focused GREEN, self-review and independent task review.** Run policy/probe/OFF/evaluator/merge owned tests as one focused set with CPU-only env and explicit checkpoint basetemp. Capture full RED/GREEN logs and a stable source hash/diff package. Report file paths and any uncertainty; do not claim real-model parity. Only root dispatches independent task/final reviewers. Root routes fixes through the selected implementer and scoped re-review.
- [ ] **Step8 — Final CPU verification once on the reviewed code.** Create parent then run: `env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q -m 'not gpu and not distributed' --basetemp=checkpoints/kv-numerical-alignment-20261024T070941Z/final-cpu`. Save complete output to matching logs. Read all failures/warnings. Commit only reviewed authorized implementation/test files after green. Root verifies committed content hashes equal reviewed/tested bytes. Do not repeat identical full suite after green.
- [ ] **Step9 — Root GPU first nail and bounded confirmation.** After review/verification gates, run source12 profiles in order with one recorded aggregate300-second budget. Fresh unobserved controls and observed trace must reproduce under each same policy. Record actual cache use and active settings/dtypes. Try math_sdpa only if earlier Q equality holds and an attention boundary remains. Stop at first completely passing candidate; confirm0/3 within remaining budget. No pass if any required evidence missing; no trial/default promotion by declaration alone. Report new-profile versus old outputs separately.
- [ ] **Step10 — Claim-specific verification and branch handoff.** Persist approval, actual agent IDs, review verdicts/minors, content/commit identity, complete test evidence and GPU outcomes. Never erase prior diagnostic TDD deviation. If correction is empirically successful, finishing-a-development-branch applies; reuse the final project CPU witness for the overlapping test-pass claim and present human-owned integration choices. If no candidate passes, report the conditional design failed to establish a fix and return to diagnosis; preserve branch/results, no misleading completion or unapproved wider change.

## Self-Review and Handoff

Every approved requirement maps to Task1 steps1–10. Names/IDs agree across scope, CLI and metadata. The one feature is tightly coupled: a policy without accurate identity and a faithful witness cannot establish the requested fix. The five Review Focus cases have owned tests above. No new coordinator, retry/recovery system, default change, training or fullTEST400 is planned. Existing source12 diagnostic evidence/reviews at55437fd remain reusable only for identical content. FullGPU budget is separate from CPU test wall time.

Execution method remains subagent-driven. **Writing-plans requires human review of this plan before implementation; do not infer plan approval from the earlier design approval.**
