# Stage1-v5 Global Token DIL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan. The user already selected this method; do not ask again.

**Goal:** Compute Stage1-v5 DIL as the global sum of tokenwise BCE divided by the eligible DIL tokens in the actual optimizer window.

**Architecture:** Keep the current per-token kernel and native Trainer lifecycle. Prefetch/reduce token and original-block counts together, use independent DIL and entropy/mass scales, and bind new normalization to a versioned sidecar. Historical sidecars remain readable but cannot resume training under the new objective.

**Tech Stack:** Repository Python 3.11 virtualenv, PyTorch, vendored Transformers Trainer, Accelerate/ZeRO-1 runtime, pytest.

---

## Bound approval, workspace and execution

Approved commitment: `docs/superpowers/specs/2026-10-17-stage1-v5-dil-token-mean-design.md`, SHA256 `64d8a8474575e540f222514d9109836a2f2353ad5671fb4e672f0cea4075406e`.
User final response after the written commitment was presented: `진행해`.
Implementation base before this plan: `1f17c6d` (product source still matches `c9d6afe`).
Use `/root/.codex/worktrees/1061/ILVR`, existing branch `stage1_v5`, already isolated. Preserve canonical source, checkpoints and all unrelated/untracked files. No training, evaluation campaign, dependency install, merge, push or cleanup.

One atomic implementation task is appropriate: the objective's raw sum, window denominator, runtime contract and its verification consumers must change together. Do not publish an intermediate implementation with token sums and a block denominator. Root writes only workflow control artifacts; the implementer/fixer owns all requested code/tests/documentation changes.

All commands use the worktree root and this prefix:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python
```

Logs/results: `logs/stage1-v5-dil-token-mean-20261017/` and matching `results/`. Save red/green outputs and exact commands. Do not store model-bearing artifacts there.

## File responsibilities

- `src/stage1/v5_objective.py`: tokenwise DIL sums, unchanged block entropy/mass and sample Text-KL.
- `src/stage1/v5_reduction.py`: immutable window counts and the two distinct pre-parent scales.
- `src/stage1/v5_trainer.py`: one shared count collective, split composition/diagnostics, truthful runtime schedule identity.
- `src/stage1/v5_resume.py`: new reduction schema and strict historical inspection versus early training-resume rejection.
- `src/stage1/v5_config.py`, `src/utils_deepseed.py`, `src/stage1/composition.py`: use existing validation entrypoints as needed to reject unsupported fresh selectors and historical resume before model loading; no new public option.
- `scripts/verify_stage1_v5.py`: update affected count/report/contract consumers and scalar witnesses without changing their protocols or executing a training mode.
- `tests/stage1/v5/test_token_mean.py` (new) and affected objective/reduction/trainer/history/resume/config/protocol tests: independent numerical, gradient, entrypoint and contract witnesses. Update only assertions whose expected semantics changed; preserve tests of unchanged behavior.
- Existing user-facing documentation, only if needed to correct a presently described current-v5 denominator: add a concise current-behavior note without rewriting historical specifications/results.

## Task 1: Atomic token-normalized DIL and safe contracts

### Step 1: Write independent failing numerical/gradient tests

- [ ] Add a literal token-BCE reference and unequal original-block partitions. Use three target rows, first block length 2 and second length 1, and an empty block. Independent token reference:

```python
def literal_token_losses(clean, reference, targets):
    lp = clean.float().log_softmax(-1)
    lm = reference.detach().float().log_softmax(-1)
    r = lp - lm
    gt = r.gather(1, targets[:, None]).squeeze(1)
    return torch.nn.functional.softplus(-gt) + (
        lm.exp() * torch.nn.functional.softplus(r)).sum(-1)

# Existing objective must now return a token SUM, not a sum of block means.
expected = literal_token_losses(clean, ref, targets).sum()
torch.testing.assert_close(actual.dil_sum, expected)
torch.testing.assert_close(
    torch.autograd.grad(actual.dil_sum, clean, retain_graph=True)[0],
    torch.autograd.grad(expected, clean)[0],
)
assert ref.grad is None
```

- [ ] Add repartitioning invariance: identical token loss rows assigned to blocks of sizes `[2, 1]` or `[1, 2]` must have the same DIL sum and clean gradient. Also verify tokenwise softplus remains before averaging, and entropy/mass retain independent per-block mean gradients.
- [ ] Add tests requesting two independent window scales. A 4-rank, GAS4, 2-microbatch tail with 5 blocks and 11 tokens must yield DIL scale `16/11`, auxiliary block scale `16/5`; zero tokens gives zero DIL scale.

```python
w = V5Window(global_original_steps=5, actual_microbatches=2,
             configured_gas=4, world_size=4, global_dil_tokens=11)
assert v5_dil_scale(w) == 16 / 11
assert v5_auxiliary_scale(w) == 16 / 5
```

- [ ] Exercise actual parent `Trainer.training_step` dispatch, as the existing CPU reduction test does. Four logical ranks have unequal microbatch token counts, an empty rank, and partial tails. Independent objective:

```python
token_mean = torch.stack(independent_all_token_losses).sum() / total_tokens
auxiliary = entropy_weight * sum(independent_block_entropy_means) / total_blocks
native = sum(independent_native_losses) / (configured_gas * world_size)
expected = token_mean + auxiliary + native
```

Compare value and rank-averaged accumulated gradients, and assert native DeepSpeed dispatch still receives `scale_wrt_gas=False`. This is a CPU witness, not a GPU ZeRO run.
- [ ] Run the new focused tests before production edits. Expected RED: old objective gives block-mean sums or lacks the independent token count/scale. Fix test setup errors before accepting red evidence. Save `red.log`.

### Step 2: Implement the sums and independent scales

- [ ] Extend `V5Window` with required nonnegative integer `global_dil_tokens`. Update all construction sites explicitly; do not silently default missing counts to zero. Preserve existing validation of GAS/world/actual microbatches.
- [ ] Add this scale alongside unchanged `v5_auxiliary_scale`:

```python
def v5_dil_scale(window: V5Window) -> float:
    return (window.configured_gas * window.world_size / window.global_dil_tokens
            if window.global_dil_tokens else 0.)
```

- [ ] In `get_batch_samples`, build a two-element int64 tensor from the sum of original blocks and `len(batch_metadata.query_keys)` over prefetched batches. Reduce once across ranks and attach one identical immutable window to each local batch:

```python
counts = torch.tensor([
    sum(b['stage1_v5_metadata'].original_step_count for b in batches),
    sum(len(b['stage1_v5_metadata'].query_keys) for b in batches),
], device=self.accelerator.device, dtype=torch.long)
global_counts = self.accelerator.reduce(counts, reduction='sum')
window = V5Window(int(global_counts[0]), len(batches),
                  self.args.gradient_accumulation_steps,
                  self.accelerator.num_processes, int(global_counts[1]))
```

- [ ] In `compute_v5_loss_sums`, replace only DIL accumulation:

```python
chosen_sum = chosen_sum + d.chosen.sum()
reference_sum = reference_sum + d.reference.sum()
```

Keep entropy/mass block means, Text-KL, detached reference, token filters and the kernel unchanged. Empty logits must remain connected to clean graph zero. Record token counts with existing scalar diagnostics; identify sum units in versioned contract metadata rather than inserting strings into tensor-only logging.
- [ ] In trainer composition use:

```python
dil_scale = v5_dil_scale(window)
block_scale = v5_auxiliary_scale(window)
loss = (self.ce_weight * ce + self.sim_weight * alignment
        + config.text_kl_weight * sums.text_kl
        + dil_scale * config.lambda_preference * sums.dil_sum
        + block_scale * (config.lambda_topk_entropy * sums.entropy_sum
                         + config.lambda_topk_mass * sums.mass_sum))
```

Add DIL token denominator/scale to existing debug diagnostics. Keep parent backward, stochastic execution, checkpointing and optimizer paths unchanged.
- [ ] Re-run the focused tests to GREEN; include empty-window and chunk-boundary cases. Save `green-reduction.log`.

### Step 3: Write failing contract/entrypoint tests, then implement safe boundaries

- [ ] Add original schema-1 fixtures independently of the new builder. Assert `load_v5_contract` still strictly reads them, while training validation rejects them with an explicit block-mean/fresh-run explanation. Test both parser and direct composition with model-loader spies that must never be called on rejection.
- [ ] Add new schema roundtrip, unknown/malformed reduction identity, new resume conflict and unsupported fresh selector tests. New token-mean training supports only `selection=all`; test all four other selectors for pre-model rejection. Read-only old contract parsing still permits its original valid selector fields.
- [ ] Run these tests RED before implementing the boundary. Save `red-contract.log`.
- [ ] Preserve the public sidecar loader and builder signatures. Split validation/reconstruction internally by supported schema. Schema 1 validates against the exact historical semantic dictionary; schema 2 records a distinct token-DIL/block-auxiliary identity. A common private payload helper may take the schema explicitly; do not create a legacy training engine.

```python
# In training resume validation, after strict sidecar validation:
if saved['schema_version'] == 1:
    raise ValueError('block-mean Stage1-v5 checkpoint requires its original '
                     'reviewed code; token-mean training requires a fresh run')

# At fresh training argument/composition validation, before any model load:
if config.selection != 'all':
    raise ValueError('token-mean Stage1-v5 training currently requires selection=all')
```

Do not put training-only rejection in `load_v5_contract`: historical artifact/reference validation depends on that read API. Do not relax source/package/reference/schedule checks. Add a new explicit DIL token denominator to runtime schedule while retaining block denominator semantics for entropy/mass; ensure consumers validate both.
- [ ] Run the focused contract/entrypoint tests GREEN; save `green-contract.log`.

### Step 4: Align consumers and complete the focused integration check

- [ ] Update verification-tool window/report fixtures and affected tests to supply token counts and use the two scales. The actual loss formula and contract must drive the witness; never hardcode success or merely rename an old block witness as a token witness.
- [ ] Run the v5 focused suite once after integration:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python -m pytest tests/stage1/v5 -q
```

Expected: all focused tests pass, no formerly valid historical sidecar read becomes a silent new objective, and no model/training launch. Include affected external parser/composition unit tests if outside this directory.
- [ ] Run a CPU inspection of both real historical `checkpoint-648/stage1_v5.json` files with `load_v5_contract`, then assert `validate_v5_resume(saved,{})` rejects before model work. Preserve files and hashes.
- [ ] Self-review diff against the approved commitment and this plan. Record exact tests/red-green outcomes, changed paths and limitations in the implementation report. Commit only requested implementation files and tests; root owns the workflow ledger.

## Required task review and final gates

- [ ] Root dispatches fresh independent spec review with full task/spec text and exact diff SHAs. Fix via the implementer/fixer agent and re-review if needed.
- [ ] After spec PASS, dispatch independent quality review using the requesting-code-review template. Fix/re-review required findings via SDD; no controller implementation fixes.
- [ ] Dispatch the mandatory final reviewer for the whole implementation; preserve unresolved findings as unresolved. Do not repeat broad review cycles beyond required gates.
- [ ] Before selecting final verification, invoke finishing-a-development-branch and verification-before-completion. This task completes feature work in an existing named-branch linked worktree, so finishing applies. Run the project full suite once on the final reviewed revision and reuse its result for both gates:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q
```

Read complete output and exit status. Also run distinct historical-sidecar/source-diff checks required by claims. No real GPU/multi-node training or generation improvement is claimed. Preserve worktree; human owns integration/keep/discard decisions. Do not repeat a known keep-worktree preference as a new cleanup approval flow.

## Plan self-review

Spec coverage: formula/weighting/empties→Steps1–2; prefetch/tails/other terms→Steps1–2; diagnostics/runtime consumers→Steps2/4; new schema/history reads/rejections→Step3; review/verification/no-training→final gates. Repartitioning and actual parent dispatch explicitly test the root first-nail concern. Types are consistently `global_dil_tokens`, `v5_dil_scale`, existing block scale and unchanged token kernel. One implementation owner avoids mixed intermediate semantics. No unresolved placeholders or new orchestration.
