# Authorized Stage1-v5 design-premise probe

This is the bounded experiment the user separately approved with
`CPU gradient 동등성 검사 실행 (권장)`. It is not the Stage1-v5 product plan.
The outer architectural brainstorming phase remains active. Product integration
requires final spec and plan approval later; experiment code is throwaway.

## Global Constraints

- CPU only; use this worktree's `./venv/ilvr-venv/bin/python` with
  `PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES=''` and
  `PYTHONPATH=src:transformers/src`. Do not load real pretrained checkpoints,
  download/install dependencies, launch training, or create model-bearing files.
- Work only in `/root/.codex/worktrees/6b9d/ILVR`. Product `src/`, `transformers/`
  and `tests/` are read-only for this experiment. Do not modify other worktrees.
- Probe source and execution logs belong in `logs/stage1-v5-premise/`;
  result metadata belongs in `results/stage1-v5-premise/`. Explicitly label the
  code throwaway, not a product implementation. No retry/recovery framework.
- Use actual patched tiny Qwen/DoRA and native RMSNorm, actual autograd and
  non-reentrant checkpointing. Do not substitute toy linear attention or stub
  out a gradient path whose equivalence is the point of the probe.
- No subagents, Paperthin invocations, broad test suite, or product-code changes.
  TDD applies to the small experiment adapter: assertions/oracle first, observed
  expected RED, then minimal candidate adapter and focused GREEN/falsification.
- A failed premise is a useful result. Do not weaken tolerances/oracle, hide
  failed cases, or redesign the architecture until it passes. Diagnose harness
  mistakes minimally; report genuine unsupported graph behavior to root.

### Task 1: Fused-loss shared-F graph versus independent masked calls

Question: can one ordinary clean F graph support its own CE/alignment/DIL and
attached packed V-off/Text-off gradients in one ordinary backward, matching
independent full-sequence masked student calls under the same parameters,
positions, helper-latent snapshot, dropout identities and loss definition?

Create a compact pytest-based experiment, suggested files
`logs/stage1-v5-premise/test_probe.py` (independent literal oracle/assertions) and
`logs/stage1-v5-premise/probe_adapter.py` (candidate capture/packing adapter).
Use existing tiny Stage1 fixtures and reviewed numerical kernels where helpful.
Read source only as needed: current `src/stage1/peft.py`, `equivalence.py`, the
patched Qwen shared-prefix API, and pinned Stage1-v3 tracing/replay at
`/root/.codex/worktrees/541e/ILVR` commit
`e555a2b6ddf43d689cceca7342fa3fb6a17b90c3`. Do not import that worktree at runtime;
copy only narrowly necessary experimental mechanisms into the probe directory,
recording their origin. Never change the frozen proposal to fit the experiment.

Fixtures must have unequal block/target lengths, nontrivial previous text,
multiple independent blocks, END as the first target predictor, and vocabulary
large enough for top50. Use fixed detached helper latent inputs, fixed text-key
selections, fixed candidate IDs and fixed alignment noise for comparisons.
Set nonzero DoRA/adapter effects so ignored parameter paths are detectable.

Candidate: one clean F captures attached prefix K/V and real inputs/positions;
separate frozen clean reference reads the identical realized history with its
own weights/K/V; one packed student decoder call evaluates V-off and Text-off.
V-off only blocks current latent-body keys throughout the replay suffix.
Text-off blocks selected previous text keys at scored queries, leaving latent
body available. All losses use optimizer-owned parameters and one backward:

```text
0.1*CE + 0.1*alignment + 1.0*DIL + 0.1*top50_entropy + 0.01*Text_KL
```

Use the real shifted CE and Stage1 cosine-alignment formula with fixed noise;
DIL is exact full-vocabulary chosen/reference BCE. Text-KL is
`4*KL(detach(softmax(F/2)) || softmax(Text_off/2))`. Fixed candidate IDs are
selected once from the ordinary candidate execution, then reused unchanged
by the independent oracle and negative controls. Preserve all trainable DoRA,
sparse input/output token-row and live non-helper input/prefix gradients.

Oracle: separately assembled full causal masks and full-sequence model calls
for each OFF branch/block; do not call the candidate mask/packing routine or
reuse its K/V. Compare clean/reference/OFF selected outputs, every loss term,
combined loss, every trainable parameter gradient and live prefix-input gradient.
Use cloned identical models and source-coordinate dropout realization, without
replacing production normalization arithmetic. State numeric tolerances before
the GREEN run and report maximum errors; unexplained relaxation is prohibited.

Cover dropout 0 and 0.05, checkpoint off and non-reentrant on, and early/full
checkpoint recomputation if the existing mechanism distinguishes them. Compare
real trainable parameter identity and reference effective state before/after
backward and one tiny in-memory native optimizer update. Reference remains
eval/frozen and receives no gradients; helper tensors receive no gradients.

Record original decoder-call hooks separately from recomputation. The critical
new subgraph has one F, one reference and one packed call. If using fixed helper
fixtures rather than the real Stage1 target builder, explicitly say the helper
pass/full four-call trainer witness is outside this probe; do not add a fake
helper call to claim end-to-end coverage. Production integration will verify
the real helper and overall four-call sequence.

Require causal negative controls: detach the reused prefix and show the gradient
oracle detects the difference; contaminate V-off with the Text-off mask and show
output/loss or gradient mismatch. They must fail their intended equivalence
assertion, not merely throw an unrelated exception. Keep failures as evidence.

Write concise machine-readable `results/stage1-v5-premise/result.json` with
case outcomes, commands, tolerances/max errors, call counts, reference and
gradient boundaries, negative controls and limits. Keep full stdout/stderr in
logs, not copied wholesale into the report. After self-review, commit only the
small probe `.py` source and result metadata (explicit `git add -f` for these
owned ignored paths is allowed); do not commit large logs or unrelated files.
Provide source/result SHA-256 hashes for audit and the exact focused command.

Return DONE for a trustworthy result even if it falsifies the premise, with
the falsification clearly identified; DONE_WITH_CONCERNS if the probe's own
validity is uncertain. Independent review of both probe specification compliance
and quality follows before root uses the result as evidence. Main feature-wide
review/finish remain later outer-lifecycle gates, not claims of this experiment.
