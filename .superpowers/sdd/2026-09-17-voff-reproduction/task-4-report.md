# Task 4 — PARTIAL / BLOCKED (in progress)

## Process deviation discovered before final review

Required procedure: supplied external source is read-only evidence. Actual action: Python subprocess imports omitted bytecode suppression, causing source-tree `__pycache__/*.pyc` creation or refresh. Source `.py` files were not edited. Missing stage: read-only import preflight. This affects cache filesystem state; original cache ownership/preexistence is not established, so no cleanup/deletion is authorized by assumption. Later correction does not retroactively establish compliance.

Worker attempts touched caches in external `src/stage2` (initializer, peft, data), `src/step_preference` (initializer, spans, negatives, packing), `src` (peft_common, artifact_paths), and external Transformers initializer and Qwen prefix-cache/latent-rollout/sidecar modules. Full observed cache list is recorded in `logs/voff-reproduction-task4/external-cache-observation.txt`; an older inference cache may belong to another task. Root notified before further progression.

Proposed recovery: explicitly suppress bytecode before imports in both controller worker environment and worker entry point; add a causal regression witness that external file stats remain unchanged during CPU comparison; retain caches and source hashes. Full run remains blocked independently by arithmetic mismatch and absent distributed evidence.

## Recovery and final CPU evidence

Root recorded/reported the deviation and routed targeted recovery before further development. Both the worker and its subprocess environment now suppress bytecode. The existing Task 3 external-controller test loader also suppresses bytecode around its actual import, restoring the previous setting afterward. Two causal RED tests on copied source fixtures observed cache writes; both pass after the fix. Original external caches were neither deleted nor claimed owned.

All 10 records in the preexisting `external-source-identity.json` still match their pinned SHA256. `logs/voff-reproduction-task4/pinned-source-hash-check.json` records the comparisons. A final pre/post size-and-nanosecond-mtime inventory of 143 external files reports no changed files in `external-no-write-witness.json`. This verifies the final comparison's no-write claim; it does not undo the earlier deviation.

## Scope, revision, lifecycle

- Implementer: `/root/repro_witness_impl`; Task 4 delegated brief/global constraints read. No subagents or paperthin invoked.
- Code/test revision: `8574968` (`test(stage2): add bounded V-off source differential witnesses`), based on reviewed Tasks 1–3 revision `6c68aa7`.
- Skills executed: TDD and writing-good-tests (RED before witness functions; causal regression for rendering spelling and read-only import); systematic-debugging (dependency failure and cache-write causes); verification-before-completion (fresh focused suite and CLI evidence). This is a partial implementer handoff, not branch finishing. Root retains independent review and continuation.
- Controller clarified partial CPU-only handoff after observing parent arithmetic disparity. Production instrumentation and any training remain outside this completed portion. No coordinator, optimizer, loss, model, or launcher production code changed.
- `scripts/verify_voff_reproduction.py` is the bounded CLI. Default invocation with `--output-dir results/<fresh-id> --external-source /raid/workspace/ILVR/ILVR/.worktrees/ours1/source` writes raw current/external JSON and a report, and returns **2/BLOCKED**. It never starts training or marks full-run authorization true. Outputs reject paths outside this worktree's results tree and existing directories.

## Independent arithmetic specification and observed counterexample

Reference fixed before observation: average each example's supervised CE tokens; average that example's valid latent-step beta=.1 preference losses at weight1; then average those sums over the **actual** number of examples in the optimizer group. Intended eventual update clips the full effective gradient once at norm1 before existing AdamW. No Trainer scaling helper participates in the reference.

Hand-calculated RED/GREEN unit cases: at parameter2, two differently sized examples produce objective14 and derivative12; a one-example partial produces objective10 and derivative7. Doubled, omitted, and configured-full-group denominators fail gradient comparison. Nonfinite/missing gradients cannot pass.

The actual imported current `Trainer.training_step` and AST-extracted supplied `Trainer.training_step` body were each executed on CPU with heterogeneous scalar losses and a plain backward boundary. Current dependency globals are shared and hashed. With `model_accepts_loss_kwargs=False`, `compute_loss_func=None`, GAS4:

| local group | actual-example expected gradient | current observed | supplied observed |
| --- | ---: | ---: | ---: |
| four examples, coefficients1,2,3,4 | 10 | 10 | 10 |
| one example, coefficient1 | 4 | 1 | 1 |

Both supplied `scale_wrt_gas=False` at the backward call. Thus the parent step's partial contribution is one quarter of the independently specified actual-group mean in this controlled witness. **This is not actual DDP/DeepSpeed gradient or optimizer evidence.** Runtime flag resolution, complete Trainer-loop grouping/remainders, collectives, clipping, and optimizer deltas remain unobserved. No arithmetic correction was made. Parent compatibility and the independent objective are distinct claims: the parents match one another in this narrow witness, while both disagree with the specified partial-group reference.

## Actual TRAIN source-level differential

Final result: `results/voff-comparative-20260917-d/report.json`; raw current/external evidence in matching files. Earlier attempts `a`/`b` preserve failures; `c` is pre-recovery comparison and must not be presented as read-only-compliant.

Current production collator uses actual TRAIN/images and the approved initializer's real processor, concatenated-v1, latent8, START/END supervision, until_next_latent, visual-only attention-mask sidecars. Supplied data/parser/sidecar source functions run independently in a subprocess. Three distinct records cover one latent block, multiple latent blocks, and adjacent final text:

| TRAIN index | canonical sample ID | latent blocks | token length |
| --- | --- | ---: | ---: |
| 0 | a3bd9f091a93b45b23ac1bfaa5d72ca69ea3473c9d3e5f6080bc11706c90f46f | 1 | 212 |
| 920 | 0c5d21162264e65972f38e662d82b10c83d635d214c18ba0dab4da061db50f27 | 3 | 1086 |
| 1 | 25b2a76cabb9daa2920b685a6c2aa7d972c213e2af303d766d93f69cf3feffb9 | 1 | 309 |

Zero mismatches: input IDs, CE labels, chosen target IDs, branch input IDs, logical replay positions, clean-prefix extent, blocked latent intervals, selected score offsets, and full dense sidecar visibility shape/hash. This is representative data/pure-plan evidence, not all-retained-cohort validation or execution of Flex kernels.

Supplied Transformers is incomplete: importing it directly fails because `dependency_versions_check.py` is absent. The corrected subprocess loads supplied pure rollout/prefix/sidecar modules in an isolated namespace, supplied data/step-preference files from its source root, and current Transformers dependencies. Unrelated Stage2 package initialization is bypassed. Current cache utilities are explicitly shared; file hashes for imported Transformers scaffold modules and each compared supplied source file are recorded. The supplied parent method is extracted via AST without rewriting its body and executed against current parent globals. Therefore these are **source-level differentials**, not authenticated external-runtime or historical-run equivalence.

## Common-input loss and gradient boundary

CPU FP32 common-logit probe executes each tree's own score and context-preference functions on identical fixed logits, targets, and masks. Two examples have two and three supervised tokens. Current `mean_completion_logps` and supplied `mean_token_logps` are aligned through their documented causal shift; each uses its own actual scoring function. The combined per-example CE plus beta=.1/weight1 preference objective has total loss `2.3219823837280273`, preference loss `0.6943811178207397`. All compared clean/ablated scores, both losses, and every clean/ablated **logit** gradient agree exactly (max absolute error0). Both gradient branches are finite and nonzero.

No model forward is part of this probe. These gradients are **not production model parameter gradients**. Pure loss agreement cannot establish shared-prefix decoder connectivity, selective-checkpoint replay equivalence, RNG, DoRA behavior, or DeepSpeed updates.

## Verification commands and witnesses

- Initial arithmetic/record tests: `logs/voff-reproduction-task4/red.log` — 5 failures for missing witness implementation; initial GREEN5.
- Rendering selector causal RED: `rendering-red.log`; corrected supported selector after exact exception.
- External dependency isolation causal RED: `external-red.log`; corrected source-level scaffold, subsequently GREEN7.
- No-write causal RED: `readonly-red.log` — two copied-fixture failures. No deliberate RED writes to original external tree. `readonly-green-logits-red.log` — no-write tests pass, new common-logit requirement fails before implementation.
- Fresh final focused suite: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q tests/stage2/test_reproduction_probe.py tests/evaluation/test_continuous_latent.py` — **34 passed in13.93s**, full output `final-tests.log`.
- Fresh final CLI in `comparison-d.log` — exit2/BLOCKED, zero data mismatches; logit comparisons all pass; source file no-write inventory unchanged. Full command is the documented default CLI with output `results/voff-comparative-20260917-d` and supplied source above.
- `git diff --check` — clean. No training or GPU test was run. No full-project-suite or independent review claim.

## Unsatisfied Task 4 requirements / continuation interfaces

1. **Actual production-model common-input loss and all relevant trainable parameter gradients** versus supplied implementation, with controlled dropout and separately production dropout/replay determinism. Common-logit comparison above is only a bounded primitive witness; incomplete supplied package needs explicit source scaffold/dependency interpretation for a model-level comparison.
2. **Actual four-rank Trainer/DeepSpeed effective gradients and AdamW deltas** for full and final partial groups; at least one full group with nonzero LR, actual LR values, finite preclip gradients and once-per-update clipping. Existing `Stage2Trainer._install_deepspeed_gradient_adapter` wraps `engine.step`; it currently yields norms/counters, not per-parameter gradient/delta reference comparisons. No new observer or flag was added. A norm/counter cannot substitute for vector or delta evidence.
3. **Observed sampled IDs, group counts, and padding**, inferred from real runtime order. Need actual concatenated-mode retained filtering at clean maximum8192; retained3379 and update212 cannot be inferred from earlier manifests. A bounded 36-example cohort could supply two full16 groups and a final4 after balancing, but its actual order/padding must be observed; this remains a proposed fixture, not evidence.
4. **Longest-retained sequence/latent-block memory check**, no OOM and actual all-four-rank participation under approved production flags through the existing coordinator. No manual-rank launcher or single-node fallback.
5. **Contract resolution before full212 training**: parent CPU evidence exposes actual-group normalization disagreement. Controller must retain the blocker and resolve it without silently changing production semantics or redefining a failed reference as a pass. Actual distributed witness is still required; this parent-level result does not make it unnecessary.

Self-review found no production mutation beyond the explicitly requested test-loader bytecode guard. Test dependencies include repository data/initializer processor and supplied source; the targeted suite is not portable to a checkout missing those local evidence artifacts. External source caches remain preserved with original ownership uncertainty. Task 4 status is **PARTIAL/BLOCKED**, pending independent review and root continuation.
