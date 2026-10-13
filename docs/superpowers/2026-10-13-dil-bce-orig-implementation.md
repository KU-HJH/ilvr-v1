# SDD ledger — plan: docs/superpowers/plans/2026-10-13-dil-bce-orig.md

Approval: user `Subagent` approves the presented plan at e597354f09b9d4f2db4925cb48ff55e9157f8da4 and selects Subagent-driven. Plan SHA256 e699acb7432bb4d116ab1b9f64c7e911ddbe45026ed4593d3f27ea92b5310722. Written spec approved earlier at 112b09c. No training execution authorized.
Workspace: /root/workspace/ILVR/.worktrees/dil-bce-orig. Branch dil-bce-orig. No prior task records exist.
Skills executed: using-superpowers lifecycle reconciliation; using-git-worktrees existing isolation check; SDD setup/plan scan; requesting-code-review routing. TDD/test-quality read and required for implementers. Controller authors only control records; no product edits.

## Preflight task/interface scan

| Tasks | Producer versus consumer / shared file | Finding |
| --- | --- | --- |
| 1,2 | Numerical kernels vs config constraints | Both require orig fixed beta/margin and detached reference; no direct coupling |
| 1,3 | Clean/ref logits shapes | Replay produces P,V full vocabulary consumed by kernels; agrees |
| 1,4 | Frozen ref ownership | Loader output remains detached before numerical helper; agrees |
| 1,5 | OrigTokenComponents/CandidateSupport/entropy/mass and topk_entropy_loss.py | Task5 extends scalar-stat dataclass only; helper ownership stays Task1 |
| 1,6 | Probe same-forward numerical oracle | Probe uses literal expression independently; does not reuse kernel as expectation |
| 2,3 | Profile validation vs replay modalities | Unsupported profiles fail, native/PEFT supported; agrees |
| 2,4 | reference_identity binding vs ReferenceSnapshot | fingerprint+compatibility binds after snapshot verification; path separately stored |
| 2,5 | StepPreferenceConfig/identity reduction tag | reducer uses orig global microbatch step mean plus occupancy and fixed GAS; agrees |
| 2,6 | CLI fields/config/dil_identity | worker must retain explicit/omitted provenance; bind identity before Trainer save |
| 3,4 | CleanHistoryReplay -> FrozenCleanReference.score | same packet and replay helper; no separate rollout or student KV |
| 3,5 | model request/output; modeling_qwen2_5_vl.py | capture after student rollout; stats added in Task5; clean-first ordering retained |
| 3,6 | native/PEFT wrapper resolution | main passes separate owner, not model child; agrees |
| 4,5 | FrozenCleanReference -> Stage2Trainer | plain owner outside optimizer/DDP; agrees |
| 4,6 | snapshot/owner -> coordinator/main | one writer before workers; worker only validation; resume original provenance vs local paths |
| 5,6 | trainer.py reduction/metrics -> save/probe callbacks | Task6 observes raw steps without changing normal gradient path; numerical oracle separate |
| 1 | Internal tests/code/files | Step3 includes chunk API before Step4 chunk RED: reorder only this TDD substep as ruled below |
| 2 | Internal tests/code/files | CLI nullable fields and explicit null need explicit_fields; plan names it; matches identity tests |
| 3 | Internal tests/code/files | owned packet clones preserve live student graph; literal ref oracle independent; consistent |
| 4 | Internal tests/code/files | immutable copies versus relocated effective base: retain provenance bytes, resolve load path separately; consistent |
| 5 | Internal tests/code/files | Step4 raw reduction is covered by trainer RED; distributed counts added after Gloo RED in Step7; consistent |
| 6 | Internal tests/code/files | worker/probe RED precedes implementation; prose documentation does not earn tests; runtime launch excluded |

Ruling: Task1 writes chunk/checkpoint assertions before implementing orig_components_chunked (move that portion of Step4 into initial RED, or implement the chunk wrapper only after Step5 RED) — the TDD skill prohibits code preceding its failing witness; equations/interfaces and scope are unchanged — cost if wrong: only test sequencing/rework, no objective change.

## Task state

- [ ] Task 1: numerical kernels — dispatch next
- [ ] Task 2: config/identity
- [ ] Task 3: replay
- [ ] Task 4: reference artifacts/owner
- [ ] Task 5: Trainer/reduction
- [ ] Task 6: entrypoints/probe/docs
- [ ] Final independent review and fresh verification

Resource witness: nvidia-smi showed local H100 NVL GPU0/GPU1 at 0 MiB and 0% with no compute processes. Each full-suite runner rechecks before CUDA unit tests; no training or remote execution authorized. Existing canonical B200 wording is historical environment description, not permission to change packages.
