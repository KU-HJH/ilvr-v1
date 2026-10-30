# Bounded native task brief: Stage1 EMA compatibility

Spec: docs/superpowers/specs/2026-10-30-stage1-ema-baseline-design.md
Approval: user `ㅇㅋ` after final design v1 presentation; implementation, tests, independent review and experiment authorized within that design. Native method inherited and retained.
This is the concrete task brief required for bounded work, not an architectural plan. No additional writing-plans phase applies under the user workflow's conditional trigger.

## Task 1: Support native EMA in matched standard fresh and resume entrypoints

Scope: src/stage1/matched_execution.py, src/stage1/matched_resume.py, and focused tests. Reuse native EMA; preserve all other profile constraints and historical markers. Invalid decays must fail, saved EMA must remain exact on resume.

1. Extend tests first: fresh acceptance of valid decays, invalid rejection, saved EMA tau0.999 contract, frozen/EMA drift rejection, native matched trainer all-token tau0.999 uninterrupted versus resumed teacher/optimizer/RNG equality and teacher mutation. Expected RED: valid non-frozen fresh and resume rejected by existing fixed-teacher guards.
2. Reuse native decay validation at both profile gates without rewriting EMA or broadening V5. Expected GREEN: targeted new tests and existing matched tests pass.
3. Run task-done with tests/stage1/test_matched_execution.py and tests/stage1/test_matched_resume.py under repo venv and project PYTHONPATH; temp model artifacts under checkpoints. Expected: all pass.
4. Commit implementation and tests with RED/GREEN evidence; independent whole-diff specification/quality review. Review focus: runtime callback dispatch, decay exactness, saved teacher restore, invalid values, historical frozen semantics, no V5 relaxation.
5. Apply required verification-before-completion + finishing gate with project pytest -q once after review; record full output. Distinct supported integration witness checks actual EMA tau0.999 mutation and checkpoint continuation before full launch. No merge or cleanup is implied by experiment authorization; keep runtime worktree for execution.
6. Dry-run and preflight exact approved training profile on both nodes; run existing Stage1 IB coordinator and automatic original/native KV-ON TEST400 then paired analysis. Do not start full training on failed critical validation.
