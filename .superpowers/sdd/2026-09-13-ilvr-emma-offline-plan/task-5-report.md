# Task 5 documentation report

Status: documentation complete; final root verification evidence pending.

Updated `docs/evaluation/protocols.md` with the actual EMMA-mini retry path,
preparation and scoring CLI usage, required run metadata, extraction precedence,
modified comparison rules, exact-long-content behavior, limits, diagnostics,
rounding, unsupported numeric words, multiple-choice wrong-label precedence and
duplicate-content limitation. It explicitly separates pure-helper comparison,
CPU contracts, model/GPU/full evaluation, and paper-protocol claims.

Updated `docs/evaluation/validation.md` from the existing artifacts. It records
400 unique rows, 614 byte-matched images, subject/type/format/slot counts, zero
exclusions, manifest SHA-256
`b41e7b210898417dfea6d48280737bc3e35c2133de183937fae1dc2ca805f961`,
the nullable-solution failure and focused 33-pass regression, and the final
23-case helper comparison (12 same selected values, 11 intended differences).
It distinguishes the same-value/nonfinite-status cases and preserves the
earlier prefix-bug comparison as diagnostic history.

Links and facts were checked against `prepare.json`,
`prepared-correspondence.json`, `actual-prepare-first-attempt.json`, the final
extraction comparison, Task 1–4 reports, the focused nullable regression log,
the checked-in config, and the implementation contracts. No preparation or
test command was run for this documentation-only assignment.

Pending: root's fresh reviewed-byte P1–P5 CPU/stub counts and final
preservation/Reference evidence. The validation document labels these pending
and does not promote the historical 167-pass run to final evidence. No model,
GPU, full 400-response evaluation, upstream full environment, per-model
first-nail, or paper-protocol verification is claimed.

## Documentation review fix round 1

The review found two overbroad protocol statements and one provenance wording
mismatch. The protocol now states that `\text{...}` unwrapping applies only to
boxed candidates, while `$...$` and `\(...\)` math-wrapper normalization applies
to every selected candidate. It also states that `empty_extraction` remains in
the affected row's `error`; the aggregate exposes only its `invalid_format`
status and does not create an `empty_extraction` status or comparison-diagnostic
key. Finally, the preparation example now reproduces the recorded successful
direct invocation of `scripts/prepare_eval.py` rather than the equivalent module
form.

These corrections were checked against `extract_answer`/`_finish`,
`emma_summary`, and `actual-prepare-r02-command.json`. Only documentation was
changed. Final reviewed-byte CPU/stub and preservation/Reference evidence remains
pending for the root controller's mechanical update.

## Final root verification evidence

Root's fresh final verification completed on the reviewed bytes. The scoped
P1–P5 CPU contract command passed 550 tests in 7.79 seconds, exit 0; it selected
the 24 listed evaluation test files and is not a full-project suite. The separate
controlled runtime-stub command passed 7 tests in 9.17 seconds, exit 0; it is not
GPU or real-model inference. Exact commands and exit codes are in
`results/evaluation/p5-final-20260913-01/verification.json`, with complete output
in `logs/evaluation/p5-final-20260913-01/cpu.log` and `runtime-stubs.log`.

The same verification found 5,026 protected files unchanged; seven Reference
roots passed root/origin/commit/clean checks and 33 evidence/license hashes;
frozen proposal v1/v2/v3 hashes were unchanged; the `data` and `checkpoints`
symlinks were preserved. A final `read_manifest` observation loaded 400 rows and
reconfirmed manifest SHA-256
`b41e7b210898417dfea6d48280737bc3e35c2133de183937fae1dc2ca805f961`.

P5 offline contracts and prepared-data correspondence are complete. Real model
responses, the model-specific extraction first-nail, GPU inference, a full
upstream environment run, 400-response model performance, and paper-protocol
verification remain not run (or false). This was a mechanical evidence update;
no tests, preparation, source, code, config, or protocol document was changed.

After the final extractor change, root repeated the bounded 23-helper comparison
on the reviewed bytes. The fresh r03 artifact and log completed with exit 0,
retained 12 same selected values and 11 intentional differences, and all 23
observation records are exactly equal to the post-prefix-fix r02 artifact. The
validation document now cites r03 as final while retaining r02 as historical
evidence. No additional test or preparation command was run for this update.
