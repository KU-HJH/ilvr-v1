# Stage1 standard loss-term logging — design commitment v1

Status: final v1 approved by user response `수정안 v1 승인, 진행` to the exact CE/alignment/raw/weighted/total interval-logging question. Native implementation and independent review authorized; active training must continue unchanged.

Understood as: correct the missing per-term logging in the current standard Stage1 EMA baseline. The same standard trainer also serves fixed-teacher baseline. This does not request changes to V5/Stage2 objectives or a fresh training experiment.

## Evidence and approach

Source revision: 01518f4e2903f36256911bc7735c3430e762b922.
`composition.py` enables standard diagnostics only for benchmark evidence; `trainer.py` additionally disables them under production policy. CE/alignment/total metrics are stored and emitted only by that diagnostic path. The existing diagnostic buffer holds the latest microbatch, whereas Trainer's `loss` averages accumulation-scaled losses across ranks and the logging interval. Rank0 runtime logs contain aggregate loss only.

Use a small independent loss accumulator in the standard trainer. Enabling all diagnostics would violate the approved production check policy; simply printing the latest microbatch would give a different statistic from interval loss. Neither is a suitable alternative. Native implementation is the user's existing selected method.

## Frozen proposal v1

- Record raw CE and alignment, their weighted contributions, and total at the existing training logging cadence. Use the same rank averaging, accumulation denominator and optimizer-update interval as Trainer's finite aggregate loss, including partial epoch tails. Field names distinguish weighted contributions.
- Accumulate detached scalar tensors; synchronize and transfer metric values only at normal log events. No extra model forward/backward, target scans or per-microbatch host reads. Production diagnostics remain off.
- Only training calls enter this accumulator. Evaluation/log-only events must not consume it. Preserve existing diagnostic behavior without allowing last-microbatch diagnostics to overwrite interval loss metrics. Record any final partial interval with explicit interval metadata rather than mixing it with whole-run train_loss.
- A resume starts its metric interval at the restored global step. Never infer or backfill earlier CE/alignment values from aggregate loss. Preserve training objective, gradients, EMA update and checkpoint semantics.
- Correct and independently review isolated source before deployment. Do not hot-patch the running Python process. Current run was observed at 1865/2160; let its authorized training/evaluation continue while preparing this change. Applying to an already-running process requires a separately explicit checkpoint restart decision; otherwise use the corrected source on the next launch/resume.

## Hate critique and synthesis

Frozen proposal above was reviewed unchanged.

{ "root": "Recording terms is misleading if their rank, gradient-accumulation or interval denominator differs from the existing aggregate loss, especially across epoch tails or resume.", "first_nail": "A small CPU Trainer witness with known unequal microbatch losses, non-unit weights and an incomplete accumulation tail must reproduce aggregate loss from weighted terms; repeat after a restored global step and verify evaluation does not consume the buffer." }

First nail is proposed, not executed: implementation tests follow approval. There is no costly or state-changing premise experiment. The arithmetic expectation is testable before deployment and no runtime equivalence is currently claimed. No material design changes or unresolved competing approaches arose from critique. Macrothink is not triggered for this bounded, diagnosed logging defect.

## Authorized work and remaining gates

User explicitly approved this precise v1 in the asynchronous question reply after its linked presentation. Approval excludes interrupting the active experiment, merging or pushing. Using-superpowers, readchk, systematic-debugging, brainstorming and hate have been applied to intake/diagnosis/design. This control artifact does not claim implementation, independent review or verification completion.

After approval: bounded native task brief, causal RED test, minimal correction, focused CPU training/logging and gradient-parity tests, independent specification/quality review, fresh completion verification and applicable branch handoff. No new full architectural plan or extra GPU training is needed merely to validate logging. Preserve active experiment artifacts and automatic evaluation. Do not claim missing historical term losses are recoverable.
