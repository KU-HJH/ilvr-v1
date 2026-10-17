# Independent problem read — facts and constraints

This is a read-only advisory input to the root controller's macrothink pass.
Do not edit files, execute training/probes, invoke Paperthin skills, or spawn
agents. Do not read this task's design proposals or other agents' findings.
You are not an implementer, an approval authority, or a code reviewer.

## Underlying problem

An existing multimodal Stage1 training system selects visual supervision
targets from helper-image candidates. The user wants an optional additional
way to form the selection query, informed by reasoning history and prior
visual information. Existing default behavior must remain usable unchanged.
The supplied history-query documents motivate recent text and relevant past
visual content, but provide no runtime measurements or performance result.

The intended experiment concerns training-target selection. Student generation,
loss coefficients, optimizer, reference identity and evaluation must retain
their existing contracts. No accuracy or speed improvement is established.

## Observed code facts

Source base: /root/.codex/worktrees/1061/ILVR at
148b0e8ead14456d02cc595f0cc32a0752ee22e8.

- Stage1-v5 calls inherited input preparation and Stage1TeacherTargetBuilder.
- Current target builder makes one full teacher language pass per sample,
  then selects per-block helper candidates. Past selected targets were not
  injected into that teacher pass.
- Existing query modes average current/all reasoning text and selected raw
  visual feature history. Candidate ranking is cosine/top-p/top-k, with a
  variable number of selected rows per latent block.
- Existing sample metadata owns token positions, latent spans, original and
  helper image ranges, and helper-to-block mapping. The return is packed
  [N,H] target values and a boolean [B,T] mask.
- The teacher is an immutable initial effective DoRA model in eval/no-grad,
  with decay1. Student and teacher use Qwen2.5-VL with image MRoPE and GQA.
- Language attention uses SDPA; vision uses FlashAttention2. Effective Q/K
  projections are module calls. Production code need not return attention
  probabilities, and the old global-intent helper is an approximate pooling.
- v5 computes native CE/alignment, a clean student context, frozen reference
  and packed intervention contexts sharing one selected latent snapshot.
- Existing v5 checkpoints validate objective, reference, execution and source
  identity. They have no setting for the new query algorithm.

## Constraints and uncertainty

Use the current v5 DoRA/frozen-teacher/ZeRO1 profile, existing candidate and
loss paths, parser/launcher/checkpoint machinery, repository venv and storage.
All actual training, including smoke, requires two nodes/four ranks and a
separate approved execution. No new training is authorized by this design pass.
Avoid general orchestration, retries, unrelated refactors and dependencies.

No implementation of the requested query addition, attention observation or
new context construction has been run. Numerical faithfulness, actual cost,
memory use, benchmark effect and old-checkpoint compatibility are open.

## Requested independent read

Starting from these facts, identify the direction you would take and any
load-bearing mismatch between the real problem and a tempting implementation.
Prefer the smallest useful scope. Describe one cheap discriminating piece of
evidence; do not execute it. Mark which claims are static observations versus
untested assumptions. Return a compact read with enough rationale for the
controller to classify it. Do not seek consensus or assume the source
documents' proposed implementation is the only answer.

You may read these source files and relevant existing tests if useful:
src/stage1/targets.py, src/stage1/types.py, src/stage1/v5_trainer.py,
src/stage1/v5_resume.py, src/stage1/ema.py, src/peft_common.py, and the vendored
Qwen2.5-VL modeling implementation. Remain read-only and bounded.
