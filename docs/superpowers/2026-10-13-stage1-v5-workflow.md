# Stage1-v5 workflow and scope continuation

Date: 2026-10-13. Active phase: architectural brainstorming, section review.
This is a controller record, not an implementation-authorizing specification.

## Scope and approvals

Understood as: continue DIL-BCE-ORIG in fresh Stage1 training, preserve alignment,
omit Gain, add independent V-off top-k entropy and existing-style Text-off
matching KL, then train and automatically evaluate the final artifact. The
latest requested workspace/branch name is `Stage1-v5`. The selected implementation
method remains Subagent-driven. No new training authorization is required.

Explicit user decisions in this continuation:

1. `유지하면서 추가`: retain Stage1 alignment and add DIL.
2. `0.1 * (L_ce + L_alignment) + dil-bce-orig with reference`: CE and alignment
   each have coefficient 0.1. This supersedes retaining the old alignment weight.
3. Omit Gain; include V-off entropy and Text-off matching KL.
4. `두 OFF branch를 묶어 총 4회 (권장)`: approve the proposed call structure:
   helper teacher, clean student, frozen clean reference, packed independent
   V-off/Text-off branches. There are five semantic contexts in four decoder
   calls at microbatch 1 with eligible auxiliary blocks. Checkpoint recomputation
   and vision encoder execution are outside this call count.
5. `k=50, entropy 가중치 0.1`: explicit top-k and entropy coefficient. The question
   also presented DIL coefficient 1 and existing Text-KL temperature 2/coefficient
   0.01. No Gain update or projected optimizer is part of this experiment.
6. `Text-off는 구현되어있나? Stage1-v3을 참고해라. 그리고 Stage1-v5라는
   worktree를만들어 거기서 진행하자`: reference Stage1-v3 and move active work.

The pending section question covers preserved Stage1 data/helper/CE semantics,
reference reuse, immediate-text targets, independent branch masks, verification
and the two-node training/TEST400 execution sequence. The workspace correction
does not itself approve that entire section or an unwritten specification.
The earlier Stage2 design/plan remains superseded. Reviewed numerical kernels
remain reusable; no Stage2 integration was implemented.

## Worktree and evidence continuity

- Native tool `mcp__codex_worktrees__create_worktree` was available and used with
  `ref=Stage1-v5`; operation `ae0b60dd-c8e9-4c29-9e11-2b84c16cff56` completed.
- Managed workspace: `/root/.codex/worktrees/6b9d/ILVR`, branch `Stage1-v5`.
  Base is `9119b55cc717d8dfe9fc047cad128d90074e3faa`, the unchanged prior feature
  revision including reviewed kernel commits `5debf2a` and `c52fc56`.
- Original workspace `/root/workspace/ILVR/.worktrees/dil-bce-orig` is preserved.
  Its ignored `.superpowers/sdd/2026-10-13-dil-bce-orig/` review records and
  `logs/dil-bce-orig-t1-20261013t163441z/` remain there. Relative historical log
  paths in the inherited workflow records refer to that original workspace.
- New workspace uses symlinks to canonical `data`, `checkpoints`, and `venv`.
  New logs/results remain inside this managed workspace.
- User explicitly requested a new worktree, overriding the skill's ordinary
  preference to reuse existing isolation. No existing worktree was moved,
  removed, overwritten, or reset.

Fresh baseline command in the new workspace:

```bash
PYTHONPATH=src:transformers/src CUDA_VISIBLE_DEVICES='' ./venv/ilvr-venv/bin/python -m pytest tests/step_preference/test_dil_bce_orig_loss.py tests/step_preference/test_topk_entropy_loss.py -q
```

Result: exit 0, **70 passed in 1.77s**. Full output was read from
`logs/stage1-v5-setup/kernel-baseline.log`. This is a numerical-kernel baseline,
not a full project suite, Text-off integration test, or four-forward witness.

Inherited product SHA-256:

- `src/step_preference/dil_bce_orig_loss.py`:
  `5c6bd9db00468220420ac1179934f3a3ed821f2fbaaa2d3c8a5b4ca56e61b256`
- `src/step_preference/topk_entropy_loss.py`:
  `e2f49f027eca3a5215fe427071bfad8197a8a9d677f052076a2ef8f235f883ff`

## Stage1-v3 source investigation

Read-only source: `/root/.codex/worktrees/541e/ILVR`, branch `stage1-v3`,
commit `e555a2b6ddf43d689cceca7342fa3fb6a17b90c3` at inspection. That workspace
has unrelated untracked files; inspected tracked source paths had no diff.
It is not modified or imported as a runtime dependency by this work.

Actual call path: Stage1 `compute_loss` builds helper latents, executes ordinary
clean F, then `compute_v3_gradients` runs a separate alias clean trace, optional
frozen clean reference for KL selection, packed R/RV, and explicit Gain/KL VJPs.
Consequently this code is a source for the Text-off semantics, not an already
implemented Stage1-v5 four-call path. The current DIL worktree has none of that
Text-off integration yet.

Observed reusable semantics:

- `v3_metadata.py`: source-item immediate text targets and previous assistant
  reasoning candidates, token-offset roundtrip and crossing-token exclusion.
- `v3_selection.py`: detached full-vocabulary student-to-reference KL at T=2,
  sample-centered rho 0.3--0.5; rank text keys by original causal attention mass;
  cap 0.3; low-mass threshold 0.1/fallback ratio 0.1; no image masking.
- `v3_objective.py:text_kl_loss`: T-squared KL(detached ordinary F || R), query
  mean within valid sample then sample mean. Current fresh weight 0.01 and no
  KL warmup; LR warmup remains separate.
- `stage1_v3_trace.py`: actual prepared embeddings/positions, attached clean
  prefix K/V and detached attention observation, with checkpoint bindings.
- `stage1_v3_replay.py`: packed independent prefixes and source-coordinate
  dropout. Existing R/RV both mask text; Stage1-v5 requires a distinct V-off
  branch with unmasked text and a Text-off branch with visible latent body.

Inspected source SHA-256, respectively:

```text
f963bf5f3e6b4d9cb4cde69296f796cd0bd5796942b6f86e82511210cb6c345c  src/stage1/v3_metadata.py
999363ff3f6968e597946cee49eb08bf1b22fa106d6a8df54509ad9b0c9fb0da  src/stage1/v3_selection.py
ff7a07625b3aed347b5e4b643b155fa0cb63645610401b63183ce36a2698f344  src/stage1/v3_objective.py
f9624a57604fbe29bbac961d1b496caf7d2e1e68c2d94c6917376c83d4864d9f  transformers/src/transformers/models/qwen2_5_vl/stage1_v3_trace.py
47770180105bcde3d0f03294d7be3b6f8cd839aca9507d9d2e51fb37985f24f8  transformers/src/transformers/models/qwen2_5_vl/stage1_v3_replay.py
```

## Remaining gates

Skills actually applied in this continuation: using-superpowers, readchk,
brainstorming context/questions/approach-and-section review, using-git-worktrees
creation/setup/focused baseline, verification-before-completion for the limited
baseline claim. No implementer or reviewer was dispatched for new integration.

Next: finish section confirmation, write/self-review revised architectural spec,
freeze its proposal, run root-only hate and applicable macrothink on identical
bytes, resolve material premise uncertainty with separately authorized evidence
or an explicit waiver, then present the exact commitment for written approval.
Only afterward create/review the revised plan and resume Subagent-driven TDD,
independent review, verification and the applicable branch finish gate.

Production scope remains one fresh run from original Qwen, canonical common
defaults plus the explicit loss overrides above, followed by final-artifact
preparation and TEST400 automatic evaluation. Derive the actual Stage1 update
budget and verify compatible code/environment and available resources on both
nodes before launch. No training, evaluation or new premise probe has run.
