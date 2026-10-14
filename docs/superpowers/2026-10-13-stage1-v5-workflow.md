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

## Section approval and proposal preparation

The user replied `진행해` to the final section question about writing the revised
design with Stage1-v3 Text-off reuse and Gain omitted. This completes the
conversational section-review prerequisite and authorizes written-spec creation,
not approval of a previously nonexistent spec or implementation plan. Earlier
pending-section statements above are historical.

Root authored `specs/2026-10-13-stage1-v5-design-proposal-v1.md`, incorporating the
confirmed formula/call structure/k=50 and existing execution authorization.
The next phase is inline spec self-review, proposal freeze, root hate and
macrothink fresh reads under the standing AGENTS invocation. These advisory
checks are warranted because a wrong graph/reduction premise would invalidate
costly distributed training and its evaluation. No integration implementation
has started.

Root self-reviewed all ten proposal sections for placeholders, inconsistent
coefficients/roles, scope drift and ambiguity. Clarified optional OFF call
pruning and the difference between the selected run profile and standalone orig
defaults. The document fixes the Stage1 reference source, source-item alignment,
independent masks, Text-KL direction, distinct reduction denominators, resume and
execution boundary; these remain specified behavior, not observed integration.

Proposal v1 is now frozen at SHA-256
`71d78c7280d7ca14f5bd28b2c77863958d447b28779936de45b2bbe52d7b893e`.
The commit containing this append freezes that exact proposal. All critique
passes must use these same bytes. Fresh document checks passed (exit 0), with
full output read from `logs/stage1-v5-design/proposal-document-check.log`.
Original DIL source and inherited product code remain unchanged. No full suite
is appropriate for this documentation-only claim; branch finishing is not next.

## Design commitment critiques in progress

Root executed hate against frozen proposal v1 and captured exactly `{root,
first_nail}` in `specs/2026-10-13-stage1-v5-hate-v1.json`. Its objection is the
unwitnessed shared-F checkpoint/dropout gradient equivalence; the first nail is
a bounded tiny CPU DoRA comparison against independent masked calls. The
separate experiment authorization question is pending. No probe has run and
the earlier Stage2 probe is not reused as evidence for this different graph.

Root invoked macrothink under the standing design-gate authorization. The
stripped prompt retained the user's objective/coefficient/call-count constraints
and observed Stage1-v3 facts but omitted the root's objection and implementation
recommendation. It requested an independent framing before reading the frozen
proposal, then evidence-bounded divergence assessment. Three fresh same-model
read-only agents are active: `/root/v5_fresh_read_1`, `/root/v5_fresh_read_2`,
`/root/v5_fresh_read_3`. They were told not to invoke Paperthin, spawn agents,
edit files or run experiments. No divergence classification or completion is
claimed until their actual results are received and synthesized by root.

All three reads returned and were classified divergent-compatible, with no
established incompatible divergence. Full clustered findings and source
references are preserved in `specs/2026-10-13-stage1-v5-critique-v1.md`.
Convergence is reassurance only. The frozen proposal bytes remain unchanged.

The user explicitly replied `CPU gradient 동등성 검사 실행 (권장)` to the
separate experiment request. Root prepared the bounded brief
`briefs/2026-10-13-stage1-v5-premise.md` and its scoped SDD ledger/constraints,
completed the one-task preflight table, and dispatched `/root/v5_premise_impl`
(gpt-6.1-sol, high, fresh context) from base
`3441a1ce9d205b175681ef229aedb8643f23718e`.

This is the separately authorized premise experiment within brainstorming,
using the selected Subagent-driven implementation/review method and TDD for its
small throwaway adapter. It is not the architectural path's terminal product
implementation transition. Product spec/plan approval and all later gates
remain pending. A fixed helper fixture narrows the probe to the new three-call
F/reference/packed subgraph; actual Stage1 helper/full four-call integration and
two-node memory/reduction remain later evidence obligations. The experiment's
independent spec-and-quality review is required before commitment. No product
code is authorized in this experimental task.

## CPU experiment and first independent review

The throwaway probe was committed as `fe9f434` (parent `3441a1c`), only
`logs/stage1-v5-premise/{probe_adapter.py,test_probe.py}` and
`results/stage1-v5-premise/result.json`. The original focused run recorded
8/8 checks: six dropout/checkpoint combinations, detached-prefix control and
contaminated-V-off control. Source-coordinate dropout/native Qwen/DoRA and
RMSNorm were exercised. Root allowed a fixed x4 initial output-head scale after
the original detached-prefix control measured below unchanged tolerances;
scale-1 failure and diagnostic measurements remain in logs. No product files,
real checkpoints, GPU or training were used.

Distinct `/root/v5_premise_review` (gpt-6.1-sol/high, fresh context) reviewed the
complete `3441a1c..fe9f434` diff and actual runtime call paths, independently
checked hashes and read the saved complete logs. Verdict: explicit probe
implementation compliant, quality **Needs fixes**. Its concrete read-only
mutation removed Text-KL and its 0.01 total contribution, yet every loss and
fused-gradient assertion passed: original K6.50994e-7, maximum fused-gradient
error2.60770e-6 below atol1e-5. This is an evidence gap, not a demonstrated broken
graph or missing formula. The original eight passes are not accepted as
complete five-term evidence. Full review and exact mutation command are in
`.superpowers/sdd/2026-10-13-stage1-v5-premise/task-1-review.md`.

Root applied receiving-code-review to the full finding and dispatched fix round1
to the original implementer, starting `fe9f434`. The bounded remedy adds isolated
raw Text-KL gradient comparisons and a causal detached-Text-off control, leaving
the requested weighted total, masks, fixture and original tolerances unchanged.
No broader test/review cycle or architecture redesign is authorized by this fix.
The final commitment draft remains unapproved and unpresented pending this
specific fix, scoped re-review and fresh evidence.

Fix `8b675e2` adds isolated raw-K gradient comparisons and detached-Text-off
control at the unchanged tolerances. The omission now produces six intended
assertion failures. Adding a diagnostic traversal exposed the probe's one-shot
coordinate context on four checkpoint cases; root authorized its minimal
reusable context replacement, preserving all bindings and actual model math.
The same four failures and restored-context assertions are the causal regression
evidence. Implementer final log records 9/9 checks and raw-K max error7.75e-9,
with detach-control error2.61e-4. Scoped re-review is pending; no clean verdict
is inferred yet.

Before completion/verification, root clarified gate applicability: the entire
authorized one-task experimental diff also receives the SDD final review after
task re-review closes. Prior statements that main-feature final review/finish
remain later still apply to the unimplemented product. The experimental final
review does not establish whole-feature completion or trigger a branch
integration/cleanup choice. No required review gate has been passed over; this
clarification occurs before experimental task completion. The final fresh
verification will be the focused probe and separate control-document/hash/scope
checks, not the unrelated project full suite.

## Accepted CPU evidence and final design commitment

Task reviewer `/root/v5_premise_review` closed the single Important finding in
`task-1-rereview-1.md`: all findings addressed, no new Critical/Important
breakage, no new out-of-scope findings. Final reviewer
`/root/v5_premise_final_review` (gpt-6-astra/high, fresh and distinct) reviewed
the complete `3441a1c..8b675e2` experimental scope and accepted it as bounded
design evidence, with no remaining Critical/Important/Minor findings or parked
issues. Its full verdict is durably preserved at
`reviews/2026-10-13-stage1-v5-premise-final-review.md`.

Root disposition of each final-review declined-to-judge item:

| Item | Decision and reason | Cost if misunderstood |
| --- | --- | --- |
| Reference-local embedding/vision | Accept as outside fixed-realized-input CPU scope; Section7 retains an actual production witness | Reference conditioning could be wrong despite a passing graph probe |
| Real helper/full four-call Trainer | Accept as untested; require real helper and runtime entry hooks during integration | Four-call or latent-history claim could be false |
| Production dropout/checkpoint bindings | Accept harness-only arithmetic evidence; require actual trainer integration | Binding/recomputation errors could appear outside the probe |
| Vision/pretrained data, selectors/mapping and arbitrary layouts | Accept fixed-fixture scope; keep specific mapping/selector tests in the plan | A correct tiny graph could still score incorrect targets |
| GPU/BF16, EMA/ZeRO, accumulation/resume and production ranks | Accept CPU-only limitation; required distributed/runtime witnesses remain gates | Full run gradients or reference restoration could be wrong |
| Memory/throughput and global/empty-step reduction | Accept not measured; actual maximum packed geometry and Trainer/global oracle remain required | Runtime OOM or biased update could invalidate training |
| Generic harness concurrency/hardening | Omit from this throwaway one-process experiment; no reusable-library claim | Reuse elsewhere could require additional engineering |
| Architectural/product/merge/training readiness | Not established; retain final human spec/plan approval and all implementation gates | Treating this result as readiness would bypass required work |

Rulings retained from the experimental ledger: fixed helper inputs isolate the
new three-call subgraph at the cost of leaving helper integration untested;
deterministic x4 initial head scale improves below-tolerance prefix sensitivity
but is an artificial fixture, not a production distribution; reusable coordinate
contexts support the extra diagnostic traversal without changing native model
math, and actual production lifecycle still needs its own witness. No other
finding was silently parked or waived.

Root executed verification-before-completion after final review. Branch finishing
is not next: Stage1-v5 product integration is unimplemented and no branch
handoff decision is ready. The claim-specific command was:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q logs/stage1-v5-premise/test_probe.py
```

Fresh complete output `logs/stage1-v5-premise/root-final.log`: **9 passed in
10.02s**, exit0. No warning/failure summary. This is the focused CPU experiment,
not a project full-suite, GPU, or training pass. The rerun left the result
artifact byte-identical to reviewed commit `8b675e2`. Hashes:

- Adapter `13014ec5b11858b69c3c69ed200b30a73fc82a0c7da81b19c83ef57b17005093`.
- Test `d1bf354a0b6f282e7f61efe7a9b6a3bd0ee21fec1a1afa00672b95ac1149e698`.
- Result `46ad32c59e28eb7da4466e5cf766f081a9112ad9bb1ca3b1f1ff9209f1a345d9`.

Root synthesized and self-reviewed design commitment v1 in
`specs/2026-10-13-stage1-v5-design.md`. It preserves the frozen proposal,
incorporates the exact selector/memory clarifications and reviewed first-nail
result, states no incompatible divergence was established, and explicitly
limits all CPU claims. Its SHA-256 is
`e3d2515ccc7b1a0de0e82de2e4b957abebbcf220b022c1baff06521fd5c45762`.
Fresh separate document/scope/hash verification exited0; complete output is
`logs/stage1-v5-design/commitment-document-check.log`. It checked all11sections,
exact coefficients/selector rules, absence of unresolved placeholders, frozen
proposal/original DIL hashes, unchanged inherited product sources and exact
reviewed source/runtime/result hashes. `git diff --check` was clean.

Actual skills executed for this experimental continuation: selected SDD task
implementation/fix/review/final-review routing; implementer TDD and systematic
debugging; root receiving-code-review, requesting-code-review and fresh
verification-before-completion. Root brainstorming remains the outer lifecycle
owner; hate and macrothink were completed on frozen proposal v1 before synthesis.
No writing-plans or product implementation transition has happened. Main branch
finishing/cleanup remains nonapplicable while the requested feature is unfinished.
Experimental scratch is retained with this durable record for active design
handoff; no audit evidence was deleted.

**Current active phase:** architectural brainstorming, exact design commitment
v1 ready for final written human approval. The user has approved the CPU probe,
not this newly written commitment. After written-spec approval, invoke
writing-plans, obtain the plan review, and execute using the already-selected
Subagent method. Existing authorization for fresh Stage1 training and automatic
TEST400 evaluation persists; do not ask for it again. No real training/evaluation
or new product integration has been launched or claimed complete.

## 2026-10-14: user authorizes the presented commitment and reiterates training

Understood as: the user wants the already specified Stage1-v5 work carried
through implementation, two-node training and automatic evaluation, without
re-asking for training scope or execution method. The reply “학습까지
진행하라니까” directly follows the linked final commitment v1 approval question;
it approves that presented design. Bound artifact is commit `a4519cd`, spec
SHA256 `e3d2515ccc7b1a0de0e82de2e4b957abebbcf220b022c1baff06521fd5c45762`.
No coefficient/reference/mask/scope change was requested. Subagent-driven method
and prior training/automatic-evaluation authorization persist.

Root reconciled the clean worktree, prior experimental reviews and hashes;
no completed probe/task was re-dispatched. Brainstorming's written-spec gate is
now satisfied and its sole architectural transition, writing-plans, is active.
Root wrote `plans/2026-10-14-stage1-v5.md`: seven independently testable
implementation tasks, final review/verification, then root-operated preflight,
four-rank runtime witness, fresh training and automatic TEST400 evaluation.
Pinned Stage1-v3 HEAD/source hashes were rechecked and still match the design.
The plan self-review checked spec coverage, exact public interfaces, existing
Trainer GAS scaling, current EMA filename and producer/consumer consistency.
The plan is a new artifact; no human plan review has been invented or inferred
from an approval preceding its creation. Actual handoff outcome will be recorded
below. Product implementation has not yet started.

## 2026-10-14: written plan approved; product implementation begins

The user answered the linked written-plan review: “계획 확인, 학습·자동 평가까지
진행”. This binds plan commit `be9d631dde645f0128ab8500f2c8d061fbef1a5a`,
SHA256 `b3b4f75d674af2fea671b9ce7e3aabb80c0cf3b2b1b67b6d219c676d4067696d`.
Design commitment v1 approval above remains unchanged. Writing-plans is complete;
the active phase is subagent-driven implementation of seven tasks, followed by
independent final review, fresh verification, two-node training and automatic
TEST400 evaluation. No further method/training approval is needed within scope.

Root read the plan/spec and SDD instructions, verified this existing linked
Stage1-v5 worktree (not a submodule), and created the plan-specific ledger at
`.superpowers/sdd/2026-10-14-stage1-v5/progress.md`. Existing baseline and CPU
premise evidence are reused; no dependency installation or repeated probe.
Prior hidden-CUDA full-suite failures are not represented as a passing suite.

Read-only checks found both nodes reachable, two idle H100 NVL GPUs each,
bond1.102 present and sufficient checkpoint storage (local37TiB/remote39TiB).
Both report torch2.7.1+cu128, transformers4.50.0.dev0, accelerate1.10.1,
deepspeed0.18.7 and peft0.17.1. TRAIN3450 and TEST400 match on both nodes:
`5bc1313a594f5713975b67388c745c6632275c37fd260055b27230c9cfedbed8` and
`6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618`.
Original base and image-root paths exist. This does not establish future resource
availability, full input validity, reviewed-code compatibility or update budget;
authoritative preflight/audit remains after implementation.

### Latest execution stopping point

The user then instructed: “학습이 되는것까지 확인하고 멈춰봐.” Root stated it
will observe actual participation by all four ranks, finite loss and optimizer
step progress, then report and stop monitoring while leaving training running
in the background. This narrows the current response's completion witness;
implementation/review/launch gates remain, and no training process is to be
killed by interpreting this as a termination request. Existing automatic
post-training evaluation authorization is retained; results need not be awaited
before the requested handoff. No objective or design change is involved.

## Product Task1 completed

Implementer `/root/v5_task1_impl` (gpt-6.1-sol/high) implemented approved contracts,
original-source metadata and opt-in CLI in `1e2a4bb`, then fixed review finding I1
in `f5f8ff3`. Task base `37dedfc`. Independent reviewer `/root/v5_task1_review`
(same model/effort, distinct fresh context) assessed both spec and quality:
initial Important finding was unsupported explicit legacy controls silently
accepted on v5. Scoped re-review accepted the fix, no remaining findings or
parked items. Broader runtime/resume questions remain assigned to Tasks3-7.

Actual task skills: implementer TDD, targeted systematic debugging and fresh
claim verification; root SDD/requesting-review/receiving-review routing. No
product training/full-suite claim. RED was48failures for missing behavior;
final initial-task command used repository Python, PYTHONDONTWRITEBYTECODE=1,
CUDA_VISIBLE_DEVICES='', PYTHONPATH=src:transformers/src and pytest -q with
`tests/stage1/v5/test_config.py tests/stage1/v5/test_metadata.py
tests/stage1/test_cli.py tests/stage1/test_data.py`:99passed/3.15s, no warnings.
Fix RED16failed/2passed; same environment and pytest with v5/test_config.py plus
stage1/test_cli.py:70passed/3.10s, no warnings, exit0. Complete logs under
`logs/stage1-v5-task1/`, including retained failed abbreviation adaptation.
Report/reviews remain in the plan-specific SDD workspace until final evidence
handoff; these exact Git revisions identify the reviewed changes.

Root Ruling: fresh explicit percentage with omitted selector infers clean,
matching original DIL Section7/existing source default; explicit all+percentage
rejects and saved config remains authoritative. Default all/None and requested
run unchanged. Cost if wrong: optional percentage-only CLI requires correction.
Independent reviewer accepted this interpretation against binding source/spec.
Task5 must load saved sidecar before eager parser validation; producer names are
`args.stage1_v5_config` and `args.stage1_v5_explicit_values`.

## Product Task2 completed

Implementer `/root/v5_task2_impl` (gpt-6.1-sol/high) implemented loss/selection
helpers at `28fbebd5c70d178fba2826de32ddbaca1b7982ce`, base`1d9fd2a`.
Independent `/root/v5_task2_review` (same model/effort, fresh/distinct) approved
both specification and quality with no findings. Reviewed numerical kernels
remain unchanged. Real-logit gathering/trace, Trainer reduction, reference
lifecycle and production evidence remain Tasks3-7 responsibilities. Reviewer
approval-provenance note is resolved by actual user replies/hash bindings above;
the approved spec bytes retain their pre-approval header intentionally.

Implemented TDD and claim verification: initial24missing-feature failures;
new empty-OFF-selector test3failures corrected. Repository Python with
PYTHONDONTWRITEBYTECODE=1 and PYTHONPATH=src:transformers/src ran pytest -q on
`tests/stage1/v5/test_selection.py tests/stage1/v5/test_objective.py
tests/step_preference/test_dil_bce_orig_loss.py
tests/step_preference/test_topk_entropy_loss.py`:97passed/2.24s, exit0.
After final connected-zero refactor, the first two files alone:27passed/2.06s,
exit0, no warnings. Logs under `logs/stage1-v5-task2/`; report and review in
plan-specific SDD workspace. No GPU/distributed/full-suite claim.

Root Ruling: corrected brief's nonexistent top-level kernel test paths to the
existing `tests/step_preference/` paths, retaining identical test semantics.
Cost if wrong: regression evidence would be missing; exact executed paths and
passing output above resolve that risk. All objective inputs are selected
[N,V] in metadata.query_keys order; Task5 gathers ordinary F at those keys.

## Task3 procedure deviation and recovery

Before its commit, implementer `/root/v5_task3_impl` reported a timing violation:
systematic-debugging must be read/invoked before a bug fix. Its scripted import
insertion failed to match `import torch.nn as nn`, producing NameError for
`v5_decoder_forward` during initial test collection. It inspected the stack and
source and inserted `from .stage1_v5_trace import v5_decoder_forward` at Qwen
modeling line53 before reading that skill. Affected change is the missing import
in `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`.
The omitted step is skill invocation timing, not claimed absence of debugging;
later mask-fixture debugging used the skill. Original failed evidence remains
`logs/stage1-v5-task3/green-initial.log` (NameError at lines27/78).

Implementation progression paused before commit; root recorded and reported
the deviation. At this point focused/native tests are reported green, but
independent specification/quality review and full feature/runtime gates remain
unverified. Root recovery is to preserve this non-retroactive record and failed
log, finish the report with exact affected scope, then continue the already
authorized task commit and independent review of the complete change (including
that import). No repeated broad test or self-review substitute is prescribed.
Later review cannot retroactively establish compliance with the missed timing.

## Product Task3 completed

Implementer `/root/v5_task3_impl` (gpt-6.1-sol/high) committed native trace/replay
at`8cb30bf` and selected-reference chunk fix at`b2c8291`, task base`dc2aabc`.
Independent `/root/v5_task3_review` (gpt-6-astra/high, fresh/distinct) found one
Important missing chunk128 projection, directly witnessed with129queries.
Scoped fix review approved both spec/quality with no remaining findings/parked
items. The previously recorded timing deviation remains non-retroactive.

CPU native six-case dropout/checkpoint oracle covers all trainable/input
combined gradients and isolated raw-K gradients (max errors2.980232e-8 and
2.478373e-9), actual F/reference/packed entries, and three causal controls.
Separate selected-reference image/token/latent preparation and native vision
checkpoint tests are bounded CPU evidence. They do not prove integrated helper,
Trainer/DDP/BF16/FA2 or capacity. Later tasks/live gates own those claims.

TDD:20missing-feature failures; retained subsequent import and fixture failures
under `logs/stage1-v5-task3/`. Named regression command used repository Python,
PYTHONDONTWRITEBYTECODE=1, PYTHONPATH=src:transformers/src, pytest -q on
`tests/stage1/test_qwen_latent_injection.py tests/stage1/test_equivalence.py
tests/stage1/test_peft.py`:442passed/1inventory-failure. Final affected run with
CUDA_VISIBLE_DEVICES='', pytest -q -s on v5/test_execution.py, v5/test_replay.py
and the exact inventory test:23passed/5.99s. Fix RED8failed; same environment,
pytest -q -s on v5/test_execution.py plus
v5/test_replay.py::test_native_combined_and_raw_K_match_literal_full_masks:
24passed/5.58s, exit0/no warnings, intentional runtime witnesses retained.
Default129queries project128+1, configured32 projects32+32+32+32+1; one reference
decoder, reversed query ordering matches its own full projection.

Root rulings: preserve standard nonempty dropout-site schema; p0 fixture
constructs declared sites then disables their probability. Cost if wrong: no
fresh zero-site profile support (not claimed). Extend strict RNG inventory only
with V5Execution.dropout_mask/torch.rand; cost if wrong: unintended random path
admission. Omit artificial peft.py edit because actual existing installer and
native fixture suffice; cost if wrong: adapter binding gap. Reviewer checked
that path independently. Add selected-only stage1_selected_chunk_positions
keyword (default128) so Task5 passes configured chunk size; cost if wrong:
inconsistent reference/OFF bounds. Runtime projection tests cover this contract.

Task5 integration uses `bind_v5_execution(decoder, execution, role='F',
visual=...)` inside graph(F) with capture around ordinary F; packed helper binds
internally inside graph(packed). Keep owner active through parent backward,
exit observation hooks beforehand. Full task/fix reports and reviews remain
in plan-specific SDD workspace for final handoff.
