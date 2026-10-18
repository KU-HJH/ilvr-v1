# Stage1-v5 GPU utilization optimization workflow

## Intake and authority

Understood as: improve actual Stage1-v5 throughput by reducing host-side
attention/selection/mask/dropout overhead and investigating rank imbalance,
while preserving the existing query/loss/teacher/reference semantics and
completed training artifacts. The user's `수정진행해` authorizes pursuing this
development; it is not final approval of a subsequently written architecture.

Source revision: `65643f08cb6e61fbbb200de825d57086f5c7a127`. The previous
feature and merge reviews plus full-suite5084pass/11skip remain evidence for
identical source; no broad baseline rerun is needed. Existing data/checkpoint/
venv links are reused. No package install, evaluation or training is requested.

User answered the numerical-policy question: selected token indices, attention
masks and dropout randomness must agree exactly; floating attention/loss/
gradient values may use small, predefined tolerances. Exact tolerances and
edge behavior still need specification. Existing comparisons of scalar logits/
loss/gradients use ATOL1e-5/RTOL1e-4; old throughput attention tests currently
require exact rows. Do not silently relax those tests without the approved
new numerical contract and an unchanged baseline oracle.

The user answered the scope/decomposition question: `연산 경로 4개 우선,
rank 배분은 후속 단계 (권장)`. This confirms attention observation, text
selection, replay masks and source-coordinate dropout as the first scope.
Keep sample order, sampler, rank assignment, accumulation and resume sample
progress unchanged. Rank balancing is deferred to a separate later design;
it is not included in this implementation. This answer confirms scope, not
the subsequently presented architecture or final written commitment.

## Isolation and actual prior-training status

Native worktree list returned no task attachments. Other existing checkouts
contain separate user branches or the prior launch's code/results. Root used
native create_worktree(ref=65643f0), operation
`2e1256dd-c3ba-49e7-b44f-13ae80012e5b`, then observed completed creation of
`/root/.codex/worktrees/d3c8/ILVR`. It is host-managed, detached, same source
revision and initially clean. The user accepted working separately from the
training source in the preceding proposal. Preserve this workspace's host
ownership; do not delete other worktrees, branches or unrelated files.

Root initially treated the previous training as still live. A fresh process/
artifact check found it already complete: both node exits0, producer evidence
global_step648, all process ranks0..3, finite_loss=true, aggregate train loss
1.3436651700808677; trainer_state at checkpoint-648 has epoch3.0. This update
was reported to the user. Existing canonical and6b9d files remain unchanged.
The fresh workspace links venv/data/checkpoints to their canonical existing
locations; no model-bearing data is copied to results or documentation.

## Active lifecycle and task state

Superpowers is controller. Read using-superpowers, brainstorming and
using-git-worktrees, canonical/worktree AGENTS and the prior throughput brief.
Readchk was applied to scope/constraints. Active phase: architectural
brainstorming for a multi-step runtime/numerical optimization. Retain the
previous user-selected Subagent-driven implementation method after approval.
Root authors only control artifacts. No product code, tests or configuration
have been changed; no implementation agent has been dispatched.

1. Context exploration: actual runtime call paths and source/test oracles read.
2. Visual companion: nonapplicable; decisions concern arithmetic, interfaces
   and validation rather than visual layouts.
3. Clarifications: numerical policy and scope/decomposition answered.
4. Alternatives: four-operator scope selected over rank redistribution and
   mask/dropout-only scope. Preserve this choice rather than asking again.
5. Architecture/components/data-flow section A1: confirmed by the user's
   subsequent `진행해` in response to the explicit A1 question. This approval
   binds A1 only and authorizes elaborating validation, not product changes.
   Error handling and testing section B1: confirmed by the subsequent `계속`
   after the explanation of approval fragmentation. Treat this as continuation
   of the pending design section, not waiver of the final written-spec gate.
   The user objected to repeated stops; consolidate the remaining design,
   critique and uncertainty presentation into one exact final commitment.
6. Complete written specification and self-review: proposal v1 committed at
   93bd9519a73ab4eda4c1c820ae5a522db508924a, SHA256
   e036cc923079233ad2ccf4bc0807164c827337742ca303a974aa506a91c9aec0.
   Placeholder/consistency/scope/ambiguity check completed, including actual
   V5Block invalid-input behavior. Frozen proposal remains unchanged.
7. Root hate completed with exactly root/first_nail; macrothink completed
   with three fresh same-model read-only agents /root/gpu_util_fresh_a,
   /root/gpu_util_fresh_b and /root/gpu_util_fresh_c. Input/results and primary
   divergent-incompatible classifications are recorded in the design-review
   file. Root synthesized commitment v1, preserving unresolved attribution/
   boundary uncertainty. No premise experiment was authorized or executed.
8. Final human approval of exact written commitment, including any explicit
   uncertainty acceptance: pending. Initial implementation request is not it.
9. Only after approval: writing-plans, retained SDD/TDD implementation and
   independent spec/quality/final review, fresh claim-specific verification,
   then applicable branch finishing. No current performance-completion claim.

## Findings and constraints to resolve in the design

The36-second prior read-only observation averaged58..64percent GPU busy time;
main-process CPU exceeded one core, worker CPU averaged about one percent.
CPU observations include runtime/communication threads and do not isolate
Python time. No stage/CUDA profile establishes a dominant function.

Actual paths: v5_trainer.compute_loss builds targets, then F with per-layer
capture_v5_trace, immutable reference, select_v5_text_keys, packed V/Text-off,
loss and native backward. Trace loops over each query in each layer;
selector bulk-transfers weights and performs Python stable ranking/prefix
accumulation; packed replay fills bias per query; SampleStableStage1Dropout
dispatches through active V5Execution to source-coordinate masks, whose
dropout_mask currently converts coordinates to host and stacks per-token rows.

Potential constraints: preserve full causal softmax denominator and original
candidate coordinates; FP32/head/layer semantics; stable position tie-break,
low-mass/cap/shortest-prefix behavior and Python-double cumulative decisions;
exact 0/-inf masks and source padding; identical random draw shape/seed/site/
step/source identity and checkpoint replay; attached F K/V gradients; immutable
reference and Teacher query rules; all existing loss reductions, GAS, sampler,
workers, resume/source-binding strictness and storage policies. Batched float
math does not itself establish exact discrete selection at a boundary.

No broad compiler, CUDA graph cache, retry/recovery or sampler infrastructure
is assumed. Existing DeepSpeed overlap_comm is already true. Future benchmarks
need identical baseline/candidate inputs, wall time and peak memory as well as
operator count and exact discrete comparisons. GPU utilization alone is not
the success metric. Separate CPU evidence, operator CUDA evidence and actual
two-node/four-rank training throughput; none may stand in for another.

## Fresh focused baseline and additional contract finding

At control-only revision492b0e8, root ran the existing unchanged focused CPU
tests with CUDA_VISIBLE_DEVICES empty, repository venv, vendored Transformers,
one CPU thread and a fresh basetemp under checkpoints. Exact command:

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=transformers/src:src:. ./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/v5/test_throughput_attention.py tests/stage1/v5/test_selection.py tests/stage1/v5/test_execution.py tests/stage1/v5/test_replay.py --basetemp=checkpoints/stage1-v5-gpu-utilization-20261018-tests/intake-baseline > logs/stage1-v5-gpu-utilization-20261018/intake-baseline.log 2>&1
```

Observed exit0, **64passed/2skipped in12.97s**; complete output read. Log
SHA25681260e8769331b8b1b2c0f964eb374a15015eabd37e903029c72b1819bc79618.
These are baseline witnesses, not tests or measurements of a new implementation.

Additional static finding: the current eight runtime source bindings cover
Teacher/query/history and parent/v5 Trainers, but not the source modules that
implement dropout ownership, student attention capture, packed replay or text
selection. The design must explicitly bind changed execution modules/profile
so a changed implementation cannot silently present the old runtime identity.
Keep old completed contracts/checkpoints inspectable and unchanged. Do not
weaken existing strict resume comparison or claim compatible continuation
without separately reviewed evidence. No new loss/query formula is proposed.

Implementation remains unauthorized pending completed architectural design
sections, critique commitment and final written approval. Scope and numerical
policy are now confirmed. No human response is inferred from elapsed time,
and no product changes have been made.

## Architecture section A1, confirmed by the subsequent `진행해`

Use the existing F -> reference -> text selection -> packed V/Text-off ->
loss/backward flow and existing public interfaces. Change four internals:

1. Replay mask: construct prefix visibility and suffix causal visibility by
   broadcast/index operations, then apply the V body and selected Text key
   exclusions. Preserve exact zero/-infinity values, padded-query handling,
   original predictor coordinates, branch/row order and attached F-prefix
   gradients. Do not allocate a second dense floating replay-sized workspace.
2. Source-coordinate dropout: retain private generator, seed, original
   source-length x hidden draw shape and scaling arithmetic; gather the masks
   with tensor indices instead of constructing one Python element per token.
   Keep per-source/site generation local; no persistent all-site mask cache.
3. Student attention observation: use bounded query chunks per sample/layer,
   with native RoPE, original positions, full visible-key FP32 softmax,
   head/layer averaging and detached observation. No candidate-only softmax
   and no full all-query attention tensor. Batching can perturb floats; it
   cannot be accepted solely on an attention tolerance if selected keys differ.
4. Text selector: stable GPU sorting with explicit original-position tie
   handling, transfer only the ranked prefix needed by the existing cap plus
   original-device masses/rhos, and retain Python-double sequential threshold
   accumulation. Preserve low-mass, zero-cap, empty, tie and boundary behavior.
   Do not replace the cumulative decision with a parallel GPU prefix sum.

Bind these execution modules in the existing runtime source identity so a new
implementation cannot silently resume under an old source contract. Preserve
historical contracts/checkpoints for read-only inspection. Do not add a new
query mode, loss coefficient, sampler, launch framework or training run.

Exact selection under batched float perturbations is not established by
static reasoning or the unchanged baseline tests. The next section must
define the comparison witnesses and numerical limits, with a mismatch blocking
activation rather than being explained away by float tolerance. Finite tests
will not be described as a proof over every future input. No GPU premise probe
has been authorized or executed for this proposal.

## Error handling and verification section B1, confirmed by `계속`

Keep the existing literal independent full-sequence oracle and frozen older
sources. Add an immutable, hash-identified comparison baseline for the four
operator modules as they exist at source65643f0; do not derive expected masks,
selected keys or random draws from the implementation under test. Do not
replace the historical oracle with the candidate. Tests that already require
exact values unrelated to attention batching keep those exact requirements.

The proposed numerical limits, fixed before any candidate GREEN result, are:

- Exact equality of selected original token positions and their order,
  zero/-infinity attention mask entries including padding, and raw source-
  coordinate dropout masks. Keep source seed/draw shape/site/step identity,
  checkpoint recomputation and unchanged global RNG state. Direct-selector
  tests receive identical attention inputs; separate joint attention/selector
  tests compare both paths starting from identical projections and logits.
- Detached FP32 attention observation: ATOL1e-7/RTOL1e-5. Preserve the full
  causal denominator, native RoPE, GQA expansion and head/layer averaging;
  compare every candidate row, not only its total or final loss.
- Native FP32 oracle logits, each loss term, attached-prefix gradients and
  trainable parameter gradients: the existing ATOL1e-5/RTOL1e-4. No changed
  loss reduction or normalization. Report the checked dtype and execution
  scope; these CPU/tiny-model bounds are not proof of production BF16 or
  two-node/four-rank behavior.

Include rejection of unsorted candidate metadata, exact weight ties, nextafter threshold
neighbors, mass on both sides of the low-mass branch, empty and zero-cap rows,
multiple source samples, duplicate/permuted source coordinates, variable
source lengths, padding and checkpoint full/early-stop recomputation. Use
multiple-query chunk boundaries and a final shorter chunk. Exercise actual
Trainer training_step and packed replay, not isolated helpers alone.

Any discrete mismatch is a failed equivalence check even if attention is
within tolerance. Nonfinite outputs, exceeded tolerance, broken attached
gradient paths, incompatible source resume and invalid inputs retain explicit
errors. Correct the cause through the selected implementation/review method;
do not relax thresholds after seeing a failure. If attention batching cannot
meet the contract, report it and revise the design rather than silently
activating it or treating three optimized paths as the approved four-path
completion. Finite comparison tests are not a universal proof of unchanged
selection for all future floating-point boundary cases.

Bind execution/profile identity to all four modules, and update the existing
verifier plus actual resume/admission consumers and tests. New-code training
resume from an old incompatible contract must fail before optimizer work with
clear guidance to use the original code; historical read-only audit remains
possible without rewriting old sidecars or weakening strict validation.

After final design approval, use TDD and the retained Subagent-driven method,
independent specification/quality/final review, then one fresh appropriate
completion suite plus distinct claim-specific integration checks. Reuse the
existing operator measurement script rather than create a new framework.
Compare old/new identical inputs for attention, selection, mask and dropout:
three warmups, twenty synchronized timed repetitions, median and raw timings,
host-transfer/operator counts and peak allocated/reserved CUDA memory. Keep
attention score/probability workspace bounded by query chunking; avoid a
second dense replay-sized floating mask. Report any operator regression.

CPU evidence, CUDA operator evidence and end-to-end distributed training
throughput are separate claims. A CUDA operator comparison contains no model
loading, optimizer steps or model state writes. The later implementation
approval may cover these local operator checks, with an idle-device check;
it does not itself launch fresh training. A separate actual training measure
uses the mandatory two-node/four-rank coordinator and unchanged comparison
profile only when authorized. GPU busy percentage alone cannot establish an
improvement; actual training claims need matched seconds/update or tokens/s.

No costly or implementation-like premise probe has been run during design.
Hate's proposed first nail will remain a proposal unless separately authorized.
Any remaining load-bearing uncertainty must be explicit in the exact final
commitment and accepted by the human before planning and implementation.

## Written-spec preparation and self-review correction

Runtime-path inspection confirms V5Block.__post_init__ already requires sorted
unique candidate positions. B1's unsorted-metadata test therefore verifies
unchanged rejection, not a new supported input. Interleaved samples/blocks
remain valid; stable sorting must preserve original position tie-break for
valid sorted candidates. This correction preserves A1's existing interfaces
and adds no new behavior or implementation authority.

## Critique commitment and final gate

Using the root-only advisory hooks did not advance implementation. Fresh
reads challenged committing to four changes without attribution; one also
challenged attention arithmetic before boundary evidence. Final commitment
retains the explicitly chosen four-path scope, stages baseline measurements
and exact mask/dropout first, and gates attention integration on joint discrete
comparison. The alternative priority of a real unchanged distributed profile
before any rewrite remains unresolved. Same-model alignment is reassurance,
not runtime evidence or review approval.

The exact final artifact is
`specs/2026-10-18-stage1-v5-gpu-utilization-commitment-v1.md`. It includes all
A1/B1 architecture/acceptance details, baseline hashes, critique synthesis,
experiment-not-run status and explicit uncertainty acceptance. Its approval
will cover bounded local development and operator checks with idle-resource
checks, not another training launch. Root will ask one consolidated written
specification approval, not repeat scope/A1/B1 questions. No product source,
tests, runtime configuration, historical checkpoints or remote files changed.

Final commitment was read in full after synthesis and checked for consistency
against A1/B1, actual source invariants and the critique record. Its SHA256 is
`843832ae9b09c10f28dea3dea60ceaa8d845f77f103f98259d4f26304cbe659a`.
The frozen proposal digest was rechecked unchanged; `git diff --check` passed.
These are document/identity checks, not new correctness or performance tests.
Await exact commitment approval and explicit uncertainty acceptance before
invoking writing-plans or dispatching implementers. No new section approval
is being requested.

## Final approval and execution handoff

The subsequent user `계속` directly answers the consolidated final commitment
question, which explicitly disclosed unrun premise experiments, attribution,
production boundary/memory uncertainty and the scope of approval. Record this
as approval of commitment v1 SHA256843832ae9b09c10f28dea3dea60ceaa8d845f77f103f98259d4f26304cbe659a,
including acceptance of proceeding under that uncertainty. It does not waive
the exact discrete checks or authorize fresh training. A1/B1 are not reopened.

Root executed writing-plans and self-reviewed the concrete four-task plan at
`plans/2026-10-18-stage1-v5-gpu-utilization.md`. Retained Subagent-driven method
requires fresh implementers and sequential spec/quality reviews per task,
then whole-change review and fresh verification/finishing. Root read SDD,
TDD, requesting-review and their prompt templates. Active phase: approved
architectural implementation handoff; no product changes by root.

Task tracker: T1 baseline/measurement pending; T2 mask/dropout pending;
T3 selector/attention pending; T4 source identity/integration pending;
independent final review and verification/finish pending. SDD will execute
continuously without further same-scope approval stops.

Fresh local resource observation before handoff: two NVIDIA H100 NVL devices,
zero used memory/zero utilization and no compute processes. This supersedes
any presumed B200 for local operator measurements; recheck before CUDA work.

## Task 1 implementation and review evidence

Implementer `/root/gpu_util_t1_impl`, base91a1440b199548afd387d76cf7a303a5e59a24a2,
head5a7df38096f6e874269e731d1d0b9f055fc0a9ec. Nine files: four byte-identical
frozen sources, fixtures/tests, direct benchmark extension and two baseline
JSONs. Root verified the head/diff and unchanged production operators plus
historical frozen_throughput via `git diff --exit-code 65643f0 -- <four paths>
tests/stage1/v5/frozen_throughput`. Worktree was clean after commit.

Causal RED logs `t1-red-operators.log` (missing replay_mask_before/frozen
files/paired execution) and `t1-red-dropout.log` (missing execution fixture)
precede implementation. Initial GREEN `t1-green.log` exposed serialization of
inherited dispatch bookkeeping; the script now explicitly serializes counters.
Corrected GREEN `t1-green-fixed.log`: exit0, 26pass/2CUDA-onlyskip in11.69s.
Root read corrected GREEN and output-validation/resource-check logs; spec
reviewer read all RED, failed/corrected GREEN and complete measurement logs.

All following commands ran from d3c8 with repository venv, exited0 and had
complete stdout/stderr redirected to the matching logs. Exact effective
commands, without the shell's subsequent output-display/exit wrapper:

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=transformers/src:src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./venv/ilvr-venv/bin/python -m pytest tests/stage1/v5/test_gpu_utilization.py tests/stage1/v5/test_throughput_attention.py -q --basetemp=checkpoints/stage1-v5-gpu-utilization-20261018-tests/t1-green-fixed > logs/stage1-v5-gpu-utilization-20261018/t1-green-fixed.log 2>&1
CUDA_VISIBLE_DEVICES= PYTHONPATH=transformers/src:src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./venv/ilvr-venv/bin/python scripts/benchmark_stage1_v5_attention_selection.py --device cpu --warmup 3 --repetitions 20 --output results/stage1-v5-gpu-utilization-20261018/baseline-cpu.json > logs/stage1-v5-gpu-utilization-20261018/baseline-cpu.log 2>&1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=transformers/src:src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./venv/ilvr-venv/bin/python scripts/benchmark_stage1_v5_attention_selection.py --device cuda:0 --warmup 3 --repetitions 20 --output results/stage1-v5-gpu-utilization-20261018/baseline-cuda.json > logs/stage1-v5-gpu-utilization-20261018/baseline-cuda.log 2>&1
```

Pre-CUDA resource check: idle two H100 NVL devices/no processes, recorded in
`baseline-cuda-resource-check.log`. CPU baseline JSON SHA256
01983a777189e9a9a103c6ba706aee11278cb34a6a4a2b277907e8241712f50c;
CUDA baseline JSON SHA256
aaedfd07f9bf692ede952dff5a125bb342654b6157fb0b535efe57c53f96cb4a.
Measurement identity includes pre-commit HEAD91a1440 and exact committed
script/fixture hashes. All four exact flags true, attention maxerror0 and
global dropout RNG unchanged. These are unchanged-code operator costs; no
improvement or production/distributed equivalence is established.

Independent spec reviewer `/root/gpu_util_t1_spec`: SPEC PASS at5a7df38;
independently checked byte sources, timing order/medians/hashes/log matches,
actual dummy replay/dropout paths and declared scope. No gaps/extras. It noted
raw logs lack argv; the implementer's exact execution record is persisted
above without rerunning measurements. Quality reviewer
`/root/gpu_util_t1_quality`: QUALITY PASS at5a7df38, no Critical/Important
issues; independently checked all four source bytes, JSON/full-log identity,
script/fixture hashes, paired order/medians/memory arithmetic and equality/RNG
flags. No GPU or test rerun. Task1 is complete; Task2 may start. Final quality
response and any minor observations will be retained when received.

Task1 final quality response confirmed no Minor corrections either; all task
reviews closed. Control record commit43d71f3 precedes Task2.

## Task 2 implementation and review evidence

Implementer `/root/gpu_util_t2_impl`, base43d71f37872ad0294e0bd465fd0b2a167c1ff31a,
headb15fa466c3fc69d5b371788e6af531e556cb22bd. Root inspected the production
diff, read the complete causal RED and implementation-evidence JSON, confirmed
HEAD and clean status, and read focused GREEN's final result. Actual evidence,
commands/exits, content/log hashes, dispatch counts and caveats are committed
in `results/stage1-v5-gpu-utilization-20261018/task2-implementation-evidence.json`.
Raw fixed CUDA fixture report is `task2-operators-cuda.json` in the same tree.

RED task2-red.log:43pass/30CUDA-skip/2expected operation-count failures, exit1,
before production edits. GREEN task2-green.log:75pass/30CUDA-skip, exit0,
existing native loss/gradients and checkpoint off/full/early unchanged at
ATOL1e-5/RTOL1e-4. Actual dispatch growth with suffix/token count is removed:
replay select/copy230/32->3/0 at suffix32; dropout select/stack256/1->0/0,
ordinary host list elements512->3 at256requested tokens/3sources.

Freshly idle-checked CUDA operator benchmark exit0, warmup3/paired20:
mask6.885023->0.955306ms, dropout4.471019->2.1165875ms. Exact mask/raw dropout/
outputs/gradients and global CPU/CUDA RNG agree on that fixed fixture.
Mask extra allocated peak1115648->1129472B; dropout1414144B unchanged.
Native model integration remains CPU; CUDA edge parametrizations were skipped.
Do not infer production BF16 or distributed-throughput equivalence.

Independent spec reviewer `/root/gpu_util_t2_spec`: SPEC PASS atb15fa466;
actual source/paths, padding/escape/body/branch/gradient semantics, complete
RED/GREEN, source656 bytes and recorded CUDA identities checked. No gaps or
extras. Quality reviewer `/root/gpu_util_t2_quality`: QUALITY PASS atb15fa46,
no Critical/Important/Minor issues. It traced actual Trainer->replay->native
checkpoint/dropout and prefix gradients; independently ran CPU mask/dropout/
dispatch cases38pass/30CUDA-skip/7deselected and checked complete logs/hashes.
Task2 review gates closed; Task3 may proceed. Exact reviewer command requested
for this record without rerunning the test.

Task2 quality review's exact independent command (exit0,3.13s):

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=transformers/src:src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./venv/ilvr-venv/bin/python -m pytest -q -p no:cacheprovider tests/stage1/v5/test_gpu_utilization.py -k 'replay_bias or raw_dropout or diagnostic_coordinate or replay_without or dispatch'
```

Result38pass/30skip/7deselected. No model fixtures or log file were created;
complete stdout remains tool transcript chunk ae935d and this durable record
retains the command/result. Root did not repeat it. Task2 complete at control
recordd63db46; final quality verdict has no unresolved findings.

## Task 3 active implementation

Fresh implementer `/root/gpu_util_t3_impl`, based63db46ea8cd134b66ed42bf1935212c4ffaadf4,
received full Task3 requirements and existing reviewed dependencies. Scope is
stable prefix ranking with unchanged original-device masses/Python decisions,
then bounded attention chunks with early joint exact-selection comparisons.
Root emphasized detached in-place scale/mask lifetime management within the
32MiB score/probability target and honest actual peak memory. Any discrete
mismatch stops affected activation; no silent fallback/tolerance change.
No new human confirmation is needed for these approved internals.

Task3 implementer committed4888e7ad4a46c5044d03124bc381eec970564dd8 with base
d63db46ea8cd134b66ed42bf1935212c4ffaadf4; root control edits were excluded.
Root read actual trace/selector diffs, full selector RED/GREEN and early
CPU/CUDA logs and implementation evidence JSON (SHA256
cbda9a1b52baa28937ce52dcbde7c518c892d91e4d858bf5bc6a59b74785f45c).
Commands/exits/content/log hashes are durably in
`results/stage1-v5-gpu-utilization-20261018/task3-implementation-evidence.json`.

Selector RED1expected fail precedes code; GREEN43pass/57skip. Attention RED
shows old140GEMMs versus required8 at35queries/sample; early CPU7pass/3skip,
early CUDA61pass (including55selector placement cases). Joint old/new selected
tuples exact for normal/ties/near-ties at70queries each and seven measured
default/cumulative/lowmass boundary configurations. Max observed attention
absolute error CPU9.313225746154785e-10, CUDA1.862645149230957e-9, within
predefined1e-7/1e-5. Chunk constants32/33554432, remainders and budget-forced
chunk3 witnessed. These are finite operator witnesses, not universal proof.

Focused CPU six-file integration171pass/90CUDA-skip,16.95s; root read its
summary/native-gradient outputs and small early logs in full. Large JSON
outputs in the full integration log remain available for programmatic review.
Benchmark-report causal RED precedes new joint/byte metadata. Narrow final
self-review8pass/3skip follows test-helper cleanup. Selector benchmark payload
4160->2496elements but16640->14848bytes (int64 ranks separate from FP32 values),
not the same percentage. Native full-sequence oracle/frozen sources unchanged.

Independent spec reviewer `/root/gpu_util_t3_spec`: SPEC PASS at exact diff
d63db46..4888e7a, no gaps/extras. It traced actual source and independently
checked hashes, complete JSON reports and non-JSON integration-log text;
confirmed boundary cases produce distinct choices (first-query cumulative
2vs3 and lowmass38vs12), so labels are backed by actual branch effects.
Quality reviewer `/root/gpu_util_t3_quality`: QUALITY PASS at4888e7a,
no Critical/Important/Minor issues. Traced production callers/native attention,
checked six source and nine log hashes and complete focused-log data; fresh
CPU selector/joint/chunk tests30pass/3skip reproduced exact keys/maxerror/chunks.
Task3 review gates closed; Task4 may proceed. Reviewer command requested for
the durable record without another run. No final performance claim from the
one-repetition CPU acceptance report.

Task3 quality review command (exit0,30pass/3skip/131deselected,3.32s), stdout
captured in tool transcript6787e5 without a separate log:

```bash
env CUDA_VISIBLE_DEVICES= PYTHONPATH=transformers/src:src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./venv/ilvr-venv/bin/python -m pytest -q -s -p no:cacheprovider tests/stage1/v5/test_gpu_utilization.py -k 'selector_ordered or selector_transfers or joint_attention or attention_dispatches' --basetemp=checkpoints/stage1-v5-gpu-utilization-20261018-tests/t3-quality-review
```

## Task 4 active implementation

Fresh implementer `/root/gpu_util_t4_impl`, base
ffec431b6abac35e470667a2e9d0a049b91c43c6, received full identity/admission/
actual-train-dispatch TDD requirements and final paired CPU/CUDA default/long
measurement task. Exact source keys/profile/policy are fixed in the plan;
actual module files must be hashed, not decorator-wrapper files. Historical
read-only parsing stays distinct from strict new runtime resume. Existing
operator algorithms, losses, sampler, rank allocation and training artifacts
are outside this task's changes. No new training is authorized.

## Task4 exactness failure and stopped progression

Task4 implementer parked WIP commitb3e13a4abcc237e10c397eb076be8bd3b79a9f55
at the first real long-profile exact-selection failure. It includes runtime
identity/admission/resume and long-geometry controls/tests, not operator
algorithm changes. Root control edits were excluded. Source/profile causal
REDs precede code; current focused result is266pass/88skip/1fail,exit1,19.02s.
Exactness is red; Task4, final review, full-suite finishing and performance
completion have NOT passed. No final CPU/CUDA measurements or training ran.

Failure: CPU B2/T8192/H8/D32/L2/Q67per sample/candidates2048 per query,
chunks32/32/3. Ten failed case/query pairs: nine adjacent-order mismatches
across three queries, and cumulative_equal at(1,8125) changed the selected
set from[289,1235] to[289,1235,2493]. This is not accepted floating drift.
Both query-GEMM and head-reduction shapes differ; the existing trace alone
does not localize the earliest numerical difference. Do not assume a cause
from method names or reclassify set differences as harmless ordering.

Committed task evidence:
`results/stage1-v5-gpu-utilization-20261018/task4-evidence.json`, SHA256
be033964bab41b85ae1b2e82470cb5c8c3fcae5448894be4e4aea1ec970a0875;
full failure and parsed diagnosis `task4-cpu-long-discrete-failure.json`, SHA256
cdb66d95c382c13e71db07958d855a2f2bbed248af8ea4b84190858d3ba4a27e.
Raw t4-green.log preserved, SHA256
768f485b123e75e8c7f6606f8fd73d84c653f4a15def53f2eb11513640a890dc.
Root inspected actual failure, summary/source hashes and actual runtime diff.
The original Task3 passes remain scoped historical evidence, not clean
approval over this newly falsified long-input claim.

User informed of failed selected set and stopped integration/measurements.
Root routed a single narrow causal diagnosis to original implementer
`/root/gpu_util_t3_impl`, with Task4 parked. First localize scores versus
softmax/head/layer reduction and compare the unchanged selector on the same
changed rows. No production fix until a concrete numerical repair is
adjudicated. Keep geometry/tolerance/seed/thresholds/oracles unchanged.
Assess whether original matrix-vector arithmetic can be retained within the
approved bounded query chunks and memory, without large K duplication,
fallback, quantization, global backend changes or new framework. A routine
internal correction may proceed under the existing approval; if it requires
a material architecture change, return to the design gate. No silent drop of
the attention path or weakening of exact selection is authorized.

## Narrow numerical diagnosis and authorized correction

Original implementer `/root/gpu_util_t3_impl` reproduced the exact recorded
long failure under CPU. Actual old/new Q and K are bitwise equal; earliest
observed difference is raw batched dot score (up to9.5367431640625e-7), before
softmax/head/layer accumulation. Both selectors agree on identical changed
rows across all seven configs. Thus the selector is not canceling a trace
error. Diagnosis/evidence JSONs and two logs are preserved under the existing
results/logs tree. First observer-only diagnostic attempt failed because the
observer returned an expanded result; corrected observer-only run passed1test.
Neither changed production code or thresholds.

The one assessed repair retains query chunks32/32/3 and batches independent
M=1 matrix-vector products per head with zero-stride K views, then stacks
heads. CPU test-only probe restored bitwise-equal candidate rows at all134
long queries and exact seven-config selections, maxrowerror0. This is finite
CPU evidence; CUDA behavior/performance is not inferred from it.

Root adjudication: replacing only the dot expression this way stays within
approved per-sample/layer bounded query chunks, same heads, native math,
memory target and exact discrete policy. It introduces no fallback, K repeat,
backend, formula, input or scope change. Root authorized the original
implementer to apply this narrow corrective change under existing approval,
using the preserved long case as causal RED and fresh CPU/CUDA gates. The
plan's kernel-count witness must reflect heads*chunks/M=1 operands instead of
the failed single all-head GEMM assumption; no functional test is relaxed.
Independent spec and quality fix re-review remain required. Task4 stays
parked; final performance/full-suite/finishing remain unpassed.

## Numerical correction implementation and fix review

Original implementer `/root/gpu_util_t3_impl` committed the narrow correction
at4440fe78b72b71303ab30f29b1b73aa11183c1f2, based onb3e13a4. Root inspected
the exact production diff: only the trace score construction changes to
per-head M=1 BMM with expanded K views. Root verified the evidence digest
`2ce0a9216a4123605e356c7ae7bb3b6fdde6655a0337c752679820ba0242116d`
for `results/stage1-v5-gpu-utilization-20261018/task3-numerical-repair-evidence.json`.
That artifact records commands, exits, source/log/report hashes, diagnosis,
RED/GREEN, unchanged oracles and finite-case numerical limits. Root control
edits were excluded from the implementation commit.

Additional causal RED: five failures/one pass/two skips before the production
repair, exposing M>1 operands and the same long discrete mismatch. After
repair, early CPU10pass/5skip and idle-checked CUDA10pass. Native CPU focused
integration95pass/2skip includes the previously failing default/long benchmark
acceptance and the unchanged loss/gradient oracle/checkpoint modes. Long
CPU/CUDA rows and seven-config tuples match exactly; `(1,8125)` is restored
to[289,1235]. CUDA default rows are close rather than bitwise, max absolute
error4.656612873077393e-10, with all tested ordered tuples exact. No final
timing, actual peak-memory or distributed/BF16-equivalence claim is made.

Independent fix spec reviewer `/root/gpu_util_t3_spec`: SPEC PASS over
b3e13a4..4440fe7. It independently checked every recorded source/log/report
hash and unchanged failure artifact, fully parsed diagnostic and acceptance
JSON plus the7,746,141-byte native log, and inspected the remaining text.
It verified M=1, head*chunk dispatch, remainders, one current-sample FP32 K
copy/storage alias, unchanged subsequent arithmetic and the restored causal
boundary. No missing requirements or unauthorized changes. QUALITY fix
re-review has been dispatched to `/root/gpu_util_t3_quality` after SPEC PASS;
Task4 remains parked until that gate closes. Fresh final verification and
whole-change review are still pending.

Independent fix quality reviewer `/root/gpu_util_t3_quality`: QUALITY PASS
at4440fe7; no Critical/Important/Minor findings. It traced actual memory
lifetimes and current-sample K alias, checked source/oracle/report/log hashes,
fully parsed the native log and reran the narrow CPU dispatch/long/default
regressions:6pass/164deselected, exit0, stdout tool chunkdbed42. Exact command:

```bash
env CUDA_VISIBLE_DEVICES= PYTHONPATH=transformers/src:src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./venv/ilvr-venv/bin/python -m pytest -q -s -p no:cacheprovider tests/stage1/v5/test_gpu_utilization.py -k 'attention_dispatches or (joint_matrix_vector and cpu)' --basetemp=checkpoints/stage1-v5-gpu-utilization-20261018-tests/t3-repair-quality-review
```

The original numerical failure is resolved for the checked cases and both
fix-review gates close. Resume the same parked Task4 implementer; no new
scope approval, input/tolerance change or training launch is involved.
