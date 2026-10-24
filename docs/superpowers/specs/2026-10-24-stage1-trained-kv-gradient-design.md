# Trained Stage1-v5 K/V reuse loss and gradient diagnostic — v2

Status: design proposal for human review; no diagnostic source implementation authorized yet.

## Intent and evidence boundary

Measure the local numerical difference between the historical Stage1-v5 native
attached-prefix replay and independent full-sequence decoder recomputation on
the trained checkpoint-648 and a real TRAIN example. Compare objective
components and every trainable parameter gradient. A scalar loss match alone
does not establish gradient equality.

This is a standalone forward/autograd diagnostic. It performs no optimizer
step, Trainer.train, checkpoint save, scheduler update or distributed training.
It cannot establish historical training-trajectory, four-rank ZeRO/DDP,
gradient-bucket, optimizer or convergence equivalence. Any later training run
still requires the canonical two-node/four-rank coordinator and its gates.

The separately approved additional GPU budget is 300 aggregate process-seconds;
historical use is 289.42876996000996 seconds. That budget approval is separate
from approval of this design.

## Grounded facts

- The checkpoint is `checkpoints/stage1-v5-mn-g16-e3-s648-20261014-150317/checkpoint-648`.
  The original Qwen2.5-VL-3B base and `stage1_v5_reference.pt` are present.
- Historical revision `07eb774281925866502e72bdfcaa4df54073103a` matches the
  checkpoint's recorded Stage1, Stage1-v5 and framework Trainer SHA256 values.
  This does not authenticate every historical runtime byte or environment detail.
- Current code uses a different DIL normalization. The probe must execute the
  historical objective, not silently substitute today's token normalization.
- The saved student uses BF16, DoRA r32/alpha64/dropout0.05, SDPA decoder,
  FlashAttention2 vision and checkpointing. The adapter artifact contains
  BF16 tensors. Saved reference identity and dtype must be preserved separately.
- Existing tiny CPU tests and a synthetic CUDA Stage2 diagnostic are useful
  prerequisites; they are not trained Stage1-v5 BF16 evidence.

## Approaches

The chosen scope is an unchanged-weight native loss/backward comparison. This
continues the user's actual-checkpoint diagnostic request and preserves the
explicit no-training/checkpoint-change condition. A full distributed training
or optimizer comparison is a larger experiment. Repeating synthetic tests
cannot answer the trained-artifact question. Neither alternative substitutes
for the local comparison proposed here.

Implementation remains through the user's selected subagent-driven method.
The controller proposes this diagnostic scope; no new training execution
profile or repair is adopted by it.

## Runtime and files

Use an isolated diagnostic descendant of `07eb774`. Add only a focused CLI
`scripts/probe_stage1_training_kv.py` and its CPU regression file
`tests/stage1/v5/test_training_kv_probe.py`. Do not edit native model,
Trainer, loss, coordinator, existing tests or production launchers.

Reuse the historical training loading, sparse-row, target-builder, reference,
collator and objective APIs. The training-shaped loading order already used
by `scripts/analyze_stage1_v5_causal.py` is evidence for the sequence, not a
dependency on another checkout. All imported native modules must resolve
inside the isolated historical descendant; no mixing current and historical
runtime modules. Record actual imported module paths and hashes for Trainer,
model, trace/replay, stochastic, target, data and loss modules, plus framework
paths/hashes and package versions. Fail explicitly on wrong import roots.

Use only the repository venv. Preserve `data/`, `checkpoints/` and venv links.
Inputs/cache remain under data; results/logs use this worktree's own trees.
All model-bearing CPU fixtures use a unique explicit checkpoint basetemp.
No gradient/model tensor dumps are required; save scalar/hash summaries only.

## Fixed real input and state

Before any GPU metric, select the first retained TRAIN row by ascending source
index with one or two original latent blocks, at least one scored target,
native collated length at most 512 and complete original/helper images.
Use native preparation and AddedToken processor construction; do not truncate,
change image order or substitute synthetic data if no row qualifies. Save
source/row/image/processor identity, the actual coordinates and lengths.

Load the student as trainable through the historical PEFT/row factories.
Restore immutable reference with its authenticated base/processor/effective
state identity. The student BF16 representation must match the saved artifact;
do not silently retain evaluation PEFT upcasting. Preserve the authenticated
reference's own dtypes. No persistent model artifact is cast or rewritten.

Keep saved CE0.1/sim0.1 and every saved DIL/entropy/TextKL coefficient and
formula. Keep source-coordinate dropout0.05 and native latent checkpointing.
Use no autocast for this explicit BF16-storage pre-engine diagnostic; record
the actual ambient/autocast/module-boundary dtypes. This does not certify the
original engine's runtime precision.

Define a local normalization witness explicitly rather than invent a historical
sampler window: use a diagnostic window representing this same example repeated
across a nominal full global-16 window. With B original blocks, the historical
V5Window is `(global_original_steps=16*B, actual_microbatches=4,
configured_gas=4, world_size=4)`, yielding the native block scale `1/B`.
These are objective coefficients, not a claim that four ranks participated.
Report the raw local objective and its loss/4 autograd gradients separately.
This is a valid controlled per-example objective, not the actual historical
heterogeneous-window gradient. Report raw components to make that boundary clear.

## Native carrier prerequisite

Invoke historical `CustomTrainerStage1V5.compute_loss` through a minimal
diagnostic carrier with authentic V5Execution and backward contexts. Reuse
native helper preparation and reductions. The carrier bypasses some framework
initialization and profile checks; enumerate those boundaries in the report.

Before GPU, a CPU witness must compare carrier-native loss and every gradient
with the actual historical tiny parent training_step dispatch under matching
local loss/GAS and dropout/checkpoint conditions. It must identify real versus
substituted engine pieces. It does not validate historical ZeRO/DDP scaling.

## Independent full-prefix counterfactual

The native arithmetic is unmodified. Only the oracle arm temporarily replaces
the native `run_v5_packed_replay` call target. A separate passive observer may
wrap the actual native entropy-support builder, returning its unmodified result.
Preserve exact original binding presence/value on success and exception exits.

The oracle recomputes complete decoder sequences from controlled original
decoder inputs. Specify their channel separately: a passive student-decoder
pre-hook records the first native F call's live prepared inputs_embeds,
position IDs and padding, before prefix computation. Preserve the live graph
from trainable token rows; helper targets remain detached by native preparation.
The oracle receives only this restricted input view and frozen block/target
coordinates. It does not extract inputs or positions from the native trace,
and cannot read captured K/V, attention rows, native visibility/planner output
or prefix-state tensors. Supply neither past_key_values nor
stage2_prefix_key_values. CPU sentinels must detect forbidden trace reads.

Sharing the original input-preparation graph is intentional: this isolates
decoder-prefix state computation, not separate full-policy embedding/image
preparation. CPU witnesses must cover a live token-row gradient, detached
helper targets and two overlapping prefixes, and detect detachment of the live
input channel. Do not claim that this oracle independently re-embeds images or
tokens for every branch.

Build masks independently from literal causal and source coordinates. Keep
the whole prefix subgraph clean. V-off removes latent-body keys only from the
same suffix/query rows affected by native replay; START/END retain visibility.
Text-off removes exactly the selected target-query/key edges. A test must
detect the wrong all-row-masked variant.

A batched collection of full sequences may preserve the single packed graph
ownership. Its source-coordinate dropout mapping must align each token with
the native arm. Use the original source-coordinate policy, not arbitrary
RNG reseeding as a substitute. Checkpoint recomputation must retain those
bindings. All native loss reduction functions remain unchanged.

Use saved seed42 and the checkpoint's recorded global_step as a declared
diagnostic context key, with the selected sample's raw/canonical identity,
valid length and exact stochastic-site inventory recorded. This defines a
common local realization; it is not evidence of the historical training draw.

## Measurements and validity

Run fresh unobserved native U, observed native A, observed native A-repeat and
observed oracle B with identical represented parameters, input, reference,
helper targets and controlled stochastic context. Verify native observer
fidelity U/A and native repeat A/A-repeat before interpreting the contrast.
If those controls change loss/gradient or helper/reference/clean state,
attribution is inconclusive; do not weaken a control after seeing results.

Clean-derived Text-off keys and original block coordinates must match across
arms. Entropy supports are different: the saved `independent` mode derives them
from each arm's V-off logits through the unmodified native
`build_entropy_support`. Preserve and passively observe each actual result;
never substitute A's support into B or force a common support. Record indices/
hashes, overlap and equality for the actual calls. If supports differ, report
the native loss/gradient contrast as including a discrete selection-mediated
effect. That difference alone does not make the native comparison invalid,
but it cannot support an isolated fixed-support rounding claim.

Use autograd without optimizer or parameter updates. Compare every trainable
named parameter, including sparse token/head rows and DoRA magnitude. Distinguish
matched unused gradients from a presence mismatch; require nonzero trained
adapters and at least one nonzero gradient. Record parameter/gradient dtypes,
finite flags, exactness, max absolute error, per-parameter and global relative
L2 and cosine, and actual loss components and local coefficients.

Runtime witnesses must show native attached-prefix suffix calls versus oracle
full-prefix calls without prefix inputs. Authenticate parameter values before
and after each arm; capture input/reference identity and no update/save events.

Native-repeat spread is repeatability evidence, not the tolerance for a
systematic cross-shape BF16 difference. BF16 results are measurements, never
declared equivalent using FP32 tolerances. A discrepancy establishes the
concrete native-versus-oracle contrast, not a located cache defect, wrong
optimizer gradients or a precision remedy.

Record active V-off/Text-off edges, scored queries and component magnitudes.
Keep the fixed first-row selection rule if a branch is inactive and label it
unexercised. Matching the total represented BF16 parameter gradients does not
establish equality of each OFF-branch Jacobian or hidden intermediate; no such
claim follows without a separate component-gradient witness.

## First nail and execution gates

The load-bearing premise is fidelity of the carrier and oracle to the same
historical local objective and stochastic realization. Before GPU, demonstrate
the CPU carrier/native-parent match and detect detached-prefix and wrong-
visibility controls with nonzero DoRA/row gradients. Also force a controlled
V-off ranking crossing in CPU fixtures: each native independent support must
remain arm-specific and be observed without replacement. The passive observer
must preserve native loss/gradient and restore all bindings. Those tests are
proposed work, not evidence already obtained for this diagnostic.

On GPU, native-repeat helper/reference/clean/selection/dtype fidelity precedes
the oracle comparison. Missing evidence, unsupported runtime, nonfinite values,
OOM or deadline yields an incomplete/inconclusive report; no synthetic,
dropout-off or shortened-input success fallback.

After human approval: plan when triggered, selected subagent TDD, independent
specification/quality task review and broad final review, one final project CPU
suite on final reviewed bytes and identical-content commit. Root then runs the
bounded GPU diagnostic with a managed process deadline. Loading, failures,
forward/backward and report time all count against the new aggregate300s.
No generalized retry/coordinator/recovery infrastructure is in scope.

## Human decision and unresolved uncertainty

Approve this local trained-artifact diagnostic, not a training repair or full
training equivalence claim. A dataset-wide or distributed-engine comparison
requires a separately scoped experiment. Carrier fidelity, independent
source-coordinate oracle behavior, the qualifying row and GPU feasibility
remain to be demonstrated. Matching two Trainer hashes is partial historical
authentication; do not overstate it.

This document is Superpowers control material. The approved budget does not
authorize source implementation before this design's final approval.

The independent builder question on v1 identified the entropy-support issue.
Its source-based answer is incorporated above. v1 was not implemented or
human-approved. Existing saved formulas remain authoritative; this revision
does not add a fixed-support objective.

Controller proposals made explicit for human review are the local repeated-
example normalization, shared original input-preparation graph, standalone
pre-engine boundary and declared diagnostic stochastic context. They are not
being represented as previously agreed historical execution details.
