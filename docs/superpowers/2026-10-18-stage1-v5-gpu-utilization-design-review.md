# Stage1-v5 GPU utilization design commitment review

## Frozen input and authority

Proposal v1: `specs/2026-10-18-stage1-v5-gpu-utilization-design.md` at
`93bd9519a73ab4eda4c1c820ae5a522db508924a`, SHA256
`e036cc923079233ad2ccf4bc0807164c827337742ca303a974aa506a91c9aec0`.
Do not edit that proposal between critique passes. Root performs the advisory
hooks under the user's AGENTS standing invocation. No implementation agent,
premise experiment or product change is authorized by these reads.

## Root hate result

```json
{
  "root": "The four-path design requires exact selected tokens, yet its batched-attention path permits numerical perturbations before a discontinuous ranking/cumulative-threshold decision. No production boundary witness or sound runtime guard establishes that these two requirements can coexist. A close attention-row comparison or selector-only test cannot establish the required joint behavior; one changed key can change the training intervention.",
  "first_nail": "Before committing to the attention rewrite, compare the frozen per-query and proposed chunked attention plus the unchanged selector on identical deterministic Q/K, padding, candidate coordinates and logits on the installed CUDA backend, concentrating cases at equal weights and nextafter neighbors of the cumulative and low-mass boundaries. One changed selected-key tuple falsifies acceptance of that batching path. This requires a small implementation-like operator probe, not a model load or training run; it is proposed only, not authorized or executed."
}
```

## Root macrothink invocation

Applicable: choosing the wrong optimization or altering discrete interventions
can consume or invalidate multi-hour distributed runs. Three bounded fresh
same-model read-only reads cost materially less. The input below is the same
stripped goal/fact/constraint extract from frozen proposal v1 for every read;
it omits the proposed implementation and the hate conclusion. Agents are not
given each other's conclusions. Root alone classifies and synthesizes later.

Frozen underlying problem: improve useful Stage1 training throughput while
preserving sample order/rank assignment, teacher/reference behavior, losses,
exact discrete token choices, attention masks and source-coordinate dropout
randomness. Small predefined tolerances are allowed for floating observation,
loss and gradients. Production uses two nodes/four ranks, variable sequences
and one clean plus packed counterfactual decoder path. Observation sampled
roughly58--64% GPU busy time; main process CPU exceeded one core, workers were
near1%; there is no stage/CUDA attribution profile. Source includes per-query
observation, host selection logic, per-query replay-mask construction and
per-token dropout assembly. The selection decision ranks candidate weights
and accumulates a prefix against a data-dependent threshold. Existing tests
provide small CPU/native-model and frozen-source comparisons. Current task
authorizes a design and local development after final approval, not a fresh
training launch. Preserve historical checkpoints and strict resume source
identity. No sampler/rank redistribution or new framework is in scope.

Question to each fresh read: Given only these facts and constraints, what
development direction should the controller choose, what load-bearing premise
must it challenge, and what is the cheapest useful evidence? Separate observed
facts from inference. Inspect source read-only if helpful, but do not read the
proposal/workflow/other critiques, modify files, run tests or GPU/model work,
invoke advisory hooks, or spawn agents. Return an independent concise analysis;
do not vote on an existing proposed solution.

## Read results and synthesis

All three reads completed independently with `fork_turns=none`, no model or
effort override (same inherited model), and the same stripped prompt above.
They were instructed not to read the chosen solution, one another's output,
or invoke any hooks/agents. Their advice is not independent implementation
review, runtime evidence or authorization. Root received the following actual
results and classified each against the frozen four-path direction.

### Divergent-incompatible: attribution before a fixed four-path rewrite

Shared root: low busy time plus CPU activity does not establish which work
limits useful distributed throughput. Readers A/B prefer attributing the
unchanged critical path before choosing one component. C adds that changed
observation arithmetic must be gated on discrete boundary evidence. This is
an alternative priority to implementing all four chosen paths now, not an
established failure of the current design. The missing runtime profile is
unresolved, rather than explained away by the baseline unit tests.

Reader `/root/gpu_util_fresh_a`, primary `divergent-incompatible`:

- Recommended measurement-gated work on one demonstrated contributor,
  retaining execution/sample assignment/objective and measuring elapsed time.
- Source observations: repeated source-coordinate transfers/token assembly;
  selector already has one host transfer and deterministic sorting/accumulation.
  These are static structures, not measured costs.
- Named the load-bearing premise as sufficient exposed critical-path time in
  those Python-heavy operations. Alternative causes include Teacher, runtime
  communication, memory pressure and sequence costs; low worker CPU does not
  establish a data-loading bottleneck.
- Proposed one authorized warmed CPU/CUDA timeline over clean/Teacher,
  selection, preparation, packed forward and backward, then a bounded target.
  For dropout, preserve generator/seed/draw shape and verify masks across
  clean/packed/recomputation before gradients. Preserve selector arithmetic;
  later changes need adversarial exact discrete checks.

Reader `/root/gpu_util_fresh_b`, primary `divergent-incompatible`:

- Recommended a measurement-led bounded optimization: find a dominant
  overhead and batch/cache repeated metadata rather than commit to broad
  vectorization before attribution.
- Observed per-query attention/mask assembly and existing packed selector
  transfer; additional transfer work might have little benefit.
- Named the untested critical-path premise; decoder, communication, rank
  imbalance and observation work remain alternatives.
- Proposed a short unchanged CPU/CUDA timeline with clean/observation,
  selection/mask/dropout preparation, packed, backward and communication.
  Fresh training profiling needs separate authorization. Then compare exact
  keys/masks/dropout and predefined gradients on representative inputs;
  component speed does not prove training speed.

Reader `/root/gpu_util_fresh_c`, primary `divergent-incompatible`:

- Recommended attribution first and a narrow demonstrated target, preserving
  original arithmetic and RNG generation.
- Source observations: selector's single packed transfer/Python decisions,
  per-query observation launches and per-token dropout assembly.
- Challenged the combined premise that batching improves critical-path time
  without changing selection; near ties/thresholds, floating tolerance does
  not protect discrete choices.
- Proposed an authorized warmed timeline plus an identical-input differential
  boundary probe before loss/gradient tolerance comparisons. Suggested
  dropout gather first if attribution supports it, and deferring attention
  GEMM/reduction shape changes until joint selection behavior is tested.

### Divergent-compatible contributions

Preserve the measured one-transfer selector baseline, rather than claiming
the optimization newly removes per-query transfers. Record transfer sizes and
sort time. Use warmed representative geometry and actual wall time; preserve
source RNG generation and exact keys/masks through recomputation; gate
attention integration on joint observation/selection comparison. These
contributions fit the user's existing four-path scope and numerical policy.

### Convergent aspects: reassurance only

The reads favor retaining execution/objective/sample assignment and original
boundary/RNG semantics. This partial alignment is reassurance only; no fresh
read is being relabeled a proof or a correctness/performance pass. Each read's
primary classification above remains divergent-incompatible. No majority
vote, average or agent count decided the design.

## Superpowers controller synthesis

Retain the user-chosen four-path scope and predefined numerical policy.
Add baseline operator costs before local changes, exact mask/dropout work
first, and joint attention boundary evidence before integration of that path.
Use paired timing and actual peak memory, report operator regressions and
keep actual distributed throughput unverified absent its own profile/run.
Do not silently switch to a one-operator scope, invent new orchestration or
launch a training profile. Profile-first before any rewrite remains a valid
unresolved priority disagreement, explicitly disclosed in commitment v1.

Hate's first nail remains **proposed, not authorized/executed**. No premise
experiment result exists. Final approval must explicitly accept the lack of
pre-implementation boundary/attribution evidence while retaining all later
implementation, equivalence, independent-review and launch requirements.

Exact final approval artifact:
`specs/2026-10-18-stage1-v5-gpu-utilization-commitment-v1.md`.
No implementation or experiment is authorized by this critique record.
