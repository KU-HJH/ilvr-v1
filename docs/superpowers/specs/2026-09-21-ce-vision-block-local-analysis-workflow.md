# CE Vision block local analysis — workflow evidence

## Approval and phase

- Active owner: root controller `/root`; Superpowers architectural design path.
- User selected: `CE-only Stage 2 baseline`.
- User specified the ordered four conditions: clean; Body masked; boundaries masked/replaced; entire block masked with length retained.
- Presented conversational definition: preserve length/position; independently fork from clean; mask root K/V; replace boundary-off END input with zero and replay; keep clean Body fixed for boundary-only. Root asked whether to write the design document using this definition. User replied `ok`.
- Bound approval: write the spec for that conversational scope. Written-spec approval, implementation-plan approval, product implementation and GPU execution are not yet evidenced.
- `understood as`: quantify teacher-forced next-text sensitivity at about 100 TEST samples with the five user metrics, on a fixed CE-only Stage 2 artifact, while independently toggling direct Body access and boundary input/access.

## Selected skills and actual work

- `using-superpowers`: read with Codex adaptation; establishes lifecycle routing.
- `readchk`: model referent was clarified with the user; no unexplained switch to preference weights or another Stage 1 initializer.
- `brainstorming`, architectural path: read-only project exploration and conversational scope confirmation complete; document drafted; written review remains pending.
- Built-in spec self-review: checked target scope, common denominator, END predictor alignment, clean prefix isolation, output storage and exclusions. Set numeric preflight thresholds before any experiment; no numerical comparison run is claimed.
- `hate`: executed by root on frozen proposal v1; exact result below.
- `macrothink`: read and applicability considered, not executed. User fixed model, four conditions and metrics. Remaining decisive uncertainty is implementation fidelity, addressed by a narrow runtime witness; broader alternative-problem fan-out does not presently justify its cost.
- `prism`: executed by root as two distinct same-session lenses on frozen proposal v1. This is not two independent code reviewers and is not runtime evidence. Results below.
- `mandela`: read-only root audit of target independence and interpretation. The potential **Wrong null hypothesis** is treating boundary-only as removal of all boundary influence, or full-mask as removal of all visual evidence. Sections 4–5 explicitly scope the intervention to current-block direct access and END input; original image/history and clean Body's START history remain. NLL targets come from dataset annotation, not generated clean predictions. No claim about absence of training contamination or universal annotation truth is made.
- `shower`: cold-read agent `/root/ce_block_spec_cold_read`, fresh context (`fork_turns=none`), supplied inline document contents and prohibited adjacent-file exploration, edits and child skill invocation. Verdict: **minor gaps**. It correctly understood the four-condition CE-only teacher-forced experiment and identified three handoff gaps: scalar-looking position shorthand, independence/entrypoint of the oracle, and post-tokenization availability of smoke samples. Root clarified original three-axis column selection, nonshared visibility construction and ordinary `_forward_stage2` clean entrypoint, plus a same-sample single-branch fallback. These edits are to controller-owned design artifacts, not product deliverables. No fresh-reader clean approval is claimed for the edited revision.
- `verification-before-completion`: read; final document/hash/metric/source checks recorded below. Finishing-a-development-branch does not apply to this design-only handoff; no product implementation branch has been completed.

## Frozen proposal identity

- Source: `docs/superpowers/specs/2026-09-21-ce-vision-block-local-analysis-design.md`.
- Frozen copy: `results/ce-vision-block-local-design-20260921/design-proposal-v1.md`.
- SHA-256: `0e6c66c61bb81f3d6179ad0377ab5c811b488aa0d4b114095a1203e6675d4c70`.
- Repository HEAD before these control artifacts: `31f745e02a81619d6ee12183b8f2e9c699c80902`.
- CE launch record lists training code commit `f8f80be29182a5d96764f1717a50382bf3b2704c`; this does not certify a future analysis checkout. Compatible reviewed source identity is a launch prerequisite.

## hate output

```text
{root: "A residual clean END/future K/V path or an unmatched replay path could create or hide apparent token changes, so the four labels alone do not establish an intervention on the intended information paths.",
 first_nail: "After authorized implementation, use a one-site/two-site tiny deterministic model: no-op replay must match a separate reference, mutating captured blocked root K/V must leave off outputs unchanged, and a deliberately constructed visible-Body control must respond. Then check clean replay fidelity on one single-site and one multi-site real CE sample before admitting the 100-sample run."}
```

First-nail status: **not run**. Current approval authorizes design writing, not the implementation or GPU probe. Static evidence from the existing sidecar shows END-before cutoff and replay support; it does not prove the future boundary-only evaluator. No waiver of this premise is requested or granted: its runtime gate remains mandatory before the main run. Design review can proceed without asserting that gate already passes.

## prism synthesis

1. **Causal/runtime fidelity — unclear:** the frozen design excludes clean END/future state and keeps Body fixed, but the novel boundary-only path has no runtime witness. Its acceptance depends on the first-nail gate.
2. **Measurement/target independence — pass at specification level:** common site/target sets, shifted predictor alignment, annotation targets and token-weighted NLL specify the requested question without replacing it with final-answer accuracy. Implementation and tokenized denominators remain unverified.

Agreement: the proposed scope measures local teacher-forced sensitivity; neither lens supplies evidence of completed runtime behavior. Difference: metric contract is assessable now, execution fidelity is conditional. Single next question: **Does the actual CE analysis path pass no-op fidelity and blocked-source perturbation checks on identically aligned targets?** Owner: future implementer plus independent reviewer; resolving command belongs in the implementation plan and must pass before the main evaluation.

## Observed source evidence

- Existing 100 selected indices: unique, with category counts 25/30/15/30. Read-only enumeration of TEST found 327 nonempty immediate-text candidates across those 100 rows. This is not the post-tokenization S or N.
- Data SHA-256: `6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618`.
- Historical selection metadata SHA-256: `fb1f15fef5339abde6ba74d140e5f9a1f8823cedafe8aef1d4bb245a8e0286aa`.
- CE prepared artifact exists; metadata identifies latent size 8, START 151666, PAD 151665, END 151667, original base and reproduced Stage 1 initializer. Model weights were not loaded.
- Source inspection: `Stage2DataCollator` rejects helper-image tensors; `SidecarBranch` has a contiguous root block plus `zero_first_embedding`; prefix visibility ends strictly before END; suffix states are branch-local causal. Existing scorer returns mean target log-prob and entropy, so per-token predictions/NLL require an analysis path. These are source observations, not a new runtime test.

## Completion evidence

Final commitment v1 is specification §9. Changes from the frozen proposal clarify position selection, oracle independence and smoke selection; no material experimental scope or behavior changed. Root rechecked the same causal/runtime and metric-contract claims after these clarifications: runtime fidelity remains gated and unverified, metric contract remains specified. No broader critique rerun or extra cold-read cycle was used as a substitute for future runtime evidence.

Fresh verification output is recorded at `results/ce-vision-block-local-design-20260921/document-verification.json`; exact design/workflow commit identity is recorded at `results/ce-vision-block-local-design-20260921/commit-receipt.json`. Missing receipts mean those steps are not evidenced. Verification re-reads the document, checks the four conditions/five metrics and approval state, validates referenced input/checkpoint paths and hashes, and recomputes the 100-sample/327-candidate structure. A path-limited staged whitespace check precedes a commit containing only these two controller-owned documents. This is document/source verification, not model or product test execution. No product files changed or GPU inference launched for this design.
