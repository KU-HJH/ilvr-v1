# ILVR visual causal analysis — mask-control amendment v3

Status: proposed implementation commitment; final written-design approval pending. The user's latest `ㅇㅋ` approves preparing this spec/plan revision in the direction presented, not starting the revised experiments. This amendment remains inside the architectural brainstorming gate.

Parent contract: [approved v1](2026-09-10-ilvr-visual-causal-analysis-design.md). This amendment replaces only the E3/E4 comparison control and its validation/launch requirements. All other v1 requirements remain in force. Companion [draft execution plan](../plans/2026-09-10-ilvr-visual-causal-mask-control.md) supplies task commands and handoff; read this amendment, its plan and v1 together. Implementation source before this amendment is `a78ea0576348885326c2708185081a2c6a2395b3` in `/root/workspace/ILVR/.worktrees/ours1-visual-causal`; original native source is `ff53706882e39c9ce2ce4ac9e75242e1438ffc14`.

## Problem and observed evidence

E0 with an ordinary 2D validity mask and E3/E4 with an explicit 4D additive mask instantiate the same SDPA decoder class but need not execute the same CUDA operator. On the current H100 NVL / PyTorch 2.7.1 / bf16 runtime, a real Stage2A probe selected Flash SDPA for 2D and efficient SDPA for neutral causal4D. The neutral mask removed no image attention edges.

At CoMT original index3, max_new_tokens64, 63 paired identical-input decoder calls covering one complete latent block had maximum last-position logit absolute difference0.75 and zero raw LM-head argmax changes. Each pair received the same embeddings and positions; the 2D output alone drove the subsequent baseline trajectory. This does not demonstrate equivalence of independent full continuations or measure an accuracy difference. Earlier tiny float32 masking witnesses remain evidence only for their tested scope.

Evidence: `results/visual-causal-real-sdpa-dispatch-probe-20260910-r1/{dispatch.json,analysis.md}` and `results/visual-causal-cuda-sdpa-probe-20260910/dispatch.json`. Reproducer: `logs/visual-causal-implementation-20260910/real_sdpa_dispatch_probe.py`. No inference about visual non-use follows from this diagnostic.

## Chosen design and alternatives

Add the opt-in CLI mode `causal_mask_control`. It executes normal free latent generation with the original image, but sends the same complete additive 4D causal/padding/sliding-window mask representation used by E3/E4 to every decoder call. It removes no otherwise-allowed image edge and does not alter free-choice logits. Initial image encoding, explicit 2D-derived mRoPE positions, latent START/PAD/END rules, and fresh no-cache recurrence stay as specified in v1.

This mode is a numerical-representation control, not an image ablation and not a token-equivalent replacement for official baseline. Native reproduction remains a separate result. Do not require 2D and neutral4D outputs to agree or subtract an aggregate backend delta from an ablation delta.

Alternatives considered: forcing one SDPA kernel across all conditions would require changing and repeating more already-running conditions; moving all analysis to eager would introduce another backend bridge and runtime cost. Neither is selected. No global kernel selector or monkeypatch is added.

| Condition/report | Paired comparison control |
| --- | --- |
| E1 no_input_image | Existing SDPA E0 with ordinary 2D validity mask |
| E2 no_latent_execution | Existing SDPA E0 with ordinary 2D validity mask |
| E5 matched images | Existing SDPA originals and verified donor-equivalent inputs |
| New representation bridge | Existing SDPA E0 versus causal_mask_control |
| E3 latent_image_attention_block | New causal_mask_control |
| E4 global_image_attention_block | New causal_mask_control |

Existing per-model SDPA main100 controls are `results/visual-causal-panel-20260910t115000z/{s1,ce,pref}/e0-main.jsonl`; the draft plan specifies new run/output paths.

Report E3/E4-versus-neutral deltas, transitions, prediction changes, paired intervals, McNemar results, block distributions and timings. The bridge receives the same paired report but is labeled a mask-representation/numerical-path effect. E1/E2 and E3/E4 now have different controls; display those controls explicitly and do not compare their deltas as if their baselines were identical.

## Implementation boundary

- `src/evaluation/visual_interventions.py`: map `causal_mask_control` to a distinct internal effect `causal_mask`. Include that effect in the existing wrapper's 4D composition path. `build_image_attention_mask` already creates the base visibility matrix and removes image edges only for latent/global effects; the new effect keeps all base-visible edges. Keep `process_logits` an identity for this mode.
- `scripts/analyze_comt_visual.py`: expose the new mode in CLI choices and apply the existing SDPA compatibility gate to it. Reuse the existing generation path and unchanged row schema; `condition=causal_mask_control` identifies this control unambiguously.
- `tests/evaluation/test_visual_interventions.py`: independent mask-edge expectations, image visibility, padding/window preservation, unchanged positions/logits, and ordinary/rollout/post-END controller routing for the new mode. Preserve existing exact-identity tests for ordinary baseline.
- `tests/evaluation/test_visual_analysis_runner.py`: new mode selection, preserved original image inputs, condition serialization, fresh policy, and incompatible-backend rejection.
- `docs/evaluation/visual-causal-analysis.md`: document the additional mode, explicit comparison-control table, commands and evidence limits.
- Explicit probe/campaign scripts under `logs/<run-id>/` and outputs under `results/<run-id>/`: actual CUDA dispatch witnesses, control runs, corrected comparison commands and reports.

No change to the controller, vendored Qwen, original evaluator entrypoints, metric formulas, training, checkpoint selection, tokenizer IDs, dataset selection or existing artifacts is needed. Do not introduce retries or a new orchestrator. The old queued E3/E4 commands remain disabled; launch fresh amended commands only after gates pass.

## Exact experiment panel and cost boundary

Use the same three prepared adapters beneath `checkpoints/evaluation-staging/`:

1. `stage1-dora-z1-ema-frozen-b1-ga4-e1-lr5e5-egb16-20260902-024225/prepared-checkpoint-215` (s1).
2. `ours1-stage2a-s1matched-egb16-lr5e5-dora-z2-20260908t022000z-r3/prepared-checkpoint-212` (ce).
3. `ours1-ctx-pref-vonly-direct-s1-dora-egb16-lr5e5-default-s212-r2-20260910t022900z/prepared-checkpoint-212` (pref).

Base: `/root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct`. Source: `data/TEST.jsonl`, SHA256 `6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618`. Main indices: `data/evaluation/visual-causal-20260910/main-indices.json`, exactly the existing seed42 main100. Original subset SHA256 is `fba2b930d959e6b55e7592854f44a6d3e2dfa18d722d86ef3732b983cb07becd`.

For each checkpoint run fresh causal_mask_control100, E3-100 and E4-100. The incremental evaluation cost over v1 is three control panels, 300 example generations, plus bounded short witnesses. E3/E4 were already in v1 but have not launched. Do not extend the neutral control to the supplementary E5 rows or change E1/E2/E5. Do not promise elapsed time from a short diagnostic.

All full panels: temperature0, top_p1, repetition_penalty1.05, max_new_tokens4096, max_sequence_length32768, batch1, bf16, latent_size8, decoder SDPA and vision FA2. Keep the original processor, scoring and 100-row coverage / 99-valid-gold distinction. Paired bootstrap: seed42,10000 paired resamples, percentile95% intervals; exact two-sided McNemar. Invalid labels and inference errors remain explicit exclusions.

## Validation gates and conditional claims

1. TDD proves new-mode masks against independently enumerated causal/key-padding/window expectations, including visible image keys. Do not use the production mask builder as the only expected-value oracle. All free-choice logits and explicit positions remain unchanged by the neutral policy itself.
2. A controller witness covers ordinary text, START/PAD rollout, post-END and following text. Every actual decoder call receives the intended 4D mask, `use_cache=False` and no prior K/V. New samples/conditions start from fresh embeddings and latent state.
3. After implementation approval and focused tests, run the real short witness on all three checkpoints: original index3, max_new_tokens64, other parameters unchanged. Collect baseline2D, neutral4D twice from fresh state, E3 and E4. If any neutral/E3/E4 condition lacks a complete block or post-END text coverage, select the first OTHER original index in ascending main-panel order whose existing SDPA E0 has a complete block, and rerun the entire five-generation witness set at that one index for that checkpoint. This permits at most two sample indices per checkpoint. If coverage is still absent or no alternative exists, preserve the observed lack of entry, block the amended full panels for that checkpoint and return to diagnosis/design; do not keep searching or force START. Selection is a protocol-coverage decision, not an accuracy estimate.
4. Inspect actual decoder kwargs in a scoped pre-hook. Independently calculate allowed query/key pairs, using recorded initial image indices and registered latent spans. Compare all mask entries; compare delivered positions to a clone captured at the input of that same policy wrapper before mask conversion, using torch.equal. This is the controller-supplied 2D-derived position tensor for that condition's own actual prefix, not a position tensor from another generated trajectory. Inspect cache absence. Do not mistake `forward_records.attention_mask_shape`, which records the incoming 2D validity shape, for delivered-mask evidence.
5. Profile CPU-dispatched `aten::` SDPA operators during those short CUDA generations. For neutral/E3/E4 require only `aten::_scaled_dot_product_efficient_attention` as the concrete SDPA implementation, and its count equal to actual decoder calls times decoder layer count. No `output_attentions=True` fallback is allowed in this witness. Record device/dtype/classes, operator counts and role coverage. A mismatch blocks the amended full panels and returns to diagnosis; it does not authorize a silent backend switch. This authenticates the sampled short runs, not every kernel invocation in future long sequences.
6. The two fresh neutral runs must have identical prompt identity, raw generated IDs and scoring. Baseline2D-versus-neutral differences are measured, not a failure. For the inactive-E3 check, capture the first decoder call's kwargs from the neutral witness before policy wrapping (before any registered span). Replay those exact embeddings, 2D validity mask and explicit positions through fresh neutral and E3 policies with empty span lists into the same eval-mode decoder, use_cache=False and no past K/V. Compare delivered masks and last-position LM-head logits exactly; discard both outputs without appending tokens or latent state. This separate diagnostic is not an independent continuation and is excluded from the generation profiler call-count accounting. It checks that inactive E3 adds no second numerical path. E3/E4 numerical image-edge tests from v1 remain required evidence.
7. Only after gates and task review pass, launch detached fresh full panels. Require exactly100 unique original indices, valid metadata, zero inference errors, recorded raw latent positions and source-rescored outcomes. Compare E3/E4 to that checkpoint's neutral100 JSONL, with file hashes in each summary. Compare existing SDPA main100 to neutral100 separately. Preserve all existing output and failure logs.

The first nail for this amendment is gates1–6, especially direct mask/position inspection and actual concrete SDPA dispatch equality. It is not yet run because the new mode is not approved for implementation. Final approval would authorize implementing and running this bounded witness; it would not waive a failed witness or authorize full panels before it passes. Mask-shape equality and review agreement alone are not evidence that this works.

## Interpretation and remaining uncertainty

This amendment controls an observed representation/kernel confound. It does not make E3 a complete image-information removal: visual information can remain in earlier text states. E4 preserves image-token geometry/mRoPE and question text. The retained indirect paths and matched-image dataset-label limitations from v1 remain. Small net accuracy changes still require paired counts and uncertainty; non-significance is not equivalence. No Phase4 activation patching or new training method is authorized.

Three short model witnesses cannot establish universal kernel equivalence at every long-sequence shape. Full panels record instantiated attention classes/dtypes, condition, per-call policy records, generation errors and result coverage as already implemented. Validate those fields and report any mismatch/error before interpreting results. Full panels do NOT profile concrete SDPA operators; a long-sequence internal dispatch change may therefore be undetected. Mask semantics rely on the tested shared policy and sampled actual-call witnesses, not a claim of full-run kernel monitoring. No such failure is pre-waived. The final report must distinguish directly observed short-run dispatch from architectural expectations for full panels.

## Design commitment v3 — critique synthesis

Frozen proposal v3 SHA256: `0d2ee69c666f1681b49836b21fb169d00f9d7e89f5d45e49b565da38358d5eee`; retained in `.superpowers/sdd/2026-09-10-ilvr-visual-causal-analysis/mask-control-proposal-v3.md`. Earlier frozen proposal v2 is retained separately. A context-free shower read of v2 identified missing handoff links and ambiguities in coverage fallback, position/input references and full-panel monitoring. V3 clarifies those exact points before repeating the applicable critique passes; the main scientific conditions and comparison table are unchanged.

Hate result: `{root: "Calling both paths causal4D still fails to isolate image-edge removal if the neutral control changes another decoder input or executes another numerical path.", first_nail: "Independently inspect delivered masks and each call's original positions/cache state; verify concrete SDPA dispatch over short ordinary/latent calls for all three checkpoints and replay identical pre-latent inputs through neutral and inactive E3."}`. This is a proposed falsification test, not evidence. Its new-mode execution is pending implementation approval; the earlier 2D-versus-neutral diagnostic establishes the problem, not the remedy.

Macrothink was not repeated: the amendment addresses a directly observed, bounded numerical confound; broad reframing would not resolve the remaining runtime premise. The original research question and v1 divergences remain intact.

Prism, numerical-correctness lens: unclear until the actual-call/dispatch witness passes; mode/class names alone do not establish numerical-path comparability. Prism, statistical-comparability lens: pass at design level because the explicit control table preserves pairing and separates the representation bridge from image-edge interventions. The unresolved question is whether the bounded real witnesses satisfy the exact mask/position/dispatch contract. Missing block coverage also leaves that premise unvalidated and blocks that checkpoint's panels. The root carries this as a gate rather than averaging the two verdicts.

Mandela advisory audit: no additional independence defect identified in this amendment's proposed scoring/validation. Ground truth remains the fixed external source annotations; mask expectations are independently enumerated. This is not evidence of benchmark-label quality, train/test independence, or implementation correctness.

Superpowers synthesis: adopt frozen proposal v3 unchanged in scope, interfaces and behavior. The agreed problem is confounded E3/E4 comparison; the final design is the neutral4D mode, explicit control table and bounded validation procedure above. No material change was made between v3's critique passes and this commitment. First nail status: NOT RUN, because the new mode awaits implementation approval. Final approval authorizes implementing and running it, followed by full panels only if it passes; it does not waive failed masks, dispatch mismatches or invalid metrics. The residual limitation is that concrete SDPA dispatch is observed only on bounded witnesses, not every full-panel invocation; the final report must disclose this limitation rather than claim universal runtime equivalence. No other explicit risk waiver is requested. Approval binds to this exact commitment v3 and its draft plan; later material changes return to the gate.
