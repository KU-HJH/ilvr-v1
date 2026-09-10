# ILVR CoMT visual causal analysis — approved design v1

Approval: user explicitly approved the seven-part read-only audit and design v1 with `ㅇㅋ` on 2026-09-10. This document records that approval, not a new design request. Understood as: implement and verify opt-in inference interventions, then evaluate the approved three checkpoints; do not change training or infer vision non-use from aggregate accuracy alone.

## Scope and source

Source is ours1 commit `ff53706882e39c9ce2ce4ac9e75242e1438ffc14`, not GitHub main. Work in `/root/workspace/ILVR/.worktrees/ours1-visual-causal`, branch `ours1-visual-causal`. Preserve ours1's untracked user documents. No merge/push or main-branch changes are authorized.

Use `./venv/ilvr-venv` exclusively, `PYTHONPATH=transformers/src:src:.`, and the existing environment. Keep model artifacts under the worktree's `checkpoints/` symlink, datasets under `data/`, outputs under `results/`, execution logs under `logs/`. Reuse existing checkpoint/data/venv targets without copying model weights. Use detached execution and fresh run IDs. Do not signal foreign processes.

## Audited implementation

- `src/multinode_runtime.py::_build_evaluation_commands_for_stage` selects `eval.py` for Stage 1, `src/evaluate_deepseed.py` for Stage 2. The former loads SDPA language/FA2 vision; the latter tries FA2 then falls back to eager for classified initialization failures. Record instantiated module classes, not just requested settings. `eval.sh` is an unconfigured template, and `eval.py` requires `--runtime_identity_path`.
- `evaluation.runtime.prepare_evaluation_input` uses only `text_input` and supplied original images. `sequence_plan`, helper images, and `original_final_answer` do not enter inference. The processor expands image-token positions from actual image grids. The scorer reads gold after generation.
- Both stages now use `ContinuousLatentController`; `Stage2InferenceController` is a compatibility wrapper. They bypass generic HF `model.generate` and the retained template logits processors. Free generation can select START, cannot select PAD/END. On START, the controller appends K PAD IDs and END, and uses `LatentRolloutEngine`. Scaffold IDs consume max-new-tokens; incomplete scaffold can exist without completed rollout.
- Latent at physical position p is the final normalized decoder hidden at p-1, inserted as input embedding at p. The first producer query is START. All prefix forwards use `use_cache=False`; earlier generated latent embeddings persist within one condition. No cross-condition hidden/cache reuse is permitted.
- Stage 1 training selects detached EMA-teacher helper-image features for some PAD slots and aligns student hidden at t-1 with target embedding at t. Stage 2 training self-generates latent slots from original-image/text prefix. Current evaluation self-generates for both stages.
- Recent `interleaved` evaluation suppresses latent execution. `all` additionally removes the original image before processor construction. Neither measures latent-image attention specifically. Training V-off masks current PAD keys in target-window sidecars, while retaining original image access and clean prefix K/V outside blocked positions. It is not E3/E4.
- Existing `stage2_final_attention_mask` affects only post-rollout final decoding, so it cannot block image input during latent construction. Eager/SDPA decoder accepts a complete additive 4D mask; FA2 does not implement this arbitrary query/key contract. Explicit mRoPE positions must remain computed from original 2D validity and image grid. SDPA with output_attentions=True falls back to eager.

## Conditions and interpretation

CLI modes: `baseline`, `no_input_image`, `no_latent_execution`, `latent_image_attention_block`, `global_image_attention_block`, `image_replacement`.

- E0 baseline: current native evaluator reproduction followed by same-backend no-op analysis equivalence. All inference defaults remain unchanged.
- E1 no_input_image: rebuild prompt with images=None and no image entries/placeholders; latent execution remains enabled. Fresh base embeddings and recurrence. No image pixels, grid, prior K/V, or prior latent inputs.
- E2 no_latent_execution: suppress only discrete START at every free choice, after the ordinary repetition penalty and PAD/END exclusion. Retain remaining vocabulary scores and normal continuation. Record LM-head and decision-stage START logits, softmax diagnostic probability (not a temperature-zero sampling probability), rank, best non-control token, pre/post intervention choices, suppression changes, repeated entry attempts. This tests computation necessity under the existing token budget, not matched computation or continuous-latent likelihood.
- E3 latent_image_attention_block: mask original image keys for START and all PAD query rows, including all latent-producing rows. Apply during each latent prefix, subsequent text prefix replay, and post-END probe. Register each span before its rollout and retain prior spans. END is not itself selected as an E3 query. Original-image information in preceding text and other unblocked states may still reach latents indirectly.
- E4 global_image_attention_block: from the first decoder call, mask original image keys for every non-image query, at every decoder layer and every replay. Recompute all latent and text states under E4. Preserve ordinary image input structure, pixel encoding, image-token count and mRoPE geometry. This removes direct decoder access to original image-content K/V under the verified graph, not image size/geometry or textual information.
- E5 image_replacement: prefer the nine whitespace-normalized identical-question/choice groups with differing golds (19 test rows). Verify source annotations and valid images; evaluate both directions and donor-answer consistency. No mismatched-image accuracy is a counterfactual accuracy without a valid gold for the resulting question/image pair. Treat repeated conditions within groups as dependent. Matched group membership is independent of predictions.

Do E5 discrimination alongside initial E0/E1/E2 before interpreting E3/E4. Activation patching is deferred until E0-E5 work; meaningful alignment and identity-patch controls need a separate later design. Other beta/margin variants and full-400 expansion are not part of this initial campaign.

## Checkpoints and data

All prepared artifacts are under `checkpoints/evaluation-staging/`:

1. `stage1-dora-z1-ema-frozen-b1-ga4-e1-lr5e5-egb16-20260902-024225/prepared-checkpoint-215`.
2. `ours1-stage2a-s1matched-egb16-lr5e5-dora-z2-20260908t022000z-r3/prepared-checkpoint-212`.
3. `ours1-ctx-pref-vonly-direct-s1-dora-egb16-lr5e5-default-s212-r2-20260910t022900z/prepared-checkpoint-212`.

Initial comparison panel: `data/TEST-vision-ablation-100-seed42-20260910.jsonl`, SHA256 `fba2b930d959e6b55e7592854f44a6d3e2dfa18d722d86ef3732b983cb07becd`. Recover original indices from ours1 `results/ours1-vision-ablation-analysis-20260910/selection.json`, verify exact rows against `data/TEST.jsonl` (400 rows, SHA256 `6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618`). Do not resample or stratify-replace it. Categories derive explicitly from the first image parent directory: subset creation/deletion/update/selection counts 25/30/15/30. Full test has 100 each. The supplementary E5 groups are 100/137; 103/111; 104/143/183; 122/134; 125/133; 139/174; 151/159; 152/153; 314/372. This panel is distinct from the main accuracy denominator.

Decoding: temperature=0, top_p=1, repetition_penalty=1.05, max_new_tokens=4096, max_sequence_length=32768, batch=1, runtime bfloat16, latent_size from checkpoint (8 in inspected checkpoints), EOS from existing generation configuration. Existing checkpoint processor, image ordering, answer extraction and scoring are reused. Backend bridge: Stage 2 native FA2 versus analysis SDPA with vision FA2 fixed; Stage 1 already uses this SDPA/FA2 pair. Same-backend no-op must match raw token IDs and prompt identity. Any backend differences are reported separately from interventions. No silent fallback in analysis masking modes.

## Architecture and files

- New `src/evaluation/visual_interventions.py`: pure token-position/mask helpers, per-generation intervention state, and scoped decoder callable. Full masks combine existing causal, padding, sliding-window restrictions and intervention. Image token IDs resolve from tokenizer and are checked against config; use processed input positions, not constants.
- Modify only `src/evaluation/continuous_latent.py::generate` for optional default-None policy/observer routing. One decoder callable covers ordinary forward, existing rollout `language_model` parameter, and post-END forward. Existing engine keeps its 2D padding contract and explicit position IDs. No changes to vendored Qwen, rollout engine, sidecars, training loss, or existing evaluators unless a concrete finding requires returning to design approval.
- New `src/evaluation/visual_metrics.py`: CPU-only paired metrics, exact McNemar, reproducible paired bootstrap, category/time/latent aggregates.
- New `scripts/analyze_comt_visual.py`: analysis CLI, existing loader/preprocessing/scoring reuse, fresh condition generation, raw evidence, native-baseline comparison, aggregate/report subcommands. Use explicit small commands or a campaign shell script in logs, not new orchestration infrastructure.
- New focused tests under `tests/evaluation/`; documentation with executable CLI examples and exact verification commands.

Each JSONL row records original sample index/id, checkpoint, condition, text/image identity and replacement donor if applicable, raw output, raw generated suffix IDs, prompt length/identity, normalized/extracted answer, gold, parse status, correctness, invalid-gold reason, START/END positions (suffix and absolute coordinates defined), completed block count and truncated-block evidence, logits events when collected, generation timing, stop reason, and token counts. Run metadata records commit, command, dataset identity, actual backend module classes, runtime dtype, latent size and generation configuration. JSON output must not emit non-standard NaN/Infinity; use null plus documented semantics for nonfinite diagnostic scores.

Aggregates: coverage and valid-gold accuracy, paired delta, all four correctness transitions, normalized-answer change and raw-token change separately, 95% paired bootstrap CI, exact two-sided McNemar on valid paired labels, exploratory multiple-comparison caveat, category breakdown, block distribution and time change. Errors are explicit and not silently treated as valid paired comparisons. Missing/duplicate IDs are rejected. E5 reports group-level discriminative correctness/donor-consistent switches; generic output changes are not useful visual reasoning by themselves.

## Gates, critique, and uncertainty

Frozen proposal v1 was critiqued before human approval. Hate root: wrong query rows or incomplete policy propagation can make the expensive experiment measure the wrong path. First nail: tiny actual eager masked-edge probability witness plus native SDPA fixed-grid image-embedding perturbation invariance under E4, with an unmasked positive control. Also check explicit position equality and same-backend baseline logits/tokens; then one short real-checkpoint witness before full ablations. First nail was not run preapproval because the user prohibited implementation/experiments before audit approval. This is an unvalidated premise, not established correctness.

Macrothink's two fresh reads gave compatible divergence: establish image-specific answer competence early; trajectory disruption is not evidence of useful visual mediation. E5 priority was strengthened without changing intervention semantics or scope. Prism found execution correctness and causal interpretation conditional on those witnesses; convergence is not proof. Residual uncertainty: numerical masking, backend drift, whole-panel cost/memory, matched-image labels, statistical power, preserved text/geometry pathways. No waiver of incorrect-mask results is given: failed gates block full ablations and trigger systematic debugging.

Use TDD, one task-scoped review per task, a final whole-change review, and fresh claim-specific verification. Do not repeat broad tests for every run. Keep the experiment detached, validate outputs once, report results immediately. Code completion, runtime witness, full panel completion, and scientific conclusions are separate claims.
