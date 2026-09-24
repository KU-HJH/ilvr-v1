# ours1 fresh context-preference defaults

This checkout defaults fresh context-preference Stage 2B to `until_next_latent`.
Omitting the scope in the launcher, Python CLI, configuration, or context collator
selects the same target definition. These are the corresponding explicit settings:

```bash
# Fresh context-preference: resolved defaults in ours1.
STEP_PREFERENCE_TARGET_SCOPE=until_next_latent
GRADIENT_CHECKPOINTING=false
STAGE2_LATENT_ACTIVATION_CHECKPOINTING=true
STEP_PREFERENCE_BETA=0.5
STEP_PREFERENCE_WEIGHT=1.0
```

Keep using the existing two-node/four-rank coordinator described in
[the multi-node operator guide](../docs/multinode-training.md). These settings
are not a standalone launch recipe. The standing training defaults, including
beta 0.5 and preference weight 1.0, still apply after sourcing run settings.

The until path uses `concatenated-v1`: text fragments are joined exactly, without
inserting whitespace between adjacent text entries. Original whitespace remains.
Each latent targets tokens after its END and before the next START; the final
latent targets through assistant content end, including the final answer.
Tokens crossing internal text/text boundaries stay in these targets. Assistant
chat termination tokens remain outside the content target.

Native decoder checkpointing, final clean checkpointing, and V-off sidecar
checkpointing are OFF; cache-free latent activation checkpointing is ON. Clean
prefix KV remains differentiable when supplied to V-off. Explicit native ON is
an error for a fresh until run, including when latent checkpointing is explicitly
OFF. `STAGE2_LATENT_ACTIVATION_CHECKPOINTING=false` (Python:
`--no-stage2_latent_activation_checkpointing`) can explicitly disable latent replay.

Set `STEP_PREFERENCE_TARGET_SCOPE=immediate_text` (Python:
`--step_preference_target_scope immediate_text`) to retain the immediate target
and `adjacent-text-space-lf-v1` rendering, with latent checkpointing OFF by default
and existing sidecar checkpointing ON. Native checkpointing remains controlled by
its existing flag. CE-only, preference-off, Stage 1, Stage 2A, and legacy-negative
objectives retain their applicable scope/rendering defaults; an explicit
incompatible until scope still errors.

Resolved rendering, native/latent/sidecar checkpoint booleans, and target scope
are saved in model config under `stage2_preference_execution`. Only an actual
resume reads this dictionary as saved execution settings. Fresh Stage 1/Stage 2A
initialization ignores the initializer's dictionary. On supported resumes, saved
settings take precedence over fresh defaults, omitted scope is restored, and an
explicit scope conflict errors before training-data construction. A preference
resume missing reliable fields fails with an actionable error; ordinary
non-preference resumes without the dictionary retain their previous rendering
and native setting, with latent replay OFF unless explicitly requested. Invalid
or incomplete saved dictionaries fail closed. Stage 2B resume remains unsupported
and retains its existing rejection; this change does not introduce a resume mode.
Length manifests created with a different renderer are rejected.

This change aligns text assembly, until targets, and checkpoint placement. V-off
boundary masking, sampler, losses, and model initialization have not been aligned
with the RAID run by this patch. Its historical record says
`uncommitted_source: true`; comparisons to preserved current source and CPU
execution witnesses cannot prove exact historical execution equivalence or
production training accuracy. No training or evaluation was launched for this
change.
