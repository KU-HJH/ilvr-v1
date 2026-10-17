# Target-scope supplemental runtime verification

## Scope and disposition

- Revision verified: `3112e9677771269ddd38a1a1e2337840e66d3458`.
- This is supplemental claim-specific runtime evidence after controller adjudication. It is not a second product fix wave.
- No product code, tests, training state, optimizer state, model weights, or checkpoints were changed or created.
- The probe used the local randomly initialized tiny fixture and called the actual `Stage2Trainer.compute_loss` route for both `immediate_text` and `until_next_latent`.
- At the production replacement-scorer boundary, it captured the actual clean embeddings, base image embeddings, and positions. It then independently assembled native original and replacement prefixes using hard-coded fixture facts: target IDs, predictor rows, latent ranges, and source rows. Expected values were not derived from returned scores or planner-produced rows.

## Exact command and code

```bash
PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python - <<'PY'
import json

import torch

from stage2.trainer import Stage2Trainer
from step_preference.config import StepPreferenceConfig
from step_preference.equivalence import _trainer
from step_preference.spans import ReasoningStepSpan
from tests.step_preference.insertion_helpers import native_case
from transformers.models.qwen2_5_vl import stage2_evidence_replacement


EXPECTED_IDS = (11, 78, 80, 79, 12, 91, 90, 90, 92, 13, 14, 91, 90, 90, 92, 15, 16)
BRANCHES = {
    "immediate_text": (
        dict(step_id=0, latent=(6, 8), prefix_length=9, predictors=(8,), targets=(13,), source="base", source_positions=(2, 2)),
        dict(step_id=1, latent=(12, 14), prefix_length=16, predictors=(14, 15), targets=(15, 16), source="clean", source_positions=(6, 7)),
    ),
    "until_next_latent": (
        dict(step_id=0, latent=(6, 8), prefix_length=10, predictors=(8, 9), targets=(13, 14), source="base", source_positions=(2, 2)),
        dict(step_id=1, latent=(12, 14), prefix_length=16, predictors=(14, 15), targets=(15, 16), source="clean", source_positions=(6, 7)),
    ),
}


def config(scope):
    return StepPreferenceConfig(
        enabled=True,
        mode="stage2b",
        stage2a_checkpoint_path="tiny-stage2a",
        prefix_mode="evidence_replacement",
        target_scope=scope,
        lambda_preference=10.0,
        lambda_entropy=0.0,
        sidecar_attention_backend="dense",
        visual_only_context_preference=True,
        visual_off_suppress_latent_boundaries=False,
        replacement_seed=42,
    )


def literal_score(model, embeds, positions, predictors, targets):
    hidden = model.model(
        inputs_embeds=embeds,
        position_ids=positions[:, :, : embeds.shape[1]],
        attention_mask=torch.ones((1, embeds.shape[1]), dtype=torch.long, device=embeds.device),
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    rows = torch.tensor(predictors, dtype=torch.long, device=embeds.device)
    target_ids = torch.tensor(targets, dtype=torch.long, device=embeds.device)
    log_probs = model.lm_head(hidden.index_select(1, rows))[0].float().log_softmax(-1)
    return log_probs.gather(-1, target_ids[:, None]).mean()


def run_scope(scope):
    model, ids, _, grid, _, pixels = native_case()
    model.eval()
    assert tuple(ids[0].tolist()) == EXPECTED_IDS
    first_target_end = 10 if scope == "immediate_text" else 11
    spans = (
        ReasoningStepSpan(0, 0, 5, 6, 8, 9, first_target_end, None),
        ReasoningStepSpan(1, 10, 11, 12, 14, 15, 17, 0),
    )
    batch = dict(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        labels=ids.clone(),
        pixel_values=pixels,
        image_grid_thw=grid,
        sample_ids=("replacement-trainer",),
        reasoning_step_spans=spans,
        step_preference={"steps": tuple({"span": span} for span in spans)},
    )
    captured = {}
    original = stage2_evidence_replacement.score_evidence_replacement

    def capture(model_arg, options, input_ids, base, clean, positions, image_grid_thw):
        captured["base"] = base.detach().clone()
        captured["clean"] = clean.detach().clone()
        captured["positions"] = positions.detach().clone()
        return original(model_arg, options, input_ids, base, clean, positions, image_grid_thw)

    stage2_evidence_replacement.score_evidence_replacement = capture
    try:
        trainer = _trainer(config(scope))
        with torch.no_grad():
            _, output = Stage2Trainer.compute_loss(trainer, model, batch, return_outputs=True)
    finally:
        stage2_evidence_replacement.score_evidence_replacement = original

    actual = {
        (item["step_id"], item["kind"]): item["mean_logp"].detach()
        for item in output.stage2_sidecar_results
    }
    observations = []
    with torch.no_grad():
        for branch in BRANCHES[scope]:
            prefix_length = branch["prefix_length"]
            original_embeds = captured["clean"][:, :prefix_length]
            replacement_embeds = original_embeds.clone()
            source = captured[branch["source"]]
            source_rows = torch.tensor(branch["source_positions"], dtype=torch.long)
            start, end = branch["latent"]
            replacement_embeds[:, start:end] = source.index_select(1, source_rows).detach()
            literal_original = literal_score(
                model, original_embeds, captured["positions"],
                branch["predictors"], branch["targets"],
            )
            literal_replacement = literal_score(
                model, replacement_embeds, captured["positions"],
                branch["predictors"], branch["targets"],
            )
            actual_original = actual[(branch["step_id"], "visual_original")]
            actual_replacement = actual[(branch["step_id"], "visual_replacement")]
            torch.testing.assert_close(actual_original, literal_original, atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(actual_replacement, literal_replacement, atol=1e-5, rtol=1e-4)
            observations.append({
                "step_id": branch["step_id"],
                "predictors": branch["predictors"],
                "targets": branch["targets"],
                "actual_original": float(actual_original),
                "literal_original": float(literal_original),
                "original_abs_error": abs(float(actual_original) - float(literal_original)),
                "actual_replacement": float(actual_replacement),
                "literal_replacement": float(literal_replacement),
                "replacement_abs_error": abs(float(actual_replacement) - float(literal_replacement)),
            })
    return observations


all_observations = {scope: run_scope(scope) for scope in BRANCHES}
immediate = all_observations["immediate_text"][0]
full = all_observations["until_next_latent"][0]
summary = {
    "status": "PASS",
    "scopes": all_observations,
    "step0_scope_score_differences": {
        "original": abs(immediate["actual_original"] - full["actual_original"]),
        "replacement": abs(immediate["actual_replacement"] - full["actual_replacement"]),
    },
}
print(json.dumps(summary, sort_keys=True, indent=2))
PY
```

## Complete output

Exit status: `0`.

```text
{
  "scopes": {
    "immediate_text": [
      {
        "actual_original": -5.150517463684082,
        "actual_replacement": -5.177169322967529,
        "literal_original": -5.150517463684082,
        "literal_replacement": -5.177169322967529,
        "original_abs_error": 0.0,
        "predictors": [
          8
        ],
        "replacement_abs_error": 0.0,
        "step_id": 0,
        "targets": [
          13
        ]
      },
      {
        "actual_original": -4.928469657897949,
        "actual_replacement": -4.916774272918701,
        "literal_original": -4.928469657897949,
        "literal_replacement": -4.916774272918701,
        "original_abs_error": 0.0,
        "predictors": [
          14,
          15
        ],
        "replacement_abs_error": 0.0,
        "step_id": 1,
        "targets": [
          15,
          16
        ]
      }
    ],
    "until_next_latent": [
      {
        "actual_original": -4.905398368835449,
        "actual_replacement": -4.954164505004883,
        "literal_original": -4.905398368835449,
        "literal_replacement": -4.954164505004883,
        "original_abs_error": 0.0,
        "predictors": [
          8,
          9
        ],
        "replacement_abs_error": 0.0,
        "step_id": 0,
        "targets": [
          13,
          14
        ]
      },
      {
        "actual_original": -4.928469657897949,
        "actual_replacement": -4.916774272918701,
        "literal_original": -4.928469657897949,
        "literal_replacement": -4.916774272918701,
        "original_abs_error": 0.0,
        "predictors": [
          14,
          15
        ],
        "replacement_abs_error": 0.0,
        "step_id": 1,
        "targets": [
          15,
          16
        ]
      }
    ]
  },
  "status": "PASS",
  "step0_scope_score_differences": {
    "original": 0.2451190948486328,
    "replacement": 0.22300481796264648
  }
}
```

## Result and remaining limitation

Observed: for both target scopes and both steps, the actual Trainer `visual_original` and `visual_replacement` scores matched independently assembled native literal replay exactly in this fixture (`0.0` absolute error for all eight comparisons). The differing step-zero scope was exercised: immediate scope used predictor `(8,)` and target `(13,)`, while full scope used predictors `(8, 9)` and targets `(13, 14)`. Their actual original scores differed by `0.2451190948486328`; replacement scores differed by `0.22300481796264648`.

Unverified: this focused CPU tiny-fixture probe does not establish GPU, distributed, full-suite, production-data, memory, throughput, or efficacy behavior. Those remain separate controller verification claims.

Remaining automated-regression limitation: `tests/step_preference/test_replacement_trainer.py` still does not contain an independent scope-specific literal-score assertion. The runtime claim is directly observed here, but the missing automated regression remains an explicitly open Minor finding.
