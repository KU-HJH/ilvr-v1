# Task 3 report: causal scores and preference losses

## Implementation

- Added `causal_token_logps`, `masked_token_sums`, `reference_adjusted_effect`, and
  direct window-masked `evidence_window_margins` in `src/evidence_window/scores.py`.
- Added global, independent-window, and weighted evidence-window preference losses
  in `src/evidence_window/losses.py` using `logsigmoid` arithmetic.
- Added shape/device validation and graph-connected finite zero returns for empty
  or fully invalid masks.
- Exported all seven primitives from `src/evidence_window/__init__.py`.
- Added hand-derived causal, masking, reference-adjustment, direct-window, loss,
  empty-mask, saturation-gradient, and shape-validation tests.

## RED evidence

Command:

```text
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_scores.py tests/evidence_window/test_preference_losses.py -q
```

Result before implementation: collection failed with the expected missing-module
errors for `evidence_window.scores` and `evidence_window.losses`.

## GREEN evidence

Targeted command:

```text
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_scores.py tests/evidence_window/test_preference_losses.py -q
```

Result: `11 passed in 1.50s`.

Package regression command:

```text
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window -q
```

Result: `35 passed in 4.72s`.

## Files

- `src/evidence_window/scores.py`
- `src/evidence_window/losses.py`
- `src/evidence_window/__init__.py`
- `tests/evidence_window/test_scores.py`
- `tests/evidence_window/test_preference_losses.py`

## Self-review

- Causal scores have output shape `[B, T]`; target `t` uses logits `t-1`, while
  target zero and `-100` labels contribute zero.
- Preference margins are summed, with no length normalization.
- EW margins are direct einsum reductions over the supplied window masks.
- Global loss aggregates valid margins per example, so a fully invalid example
  does not contribute a spurious `softplus(0)` term.
- The saturation test verifies that a large first global margin suppresses the
  second margin's gradient while the independent-window loss retains signal.
- Bytecode generated during testing was removed and the tracked pre-existing
  bytecode artifact was restored before staging.

## Concerns

- The public loss API currently accepts a Python numeric `beta` and requires it
  to be finite and positive; tensor-valued beta is intentionally out of scope.
- Masks are shape/device validated but are cast to the score dtype, allowing
  boolean masks and numeric weights.

## Commit

Implementation commit: `c7baa39` (`feat: add evidence-window preference primitives`).

