# Task 2 implementation report: pinned VSP source preparation

Status: implemented and focused verification passed. No commit, merge, push, production prepare CLI run, model run, or Reference code execution was performed.

## Delivered changes

- Added `src/evaluation/vsp_prepare.py` with the fixed `LEVELS`, `COMMON_IMAGES`, and `SOURCE_COMMIT` values; strict table/pure-text agreement parsing; the approved three-image prompt rewrite and identity; pinned Git/lock verification; and preflight-before-copy preparation of 600 full and 6 smoke rows.
- Added `scripts/prepare_eval.py` with the required four arguments, existing registry validation, direct-child/fresh output enforcement, separate input (2) and infrastructure (1) exit behavior, and prepare result/log output.
- Added reusable `tests/evaluation/vsp_fixtures.py`. Its manifests use `model_identity='synthetic:model'`; assets live below active `data/`; fake model metadata lives below `checkpoints/evaluation-staging/<run-id>/`; and fixture result paths live below `results/evaluation/`.
- Added `tests/evaluation/test_vsp_prepare.py` covering table/description parsing, malformed inputs, prompt markers, fixture compatibility with P1 `read_manifest`, 600/6 manifest behavior, ordering, byte preservation, prompt/target separation, hashes, source/license records, preflight-before-output, CLI behavior, and the expanded real lock.
- Extended only the Visual-Spatial-Planning `evidence_files` in `references.lock.json`, adding the prompt and two common-image SHA-256 records. Existing records and other repository entries were not changed.

## RED evidence

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. ./venv/ilvr-venv/bin/python -m pytest -q tests/evaluation/test_vsp_prepare.py
```

Observed before production implementation: exit nonzero, `20 failed`; each failure was the expected missing `evaluation.vsp_prepare` or `scripts.prepare_eval` API.

Full output: `logs/evaluation/p2-task2-red-20260910/pytest.log`

The first implementation run found two failures caused by the missing destination parent for the raw prompt. Creating that required parent was the smallest causal fix. The CLI exception split was also narrowed so source contract failures return 2 and copy/output failures return 1.

## GREEN evidence

Final command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. ./venv/ilvr-venv/bin/python -m pytest -q tests/evaluation/test_vsp_prepare.py tests/evaluation/test_manifest_preflight.py
```

Observed: exit 0, `30 passed in 1.16s`.

Full output: `logs/evaluation/p2-task2-green-final-20260910/pytest.log`

The read-only source compatibility probe, run only after the existing lock and Git root/origin/HEAD/clean state were verified, parsed all 600 pinned table/pure-text pairs. It observed 566 hole-list records and 34 exact no-hole records. This was not a production preparation run and created no production dataset.

## Self-review

- Parsing requires exact numbered headers and rows, only `_/#/@/*`, exact four-line descriptions, in-bounds nonduplicate coordinates, one start and goal, and complete table/description equality. The existing `normalize_map(VspTarget(...))` remains the structural validator.
- Verification checks Git root, origin, pinned HEAD, clean state, and every locked license/evidence hash. Every resolved evidence, prompt, image, map, and license path must remain inside the source checkout.
- Preparation validates all prompt/common images/license and all 600 table/pure-text/PNG triplets before creating `dataset_root`; it requires exact `0..99` filename sets per level. Pillow is imported only inside image validation.
- Raw files are copied byte-for-byte with their approved layout. Manifest rows are serialized from `dataclasses.asdict(Sample)` through P1 `canonical_json`; no manifest hash is embedded in rows. Model input contains only the processed shared prompt, ordered three image paths, and empty choices.
- Full order is numeric level then numeric source ID; smoke contains ID 0 from each level. `SOURCE.json`, copied `LICENSE`, raw and processed prompts, manifests, and the seven-field summary follow the approved modified protocol.
- The reusable fixture is accepted by P1 `read_manifest`. The final focused suite also reruns every Task 1 manifest-preflight regression.
- Mutation check: the suite fails for symbol/label/dimension/coordinate parser relaxations, missing preflight, reordered IDs/images, changed hashes or bytes, target leakage into `model_input`, overwrite allowance, wrong exit classification, and omitted lock evidence.

## Remaining evidence boundary

The real production prepare CLI invocation belongs to Task 5 and was deliberately not run here. Synthetic colored fixture PNGs establish file/order/hash behavior only; they do not independently establish the semantic correspondence of all 600 upstream maze images. No paper-protocol or model/GPU claim is made.

## Fix round 1

The task-scoped review found two important defects and three narrow evidence gaps. The following changes address them without expanding Task 2:

- Replaced both malformed record checks with `not {'path', 'sha256'} <= set(record)`. A license or evidence record that omits a required key while adding an unrelated key now raises `ContractError`, preserving CLI input-error classification.
- Replaced the hard-coded/skipped Reference test with an independent Git repository and reference lock created beneath `data/evaluation/`. The tests patch only the pinned commit constant to the synthetic commit and exercise accepted clean state plus wrong root, wrong origin, wrong HEAD, dirty state, and bad evidence hash.
- Asserted the exact ordered list of all 600 manifest IDs instead of checking only the first 102.
- Derived the expected prompt identity directly from the required literal prefix and `hashlib.sha256(prompt.encode('utf-8'))`, without calling production `prompt_identity` in the expectation.
- Compared the pre-existing `pure_text/0.txt` lock record with `.superpowers/sdd/2026-09-10-ilvr-vsp-inference-plan/task-2-before/references.lock.json`; its original fields and values are preserved exactly. The only VSP evidence additions remain the prompt and two common images.

Fix RED command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. ./venv/ilvr-venv/bin/python -m pytest -q tests/evaluation/test_vsp_prepare.py -k 'missing_required_key_with_extra'
```

Observed before the subset fix: exit nonzero, `3 failed, 22 deselected`; all three escaped as the expected `KeyError: 'sha256'` rather than `ContractError`.

Full output: `logs/evaluation/p2-task2-fix1-red-20260910/pytest.log`

Fix GREEN command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. ./venv/ilvr-venv/bin/python -m pytest -q tests/evaluation/test_vsp_prepare.py
```

Observed after all five fixes: exit 0, `30 passed in 1.32s`.

Full output: `logs/evaluation/p2-task2-fix1-green-20260910/pytest.log`

The real-source 600-pair parser observation documented earlier was a separate, console-only read-only probe from the original implementation round. It was not rerun during this fix round and is not part of the Reference-free 30-test CPU result. Real preparation remains assigned to Task 5.
