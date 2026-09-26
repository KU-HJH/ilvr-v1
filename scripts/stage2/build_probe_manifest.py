"""Freeze the pre-registered Vision gain probe selection from data/TEST.jsonl.

Rows have no id field, so row_id is the 0-based line index among non-empty
lines. Rows are ordered by sha256(f"42:{row_id}") and taken until the
cumulative count of latent steps in their sequence_plan reaches 256.
The probe is diagnostic only: never use it for checkpoint selection or tuning.
"""

import argparse
import hashlib
import json
from pathlib import Path

SEED = 42
MIN_BLOCKS = 256


def build_manifest(test_path: Path) -> dict:
    raw = test_path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    order = sorted(range(len(rows)), key=lambda i: hashlib.sha256(f"{SEED}:{i}".encode()).hexdigest())
    selected, total = [], 0
    for row_id in order:
        blocks = sum(step.get("type") == "latent" for step in rows[row_id]["sequence_plan"])
        if blocks == 0:
            continue
        selected.append({"row_id": row_id, "latent_blocks": blocks})
        total += blocks
        if total >= MIN_BLOCKS:
            break
    if total < MIN_BLOCKS:
        raise RuntimeError(f"only {total} latent blocks available")
    return {
        "policy": "vision-gain-probe-v1",
        "row_id_definition": "0-based index of non-empty lines in data/TEST.jsonl",
        "ordering": f'sha256("{SEED}:{{row_id}}") ascending; rows without latent steps skipped',
        "test_jsonl": str(test_path),
        "test_jsonl_sha256": hashlib.sha256(raw).hexdigest(),
        "selected_row_ids": [item["row_id"] for item in selected],
        "selected": selected,
        "total_blocks": total,
        "s_m_mode": "full_replay",
        "model_mode": "eval",
        "use": "diagnostic thresholds only; never checkpoint selection, tuning, or early stopping",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=Path("data/TEST.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("test_jsonl_sha256", "total_blocks")}
                     | {"rows": len(manifest["selected_row_ids"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
