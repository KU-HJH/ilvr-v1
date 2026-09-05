import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.audit_packed_stage2b_lengths import (
    publish_packed_length_audit,
    select_packed_length_winner,
)


def test_audit_documented_direct_execution_reaches_cli_help():
    """Catches package-only imports that fail when Python executes the script path."""
    repository = Path(__file__).resolve().parents[2]
    script = repository / "scripts" / "audit_packed_stage2b_lengths.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--checkpoint_path" in completed.stdout
    assert "--subset_output_path" in completed.stdout


def _records():
    return (
        {
            "source_index": 0,
            "sample_id": "sample-zero",
            "clean_length": 900,
            "step_count": 1,
            "branch_count": 2,
            "unpadded_length": 299,
            "padded_length": 384,
        },
        {
            "source_index": 1,
            "sample_id": "sample-one",
            "clean_length": 100,
            "step_count": 2,
            "branch_count": 4,
            "unpadded_length": 300,
            "padded_length": 384,
        },
        {
            "source_index": 2,
            "sample_id": "sample-two",
            "clean_length": 100,
            "step_count": 3,
            "branch_count": 6,
            "unpadded_length": 300,
            "padded_length": 384,
        },
    )


def test_audit_winner_uses_exact_padded_unpadded_clean_source_tuple():
    """Catches clean length or first-row tie breaking overriding packed capacity."""
    records = _records()

    assert select_packed_length_winner(records[:2])["source_index"] == 1
    assert select_packed_length_winner(records)["source_index"] == 2

    clean_breaks_tie = (records[1], {**records[2], "clean_length": 101})
    assert select_packed_length_winner(clean_breaks_tie)["clean_length"] == 101


def test_audit_publishes_exact_report_and_winner_source_row_exclusively(tmp_path):
    """Catches an audit report and source subset disagreeing on the winner."""
    source = tmp_path / "source.jsonl"
    report_path = tmp_path / "outputs" / "audit.json"
    subset_path = tmp_path / "outputs" / "winner.jsonl"
    source.write_bytes(b'{"row":0}\n{"row":1}\n{"row":2}\n')
    identity = {"negative_sha256": "a" * 64}

    publish_packed_length_audit(
        input_path=source,
        report_path=report_path,
        subset_output_path=subset_path,
        identity=identity,
        records=_records(),
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(report) == {"schema_version", "identity", "records", "winner"}
    assert report["schema_version"] == 1
    assert report["identity"] == identity
    assert report["records"] == list(_records())
    assert report["winner"] == _records()[2]
    assert subset_path.read_bytes() == b'{"row":2}\n'

    report_bytes = report_path.read_bytes()
    subset_bytes = subset_path.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        publish_packed_length_audit(
            input_path=source,
            report_path=report_path,
            subset_output_path=subset_path,
            identity=identity,
            records=_records(),
        )
    assert report_path.read_bytes() == report_bytes
    assert subset_path.read_bytes() == subset_bytes


@pytest.mark.parametrize("existing", ["report", "subset"])
def test_audit_rejects_either_existing_output_before_publishing(tmp_path, existing):
    """Catches partial publication when either exclusive destination is occupied."""
    source = tmp_path / "source.jsonl"
    report_path = tmp_path / "audit.json"
    subset_path = tmp_path / "winner.jsonl"
    source.write_bytes(b"zero\none\ntwo\n")
    occupied = report_path if existing == "report" else subset_path
    occupied.write_bytes(b"preserve\n")

    with pytest.raises(ValueError, match="already exists"):
        publish_packed_length_audit(
            input_path=source,
            report_path=report_path,
            subset_output_path=subset_path,
            identity={},
            records=_records(),
        )

    assert occupied.read_bytes() == b"preserve\n"
    other = subset_path if existing == "report" else report_path
    assert not other.exists()


def test_audit_rejects_one_path_for_both_outputs_without_partial_publication(tmp_path):
    """Catches the report occupying the subset destination during publication."""
    source = tmp_path / "source.jsonl"
    output = tmp_path / "shared-output"
    source.write_bytes(b"zero\none\ntwo\n")

    with pytest.raises(ValueError, match="distinct"):
        publish_packed_length_audit(
            input_path=source,
            report_path=output,
            subset_output_path=output,
            identity={},
            records=_records(),
        )

    assert not output.exists()


def test_audit_preserves_new_capacity_features_and_old_winner_order(tmp_path):
    """Catches dropping the four measured capacity axes from the audit report."""
    extended = tuple({**row, "vision_token_count": 4, "image_count": 1,
                      "reasoning_block_count": row["step_count"],
                      "scored_token_count": 8} for row in _records())
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"zero\none\ntwo\n")
    report = tmp_path / "audit.json"
    publish_packed_length_audit(input_path=source, report_path=report,
                               subset_output_path=tmp_path / "winner.jsonl",
                               identity={}, records=extended)
    payload = json.loads(report.read_bytes())
    assert payload["records"] == list(extended)
    assert payload["winner"]["source_index"] == 2
