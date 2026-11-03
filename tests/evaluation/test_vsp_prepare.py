import hashlib
import json
from pathlib import Path
import subprocess
import uuid

import pytest

from evaluation.schema import ContractError
from vsp_fixtures import SOURCE_PROMPT, make_vsp_case, write_source_fixture


TABLE = "| | Col 1 | Col 2 |\n| Row 1 | @ | * |\n| Row 2 | _ | # |"
DESC = (
    "This is a 2x2 map.\nThe player is at: row 1, column 1;\n"
    "The hole(s) are at: Row 2, Column 2;\n"
    "The goal is at: Row 1, Column 2."
)


def _paths():
    root = Path(__file__).resolve().parents[2]
    token = uuid.uuid4().hex
    return (
        root,
        root / "data/evaluation" / ("vsp-source-" + token),
        root / "data/evaluation" / ("vsp-prepared-" + token),
    )


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _verification_case() -> dict:
    root = Path(__file__).resolve().parents[2]
    token = uuid.uuid4().hex
    source_root = root / "data/evaluation" / ("vsp-git-" + token)
    lock_root = root / "data/evaluation" / ("vsp-lock-" + token)
    source_root.mkdir(parents=True, exist_ok=False)
    lock_root.mkdir(parents=True, exist_ok=False)
    files = {
        "LICENSE": b"Synthetic verification license\n",
        "tracked/root-marker.txt": b"tracked root marker\n",
        "evidence/prompt.txt": b"synthetic prompt\n",
        "evidence/legend.png": b"synthetic legend evidence\n",
        "evidence/example.png": b"synthetic example evidence\n",
    }
    for relative, payload in files.items():
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    _git(source_root, "init", "-q")
    _git(source_root, "config", "user.email", "vsp-fixture@example.invalid")
    _git(source_root, "config", "user.name", "VSP fixture")
    _git(source_root, "add", ".")
    _git(source_root, "commit", "-q", "-m", "synthetic fixture")
    origin = "https://example.invalid/Visual-Spatial-Planning.git"
    _git(source_root, "remote", "add", "origin", origin)
    commit = _git(source_root, "rev-parse", "HEAD")
    entry = {
        "name": "Visual-Spatial-Planning",
        "upstream_url": origin,
        "commit": commit,
        "license_files": [
            {
                "path": "LICENSE",
                "sha256": hashlib.sha256(files["LICENSE"]).hexdigest(),
            }
        ],
        "evidence_files": [
            {
                "path": relative,
                "sha256": hashlib.sha256(files[relative]).hexdigest(),
            }
            for relative in (
                "evidence/prompt.txt",
                "evidence/legend.png",
                "evidence/example.png",
            )
        ],
    }
    (lock_root / "references.lock.json").write_text(
        json.dumps({"schema_version": 1, "repositories": [entry]}),
        encoding="utf-8",
    )
    return {
        "source_root": source_root,
        "lock_root": lock_root,
        "commit": commit,
        "entry": entry,
    }


@pytest.mark.parametrize("field", ["license_files", "evidence_files"])
def test_source_verification_rejects_record_missing_required_key_with_extra(
    monkeypatch, field
):
    from evaluation import vsp_prepare as prep

    case = _verification_case()
    lock_path = case["lock_root"] / "references.lock.json"
    lock = json.loads(lock_path.read_text())
    lock["repositories"][0][field][0].pop("sha256")
    lock["repositories"][0][field][0]["extra"] = "must not hide missing sha256"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    monkeypatch.setattr(prep, "SOURCE_COMMIT", case["commit"])

    with pytest.raises(ContractError, match="malformed file record"):
        prep.verify_source_reference(case["source_root"], case["lock_root"])


def test_prepare_rejects_license_record_missing_required_key_with_extra():
    from evaluation import vsp_prepare as prep

    _, source_root, dataset_root = _paths()
    identity = write_source_fixture(source_root)
    identity["license_files"] = [{"path": "LICENSE", "extra": "missing sha256"}]

    with pytest.raises(ContractError, match="invalid source license record"):
        prep.prepare_vsp(source_root, dataset_root, source_identity=identity)
    assert not dataset_root.exists()


def test_table_and_description_must_agree():
    from evaluation import vsp_prepare as prep

    assert prep.parse_map_pair(TABLE, DESC, 2) == ((1, 2), (0, -1))
    with pytest.raises(ContractError):
        prep.parse_map_pair(
            TABLE,
            DESC.replace("Row 1, Column 2.", "Row 2, Column 1."),
            2,
        )


def test_no_hole_description_is_supported():
    from evaluation import vsp_prepare as prep

    table = TABLE.replace("#", "_")
    description = DESC.replace(
        "The hole(s) are at: Row 2, Column 2;", "There is no holes in this map;"
    )
    assert prep.parse_map_pair(table, description, 2) == ((1, 2), (0, 0))


@pytest.mark.parametrize(
    ("table", "description"),
    [
        (
            TABLE,
            DESC.replace(
                "Row 2, Column 2;", "Row 2, Column 2; Row 2, Column 2;"
            ),
        ),
        (TABLE, DESC.replace("Row 2, Column 2;", "Row 3, Column 2;")),
        (TABLE, DESC.replace("2x2", "3x3")),
        (TABLE.replace("Row 2", "row 2"), DESC),
        (TABLE.replace("Col 2", "Column 2"), DESC),
        (TABLE.replace("_", "X"), DESC),
        (TABLE.replace("_", "@"), DESC),
        (TABLE.replace("_", "*"), DESC),
        (TABLE, DESC.replace("The goal is at:", "Goal is at:")),
    ],
)
def test_parser_rejects_malformed_or_ambiguous_map_pairs(table, description):
    from evaluation import vsp_prepare as prep

    with pytest.raises(ContractError):
        prep.parse_map_pair(table, description, 2)


@pytest.mark.parametrize("marker", ["<IMAGE-1>", "<IMAGE-2>", "<TEST-IMAGE>"])
def test_prompt_requires_each_marker_exactly_once(marker):
    from evaluation import vsp_prepare as prep

    with pytest.raises(ContractError):
        prep.build_prompt(SOURCE_PROMPT.replace(marker, "missing"))
    with pytest.raises(ContractError):
        prep.build_prompt(SOURCE_PROMPT + marker)


def test_reusable_vsp_case_is_a_real_preflight_input():
    from evaluation.io import parse_config, read_manifest

    case = make_vsp_case(2)
    config = parse_config(case["config"], case["root"])
    prepared = read_manifest(case["manifest"], config, case["root"])
    assert [sample.id for sample in prepared.samples] == [
        "vsp/level3/0",
        "vsp/level3/1",
    ]
    assert prepared.samples[0].model_input.images == (
        "prompt-visual-images/system-figure-1.png",
        "prompt-visual-images/system-figure-2.png",
        "maps/level3/img/0.png",
    )
    assert prepared.samples[0].target.map_desc == (
        (1, 2, 0),
        (0, 0, 0),
        (0, 0, 0),
    )


def test_prepare_writes_ordered_manifests_and_preserves_source_bytes():
    from evaluation import vsp_prepare as prep
    from evaluation.io import parse_config, read_manifest

    root, source_root, dataset_root = _paths()
    identity = write_source_fixture(source_root)
    source_task = source_root / "VSP-main/maze/task-main"
    source_image = source_task / "maps/level8/img/99.png"

    summary = prep.prepare_vsp(source_root, dataset_root, source_identity=identity)

    full_payload = (dataset_root / "manifest.jsonl").read_bytes()
    smoke_payload = (dataset_root / "smoke-manifest.jsonl").read_bytes()
    full = [json.loads(line) for line in full_payload.splitlines()]
    smoke = [json.loads(line) for line in smoke_payload.splitlines()]
    assert len(full) == 600
    assert len(smoke) == 6
    assert [row["id"] for row in full] == [
        f"vsp/level{size}/{index}"
        for size in range(3, 9)
        for index in range(100)
    ]
    assert [row["id"] for row in smoke] == [
        f"vsp/level{size}/0" for size in range(3, 9)
    ]
    assert (
        dataset_root / "maps/level8/img/99.png"
    ).read_bytes() == source_image.read_bytes()
    assert (
        dataset_root / "prompt-text/prompt-text.txt"
    ).read_bytes() == SOURCE_PROMPT.encode()
    prompt = (dataset_root / "prompt-text/ilvr-three-images-v1.txt").read_text()
    assert prompt.startswith("The three images are provided in this order:")
    assert "<IMAGE-1>" not in prompt and "Image 3 (maze to solve)" in prompt
    assert all(row["model_input"]["text"] == prompt for row in full)
    assert all("map_desc" not in row["model_input"] for row in full)
    assert full[0]["model_input"]["images"] == [
        "prompt-visual-images/system-figure-1.png",
        "prompt-visual-images/system-figure-2.png",
        "maps/level3/img/0.png",
    ]
    assert full[0]["target"] == {
        "map_desc": [[1, 2, 0], [0, 0, 0], [0, 0, 0]],
        "max_steps": 100,
    }
    assert summary == {
        "N_expected": 600,
        "categories": {f"level{size}": 100 for size in range(3, 9)},
        "manifest_sha256": hashlib.sha256(full_payload).hexdigest(),
        "smoke_manifest_sha256": hashlib.sha256(smoke_payload).hexdigest(),
        "prompt_template_identity": "vsp-ilvr-three-images-v1:"
        + hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "protocol_kind": "modified",
        "source_commit": "fixture-v1",
    }
    source = json.loads((dataset_root / "SOURCE.json").read_text())
    assert source["repository"] == {
        "name": "synthetic:vsp-fixture",
        "upstream_url": "synthetic:vsp-fixture",
        "commit": "fixture-v1",
    }
    assert source["protocol_kind"] == "modified"
    assert source["license_files"] == identity["license_files"]
    assert (dataset_root / "LICENSE").read_bytes() == (
        source_root / "LICENSE"
    ).read_bytes()
    config = parse_config(
        {
            "schema_version": 1,
            "benchmark": "vsp",
            "profile": "vsp_validated_v1",
            "categories": list(prep.LEVELS),
            "dataset_root": str(dataset_root),
            "max_steps": 100,
            "run_metadata": {
                "model_identity": "synthetic:model",
                "prompt_template_identity": summary["prompt_template_identity"],
                "generation_config": {"max_new_tokens": 8},
            },
        },
        root,
    )
    assert (
        len(read_manifest(dataset_root / "manifest.jsonl", config, root).samples)
        == 600
    )


@pytest.mark.parametrize("corruption", ["missing_file", "coordinate_mismatch"])
def test_prepare_preflight_failure_creates_no_dataset(corruption):
    from evaluation import vsp_prepare as prep

    _, source_root, dataset_root = _paths()
    identity = write_source_fixture(source_root)
    task_root = source_root / "VSP-main/maze/task-main"
    if corruption == "missing_file":
        (task_root / "maps/level8/img/99.png").unlink()
    else:
        description = task_root / "maps/level7/pure_text/42.txt"
        description.write_text(
            description.read_text().replace(
                "Row 1, Column 2.", "Row 2, Column 2."
            )
        )

    with pytest.raises(ContractError):
        prep.prepare_vsp(source_root, dataset_root, source_identity=identity)
    assert not dataset_root.exists()


def test_prepare_cli_writes_summary_and_log(monkeypatch):
    import scripts.prepare_eval as cli

    root, source_root, dataset_root = _paths()
    identity = write_source_fixture(source_root)
    monkeypatch.setattr(cli, "verify_source_reference", lambda source, repo: identity)
    code = cli.main(
        [
            "--benchmark", "vsp",
            "--profile", "vsp_validated_v1",
            "--source-root", str(source_root),
            "--output-dir", str(dataset_root),
        ],
        repo_root=root,
    )
    result_dir = root / "results/evaluation" / dataset_root.name
    log_dir = root / "logs/evaluation" / dataset_root.name
    assert code == 0
    assert json.loads((result_dir / "prepare.json").read_text())["N_expected"] == 600
    assert "complete" in (log_dir / "prepare.log").read_text()


def test_prepare_cli_rejects_unknown_or_existing_destinations(monkeypatch):
    import scripts.prepare_eval as cli

    root, source_root, dataset_root = _paths()
    identity = write_source_fixture(source_root)
    monkeypatch.setattr(cli, "verify_source_reference", lambda source, repo: identity)
    common = ["--source-root", str(source_root), "--output-dir", str(dataset_root)]
    assert (
        cli.main(
            ["--benchmark", "vsp", "--profile", "unknown", *common],
            repo_root=root,
        )
        == 2
    )
    dataset_root.mkdir(parents=True)
    assert (
        cli.main(
            ["--benchmark", "vsp", "--profile", "vsp_validated_v1", *common],
            repo_root=root,
        )
        == 2
    )


def test_prepare_cli_reports_copy_failure_as_infrastructure_error(monkeypatch):
    import scripts.prepare_eval as cli

    root, source_root, dataset_root = _paths()
    identity = write_source_fixture(source_root)
    monkeypatch.setattr(cli, "verify_source_reference", lambda source, repo: identity)
    def fail_copy(*args, **kwargs):
        raise OSError("copy failed")

    monkeypatch.setattr(cli, "prepare_vsp", fail_copy)
    code = cli.main(
        [
            "--benchmark", "vsp",
            "--profile", "vsp_validated_v1",
            "--source-root", str(source_root),
            "--output-dir", str(dataset_root),
        ],
        repo_root=root,
    )
    assert code == 1
    assert "copy failed" in (
        root / "logs/evaluation" / dataset_root.name / "prepare.log"
    ).read_text()


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("wrong_root", "source root is not the Git checkout root"),
        ("wrong_origin", "source origin does not match reference lock"),
        ("wrong_head", "source HEAD does not match reference lock"),
        ("dirty", "source checkout is not clean"),
        ("bad_hash", "source evidence SHA-256 mismatch"),
    ],
)
def test_source_verification_rejects_git_or_hash_mismatch(
    monkeypatch, corruption, message
):
    from evaluation import vsp_prepare as prep

    case = _verification_case()
    monkeypatch.setattr(prep, "SOURCE_COMMIT", case["commit"])
    source_root = case["source_root"]
    if corruption == "wrong_root":
        source_root = source_root / "tracked"
    elif corruption == "wrong_origin":
        _git(source_root, "remote", "set-url", "origin", "https://wrong.invalid/vsp.git")
    elif corruption == "wrong_head":
        (source_root / "second-commit.txt").write_text("second commit\n")
        _git(source_root, "add", "second-commit.txt")
        _git(source_root, "commit", "-q", "-m", "second synthetic commit")
    elif corruption == "dirty":
        (source_root / "dirty.txt").write_text("untracked\n")
    else:
        lock_path = case["lock_root"] / "references.lock.json"
        lock = json.loads(lock_path.read_text())
        lock["repositories"][0]["evidence_files"][0]["sha256"] = "0" * 64
        lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(ContractError, match=message):
        prep.verify_source_reference(source_root, case["lock_root"])


def test_source_verification_accepts_independent_clean_git_fixture(monkeypatch):
    from evaluation import vsp_prepare as prep

    case = _verification_case()
    monkeypatch.setattr(prep, "SOURCE_COMMIT", case["commit"])

    entry = prep.verify_source_reference(case["source_root"], case["lock_root"])

    assert entry == case["entry"]
