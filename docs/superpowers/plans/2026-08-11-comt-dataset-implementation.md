# CoMT Dataset Construction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Download, safely construct, validate, and wire the complete official CoMT dataset into ILVR training and evaluation.

**Architecture:** A small `scripts` package owns dataset validation, safe archive extraction, Hugging Face download orchestration, manifest generation, and a command-line entry point. Training and evaluation receive an explicit image root instead of relying on machine-specific globals, while the launch scripts provide repository-local defaults.

**Tech Stack:** Python 3.11, standard-library `argparse`/`dataclasses`/`hashlib`/`json`/`tarfile`, Pillow 12, `huggingface_hub` 0.35.3, `unittest`, Bash.

## Global Constraints

- Run every Python command through `./venv/ilvr-venv` by first executing `source ./venv/ilvr-venv/bin/activate`.
- Download only `TRAIN.jsonl`, `TEST.jsonl`, and `comt.tar.gz` from the official `shuai22/comt` Hugging Face dataset repository.
- Use the final layout `data/TRAIN.jsonl`, `data/TEST.jsonl`, `data/images_comt/`, and `data/comt_manifest.json`.
- Keep resumable work under `data/.staging/`; never replace an existing complete dataset implicitly.
- Do not transform examples, generate rationales, or change train/test membership.
- Use the official Hugging Face endpoint unless the user has already supplied `HF_ENDPOINT`; never select a third-party mirror automatically.
- Validate every JSONL record and every unique referenced image before declaring the dataset ready.
- Preserve unrelated dirty-worktree changes and stage only the files named in each task.

---

## File Map

- `scripts/__init__.py`: marks the dataset tooling as an importable package.
- `scripts/comt_dataset.py`: pure validation, archive safety, hashing, download, preparation, and manifest APIs.
- `scripts/prepare_comt.py`: thin `prepare` and `validate` command-line interface.
- `tests/test_comt_dataset.py`: synthetic dataset, archive, orchestration, and rerun tests.
- `tests/test_image_root_configuration.py`: training/evaluation image-root behavior tests.
- `tests/test_launcher_defaults.py`: repository-local shell launcher default tests.
- `src/task_deepseed.py`: preprocessing functions that consume an explicit image root.
- `src/utils_deepseed.py`: training `--image-root` option.
- `src/main.py`: binds the chosen image root into the selected preprocessor.
- `eval.py`: primary evaluation `--image-root` option and path resolution.
- `src/evaluate_deepseed.py`: legacy evaluation entry point kept behaviorally consistent.
- `run_training.sh`: repository-local training data and image defaults.
- `eval.sh`: repository-local test data and image defaults.
- `.gitignore`: excludes downloaded data and staging artifacts.
- `README.md`: documents preparation, validation, and launcher overrides.

---

### Task 1: Full JSONL and Image Validator

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/comt_dataset.py`
- Create: `tests/__init__.py`
- Create: `tests/test_comt_dataset.py`

**Interfaces:**
- Produces: `DatasetValidationError(ValueError)`.
- Produces: `ValidationReport(train_records: int, test_records: int, unique_images: int, missing_images: int, corrupt_images: int)`.
- Produces: `validate_dataset(train_path: Path, test_path: Path, image_root: Path, verify_images: bool = True) -> ValidationReport`.
- Consumes: JSONL fields `text_input`, `image_input`, and training-only `sequence_plan` entries with `type`, `content`, and `helper_image`.

- [ ] **Step 1: Write a passing-fixture test and failing validation cases**

Create a temporary `images_comt/creation/` directory, write two 2×2 RGB PNGs with Pillow, and write one train and one test record. The central valid fixture is:

```python
train_record = {
    "text_input": "Choose the correct path.",
    "image_input": "creation/input.png",
    "sequence_plan": [
        {"type": "text", "content": "Inspect the board."},
        {"type": "latent", "helper_image": "creation/helper.png"},
    ],
}
test_record = {
    "text_input": "Choose the correct path.",
    "image_input": ["creation/input.png"],
    "original_final_answer": "A",
}
```

Add these test methods to `tests/test_comt_dataset.py`:

```python
def test_validate_complete_dataset(self):
    report = validate_dataset(self.train, self.test, self.image_root)
    self.assertEqual(report.train_records, 1)
    self.assertEqual(report.test_records, 1)
    self.assertEqual(report.unique_images, 2)
    self.assertEqual(report.missing_images, 0)
    self.assertEqual(report.corrupt_images, 0)

def test_rejects_malformed_json(self):
    self.train.write_text("{not-json}\n", encoding="utf-8")
    with self.assertRaisesRegex(DatasetValidationError, "TRAIN.jsonl:1"):
        validate_dataset(self.train, self.test, self.image_root)

def test_rejects_missing_required_field(self):
    self.write_jsonl(self.test, [{"image_input": "creation/input.png"}])
    with self.assertRaisesRegex(DatasetValidationError, "text_input"):
        validate_dataset(self.train, self.test, self.image_root)

def test_rejects_invalid_sequence_step(self):
    record = dict(self.train_record)
    record["sequence_plan"] = [{"type": "video", "content": "x"}]
    self.write_jsonl(self.train, [record])
    with self.assertRaisesRegex(DatasetValidationError, "Unknown sequence step type"):
        validate_dataset(self.train, self.test, self.image_root)

def test_reports_missing_and_corrupt_images(self):
    (self.image_root / "creation/helper.png").unlink()
    (self.image_root / "creation/input.png").write_bytes(b"not-an-image")
    with self.assertRaisesRegex(DatasetValidationError, "missing=1, corrupt=1"):
        validate_dataset(self.train, self.test, self.image_root)
```

- [ ] **Step 2: Run the new test module and confirm the import failure**

Run:

```bash
source ./venv/ilvr-venv/bin/activate
python -m unittest tests.test_comt_dataset -v
```

Expected: FAIL because `scripts.comt_dataset` does not exist.

- [ ] **Step 3: Implement strict JSONL parsing and image collection**

In `scripts/comt_dataset.py`, define the public types and entry point exactly as follows:

```python
from dataclasses import dataclass
from pathlib import Path


class DatasetValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ValidationReport:
    train_records: int
    test_records: int
    unique_images: int
    missing_images: int
    corrupt_images: int


def validate_dataset(
    train_path: Path,
    test_path: Path,
    image_root: Path,
    verify_images: bool = True,
) -> ValidationReport:
    train_records, train_images = _parse_split(train_path, require_plan=True)
    test_records, test_images = _parse_split(test_path, require_plan=False)
    if train_records == 0 or test_records == 0:
        raise DatasetValidationError("TRAIN.jsonl and TEST.jsonl must both be non-empty")
    references = train_images | test_images
    missing, corrupt = _check_images(references, image_root, verify_images)
    if missing or corrupt:
        raise DatasetValidationError(
            f"Referenced image validation failed: missing={missing}, corrupt={corrupt}"
        )
    return ValidationReport(train_records, test_records, len(references), 0, 0)
```

Implement `_parse_split()` so it reports `path.name` and one-based line number for malformed JSON, requires a string `text_input`, accepts `image_input` as one string or a list of strings, requires a list `sequence_plan` for training, and adds every latent `helper_image` to the reference set. Empty text is valid; missing or wrongly typed fields are not.

Implement `_check_images()` so relative references resolve under `image_root`, absolute references remain absolute, missing files are counted, and each existing unique file is opened in a context manager and checked with `Image.verify()` when `verify_images=True`.

- [ ] **Step 4: Run the validator tests**

Run:

```bash
source ./venv/ilvr-venv/bin/activate
python -m unittest tests.test_comt_dataset -v
```

Expected: all five validator tests PASS.

- [ ] **Step 5: Commit the validator**

```bash
git add scripts/__init__.py scripts/comt_dataset.py tests/__init__.py tests/test_comt_dataset.py
git commit -m "feat: validate CoMT dataset records and images"
```

---

### Task 2: Safe CoMT Archive Extraction

**Files:**
- Modify: `scripts/comt_dataset.py`
- Modify: `tests/test_comt_dataset.py`

**Interfaces:**
- Produces: `UnsafeArchiveError(ValueError)`.
- Produces: `safe_extract_tar(archive_path: Path, destination: Path) -> None`.
- Produces: `locate_image_root(extraction_root: Path, references: set[str]) -> Path`.
- Consumes: validated relative image references collected by the validator parser.

- [ ] **Step 1: Add archive safety and root-discovery tests**

Add tests that build tar files in memory-backed temporary directories:

```python
def test_safe_extract_rejects_parent_traversal(self):
    archive = self.root / "bad.tar.gz"
    self.write_tar(archive, {"../escape.txt": b"escape"})
    with self.assertRaisesRegex(UnsafeArchiveError, "Unsafe archive member"):
        safe_extract_tar(archive, self.root / "extract")

def test_safe_extract_rejects_escaping_symlink(self):
    archive = self.root / "link.tar.gz"
    self.write_symlink_tar(archive, "images/link", "../../escape")
    with self.assertRaisesRegex(UnsafeArchiveError, "Unsafe archive link"):
        safe_extract_tar(archive, self.root / "extract")

def test_extract_and_locate_single_top_level_image_root(self):
    archive = self.root / "comt.tar.gz"
    self.write_tar(archive, {"COMT/creation/input.png": self.png_bytes()})
    extracted = self.root / "extract"
    safe_extract_tar(archive, extracted)
    located = locate_image_root(extracted, {"creation/input.png"})
    self.assertEqual(located, extracted / "COMT")
```

- [ ] **Step 2: Run the three archive tests and confirm missing-symbol failures**

Run:

```bash
source ./venv/ilvr-venv/bin/activate
python -m unittest \
  tests.test_comt_dataset.ComtDatasetTests.test_safe_extract_rejects_parent_traversal \
  tests.test_comt_dataset.ComtDatasetTests.test_safe_extract_rejects_escaping_symlink \
  tests.test_comt_dataset.ComtDatasetTests.test_extract_and_locate_single_top_level_image_root -v
```

Expected: FAIL because the archive APIs are not defined.

- [ ] **Step 3: Implement member-by-member safety checks before extraction**

Add this structure to `scripts/comt_dataset.py`:

```python
class UnsafeArchiveError(ValueError):
    pass


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            target = (root / member.name).resolve()
            if not target.is_relative_to(root) or member.isdev():
                raise UnsafeArchiveError(f"Unsafe archive member: {member.name}")
            if member.issym():
                link_target = (target.parent / member.linkname).resolve()
                if not link_target.is_relative_to(root):
                    raise UnsafeArchiveError(f"Unsafe archive link: {member.name}")
            if member.islnk():
                link_target = (root / member.linkname).resolve()
                if not link_target.is_relative_to(root):
                    raise UnsafeArchiveError(f"Unsafe archive link: {member.name}")
        archive.extractall(destination, members=members, filter="data")
```

Implement `locate_image_root()` by checking the extraction root and its immediate child directories. A candidate matches only when every supplied relative reference exists beneath it. Require exactly one match and raise `DatasetValidationError` with the candidates checked otherwise. Reject absolute references during root discovery because they do not identify archive layout.

- [ ] **Step 4: Run all dataset tests**

Run:

```bash
source ./venv/ilvr-venv/bin/activate
python -m unittest tests.test_comt_dataset -v
```

Expected: all validator and archive tests PASS.

- [ ] **Step 5: Commit safe extraction**

```bash
git add scripts/comt_dataset.py tests/test_comt_dataset.py
git commit -m "feat: safely extract CoMT image archive"
```

---

### Task 3: Resumable Preparation CLI and Manifest

**Files:**
- Modify: `scripts/comt_dataset.py`
- Create: `scripts/prepare_comt.py`
- Modify: `tests/test_comt_dataset.py`

**Interfaces:**
- Produces: `SOURCE_REPO = "shuai22/comt"` and `SOURCE_FILES = ("TRAIN.jsonl", "TEST.jsonl", "comt.tar.gz")`.
- Produces: `sha256_file(path: Path) -> str`.
- Produces: `resolve_revision(repo_id: str, requested_revision: str | None) -> str`.
- Produces: `download_sources(repo_id: str, revision: str, staging_dir: Path) -> dict[str, Path]`.
- Produces: `prepare_dataset(data_dir: Path, repo_id: str = SOURCE_REPO, revision: str | None = None, force: bool = False) -> Path` returning the manifest path.
- Produces: CLI forms `python -m scripts.prepare_comt prepare --data-dir data` and `python -m scripts.prepare_comt validate --data-dir data`.

- [ ] **Step 1: Add orchestration tests with the network boundary mocked**

Build a synthetic archive containing `COMT/creation/input.png` and `COMT/creation/helper.png`. Patch `resolve_revision()` to return `abc123` and `download_sources()` to return the fixture files. Add:

```python
def test_prepare_promotes_only_validated_dataset_and_writes_manifest(self):
    manifest_path = prepare_dataset(self.root / "data", revision="abc123")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    self.assertEqual(manifest["source"]["repository"], "shuai22/comt")
    self.assertEqual(manifest["source"]["revision"], "abc123")
    self.assertEqual(manifest["records"], {"train": 1, "test": 1})
    self.assertEqual(manifest["images"]["missing"], 0)
    self.assertEqual(manifest["images"]["corrupt"], 0)
    self.assertTrue((self.root / "data/images_comt/creation/input.png").is_file())

def test_prepare_does_not_replace_existing_complete_dataset(self):
    prepare_dataset(self.root / "data", revision="abc123")
    with mock.patch("scripts.comt_dataset.download_sources") as download:
        prepare_dataset(self.root / "data", revision="abc123")
    download.assert_not_called()

def test_prepare_rejects_incomplete_existing_final_layout(self):
    data_dir = self.root / "data"
    data_dir.mkdir()
    (data_dir / "TRAIN.jsonl").write_text("{}\n", encoding="utf-8")
    with self.assertRaisesRegex(FileExistsError, "incomplete dataset"):
        prepare_dataset(data_dir, revision="abc123")
```

- [ ] **Step 2: Run orchestration tests and confirm they fail**

Run:

```bash
source ./venv/ilvr-venv/bin/activate
python -m unittest tests.test_comt_dataset -v
```

Expected: FAIL because preparation and manifest APIs do not exist.

- [ ] **Step 3: Implement official Hub download and staged promotion**

Use `HfApi().dataset_info(repo_id, revision=requested_revision)` to resolve `info.sha`. Download each source with:

```python
path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type="dataset",
    revision=revision,
    local_dir=staging_dir / "downloads",
)
```

`prepare_dataset()` must perform these exact phases:

1. If all four final artifacts exist, call `validate_dataset()` and return the manifest without downloading.
2. If only some final artifacts exist and `force=False`, raise `FileExistsError` without changing them.
3. Resolve the immutable revision and download into `data/.staging/downloads/`.
4. Recreate only `data/.staging/extracted/`, then call `safe_extract_tar()`.
5. Parse relative references from the staged JSONL files and call `locate_image_root()`.
6. Assemble `data/.staging/ready/` with the two unchanged JSONL files and an `images_comt` directory moved from extraction.
7. Run `validate_dataset()` against the ready directory.
8. Write `comt_manifest.json` using the exact top-level keys `source`, `files`, `records`, `images`, and `validated_at_utc`.
9. Confirm no final artifact exists, then move all four ready artifacts into `data/`.

For `force=True`, remove only the four exact final artifact paths after a successful staged validation and immediately before promotion. Never remove `data/` recursively.

- [ ] **Step 4: Implement the thin CLI**

`scripts/prepare_comt.py` must parse `prepare` and `validate` subcommands. Both accept `--data-dir` defaulting to `data`; `prepare` additionally accepts `--repo-id`, `--revision`, and `--force`. The `validate` command prints `dataclasses.asdict(report)` as indented JSON. The `prepare` command prints the returned manifest path.

The module must end with:

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests and exercise CLI help**

Run:

```bash
source ./venv/ilvr-venv/bin/activate
python -m unittest tests.test_comt_dataset -v
python -m scripts.prepare_comt --help
python -m scripts.prepare_comt prepare --help
python -m scripts.prepare_comt validate --help
```

Expected: all tests PASS and every help command exits zero.

- [ ] **Step 6: Commit preparation tooling**

```bash
git add scripts/comt_dataset.py scripts/prepare_comt.py tests/test_comt_dataset.py
git commit -m "feat: prepare complete CoMT dataset"
```

---

### Task 4: Explicit Training Image Root

**Files:**
- Modify: `src/task_deepseed.py:1-175`
- Modify: `src/utils_deepseed.py:15-63`
- Modify: `src/main.py:163-165`
- Create: `tests/test_image_root_configuration.py`

**Interfaces:**
- Produces: every function in `task_preporcess_config` and `task_test_preporcess_config` accepts `sample` plus keyword-only `image_root: str | os.PathLike[str]`.
- Produces: training argument `--image-root`, stored as `args.image_root`, default `data/images_comt`.
- Consumes: `functools.partial(preprocess_function, image_root=resolved_image_root)` in `src/main.py`.

- [ ] **Step 1: Write training preprocessing path tests**

Add a temporary image and a zebra-style record, then assert the returned conversation contains its resolved absolute path:

```python
def test_training_preprocessor_uses_explicit_image_root(self):
    conversation = interleaved_latent_cot_preprocess_function(
        self.train_record,
        image_root=self.image_root,
    )
    self.assertEqual(
        conversation[0]["content"][0]["image"],
        str(self.image_root / "creation/input.png"),
    )
    self.assertEqual(
        conversation[1]["content"][1]["image"],
        str(self.image_root / "creation/helper.png"),
    )

def test_training_preprocessor_rejects_wrong_image_root(self):
    with self.assertRaises(FileNotFoundError):
        interleaved_latent_cot_preprocess_function(
            self.train_record,
            image_root=self.root / "wrong",
        )
```

- [ ] **Step 2: Run the tests and verify the unexpected-keyword failure**

Run:

```bash
source ./venv/ilvr-venv/bin/activate
python -m unittest tests.test_image_root_configuration -v
```

Expected: FAIL because preprocessors do not accept `image_root`.

- [ ] **Step 3: Remove globals and add a keyword-only root to every preprocessor**

Delete `TRAIN_IMAGE_ROOT` and `TEST_IMAGE_ROOT`. Change all six preprocessor signatures using this pattern:

```python
def interleaved_latent_cot_preprocess_function(sample, *, image_root):
    root = os.fspath(image_root)
    full_image_path = _resolve_path(root, image_rel)
```

Apply the same `image_root` keyword to single-image, multiple-image, and test preprocessors so both dispatch dictionaries expose one consistent interface. Keep absolute image paths unchanged.

- [ ] **Step 4: Wire the CLI argument through training**

Add to `get_args()`:

```python
parser.add_argument(
    "--image-root",
    default="data/images_comt",
    help="Root directory for relative image paths in the dataset JSONL.",
)
```

In `src/main.py`, replace direct preprocessor calls with:

```python
image_root = os.path.abspath(os.path.expanduser(args.image_root))
preprocess_function = partial(
    task_preporcess_config[args.task],
    image_root=image_root,
)
train_dataset = load_jsonl_dataset(args.data_path)
train_dataset = [preprocess_function(sample) for sample in train_dataset]
```

- [ ] **Step 5: Run focused and existing utility tests**

Run:

```bash
source ./venv/ilvr-venv/bin/activate
python -m unittest tests.test_image_root_configuration -v
python -m unittest tests.test_comt_dataset -v
```

Expected: both modules PASS.

- [ ] **Step 6: Commit training path configuration**

```bash
git add src/task_deepseed.py src/utils_deepseed.py src/main.py tests/test_image_root_configuration.py
git commit -m "feat: configure CoMT training image root"
```

---

### Task 5: Explicit Evaluation Image Root

**Files:**
- Modify: `eval.py:33-133,292-304,382-403`
- Modify: `src/evaluate_deepseed.py:27-105` and its `run_one_example()` call path
- Modify: `tests/test_image_root_configuration.py`

**Interfaces:**
- Produces: evaluation argument `--image-root`, stored as `args.image_root`, default `data/images_comt`.
- Produces: `_resolve_image_paths(image_input: str | list[str] | None, image_root: str) -> list[str]` in both evaluation modules.
- Produces: `run_one_example(..., image_root: str) -> dict[str, Any]` in both evaluation modules.

- [ ] **Step 1: Add evaluation path-resolution tests**

Load root `eval.py` via `importlib.util.spec_from_file_location` and add:

```python
def test_evaluation_resolves_relative_paths_under_explicit_root(self):
    resolved = self.eval_module._resolve_image_paths(
        ["creation/input.png"],
        str(self.image_root),
    )
    self.assertEqual(resolved, [str(self.image_root / "creation/input.png")])

def test_evaluation_preserves_absolute_paths(self):
    absolute = str(self.image_root / "creation/input.png")
    self.assertEqual(
        self.eval_module._resolve_image_paths(absolute, "/ignored"),
        [absolute],
    )
```

- [ ] **Step 2: Run the focused tests and confirm the signature failure**

Run:

```bash
source ./venv/ilvr-venv/bin/activate
python -m unittest tests.test_image_root_configuration -v
```

Expected: FAIL because `_resolve_image_paths()` accepts only one argument.

- [ ] **Step 3: Thread `image_root` through both evaluators**

Delete `BASE_DATASET_DIR`. Add `--image-root` to both parsers with the same default and help text as training. Use:

```python
def _resolve_image_paths(image_input, image_root: str) -> List[str]:
    paths = [] if image_input is None else ([image_input] if isinstance(image_input, str) else image_input)
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        raise ValueError("image_input must be a string, a list of strings, or null")
    return [path if os.path.isabs(path) else os.path.join(image_root, path) for path in paths]
```

Add `image_root` to `run_one_example()` and call `_resolve_image_paths(sample.get("image_input", []), image_root)`. In each `main()`, resolve `args.image_root` once and pass it to every `run_one_example()` call.

- [ ] **Step 4: Run both focused test modules**

Run:

```bash
source ./venv/ilvr-venv/bin/activate
python -m unittest tests.test_image_root_configuration -v
python -m unittest tests.test_comt_dataset -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit evaluation path configuration**

```bash
git add eval.py src/evaluate_deepseed.py tests/test_image_root_configuration.py
git commit -m "feat: configure CoMT evaluation image root"
```

---

### Task 6: Launcher Defaults, Documentation, and Full Dataset Build

**Files:**
- Modify: `run_training.sh:1-51`
- Modify: `eval.sh:1-62`
- Modify: `.gitignore`
- Modify: `README.md:75-132`
- Create: `tests/test_launcher_defaults.py`
- Generate locally: `data/TRAIN.jsonl`
- Generate locally: `data/TEST.jsonl`
- Generate locally: `data/images_comt/`
- Generate locally: `data/comt_manifest.json`

**Interfaces:**
- Consumes: `python -m scripts.prepare_comt prepare --data-dir data`.
- Consumes: `--data_path`, `--test_data_path`, and common `--image-root` CLI arguments.
- Produces: repository-root-independent launcher defaults derived from each shell script's own directory.

- [ ] **Step 1: Add failing launcher-default tests**

Create `tests/test_launcher_defaults.py` with repository-relative assertions:

```python
import unittest
from pathlib import Path


class LauncherDefaultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[1]

    def test_training_launcher_uses_repository_dataset_and_venv(self):
        script = (self.repo / "run_training.sh").read_text(encoding="utf-8")
        self.assertIn("data/TRAIN.jsonl", script)
        self.assertIn("data/images_comt", script)
        self.assertIn('VENV_BIN="${REPO_ROOT}/venv/ilvr-venv/bin"', script)
        self.assertIn('--image-root "${IMAGE_ROOT}"', script)

    def test_evaluation_launcher_uses_repository_test_split(self):
        script = (self.repo / "eval.sh").read_text(encoding="utf-8")
        self.assertIn("data/TEST.jsonl", script)
        self.assertIn("data/images_comt", script)
        self.assertIn('VENV_BIN="${REPO_ROOT}/venv/ilvr-venv/bin"', script)
        self.assertIn('--image-root "${IMAGE_ROOT}"', script)
```

- [ ] **Step 2: Run the launcher tests and confirm current defaults fail**

Run:

```bash
source ./venv/ilvr-venv/bin/activate
python -m unittest tests.test_launcher_defaults -v
```

Expected: FAIL because both launchers still contain empty data paths and do not pass `--image-root`.

- [ ] **Step 3: Set portable launcher defaults**

At the start of both shell scripts, calculate:

```bash
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV_BIN="${REPO_ROOT}/venv/ilvr-venv/bin"
```

In `run_training.sh`, use environment-overridable defaults:

```bash
HF_HOME="${HF_HOME:-${REPO_ROOT}/cache}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/TRAIN.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-${REPO_ROOT}/data/images_comt}"
SAVE_MODEL_PATH="${SAVE_MODEL_PATH:-${REPO_ROOT}/checkpoints/model_stage1}"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/logs/train.log}"
```

Invoke `${VENV_BIN}/python` for device detection, `${VENV_BIN}/accelerate` for launch, and pass `--image-root "${IMAGE_ROOT}"`. Preserve the existing model and hyperparameter values because changing the base model is outside this dataset milestone.

In `eval.sh`, replace the directory loop with one `TEST_DATA_PATH` defaulting to `${REPO_ROOT}/data/TEST.jsonl`, set `IMAGE_ROOT`, `OUTPUT_DIR`, `EVAL_SCRIPT_PATH`, and `CACHE_DIR` from the repository root, require a non-empty existing `MODEL_DIR`, invoke `${VENV_BIN}/python`, and pass `--image-root "${IMAGE_ROOT}"`.

- [ ] **Step 4: Ignore local dataset artifacts and document exact commands**

Add `/data/` to `.gitignore`. Update README Data Preparation with:

```bash
source ./venv/ilvr-venv/bin/activate
python -m scripts.prepare_comt prepare --data-dir data
python -m scripts.prepare_comt validate --data-dir data
```

Document the final layout, manifest purpose, official endpoint behavior, resumability, `DATA_PATH`/`TEST_DATA_PATH`/`IMAGE_ROOT` overrides, and the fact that model training is not started by dataset preparation.

- [ ] **Step 5: Run all automated and shell syntax checks**

Run:

```bash
source ./venv/ilvr-venv/bin/activate
python -m unittest discover -s tests -v
bash -n run_training.sh
bash -n eval.sh
git diff --check
```

Expected: all tests PASS, both Bash checks exit zero, and `git diff --check` emits no errors.

- [ ] **Step 6: Commit launcher and documentation integration**

```bash
git add run_training.sh eval.sh .gitignore README.md tests/test_launcher_defaults.py
git commit -m "docs: integrate complete CoMT dataset workflow"
```

- [ ] **Step 7: Download and validate the complete official dataset**

Run from the repository root:

```bash
source ./venv/ilvr-venv/bin/activate
unset HF_HUB_OFFLINE
python -m scripts.prepare_comt prepare --data-dir data
python -m scripts.prepare_comt validate --data-dir data
```

Expected: the preparation command prints `data/comt_manifest.json`; validation prints nonzero train/test/image counts with `missing_images` and `corrupt_images` both zero. If the official endpoint remains unreachable, retain `data/.staging/`, report the network error, and do not use an unapproved mirror.

- [ ] **Step 8: Verify one real training and evaluation sample resolves locally**

Run:

```bash
source ./venv/ilvr-venv/bin/activate
python - <<'PY'
import json
from pathlib import Path
from src.task_deepseed import interleaved_latent_cot_preprocess_function

root = Path("data/images_comt").resolve()
with Path("data/TRAIN.jsonl").open(encoding="utf-8") as handle:
    train = json.loads(next(line for line in handle if line.strip()))
conversation = interleaved_latent_cot_preprocess_function(train, image_root=root)
images = [item["image"] for turn in conversation for item in turn["content"] if item["type"] == "image"]
assert images and all(Path(path).is_file() for path in images)

with Path("data/TEST.jsonl").open(encoding="utf-8") as handle:
    test = json.loads(next(line for line in handle if line.strip()))
test_paths = test["image_input"] if isinstance(test["image_input"], list) else [test["image_input"]]
assert test_paths and all((root / path).is_file() if not Path(path).is_absolute() else Path(path).is_file() for path in test_paths)
print(f"training_images={len(images)} evaluation_images={len(test_paths)}")
PY
```

Expected: the command prints positive training and evaluation image counts and exits zero.

- [ ] **Step 9: Perform final evidence review**

Run:

```bash
source ./venv/ilvr-venv/bin/activate
python -m unittest discover -s tests -v
python -m scripts.prepare_comt validate --data-dir data
git status --short
```

Expected: tests PASS; full validation reports zero missing/corrupt images; `data/` does not appear in Git status; only intentional source/documentation changes or pre-existing user changes remain.
