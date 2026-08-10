import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.download_assets import configure_online_environment, download_assets
from scripts.model_assets import (
    CHECKPOINT_SPEC,
    MODEL_SPEC,
    SnapshotValidationError,
    ensure_model_snapshot,
    verify_model_snapshot,
)


REQUIRED_METADATA = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def write_snapshot(root: Path, index_name: str, shard_names: tuple[str, ...]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_METADATA:
        (root / name).write_text("{}\n", encoding="utf-8")
    weight_map = {
        f"layer.{number}.weight": shard
        for number, shard in enumerate(shard_names)
    }
    (root / index_name).write_text(
        json.dumps({"metadata": {"total_size": len(shard_names)}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    for shard in shard_names:
        (root / shard).write_bytes(b"weight")


class ModelSnapshotVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "model"
        self.shards = (
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_accepts_complete_indexed_snapshot(self):
        write_snapshot(self.root, "model.safetensors.index.json", self.shards)
        self.assertEqual(verify_model_snapshot(self.root, MODEL_SPEC), self.root)

    def test_rejects_missing_indexed_shard(self):
        write_snapshot(self.root, MODEL_SPEC.index_filename, self.shards)
        (self.root / self.shards[1]).unlink()
        with self.assertRaisesRegex(SnapshotValidationError, "missing or empty shard"):
            verify_model_snapshot(self.root, MODEL_SPEC)

    def test_rejects_empty_indexed_shard(self):
        write_snapshot(self.root, MODEL_SPEC.index_filename, self.shards)
        (self.root / self.shards[0]).write_bytes(b"")
        with self.assertRaisesRegex(SnapshotValidationError, "missing or empty shard"):
            verify_model_snapshot(self.root, MODEL_SPEC)

    def test_rejects_missing_required_metadata(self):
        write_snapshot(self.root, MODEL_SPEC.index_filename, self.shards)
        (self.root / "preprocessor_config.json").unlink()
        with self.assertRaisesRegex(SnapshotValidationError, "missing or empty file"):
            verify_model_snapshot(self.root, MODEL_SPEC)

    def test_rejects_parent_traversing_index_path(self):
        write_snapshot(self.root, MODEL_SPEC.index_filename, self.shards)
        index_path = self.root / MODEL_SPEC.index_filename
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["weight_map"]["layer.0.weight"] = "../outside.safetensors"
        index_path.write_text(json.dumps(index), encoding="utf-8")
        with self.assertRaisesRegex(SnapshotValidationError, "unsafe shard path"):
            verify_model_snapshot(self.root, MODEL_SPEC)

    def test_rejects_absolute_index_path(self):
        write_snapshot(self.root, MODEL_SPEC.index_filename, self.shards)
        index_path = self.root / MODEL_SPEC.index_filename
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["weight_map"]["layer.0.weight"] = "/tmp/outside.safetensors"
        index_path.write_text(json.dumps(index), encoding="utf-8")
        with self.assertRaisesRegex(SnapshotValidationError, "unsafe shard path"):
            verify_model_snapshot(self.root, MODEL_SPEC)

    def test_rejects_unindexed_weight_file(self):
        write_snapshot(self.root, MODEL_SPEC.index_filename, self.shards)
        (self.root / "model-00003-of-00003.safetensors").write_bytes(b"extra")
        with self.assertRaisesRegex(SnapshotValidationError, "unindexed weight files"):
            verify_model_snapshot(self.root, MODEL_SPEC)


class ModelSnapshotAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.temporary.name) / "models"

    def tearDown(self):
        self.temporary.cleanup()

    def test_fetches_expected_repository_into_expected_directory(self):
        calls = []

        def fake_fetch(**kwargs):
            calls.append(kwargs)
            destination = Path(kwargs["local_dir"])
            write_snapshot(
                destination,
                MODEL_SPEC.index_filename,
                ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"),
            )
            return str(destination)

        result = ensure_model_snapshot(MODEL_SPEC, self.models_dir, fake_fetch)

        self.assertEqual(result, self.models_dir / "Qwen2.5-VL-3B-Instruct")
        self.assertEqual(
            calls,
            [{
                "repo_id": "Qwen/Qwen2.5-VL-3B-Instruct",
                "local_dir": self.models_dir / "Qwen2.5-VL-3B-Instruct",
            }],
        )

    def test_reuses_complete_snapshot_without_fetching(self):
        destination = self.models_dir / MODEL_SPEC.directory_name
        write_snapshot(
            destination,
            MODEL_SPEC.index_filename,
            ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"),
        )

        def forbidden_fetch(**kwargs):
            self.fail(f"unexpected network fetch: {kwargs}")

        self.assertEqual(
            ensure_model_snapshot(MODEL_SPEC, self.models_dir, forbidden_fetch),
            destination,
        )


class UnifiedAssetDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def make_fetch(self, events):
        def fake_fetch(**kwargs):
            repo_id = kwargs["repo_id"]
            destination = Path(kwargs["local_dir"])
            events.append(("snapshot", repo_id))
            if repo_id == MODEL_SPEC.repo_id:
                shards = (
                    "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors",
                )
                write_snapshot(destination, MODEL_SPEC.index_filename, shards)
            else:
                shards = tuple(
                    f"pytorch_model-{number:05d}-of-00007.bin"
                    for number in range(1, 8)
                )
                write_snapshot(destination, CHECKPOINT_SPEC.index_filename, shards)
            return str(destination)

        return fake_fetch

    def test_downloads_all_assets_in_documented_order(self):
        events = []

        def fake_dataset_prepare(data_dir):
            events.append(("dataset", "shuai22/comt"))
            data_dir.mkdir(parents=True, exist_ok=True)
            manifest = data_dir / "comt_manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            return manifest

        results = download_assets(
            "all",
            self.project_root,
            snapshot_fetch=self.make_fetch(events),
            dataset_prepare=fake_dataset_prepare,
        )

        self.assertEqual(
            events,
            [
                ("snapshot", "Qwen/Qwen2.5-VL-3B-Instruct"),
                ("dataset", "shuai22/comt"),
                ("snapshot", "shuai22/comt_ckpt"),
            ],
        )
        self.assertEqual(set(results), {"model", "dataset", "checkpoint"})

    def test_only_model_never_prepares_dataset_or_checkpoint(self):
        events = []

        def forbidden_dataset_prepare(data_dir):
            self.fail(f"unexpected dataset preparation: {data_dir}")

        results = download_assets(
            "model",
            self.project_root,
            snapshot_fetch=self.make_fetch(events),
            dataset_prepare=forbidden_dataset_prepare,
        )

        self.assertEqual(events, [("snapshot", "Qwen/Qwen2.5-VL-3B-Instruct")])
        self.assertEqual(set(results), {"model"})

    def test_online_setup_removes_only_offline_flag(self):
        environment = {
            "HF_HUB_OFFLINE": "1",
            "HTTPS_PROXY": "http://proxy.example:8080",
            "HF_TOKEN": "secret-token",
            "HF_ENDPOINT": "https://configured.example",
        }
        configure_online_environment(environment)
        self.assertEqual(
            environment,
            {
                "HTTPS_PROXY": "http://proxy.example:8080",
                "HF_TOKEN": "secret-token",
                "HF_ENDPOINT": "https://configured.example",
            },
        )

    def test_tags_the_asset_that_failed_during_an_all_run(self):
        from scripts.download_assets import AssetDownloadError

        events = []

        def failing_dataset_prepare(data_dir):
            events.append(("dataset", "shuai22/comt"))
            raise OSError("dataset endpoint timed out")

        with self.assertRaises(AssetDownloadError) as raised:
            download_assets(
                "all",
                self.project_root,
                snapshot_fetch=self.make_fetch(events),
                dataset_prepare=failing_dataset_prepare,
            )

        self.assertEqual(raised.exception.key, "dataset")
        self.assertEqual(str(raised.exception), "dataset endpoint timed out")

    def test_main_reports_single_asset_retry_after_failure(self):
        from scripts import download_assets as command

        error = io.StringIO()
        failure = command.AssetDownloadError(
            "checkpoint", OSError("connection timed out")
        )
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}, clear=False), patch.object(
            command, "download_assets", side_effect=failure
        ), patch.object(sys, "stderr", error):
            status = command.main(["--only", "checkpoint"])
            self.assertNotIn("HF_HUB_OFFLINE", os.environ)

        self.assertEqual(status, 1)
        self.assertIn("checkpoint download failed: connection timed out", error.getvalue())
        self.assertIn(
            "./venv/ilvr-venv/bin/python -m scripts.download_assets --only checkpoint",
            error.getvalue(),
        )
