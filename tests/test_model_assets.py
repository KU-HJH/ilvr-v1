import json
import tempfile
import unittest
from pathlib import Path

from scripts.model_assets import MODEL_SPEC, SnapshotValidationError, verify_model_snapshot


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
