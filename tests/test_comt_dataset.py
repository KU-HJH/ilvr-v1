import json
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.comt_dataset import (
    DatasetValidationError,
    UnsafeArchiveError,
    locate_image_root,
    safe_extract_tar,
    validate_dataset,
)


class ComtDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.image_root = self.root / "images_comt"
        (self.image_root / "creation").mkdir(parents=True)
        for name, color in (("input.png", (255, 0, 0)), ("helper.png", (0, 255, 0))):
            Image.new("RGB", (2, 2), color).save(self.image_root / "creation" / name)
        self.train = self.root / "TRAIN.jsonl"
        self.test = self.root / "TEST.jsonl"
        self.train_record = {
            "text_input": "Choose the correct path.",
            "image_input": "creation/input.png",
            "sequence_plan": [
                {"type": "text", "content": "Inspect the board."},
                {"type": "latent", "helper_image": "creation/helper.png"},
            ],
        }
        self.test_record = {
            "text_input": "Choose the correct path.",
            "image_input": ["creation/input.png"],
            "original_final_answer": "A",
        }
        self.write_jsonl(self.train, [self.train_record])
        self.write_jsonl(self.test, [self.test_record])

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def write_jsonl(path, records):
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    @staticmethod
    def png_bytes():
        buffer = io.BytesIO()
        Image.new("RGB", (2, 2), (255, 0, 0)).save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def write_tar(path, files):
        with tarfile.open(path, "w:gz") as archive:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))

    @staticmethod
    def write_symlink_tar(path, name, target):
        with tarfile.open(path, "w:gz") as archive:
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            archive.addfile(info)

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


if __name__ == "__main__":
    unittest.main()
