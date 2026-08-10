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
