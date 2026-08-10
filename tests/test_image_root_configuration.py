import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.task_deepseed import interleaved_latent_cot_preprocess_function


class ImageRootConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.image_root = self.root / "images_comt"
        creation = self.image_root / "creation"
        creation.mkdir(parents=True)
        for name, color in (("input.png", (255, 0, 0)), ("helper.png", (0, 255, 0))):
            Image.new("RGB", (2, 2), color).save(creation / name)
        self.train_record = {
            "text_input": "Choose the correct path.",
            "image_input": "creation/input.png",
            "sequence_plan": [
                {"type": "text", "content": "Inspect the board."},
                {"type": "latent", "helper_image": "creation/helper.png"},
            ],
        }

    def tearDown(self):
        self.temp_dir.cleanup()

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
