import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class LauncherDefaultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[1]

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.sandbox = Path(self.temporary.name)
        self.bin_dir = self.sandbox / "venv/ilvr-venv/bin"
        self.bin_dir.mkdir(parents=True)
        (self.sandbox / "data/images_comt").mkdir(parents=True)
        (self.sandbox / "data/TRAIN.jsonl").write_text("{}\n", encoding="utf-8")
        (self.sandbox / "data/TEST.jsonl").write_text("{}\n", encoding="utf-8")
        (self.sandbox / "models/Qwen2.5-VL-3B-Instruct").mkdir(parents=True)
        (self.sandbox / "models/comt_ckpt").mkdir(parents=True)
        self.capture = self.sandbox / "arguments.txt"

    def tearDown(self):
        self.temporary.cleanup()

    def write_executable(self, path: Path, body: str) -> None:
        path.write_text("#!/bin/bash\nset -eu\n" + body, encoding="utf-8")
        path.chmod(0o755)

    def run_launcher(
        self, name: str, overrides: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        shutil.copy2(self.repo / name, self.sandbox / name)
        environment = os.environ.copy()
        environment.pop("MODEL_NAME", None)
        environment.pop("MODEL_DIR", None)
        environment["CAPTURE_FILE"] = str(self.capture)
        environment.update(overrides or {})
        return subprocess.run(
            ["bash", str(self.sandbox / name)],
            cwd=self.sandbox,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_training_launcher_passes_local_3b_model_and_data(self):
        self.write_executable(self.bin_dir / "python", "echo 1\n")
        self.write_executable(
            self.bin_dir / "accelerate",
            'printf "%s\\n" "$@" > "$CAPTURE_FILE"\n',
        )

        result = self.run_launcher("run_training.sh")

        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self.capture.read_text(encoding="utf-8").splitlines()
        model_index = arguments.index("--model")
        data_index = arguments.index("--data_path")
        image_index = arguments.index("--image-root")
        self.assertEqual(
            arguments[model_index + 1],
            str(self.sandbox / "models/Qwen2.5-VL-3B-Instruct"),
        )
        self.assertEqual(arguments[data_index + 1], str(self.sandbox / "data/TRAIN.jsonl"))
        self.assertEqual(arguments[image_index + 1], str(self.sandbox / "data/images_comt"))

    def test_evaluation_launcher_passes_published_checkpoint_and_test_data(self):
        self.write_executable(
            self.bin_dir / "python",
            'printf "%s\\n" "$@" > "$CAPTURE_FILE"\n',
        )

        result = self.run_launcher("eval.sh")

        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self.capture.read_text(encoding="utf-8").splitlines()
        model_index = arguments.index("--model_dir")
        data_index = arguments.index("--test_data_path")
        self.assertEqual(arguments[model_index + 1], str(self.sandbox / "models/comt_ckpt"))
        self.assertEqual(arguments[data_index + 1], str(self.sandbox / "data/TEST.jsonl"))

    def test_training_launcher_honors_model_override(self):
        self.write_executable(self.bin_dir / "python", "echo 1\n")
        self.write_executable(
            self.bin_dir / "accelerate",
            'printf "%s\\n" "$@" > "$CAPTURE_FILE"\n',
        )

        result = self.run_launcher(
            "run_training.sh", {"MODEL_NAME": "/custom/three-billion"}
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self.capture.read_text(encoding="utf-8").splitlines()
        self.assertEqual(
            arguments[arguments.index("--model") + 1], "/custom/three-billion"
        )

    def test_evaluation_launcher_honors_model_override(self):
        custom_model = self.sandbox / "custom-checkpoint"
        custom_model.mkdir()
        self.write_executable(
            self.bin_dir / "python",
            'printf "%s\\n" "$@" > "$CAPTURE_FILE"\n',
        )

        result = self.run_launcher("eval.sh", {"MODEL_DIR": str(custom_model)})

        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self.capture.read_text(encoding="utf-8").splitlines()
        self.assertEqual(
            arguments[arguments.index("--model_dir") + 1], str(custom_model)
        )
