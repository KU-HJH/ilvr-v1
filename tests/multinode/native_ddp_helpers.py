"""Real CPU Trainer fixtures and a filesystem transport for artifact tests only."""

import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def result_directory(checkpoint_root):
    directory = ROOT / "results/stage2-ddp-dev" / checkpoint_root.name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ddp_identity(method="full", **changes):
    value = dict(stage="stage2", tuning_method=method, zero_stage=None,
                 deepspeed_config_digest=None, world_size=4,
                 rank_mapping={"0": [0, 1], "1": [2, 3]}, global_step=2,
                 model_state_layout="trainer", optimizer=True, scheduler=True,
                 rng_ranks=[0, 1, 2, 3], stage_sidecar=("stage2-sampler.json" if method == "full" else "stage2_peft_metadata.json"),
                 commit="b" * 40, run_id="ddp-checkpoint-contract",
                 execution_backend="ddp", objective_enabled=True)
    value.update(changes)
    return value


class FilesystemTransport:
    """Exercise production coordinator flow with two explicit filesystem roots."""

    def __init__(self, root):
        self.root = root.resolve()
        self.pushed = []
        self.pulled = []

    def path(self, node):
        return self.root / f"node{node}" / "ddp-checkpoint-contract" / "checkpoint-2"

    def read_artifact_manifest(self, node, checkpoint, identity, *, canonical_owner=False, resume_destination=False):
        from src.multinode_artifacts import build_node_manifest, native_resume_destination_is_canonical
        if resume_destination:
            assert node == 1 and not canonical_owner
            canonical_owner = native_resume_destination_is_canonical(self.path(node),
                runtime_identity=identity, checkpoint_root=self.root / f"node{node}")
        return build_node_manifest(self.path(node), node_rank=node,
                                   process_ranks=(0, 1, 2, 3) if canonical_owner else ((0, 1) if node == 0 else (2, 3)),
                                   runtime_identity=identity, canonical_owner=canonical_owner,
                                   checkpoint_root=self.root / f"node{node}")

    def read_canonical_artifact_manifest(self, checkpoint, identity):
        return self.read_artifact_manifest(0, checkpoint, identity, canonical_owner=True)

    def validate_push_destination(self, manifest, entries):
        from src.multinode_artifacts import validate_transfer_destination
        validate_transfer_destination([self.path(0) / entry.relative_path for entry in entries])

    def push_artifacts(self, manifest, entries):
        self.pushed.extend(str(e.relative_path) for e in entries)
        self._copy(entries, 1, 0)

    def validate_resume_destination(self, entries, target):
        from src.multinode_artifacts import validate_transfer_destination
        validate_transfer_destination([self.path(1) / entry.relative_path for entry in entries])

    def pull_resume_artifacts(self, manifest, entries, target):
        self.pulled.extend(str(e.relative_path) for e in entries)
        self._copy(entries, 0, 1)

    def _copy(self, entries, source, target):
        for entry in entries:
            destination = self.path(target) / entry.relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with (self.path(source) / entry.relative_path).open("rb") as src, destination.open("xb") as dst:
                shutil.copyfileobj(src, dst)


def launch_workers(method, root, log_path, phase):
    root.resolve().relative_to((ROOT / "checkpoints").resolve())
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", ACCELERATE_MIXED_PRECISION="no", ACCELERATE_TORCH_DEVICE="cpu",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
               PYTHONPATH=os.pathsep.join(map(str, (ROOT, ROOT / "src", ROOT / "transformers/src"))))
    for key in ("ACCELERATE_USE_DEEPSPEED", "ACCELERATE_USE_FSDP"):
        env.pop(key, None)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
               str(Path(__file__)), method, str(root), phase]
    with log_path.open("a") as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=150)
    assert completed.returncode == 0, log_path.read_text()[-18000:]


def assert_nested_equal(actual, expected):
    """Compare independently captured tensors/optimizer/RNG data, not success flags."""
    import numpy as np
    import torch
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        assert np.array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_nested_equal(a, e)
    else:
        assert actual == expected


def run_native_checkpoint_probe(method: str, *, output_root: Path, log_path: Path) -> dict:
    import torch
    from src.multinode_runtime import collect_checkpoint_artifacts, redistribute_resume_checkpoint
    from src.multinode_artifacts import validate_loader_artifact
    launch_workers(method, output_root, log_path, "save")
    runner = FilesystemTransport(output_root)
    identity = ddp_identity(method)
    original_files = [sorted(p.name for p in runner.path(node).iterdir()) for node in (0, 1)]
    assert original_files[1] == ["rng_state_2.pth", "rng_state_3.pth"]
    validate_loader_artifact(runner.path(0), identity)
    collect_checkpoint_artifacts(runner, checkpoint_path=runner.path(0), runtime_identity=identity,
                                 exit_codes={0: 0, 1: 0}, world_size=4)
    assert runner.pushed == ["rng_state_2.pth", "rng_state_3.pth"]
    for _ in range(2):
        redistribute_resume_checkpoint(runner, checkpoint_path=runner.path(0),
            checkpoint_identity=identity, requested_identity=identity, destination=runner.path(0))
    assert len(runner.pulled) == len(original_files[0])
    for name in original_files[0] + original_files[1]:
        assert (runner.path(0) / name).read_bytes() == (runner.path(1) / name).read_bytes()
    source_before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (output_root / "source").iterdir()}
    launch_workers(method, output_root, log_path, "resume")
    launch_workers(method, output_root, log_path, "control")
    results = result_directory(output_root)
    summaries = [json.loads((results / f"resume-{rank}.json").read_text()) for rank in range(4)]
    max_model_delta = []
    for rank, summary in enumerate(summaries):
        control_summary = json.loads((results / f"control-{rank}.json").read_text())
        assert summary["state"] == control_summary["state"] == dict(global_step=3, epoch=1.0, max_steps=3)
        assert summary["scheduler"] == control_summary["scheduler"]
        checkpoint = runner.path(rank // 2)
        loaded = torch.load(output_root / f"loaded-{rank}.pt", weights_only=False)
        assert loaded["step"] == 2
        assert_nested_equal(loaded["optimizer"], torch.load(checkpoint / "optimizer.pt", weights_only=False))
        assert_nested_equal(loaded["scheduler"], torch.load(checkpoint / "scheduler.pt", weights_only=False))
        assert_nested_equal(loaded["rng"], torch.load(checkpoint / f"rng_state_{rank}.pth", weights_only=False))
        actual = torch.load(output_root / f"resume-{rank}.pt", weights_only=False)
        expected = torch.load(output_root / f"control-{rank}.pt", weights_only=False)
        assert_nested_equal(actual, expected)
        max_model_delta.append(max((actual["model"][key] - value).abs().max().item() for key, value in expected["model"].items()))
    assert source_before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (output_root / "source").iterdir()}
    assert all(not summary["reference_changed"] for summary in summaries)
    assert all(summary["reference_identity"] == summaries[0]["reference_identity"] for summary in summaries)
    result = dict(original_remote_files=original_files[0], original_local_files=original_files[1],
                  pushed=runner.pushed, pulled=runner.pulled, resumed_steps=[s["step"] for s in summaries],
                  reference_changed=[s["reference_changed"] for s in summaries],
                  max_model_delta=max_model_delta, matches_uninterrupted=True,
                  scheduler=summaries[0]["scheduler"], reference_identity=summaries[0]["reference_identity"])
    (results / "comparison.json").write_text(json.dumps(result, indent=2))
    return result


def _worker(method, root, phase):
    import platform
    platform.python_version = lambda: ".".join(map(str, sys.version_info[:3]))
    import torch
    import random
    import numpy as np
    from copy import deepcopy
    from types import SimpleNamespace
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import (TrainingArguments, TrainerCallback, Qwen2TokenizerFast,
                              Qwen2_5_VLProcessor, Qwen2VLImageProcessor)
    from tests.stage2.helpers import tiny_stage2_qwen
    from tests.stage2.test_objective_trainer import batch, runtime
    from stage2.objective import Stage2ObjectiveConfig
    from stage2.trainer import Stage2Trainer

    rank = int(os.environ["RANK"])
    torch.set_num_threads(1)
    torch.manual_seed(41)
    initial = tiny_stage2_qwen()
    initial.config.supervise_latent_boundaries = True
    vocab = {f"t{i}": i for i in range(128)}
    for number, token in {0: "<|endoftext|>", 80: "<|image_pad|>", 81: "<|vision_start|>",
                          82: "<|video_pad|>", 83: "<|vision_end|>",
                          90: "<|latent_pad|>", 91: "<|latent_start|>", 92: "<|latent_end|>"}.items():
        del vocab[f"t{number}"]
        vocab[token] = number
    tokenizer = Qwen2TokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<|endoftext|>")))
    processor = Qwen2_5_VLProcessor(Qwen2VLImageProcessor(), tokenizer)
    model = deepcopy(initial)
    metadata = None
    if method != "full":
        from stage2.peft import Stage2TuningConfig, build_stage2_tuned_model, resolve_stage2_target_module_names
        from stage2.checkpointing import build_stage2_peft_metadata
        from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds
        tuning = Stage2TuningConfig(method, 2, 4, 0.05, ("q_proj", "v_proj"))
        metadata = build_stage2_peft_metadata(tuning=tuning, base_model_reference=str(root / "source"),
                    model_config=model.config, token_ids=LatentTokenIds(90, 91, 92),
                    resolved_target_names=resolve_stage2_target_module_names(model, tuning.target_modules))
        model = build_stage2_tuned_model(model, tuning)
    config = Stage2ObjectiveConfig(True, 0.5, 0.5, str(root / "source"))
    rt = runtime(config, initial)
    output = root / (f"control{rank // 2}" if phase == "control" else f"node{rank // 2}") / "ddp-checkpoint-contract"

    class Witness(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            if phase == "resume" and state.global_step == 2:
                torch.save(dict(step=state.global_step, optimizer=kwargs["optimizer"].state_dict(),
                    scheduler=kwargs["lr_scheduler"].state_dict(), rng={"python": random.getstate(),
                    "numpy": np.random.get_state(), "cpu": torch.get_rng_state()}), root / f"loaded-{rank}.pt")

        def on_step_end(self, args, state, control, **kwargs):
            if phase == "save" and state.global_step == 2:
                control.should_training_stop = True
                control.should_save = True
            return control

    args = TrainingArguments(str(output), use_cpu=True, ddp_backend="gloo", report_to=[],
                disable_tqdm=True, per_device_train_batch_size=1, gradient_accumulation_steps=2,
                remove_unused_columns=False, max_steps=3, learning_rate=5e-5, seed=42,
                save_strategy="steps", save_steps=2, logging_strategy="no",
                dataloader_pin_memory=False, ddp_find_unused_parameters=True)
    assert str(args.device) == "cpu" and args.world_size == 4
    print(f"NATIVE_CPU_WITNESS rank={rank} world_size={args.world_size} device={args.device}", flush=True)
    trainer = Stage2Trainer(model=model, args=args, processing_class=processor,
                train_dataset=[SimpleNamespace(latent_block_count=1, index=i) for i in range(24)],
                data_collator=lambda examples: batch(), callbacks=[Witness()],
                stage2_objective_config=config, stage2_objective_runtime=rt, stage2_peft_metadata=metadata)
    if rank == 0 and phase == "save":
        initial.save_pretrained(root / "source")
        processor.save_pretrained(root / "source")
    trainer.accelerator.wait_for_everyone()
    # Reload the fixed source independently on every rank, also on resume.
    from stage2.reference import FrozenReference
    from transformers import Qwen2_5_VLForConditionalGeneration
    fixed_reference = Qwen2_5_VLForConditionalGeneration.from_pretrained(root / "source").requires_grad_(False).eval()
    rt.reference = FrozenReference(fixed_reference, {"source": str(root / "source"),
        "model_sha256": hashlib.sha256((root / "source/model.safetensors").read_bytes()).hexdigest()})
    reference_before = {name: p.detach().clone() for name, p in fixed_reference.named_parameters()}
    trainer.train(resume_from_checkpoint=str(output / "checkpoint-2") if phase == "resume" else None)
    observed = {"step": trainer.state.global_step,
                "reference_changed": any(not torch.equal(p, reference_before[name]) for name, p in rt.reference.model.named_parameters()),
                "reference_identity": rt.reference.identity,
                "scheduler": trainer.lr_scheduler.state_dict(),
                "state": {"global_step": trainer.state.global_step, "epoch": trainer.state.epoch,
                          "max_steps": trainer.state.max_steps}}
    root.mkdir(parents=True, exist_ok=True)
    (result_directory(root) / f"{phase}-{rank}.json").write_text(json.dumps(observed))
    torch.save({"model": model.state_dict(), "optimizer": trainer.optimizer.state_dict(),
                "rng": {"cpu": torch.get_rng_state(), "python": random.getstate(),
                        "numpy": np.random.get_state()}}, root / f"{phase}-{rank}.pt")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    _worker(sys.argv[1], Path(sys.argv[2]), sys.argv[3])
