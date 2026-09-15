"""Two-process native Trainer witness with an independently assembled objective."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest
import torch


@pytest.mark.parametrize(
    ("m_execution", "expected_model_m", "expected_decoder_m"),
    (
        ("full_replay", [3, 0, 1], [15, 0, 5]),
        ("packed_suffix", [0, 0, 0], [2, 0, 1]),
    ),
)
def test_native_gloo_trainer_global_block_mean_and_partial_window(
    monkeypatch, m_execution, expected_model_m, expected_decoder_m
):
    # Earlier training tests may leave this global setting behind. This witness
    # specifies FP32, independently of its parent's precision configuration.
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "bf16")
    root = Path(__file__).resolve().parents[2]
    logs = root / "logs/stage2-objective-dev"
    logs.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="gloo-", suffix=".log", dir=logs, mode="w+", delete=False) as log:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   ACCELERATE_MIXED_PRECISION="no",
                   PYTHONPATH=f"{root}:{root / 'transformers/src'}:{root / 'src'}")
        result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                                 str(Path(__file__).resolve()), "--worker", m_execution], env=env, cwd=root,
                                stdout=log, stderr=subprocess.STDOUT, timeout=100)
        log.seek(0)
        output = log.read()
    assert result.returncode == 0, output
    assert '"window_lengths": [2, 2, 1]' in output
    assert '"global_counts": [3, 0, 1]' in output
    assert f'"hooked_model_m": {expected_model_m}' in output
    hooked_decoder_m = [0, 0, 0] if m_execution == "full_replay" else expected_decoder_m
    assert f'"hooked_decoder_m": {hooked_decoder_m}' in output
    assert '"matched_steps": 3' in output


def worker(m_execution="full_replay"):
    from types import SimpleNamespace
    from transformers import TrainingArguments, TrainerCallback
    from stage2.objective import Stage2ObjectiveConfig
    from stage2.trainer import Stage2Trainer
    from tests.stage2.helpers import tiny_stage2_qwen
    from tests.stage2.test_objective_trainer import batch, runtime

    torch.set_num_threads(1)
    torch.manual_seed(23)
    rank = int(os.environ["RANK"])
    model = tiny_stage2_qwen()
    model.visual.requires_grad_(False)
    model.register_buffer("objective_ddp_witness", torch.tensor(7.0))
    config = Stage2ObjectiveConfig(
        True, 0.7, 0.9, "original", stage2_m_execution=m_execution
    )
    rt = runtime(config, model)
    oracle = deepcopy(model)
    oracle_optimizer = torch.optim.SGD([p for p in oracle.parameters() if p.requires_grad], lr=0.01)
    output_root = Path("checkpoints/stage2-objective-dev")
    output_root.mkdir(exist_ok=True)
    directory = tempfile.mkdtemp(prefix=f"gloo-rank{rank}-", dir=output_root)
    settings = TrainingArguments(directory, use_cpu=True, report_to=[], disable_tqdm=True,
                                per_device_train_batch_size=1, gradient_accumulation_steps=2,
                                remove_unused_columns=False, max_steps=3, learning_rate=0.01,
                                lr_scheduler_type="constant", max_grad_norm=0, save_strategy="no",
                                logging_steps=1, ddp_find_unused_parameters=False, dataloader_pin_memory=False)
    upcoming = []
    expected_metric_rows = []
    lengths, global_counts = [], []
    call_number = 0
    expected_counts = [2, 1, 0, 0, 0] if rank == 0 else [0, 0, 0, 0, 1]
    observed_model_m = [0]
    observed_packed_decoder_m = [0]

    def observe_student_forward(_module, _args, _kwargs, _output):
        if not torch.is_grad_enabled():
            observed_model_m[0] += 1

    def observe_packed_decoder(_module, _args, kwargs):
        if kwargs.get("past_key_values") is not None:
            observed_packed_decoder_m[0] += 1

    model_handle = model.register_forward_hook(observe_student_forward, with_kwargs=True)
    decoder_handle = model.model.register_forward_pre_hook(
        observe_packed_decoder, with_kwargs=True
    )
    hooked_model_m, hooked_decoder_m = [], []

    def collator(items):
        nonlocal call_number
        # Accelerate shards an already constructed loader. Encode the local
        # sequence after get_batch_samples instead, preserving the real sampler.
        return batch(two=True)

    class WitnessTrainer(Stage2Trainer):
        def get_batch_samples(self, iterator, number):
            nonlocal call_number

            def local_batches():
                nonlocal call_number
                for item in iterator:
                    count = expected_counts[call_number]
                    call_number += 1
                    if count < 2:
                        item["labels"][0, 12:14] = -100
                    if count < 1:
                        item["labels"][0, 6:8] = -100
                    item["witness_count"] = count
                    yield item

            # witness_count is test metadata; remove it before the production hook.
            counts = []

            def stripped():
                for item in local_batches():
                    counts.append(item.pop("witness_count"))
                    yield item

            batches, num_items = super().get_batch_samples(stripped(), number)
            upcoming[:] = [(deepcopy({k: v for k, v in item.items() if not k.startswith("_stage2") and k != "stage2_text_segments"}), n)
                           for item, n in zip(batches, counts)]
            lengths.append(len(batches))
            if batches:
                global_counts.append(batches[0].get("_stage2_objective_global_count", -1))
            return batches, num_items

    class CheckOracle(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            observed = [None, None]
            torch.distributed.all_gather_object(
                observed, (observed_model_m[0], observed_packed_decoder_m[0])
            )
            hooked_model_m.append(sum(value[0] for value in observed))
            hooked_decoder_m.append(sum(value[1] for value in observed))
            observed_model_m[0] = 0
            observed_packed_decoder_m[0] = 0
            all_batches = [None, None]
            torch.distributed.all_gather_object(all_batches, upcoming)
            batches = [item for worker_batches in all_batches for item in worker_batches]
            dyn, vis, ce = [], [], []
            # These literal positions derive from the 14-token fixture, not the
            # production window selector, objective helpers, or scaling helper.
            for inputs, count in batches:
                f = oracle(**inputs, stage2_finite_ce=True)
                ce.append(f.loss)
                if count:
                    with torch.no_grad():
                        ref = rt.reference.model(**{k: v for k, v in inputs.items() if k != "labels"})
                for pre, post, start, end, target in [(1, 5, 2, 5, 9), (7, 11, 8, 11, 11)][:count]:
                    text_ids = [i for i in range(128) if i not in (80, 81, 82, 90, 91, 92)]

                    def entropy(logits):
                        lp = logits[text_ids].float().log_softmax(-1)
                        return -(lp.exp() * lp).sum()

                    d = ((entropy(f.logits[0, pre]) + 1e-6) / (entropy(ref.logits[0, pre]) + 1e-6)
                         - (entropy(f.logits[0, post]) + 1e-6) / (entropy(ref.logits[0, post]) + 1e-6))
                    dyn.append(-torch.sigmoid(d))
                    with torch.no_grad():
                        m = oracle(**{k: v for k, v in inputs.items() if k != "labels"}, stage2_ablation=(start, end))
                        sm = m.logits[0, post].float().log_softmax(-1)[target]
                    sf = f.logits[0, post].float().log_softmax(-1)[target]
                    vis.append(torch.logaddexp(torch.zeros_like(sf), -(sf - sm)))
            # Native CE divides by configured A=2 even for the 1-batch remainder.
            loss = torch.stack(ce).sum() / 4
            expected_metrics = {"loss_ce": torch.stack(ce).mean().item()}
            if dyn:
                loss = loss + 0.7 * torch.stack(dyn).mean() + 0.9 * torch.stack(vis).mean()
                expected_metrics.update(loss_dyn=torch.stack(dyn).mean().item(), loss_vis=torch.stack(vis).mean().item())
            expected_metrics["loss_total"] = expected_metrics["loss_ce"] + 0.7 * expected_metrics.get("loss_dyn", 0) + 0.9 * expected_metrics.get("loss_vis", 0)
            expected_metric_rows.append(expected_metrics)
            loss.backward()
            for (name, p), (other_name, q) in zip(model.named_parameters(), oracle.named_parameters()):
                assert name == other_name
                if p.requires_grad:
                    torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=3e-5, msg=name)
            oracle_optimizer.step()
            oracle_optimizer.zero_grad()

        def on_step_end(self, args, state, control, **kwargs):
            for p, q in zip(model.parameters(), oracle.parameters()):
                torch.testing.assert_close(p, q, atol=3e-6, rtol=3e-5)

    trainer = WitnessTrainer(model=model, args=settings, train_dataset=[SimpleNamespace(latent_block_count=2) for _ in range(10)],
                             data_collator=collator, stage2_objective_config=config, stage2_objective_runtime=rt,
                             optimizers=(torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01), None),
                             callbacks=[CheckOracle()])
    assert trainer.accelerator.mixed_precision == "no", trainer.accelerator.mixed_precision
    trainer.train()
    assert lengths == [2, 2, 1]
    assert global_counts == [3, 0, 1]
    assert trainer.accelerator.gradient_accumulation_steps == 1
    logs = [entry for entry in trainer.state.log_history if "stage2/objective/loss_ce" in entry]
    assert [entry["stage2/objective/valid_blocks"] for entry in logs] == [3, 0, 1]
    expected_model_m = [3, 0, 1] if m_execution == "full_replay" else [0, 0, 0]
    expected_decoder_m = [15, 0, 5] if m_execution == "full_replay" else [2, 0, 1]
    assert [entry.get("stage2/objective/model_forwards_M", 0) for entry in logs] == expected_model_m
    assert [entry.get("stage2/objective/decoder_forwards_M", 0) for entry in logs] == expected_decoder_m
    assert [entry.get("stage2/objective/m_branches", 0) for entry in logs] == [3, 0, 1]
    assert [entry.get("stage2/objective/m_query_rows", 0) for entry in logs] == [3, 0, 1]
    assert hooked_model_m == expected_model_m
    assert hooked_decoder_m == ([0, 0, 0] if m_execution == "full_replay" else expected_decoder_m)
    for actual, expected in zip(logs, expected_metric_rows, strict=True):
        for key, value in expected.items():
            assert actual["stage2/objective/" + key] == pytest.approx(value, abs=1e-6)
    model_handle.remove()
    decoder_handle.remove()
    print(json.dumps(dict(rank=rank, window_lengths=lengths, global_counts=global_counts,
                         hooked_model_m=hooked_model_m, hooked_decoder_m=hooked_decoder_m,
                         matched_steps=trainer.state.global_step)), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    worker(*(sys.argv[2:] if len(sys.argv) > 2 else ()))
