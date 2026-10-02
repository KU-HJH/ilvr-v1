from dataclasses import replace
import json

import pytest
from src.multinode import build_launch_plan, _training_main
import src.multinode_runtime as runtime


def builder():
    api = getattr(runtime, "build_owned_vision_gain_probe_commands", None)
    assert callable(api), "missing fixed owned vision-gain probe builder"
    return api


def test_probe_is_fixed_dora_zero12_four_ranks():
    api = builder()
    for stage in (1, 2):
        plan = build_launch_plan(stage="stage1", tuning_method="dora", zero_stage=stage,
                                 run_id=f"vg-probe-z{stage}", main_process_port=29590)
        commands = api(plan, commit="a" * 40)
        assert [s.machine_rank for s in commands.node_specs] == [0, 1]
        for rank, spec in enumerate(commands.node_specs):
            cmd = spec.command
            assert cmd[cmd.index("--num_machines") + 1] == "2"
            assert cmd[cmd.index("--num_processes") + 1] == "4"
            assert cmd[cmd.index("--machine_rank") + 1] == str(rank)
            assert str(plan.cluster.remote_repo_root / "tests/stage1/vision_gain_zero_probe.py") in cmd
            assert cmd[cmd.index("--phase") + 1] == "initial"
        with pytest.raises((ValueError, TypeError)):
            api(plan, commit="a" * 40, harness_path="evil.py")
        with pytest.raises(ValueError):
            api(replace(plan, tuning_method="full"), commit="a" * 40)
        for changed in (dict(world_size=2), dict(cluster=replace(plan.cluster, num_machines=1)),
                        dict(cluster=replace(plan.cluster, local_rank=0))):
            with pytest.raises(ValueError):
                api(replace(plan, **changed), commit="a" * 40)
    zero3 = build_launch_plan(stage="stage1", tuning_method="dora", zero_stage=3,
                              run_id="vg-probe-z3", main_process_port=29590)
    with pytest.raises(ValueError):
        api(zero3, commit="a" * 40)


def test_fixed_probe_coordinator_dry_run_has_both_attempts(capsys):
    assert _training_main("stage1", ["--vision-gain-runtime-probe", "--run-id", "vg-route",
                                    "--zero-stage", "1", "--main-process-port", "29590", "--dry-run"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["kind"] == "stage1-vision-gain-runtime-probe"
    assert set(record["attempts"]) == {"initial", "resume"}
    assert record["resolved_profile"] == dict(microbatch=1, accumulation=4, global_batch=16,
           learning_rate=5e-5, seed=42, warmup=10, ce_weight=0.1, alpha=0.001, max_steps=2)
