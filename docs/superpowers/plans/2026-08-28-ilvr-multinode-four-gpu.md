# ILVR 2-Node 4-GPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run ILVR Stage 1/2 Full, LoRA, and DoRA with DeepSpeed ZeRO-1/2/3 on two H100 nodes and evaluate one checkpoint with four independent GPU workers.

**Architecture:** Keep the existing Stage launchers and trainers as the only training entrypoints. Add a small Python CLI facade for fixed topology, a runtime module for preflight and owned process control, an artifact module for non-shared checkpoint collection, and an evaluation module for deterministic four-way sharding and merge.

**Tech Stack:** Bash, Python 3.11, PyTorch 2.7.1+cu128, Accelerate 1.10.1, DeepSpeed 0.18.0, pytest, SSH, Docker, tmux, rsync

**Spec:** `docs/superpowers/specs/2026-08-28-ilvr-multinode-four-gpu-design.md`

## Global Constraints

- Invoke Python, pytest, Accelerate, torchrun, DeepSpeed, training, and evaluation through `./venv/ilvr-venv`.
- Use local H100 GPUs 0-1 and `korea@172.16.11.41` container `llm_kd` H100 GPUs 0-1.
- Use `/root/workspace/ILVR` on both nodes, remote rank 0, local rank 1, `num_machines=2`, and cluster-wide `num_processes=4`.
- Set `MASTER_ADDR=172.16.11.41`, `GLOO_SOCKET_IFNAME=bond0.11`, and `NCCL_SOCKET_IFNAME=bond0.11`; require an explicit free `MASTER_PORT`.
- Select only `configs/deepspeed_zero1.json`, `configs/deepspeed_zero2.json`, or `configs/deepspeed_zero3.json`.
- Store checkpoints, results, and logs below `/root/workspace/ILVR/checkpoints/`, `/root/workspace/ILVR/results/`, and `/root/workspace/ILVR/logs/` with one fresh run ID.
- Do not automatically sync source, venv, dataset, or pretrained model during a launch. One-time remote provisioning is a separate task.
- Do not overwrite existing artifacts, retry automatically, call `pkill`, or signal a process without an exact ownership match.
- Stage 1 capacity candidates are 2, 4, and 8 per GPU; test 1 only after candidate 2 is classified as a CUDA capacity OOM. Stage 2 remains batch/GPU 1.
- Hardware completion requires 18 tiny lifecycles: Stage 1/2 x Full/LoRA/DoRA x ZeRO-1/2/3, each with two initial optimizer steps and one resumed step.
- Preserve unrelated user changes and existing failure logs. After a runtime failure, add one causal regression test and retry with a fresh run ID.

---

### Task 1: Commit the validated `/root` path baseline

**Files:**
- Modify: `README.md`
- Modify: `docs/zero-stage-training.md`
- Modify: `docs/superpowers/plans/2026-08-26-ilvr-stage1-dora-batched-training.md`
- Modify: `docs/superpowers/plans/2026-08-27-zero-stage-compatibility.md`
- Modify: `docs/superpowers/plans/2026-08-28-stage1-stage2-zero123-lora-integration.md`
- Modify: `docs/superpowers/specs/2026-08-27-zero-stage-compatibility-design.md`
- Modify: `docs/superpowers/specs/2026-08-28-stage1-stage2-zero123-lora-integration-design.md`
- Modify: `run_stage1_batch_probe.sh`
- Modify: `run_stage1_training.sh`
- Modify: `run_stage1_zero_matrix.sh`
- Modify: `run_stage2_training.sh`
- Modify: `src/artifact_paths.py`
- Modify: `src/stage1/benchmark.py`
- Test: `tests/test_artifact_paths.py`

**Interfaces:**
- Consumes: the already-implemented migration from the obsolete host-specific absolute prefix to `/root`
- Produces: a committed baseline whose canonical repository root is `/root/workspace/ILVR`

- [ ] **Step 1: Verify the path migration and symlink-preserving regression**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src:$PWD/tests/stage1" \
  ./venv/ilvr-venv/bin/pytest \
  tests/test_artifact_paths.py \
  tests/stage1/test_launcher.py::test_batch_probe_keeps_attempt_record_below_its_per_run_result_directory \
  tests/stage1/test_launcher.py::test_live_launcher_rejects_bare_preclaimed_artifact_bypass -q
```

Expected: `11 passed`; canonical paths retain `/root/workspace/ILVR/checkpoints/...` even though `checkpoints` resolves through the repository symlink.

- [ ] **Step 2: Verify the obsolete prefix is absent**

Run:

```bash
obsolete_prefix='/NHN''HOME/WORKSPACE/26msit001_A/PRML_Lab/hjh'
test "$(rg -l --hidden --no-ignore --fixed-strings \
  "$obsolete_prefix" . \
  -g '!.git/**' -g '!**/.git/**' -g '!**/__pycache__/**' | wc -l)" -eq 0
```

Expected: exit 0.

- [ ] **Step 3: Commit only the path migration files**

```bash
git add README.md docs/zero-stage-training.md \
  docs/superpowers/plans/2026-08-26-ilvr-stage1-dora-batched-training.md \
  docs/superpowers/plans/2026-08-27-zero-stage-compatibility.md \
  docs/superpowers/plans/2026-08-28-stage1-stage2-zero123-lora-integration.md \
  docs/superpowers/specs/2026-08-27-zero-stage-compatibility-design.md \
  docs/superpowers/specs/2026-08-28-stage1-stage2-zero123-lora-integration-design.md \
  run_stage1_batch_probe.sh run_stage1_training.sh run_stage1_zero_matrix.sh \
  run_stage2_training.sh src/artifact_paths.py src/stage1/benchmark.py \
  tests/test_artifact_paths.py
git commit -m "fix: canonicalize ILVR paths below root"
```

---

### Task 2: Define the fixed topology and immutable launch plan

**Files:**
- Create: `src/multinode.py`
- Create: `tests/multinode/__init__.py`
- Create: `tests/multinode/test_topology.py`

**Interfaces:**
- Consumes: canonical repository and DeepSpeed config paths
- Produces: `ClusterSpec`, `LaunchPlan`, `canonical_deepspeed_config()`, `build_launch_plan()`, and `python -m src.multinode plan`

- [ ] **Step 1: Write the failing topology tests**

```python
from pathlib import Path

import pytest

from src.multinode import ClusterSpec, build_launch_plan


def test_default_cluster_is_fixed_two_node_four_process_topology():
    spec = ClusterSpec.default()
    assert spec.remote_host == "korea@172.16.11.41"
    assert spec.remote_container == "llm_kd"
    assert spec.remote_repo_root == Path("/root/workspace/ILVR")
    assert spec.remote_host_repo_root == Path(
        "/home/korea/Workspace/hjh/workspace/ILVR"
    )
    assert (spec.num_machines, spec.gpus_per_node, spec.num_processes) == (2, 2, 4)
    assert (spec.remote_rank, spec.local_rank) == (0, 1)
    assert spec.master_addr == "172.16.11.41"
    assert spec.socket_ifname == "bond0.11"


@pytest.mark.parametrize("zero_stage", (1, 2, 3))
def test_launch_plan_selects_exact_zero_config(zero_stage):
    plan = build_launch_plan(
        stage="stage1",
        tuning_method="lora",
        zero_stage=zero_stage,
        run_id=f"stage1-lora-zero{zero_stage}-four-gpu",
        main_process_port=29611 + zero_stage,
    )
    assert plan.world_size == 4
    assert plan.deepspeed_config == Path(
        f"/root/workspace/ILVR/configs/deepspeed_zero{zero_stage}.json"
    )
```

- [ ] **Step 2: Run the tests and confirm the module is missing**

Run: `PYTHONPATH="$PWD/transformers/src:$PWD" ./venv/ilvr-venv/bin/pytest tests/multinode/test_topology.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'src.multinode'`.

- [ ] **Step 3: Implement the immutable types and validation**

```python
@dataclass(frozen=True)
class ClusterSpec:
    remote_host: str
    remote_container: str
    remote_repo_root: Path
    remote_host_repo_root: Path
    master_addr: str
    socket_ifname: str
    remote_rank: int = 0
    local_rank: int = 1
    num_machines: int = 2
    gpus_per_node: int = 2
    num_processes: int = 4

    @classmethod
    def default(cls) -> "ClusterSpec":
        return cls(
            remote_host="korea@172.16.11.41",
            remote_container="llm_kd",
            remote_repo_root=Path("/root/workspace/ILVR"),
            remote_host_repo_root=Path(
                "/home/korea/Workspace/hjh/workspace/ILVR"
            ),
            master_addr="172.16.11.41",
            socket_ifname="bond0.11",
        )


@dataclass(frozen=True)
class LaunchPlan:
    stage: Literal["stage1", "stage2"]
    tuning_method: Literal["full", "lora", "dora"]
    zero_stage: Literal[1, 2, 3]
    run_id: str
    main_process_port: int
    world_size: int
    deepspeed_config: Path
    cluster: ClusterSpec
    attempt_id: str = "initial"
```

Validate run and attempt IDs with `^[a-z0-9][a-z0-9-]{2,127}$`, ports with `1024 <= port <= 65535`, Stage 2 batch/GPU as 1, and only the three canonical ZeRO files. An initial launch uses attempt `initial`; a resume retains the checkpoint's run ID and uses a fresh descriptive attempt such as `resume-step10-to11`. Add a `plan` subcommand that prints the dataclass as sorted JSON and performs no writes or SSH calls.

Before any CUDA initialization, call the existing `src.deepspeed_runtime` validator with `world_size=4` and additionally require that the JSON's `zero_optimization.stage` equals `zero_stage`, bf16 is enabled, and offload/batch fields satisfy the existing Stage contract. Add one parameterized test per ZeRO file plus rejection tests for a copied config whose stage or world-size-derived batch value is changed.

- [ ] **Step 4: Run topology and CLI tests**

Run: `PYTHONPATH="$PWD/transformers/src:$PWD" ./venv/ilvr-venv/bin/pytest tests/multinode/test_topology.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/multinode.py tests/multinode/__init__.py tests/multinode/test_topology.py
git commit -m "feat: define fixed ILVR multi-node topology"
```

---

### Task 3: Extend Stage launchers with explicit Accelerate topology

**Files:**
- Modify: `run_stage1_training.sh:45-58,141-147,205-206,255-259`
- Modify: `run_stage2_training.sh:240-276`
- Modify: `tests/stage1/test_launcher.py`
- Modify: `tests/stage2/test_launcher.py`
- Modify: `tests/stage2/integration/test_distributed_contract.py`

**Interfaces:**
- Consumes: `ILVR_NUM_MACHINES`, `ILVR_NUM_PROCESSES`, `ILVR_MACHINE_RANK`, `ILVR_MAIN_PROCESS_IP`, `ILVR_MAIN_PROCESS_PORT`, `ILVR_SOCKET_IFNAME`
- Produces: existing launchers that preserve the default two-process single-node command and emit a complete four-process multi-node command when all distributed variables are set

- [ ] **Step 1: Write failing Stage 1 and Stage 2 dry-run tests**

```python
def assert_four_rank_prefix(tokens, rank, port):
    expected = {
        "--num_machines": "2",
        "--num_processes": "4",
        "--machine_rank": str(rank),
        "--main_process_ip": "172.16.11.41",
        "--main_process_port": str(port),
    }
    for flag, value in expected.items():
        assert tokens[tokens.index(flag) + 1] == value


def test_stage1_multinode_dry_run_emits_rank_one_four_process_command(tmp_path):
    env = os.environ.copy()
    env.update({
        "ILVR_NUM_MACHINES": "2", "ILVR_NUM_PROCESSES": "4",
        "ILVR_MACHINE_RANK": "1", "ILVR_MAIN_PROCESS_IP": "172.16.11.41",
        "ILVR_MAIN_PROCESS_PORT": "29621", "ILVR_SOCKET_IFNAME": "bond0.11",
    })
    completed = run_launcher(LAUNCHER, *base_args(tmp_path / "out"), env=env)
    assert completed.returncode == 0
    assert_four_rank_prefix(shlex.split(completed.stdout), rank=1, port=29621)
```

Use the same assertion for Stage 2 and parameterize the invalid contract:

```python
def test_stage2_multinode_dry_run_emits_rank_zero_four_process_command(tmp_path):
    env = stage2_env(tmp_path)
    env.update({
        "ILVR_NUM_MACHINES": "2", "ILVR_NUM_PROCESSES": "4",
        "ILVR_MACHINE_RANK": "0", "ILVR_MAIN_PROCESS_IP": "172.16.11.41",
        "ILVR_MAIN_PROCESS_PORT": "29622", "ILVR_SOCKET_IFNAME": "bond0.11",
        "STAGE2_DRY_RUN": "true",
    })
    completed = run_stage2_launcher(env)
    assert completed.returncode == 0
    assert_four_rank_prefix(shlex.split(completed.stdout), rank=0, port=29622)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"ILVR_NUM_MACHINES": "2"}, "all distributed variables"),
        (four_gpu_env(0) | {"ILVR_NUM_PROCESSES": "3"}, "exactly 4"),
        (four_gpu_env(2), "machine rank"),
        (four_gpu_env(0) | {"ILVR_SOCKET_IFNAME": ""}, "interface"),
    ],
)
@pytest.mark.parametrize("run", (run_stage1_dry, run_stage2_dry))
def test_launcher_rejects_invalid_multinode_contract(run, updates, message):
    completed = run(updates)
    assert completed.returncode == 2
    assert message in completed.stderr
```

- [ ] **Step 2: Run tests and verify missing topology flags**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" ./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_launcher.py tests/stage2/test_launcher.py \
  tests/stage2/integration/test_distributed_contract.py -q
```

Expected: new tests fail because the launchers only pass `--num_processes` and a static port.

- [ ] **Step 3: Build the shared Accelerate argument contract in each launcher**

Use this exact behavior in both scripts:

```bash
ilvr_num_machines=${ILVR_NUM_MACHINES:-1}
ilvr_num_processes=${ILVR_NUM_PROCESSES:-2}
ilvr_machine_rank=${ILVR_MACHINE_RANK:-0}
ilvr_main_process_ip=${ILVR_MAIN_PROCESS_IP:-}
ilvr_main_process_port=${ILVR_MAIN_PROCESS_PORT:-29501}
ilvr_socket_ifname=${ILVR_SOCKET_IFNAME:-}

accelerate_topology=(
  --num_processes "$ilvr_num_processes"
  --main_process_port "$ilvr_main_process_port"
)
if [[ $ilvr_num_machines == 2 ]]; then
  accelerate_topology+=(
    --num_machines 2
    --machine_rank "$ilvr_machine_rank"
    --main_process_ip "$ilvr_main_process_ip"
  )
  export GLOO_SOCKET_IFNAME="$ilvr_socket_ifname"
  export NCCL_SOCKET_IFNAME="$ilvr_socket_ifname"
  export MASTER_ADDR="$ilvr_main_process_ip"
fi
```

Validate the six distributed variables as an all-or-none group. Pass `"${accelerate_topology[@]}"` before the Python entrypoint and use `ilvr_num_processes` for DeepSpeed world-size validation. Keep single-node defaults byte-for-byte equivalent to existing behavior.

- [ ] **Step 4: Run launcher tests and shell syntax**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" ./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_launcher.py tests/stage2/test_launcher.py \
  tests/stage2/integration/test_distributed_contract.py -q
bash -n run_stage1_training.sh run_stage2_training.sh
```

Expected: tests pass; no shell syntax errors.

- [ ] **Step 5: Commit**

```bash
git add run_stage1_training.sh run_stage2_training.sh \
  tests/stage1/test_launcher.py tests/stage2/test_launcher.py \
  tests/stage2/integration/test_distributed_contract.py
git commit -m "feat: pass explicit multi-node topology to Accelerate"
```

---

### Task 4: Implement local and remote preflight

**Files:**
- Create: `src/multinode_runtime.py`
- Create: `tests/multinode/test_preflight.py`
- Modify: `src/multinode.py`

**Interfaces:**
- Consumes: `ClusterSpec`, `LaunchPlan`, input paths, and a command runner `Callable[[Sequence[str]], CompletedProcess[str]]`
- Produces: `NodeSnapshot`, `PreflightReport`, `build_remote_exec()`, `collect_node_snapshot()`, and `validate_preflight()`

- [ ] **Step 1: Write failing pure preflight tests**

```python
from src.multinode_runtime import NodeSnapshot, validate_preflight


def snapshot(node, rank):
    return NodeSnapshot(
        node=node, machine_rank=rank, commit="a" * 40, tracked_clean=True,
        repository=Path("/root/workspace/ILVR"), container_running=True,
        python_version="3.11", torch_version="2.7.1+cu128",
        accelerate_version="1.10.1", deepspeed_version="0.18.0",
        torchrun_ok=True,
        gpu_names=("NVIDIA H100 NVL", "NVIDIA H100 NVL"),
        interface="bond0.11", input_paths_ok=True, artifact_roots_writable=True,
        artifact_free_bytes=500_000_000_000, run_paths_fresh=True,
        tmux_sessions_fresh=True, master_port_free=True,
    )


def test_preflight_accepts_symmetric_clean_nodes():
    report = validate_preflight(snapshot("remote", 0), snapshot("local", 1))
    assert report.world_size == 4
    assert report.commit == "a" * 40


def test_preflight_rejects_commit_mismatch_before_launch():
    remote = snapshot("remote", 0)
    local = dataclasses.replace(snapshot("local", 1), commit="b" * 40)
    with pytest.raises(ValueError, match="same tracked commit"):
        validate_preflight(remote, local)
```

Cover every rejection with one parameterized test so each field names its own failure:

```python
@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"tracked_clean": False}, "tracked files"),
        ({"container_running": False}, "llm_kd"),
        ({"torch_version": "2.7.0+cu126"}, "environment versions"),
        ({"gpu_names": ("NVIDIA H100 NVL",)}, "exactly two GPUs"),
        ({"gpu_names": ("NVIDIA A100", "NVIDIA A100")}, "H100"),
        ({"interface": "eth0"}, "bond0.11"),
        ({"input_paths_ok": False}, "input paths"),
        ({"artifact_roots_writable": False}, "artifact roots"),
        ({"artifact_free_bytes": 1024}, "free space"),
        ({"run_paths_fresh": False}, "fresh run paths"),
        ({"tmux_sessions_fresh": False}, "tmux session"),
        ({"master_port_free": False}, "master port"),
    ],
)
def test_preflight_rejects_invalid_node_snapshot(changes, message):
    remote = dataclasses.replace(snapshot("remote", 0), **changes)
    with pytest.raises(ValueError, match=message):
        validate_preflight(remote, snapshot("local", 1))
```

- [ ] **Step 2: Run tests and verify imports fail**

Run: `PYTHONPATH="$PWD/transformers/src:$PWD" ./venv/ilvr-venv/bin/pytest tests/multinode/test_preflight.py -q`

Expected: `ModuleNotFoundError: No module named 'src.multinode_runtime'`.

- [ ] **Step 3: Implement node snapshots and fixed command builders**

`build_remote_exec(argv)` must return:

```python
[
    "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
    "korea@172.16.11.41", "docker", "exec", "llm_kd",
    "bash", "-lc", shlex.join(argv),
]
```

Implement `node-info` in `src.multinode.py` to emit one JSON object containing `git rev-parse HEAD`, `git status --porcelain --untracked-files=no`, executable checks, `torch.cuda.get_device_name()`, `ip -o link show bond0.11`, exact input existence, and artifact-root writability. Only remote rank 0 attempts `socket.bind(("172.16.11.41", port))` and closes the socket before returning.

Extend that snapshot with the pinned Python, PyTorch, Accelerate, and DeepSpeed versions, `docker inspect` running state for `llm_kd`, `shutil.disk_usage()` free bytes, and fresh run-directory/tmux-name booleans. Reject less than the launch plan's explicit `required_free_bytes` estimate without scanning or hashing model weights.

- [ ] **Step 4: Add `preflight` and `--dry-run` CLI behavior**

`python -m src.multinode preflight` executes local and remote probes. `--dry-run` prints the two command arrays and immutable launch plan without SSH, filesystem creation, or GPU initialization.

- [ ] **Step 5: Run focused tests**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD" ./venv/ilvr-venv/bin/pytest \
  tests/multinode/test_topology.py tests/multinode/test_preflight.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/multinode.py src/multinode_runtime.py tests/multinode/test_preflight.py
git commit -m "feat: preflight fixed ILVR two-node cluster"
```

---

### Task 5: Add owned tmux launch and Stage coordinator wrappers

**Files:**
- Create: `run_multinode_stage1_training.sh`
- Create: `run_multinode_stage2_training.sh`
- Modify: `src/multinode.py`
- Modify: `src/multinode_runtime.py`
- Create: `tests/multinode/test_coordinator.py`

**Interfaces:**
- Consumes: validated `LaunchPlan`, stage arguments/environment, and exact node commands
- Produces: `OwnershipRecord`, `write_ownership()`, `validate_ownership()`, `start_rank0_then_rank1()`, `monitor_owned_jobs()`, and the two root wrappers

- [ ] **Step 1: Write failing ownership tests**

```python
def test_ownership_requires_exact_run_pid_start_commit_and_command():
    record = OwnershipRecord.create(
        run_id="stage1-lora-zero3-four-gpu", attempt_id="initial",
        machine_rank=0, pid=4312,
        process_start_ticks=9981, commit="a" * 40,
        command=("./run_stage1_training.sh", "--method", "lora"),
    )
    assert validate_ownership(record, expected_run_id=record.run_id,
                              expected_attempt_id="initial",
                              expected_commit="a" * 40,
                              observed_start_ticks=9981)
    with pytest.raises(ValueError, match="start time"):
        validate_ownership(record, expected_run_id=record.run_id,
                           expected_attempt_id="initial",
                           expected_commit="a" * 40,
                           observed_start_ticks=9982)


def test_coordinator_starts_remote_rank_zero_before_local_rank_one(fake_runner):
    start_rank0_then_rank1(plan(), fake_runner, ready_timeout_seconds=30)
    assert fake_runner.events[:3] == ["remote-tmux-start", "remote-ready", "local-tmux-start"]
```

Add these causal coordinator tests:

```python
def test_rank_zero_timeout_prevents_rank_one_start(fake_runner):
    fake_runner.remote_ready = False
    with pytest.raises(TimeoutError, match="rank 0 readiness"):
        start_rank0_then_rank1(plan(), fake_runner, ready_timeout_seconds=1)
    assert "local-tmux-start" not in fake_runner.events


def test_dry_run_starts_neither_node(fake_runner):
    commands = start_rank0_then_rank1(plan(), fake_runner, dry_run=True)
    assert len(commands) == 2
    assert fake_runner.events == []


def test_peer_stop_requires_exact_ownership(fake_runner):
    fake_runner.observed_start_ticks = 9982
    with pytest.raises(ValueError, match="start time"):
        stop_owned_peer(record(start_ticks=9981), fake_runner)
    assert "signal" not in fake_runner.events
```

- [ ] **Step 2: Run tests and verify missing ownership API**

Run: `PYTHONPATH="$PWD/transformers/src:$PWD" ./venv/ilvr-venv/bin/pytest tests/multinode/test_coordinator.py -q`

Expected: import failure for `OwnershipRecord`.

- [ ] **Step 3: Implement ownership and condition polling**

The JSON schema is exact:

```python
@dataclass(frozen=True)
class OwnershipRecord:
    schema_version: int
    run_id: str
    attempt_id: str
    machine_rank: int
    pid: int
    process_start_ticks: int
    commit: str
    command_digest: str
```

Write ownership to the fresh `$log_root/$run_id/$attempt_id/` directory through a same-directory temporary file, `os.chmod(path, 0o600)`, then `os.replace`. `node-run` writes ownership before `os.execvpe()` of the Stage launcher. Poll every 250 ms for at most 30 seconds, checking a condition rather than sleeping a fixed launch delay.

Serialize the immutable `LaunchPlan` to `$local_log_root/$run_id/launch-plan.json` with exclusive creation, then push that one small file to the fresh remote log directory before rank 0 starts. The remote copy's byte size and SHA-256 must equal the local plan file; this digest covers only the small plan JSON, never a model or checkpoint.

- [ ] **Step 4: Implement fixed tmux and wrapper commands**

Remote session: `ilvr-{run_id}-{attempt_id}-rank0`; local session: `ilvr-{run_id}-{attempt_id}-rank1`. Both use `tmux new-session -d -s SESSION -- COMMAND`. The Stage 1 wrapper calls `python -m src.multinode train-stage1`; the Stage 2 wrapper calls `python -m src.multinode train-stage2`. Coordinator-owned flags are `--run-id`, optional `--attempt-id` (default `initial`), `--zero-stage`, `--main-process-port`, optional `--result-root` (default `/root/workspace/ILVR/results`), and `--dry-run`; Stage 1 forwards remaining allowlisted launcher flags, while Stage 2 forwards the existing allowlisted environment. On success, create `$result_root/$run_id/$attempt_id-train-witness.json` exclusively with stage, method, ZeRO stage, batch/GPU, world size, ranks, exit codes, checkpoint path, and finite-loss evidence.

- [ ] **Step 5: Run coordinator tests and help checks**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD" ./venv/ilvr-venv/bin/pytest \
  tests/multinode/test_coordinator.py tests/multinode/test_preflight.py -q
bash -n run_multinode_stage1_training.sh run_multinode_stage2_training.sh
./run_multinode_stage1_training.sh --help
./run_multinode_stage2_training.sh --help
```

Expected: tests and syntax pass; help exits 0 without SSH.

- [ ] **Step 6: Commit**

```bash
git add run_multinode_stage1_training.sh run_multinode_stage2_training.sh \
  src/multinode.py src/multinode_runtime.py tests/multinode/test_coordinator.py
git commit -m "feat: coordinate owned two-node training jobs"
```

---

### Task 6: Collect and redistribute non-shared checkpoint shards

**Files:**
- Create: `src/multinode_artifacts.py`
- Create: `tests/multinode/test_artifacts.py`
- Modify: `src/multinode.py`
- Modify: `src/multinode_runtime.py`

**Interfaces:**
- Consumes: completed node run directories and their owned process ranks
- Produces: `ArtifactEntry`, `NodeArtifactManifest`, `build_node_manifest()`, `validate_collection()`, `build_push_command()`, `build_pull_command()`, resume preparation, and a fresh local inference copy

- [ ] **Step 1: Write failing manifest and transfer tests**

```python
def test_manifest_records_size_and_rank_without_weight_hash(tmp_path):
    shard = tmp_path / "global_step2/zero_pp_rank_2_mp_rank_00_optim_states.pt"
    shard.parent.mkdir()
    shard.write_bytes(b"rank-two")
    manifest = build_node_manifest(tmp_path, node_rank=1, process_ranks=(2, 3))
    assert manifest.entries == (
        ArtifactEntry(
            relative_path=Path("global_step2/zero_pp_rank_2_mp_rank_00_optim_states.pt"),
            byte_size=8, node_rank=1,
        ),
    )


def test_collection_requires_all_four_process_ranks():
    with pytest.raises(ValueError, match="ranks 0 through 3"):
        validate_collection((manifest_for((0, 1)),), world_size=4)


def test_push_never_overwrites_destination():
    command = build_push_command(local_manifest())
    assert "--ignore-existing" in command
    assert "--files-from" in command
    assert "--checksum" not in command
```

Use a parameterized manifest rejection test plus a destination precondition test:

```python
@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (manifest_with_symlink(), "symlink"),
        (manifest_with_path(Path("../escape.pt")), "relative path"),
        (manifests_with_conflicting_duplicate_sizes(), "different sizes"),
        (manifests_with_run_ids("run-a", "run-b"), "run ID"),
        (manifests_without_completion_marker(), "completion marker"),
    ],
)
def test_collection_rejects_invalid_manifests(manifest, message):
    with pytest.raises(ValueError, match=message):
        validate_collection(manifest, world_size=4)


def test_transfer_refuses_preexisting_destination(tmp_path):
    destination = tmp_path / "global_step2/model.safetensors"
    destination.parent.mkdir()
    destination.write_bytes(b"owned-by-earlier-run")
    with pytest.raises(FileExistsError, match="destination"):
        validate_transfer_destination((destination,))


@pytest.mark.parametrize(
    "field", ("stage", "tuning_method", "zero_stage", "world_size", "run_id")
)
def test_resume_rejects_identity_change_before_transfer(field):
    requested = dataclasses.replace(checkpoint_identity(), **{field: changed(field)})
    with pytest.raises(ValueError, match=field):
        validate_resume_identity(checkpoint_identity(), requested)


def test_inference_copy_requires_fresh_local_destination(tmp_path):
    destination = tmp_path / "inference-copy"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="fresh inference destination"):
        build_inference_pull_command(remote_inference_manifest(), destination)
```

- [ ] **Step 2: Run tests and verify the module is absent**

Run: `PYTHONPATH="$PWD/transformers/src:$PWD" ./venv/ilvr-venv/bin/pytest tests/multinode/test_artifacts.py -q`

Expected: module import failure.

- [ ] **Step 3: Implement small manifests and rsync command construction**

```python
@dataclass(frozen=True)
class ArtifactEntry:
    relative_path: Path
    byte_size: int
    node_rank: int


@dataclass(frozen=True)
class NodeArtifactManifest:
    schema_version: int
    run_id: str
    node_rank: int
    process_ranks: tuple[int, ...]
    entries: tuple[ArtifactEntry, ...]
    runtime_identity: Mapping[str, object]
```

Use `lstat`, reject symlinks and non-regular files, and never read weight contents. Build `rsync --archive --ignore-existing --files-from MANIFEST` commands between local `/root/workspace/ILVR` and remote host `/home/korea/Workspace/hjh/workspace/ILVR`. Validate the small common metadata by runtime identity and byte size before collection.

- [ ] **Step 4: Integrate collection and resume preparation**

After both nodes exit 0, collect local rank files into remote rank 0's fresh canonical run directory and validate exactly ranks 0-3. Before resume, compare stage, method, ZeRO stage/config digest, world size/rank mapping, optimizer/scheduler/global step/RNG, Stage-specific sidecar, commit, and run ID before any process group or transfer; then pull only process-rank 2/3 files and required common metadata. For evaluation, validate the final inference manifest and pull it once from remote owner into a fresh local evaluation model directory. Preserve both source trees after transfer failure.

- [ ] **Step 5: Run artifact and coordinator tests**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD" ./venv/ilvr-venv/bin/pytest \
  tests/multinode/test_artifacts.py tests/multinode/test_coordinator.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/multinode.py src/multinode_runtime.py src/multinode_artifacts.py \
  tests/multinode/test_artifacts.py tests/multinode/test_coordinator.py
git commit -m "feat: collect run-owned DeepSpeed checkpoint shards"
```

---

### Task 7: Add deterministic evaluation sharding and merge

**Files:**
- Create: `src/multinode_evaluation.py`
- Create: `tests/multinode/test_evaluation.py`
- Modify: `eval.py:51-68,458-564`
- Modify: `src/evaluate_deepseed.py:44-55,332-414`
- Modify: `tests/stage2/test_eval_integration.py`

**Interfaces:**
- Consumes: one JSONL dataset, `num_shards=4`, and `shard_index=0..3`
- Produces: `iter_owned_samples()`, `write_shard_manifest()`, `merge_shards()`, global source indices, one merged JSONL, and one summary JSON

- [ ] **Step 1: Write failing pure sharding tests**

```python
def test_four_shards_cover_every_source_index_once():
    samples = [{"id": i} for i in range(11)]
    owned = [
        [index for index, _ in iter_owned_samples(samples, 4, shard)]
        for shard in range(4)
    ]
    assert owned == [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7]]
    assert sorted(index for shard in owned for index in shard) == list(range(11))


def test_merge_rejects_duplicate_source_index(tmp_path):
    shards = write_fake_shards(tmp_path, indices=((0, 4), (1, 5), (2, 6), (3, 4)))
    with pytest.raises(ValueError, match="duplicate source index 4"):
        merge_shards(shards, expected_sample_count=7,
                     output_jsonl=tmp_path / "merged.jsonl",
                     summary_json=tmp_path / "summary.json")
```

Add explicit merge rejection cases:

```python
@pytest.mark.parametrize(
    ("shards", "message"),
    [
        (fake_shards(indices=((0,), (1,), (2,), ())), "missing source index 3"),
        (fake_shards(dataset_sizes=(10, 10, 11, 10)), "dataset identity"),
        (fake_shards(model_ids=("a", "a", "b", "a")), "model identity"),
        (fake_shards(raw_rows=("{not-json}",)), "invalid JSON"),
        (fake_shards(rows=({"index": 0, "error": "OOM"},)), "error row"),
        (fake_shards(indices=((0, 4), (1,), (2,), (3,))), "out of range"),
    ],
)
def test_merge_rejects_invalid_worker_output(tmp_path, shards, message):
    with pytest.raises(ValueError, match=message):
        merge_shards(shards, expected_sample_count=4,
                     output_jsonl=tmp_path / "merged.jsonl",
                     summary_json=tmp_path / "summary.json")
```

- [ ] **Step 2: Run tests and verify the module is absent**

Run: `PYTHONPATH="$PWD/transformers/src:$PWD" ./venv/ilvr-venv/bin/pytest tests/multinode/test_evaluation.py -q`

Expected: module import failure.

- [ ] **Step 3: Implement shard selection, manifest, and merge**

```python
def iter_owned_samples(samples, num_shards: int, shard_index: int):
    if num_shards != 4 or shard_index not in range(4):
        raise ValueError("multi-node evaluation requires four shards indexed 0-3")
    for source_index, sample in enumerate(samples):
        if source_index % num_shards == shard_index:
            yield source_index, sample
```

Each sidecar manifest records dataset path/stat, model identity, shard index/count, expected source indices, processed indices, and worker exit status. `merge_shards()` excludes existing summary rows, rejects any error row, sorts by `index`, writes with exclusive creation, and computes accuracy from `match` once.

- [ ] **Step 4: Integrate both evaluation entrypoints**

Add `--num-shards` default 1 and `--shard-index` default 0. Preserve current single-worker output exactly. In four-shard mode enumerate the unfiltered dataset, process only owned samples, store the global source index in `index`, report progress using a local processed counter, and write the sidecar after the JSONL is closed.

- [ ] **Step 5: Run evaluation tests**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src:$PWD/tests/stage2" \
  ./venv/ilvr-venv/bin/pytest \
  tests/multinode/test_evaluation.py tests/stage2/test_eval_integration.py -q
```

Expected: all tests pass for both `eval` and `evaluate_deepseed`.

- [ ] **Step 6: Commit**

```bash
git add src/multinode_evaluation.py eval.py src/evaluate_deepseed.py \
  tests/multinode/test_evaluation.py tests/stage2/test_eval_integration.py
git commit -m "feat: shard ILVR evaluation across four workers"
```

---

### Task 8: Coordinate four evaluation workers

**Files:**
- Create: `run_multinode_evaluation.sh`
- Modify: `src/multinode.py`
- Modify: `src/multinode_runtime.py`
- Modify: `src/multinode_artifacts.py`
- Create: `tests/multinode/test_evaluation_coordinator.py`

**Interfaces:**
- Consumes: replicated inference checkpoint, TEST JSONL, run ID, task name, and output root
- Produces: remote workers for shards 0/1, local workers for shards 2/3, transferred remote JSONL/sidecars, merged JSONL, and summary JSON

- [ ] **Step 1: Write failing assignment and completion tests**

```python
def test_evaluation_assigns_remote_zero_one_and_local_two_three():
    assignments = build_evaluation_assignments(run_id="eval-four-gpu")
    assert [(a.node, a.gpu, a.shard_index) for a in assignments] == [
        ("remote", 0, 0), ("remote", 1, 1),
        ("local", 0, 2), ("local", 1, 3),
    ]


def test_incomplete_worker_set_is_not_merged(fake_runner):
    fake_runner.exit_codes["remote-shard-1"] = 9
    with pytest.raises(RuntimeError, match="all four evaluation workers"):
        run_evaluation(plan(), fake_runner)
    assert "merge" not in fake_runner.events
```

- [ ] **Step 2: Run tests and verify assignment API is missing**

Run: `PYTHONPATH="$PWD/transformers/src:$PWD" ./venv/ilvr-venv/bin/pytest tests/multinode/test_evaluation_coordinator.py -q`

Expected: import failure.

- [ ] **Step 3: Implement worker launch and remote result pull**

Launch one owned tmux session per worker with `CUDA_VISIBLE_DEVICES=0` or `1`. Pull only the remote shard JSONL and sidecar files with `rsync --archive --ignore-existing`; never transfer caches or the output directory recursively. Call `merge_shards()` only after four exit-0 ownership records and four valid sidecars.

- [ ] **Step 4: Implement the shell wrapper and dry run**

`run_multinode_evaluation.sh` calls `python -m src.multinode evaluate`. Required flags are `--run-id`, `--model-dir`, `--test-data-path`, `--task-name`, `--output-dir`; optional `--base-model-dir`, generation settings, and `--dry-run` map directly to the existing evaluator.

- [ ] **Step 5: Run tests and syntax checks**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD" ./venv/ilvr-venv/bin/pytest \
  tests/multinode/test_evaluation.py tests/multinode/test_evaluation_coordinator.py -q
bash -n run_multinode_evaluation.sh
./run_multinode_evaluation.sh --help
```

Expected: tests pass and help performs no remote call.

- [ ] **Step 6: Commit**

```bash
git add run_multinode_evaluation.sh src/multinode.py src/multinode_runtime.py \
  src/multinode_artifacts.py tests/multinode/test_evaluation_coordinator.py
git commit -m "feat: coordinate four-worker ILVR evaluation"
```

---

### Task 9: Add the Stage 1 multi-node capacity controller

**Files:**
- Create: `src/multinode_capacity.py`
- Create: `run_multinode_stage1_batch_probe.sh`
- Create: `tests/multinode/test_capacity.py`
- Modify: `src/multinode.py`

**Interfaces:**
- Consumes: Stage 1 run arguments, zero stage, tuning method, explicit port sequence, and node logs
- Produces: `capacity_candidates()`, `classify_failure()`, selected batch evidence, and a probe wrapper whose first launch is batch/GPU 2

- [ ] **Step 1: Write failing candidate and classification tests**

```python
def test_primary_candidates_start_at_two_and_increase():
    assert capacity_candidates(start=2, maximum=8) == (2, 4, 8)


def test_batch_one_requires_candidate_two_cuda_oom():
    with pytest.raises(ValueError, match="candidate 2 CUDA capacity OOM"):
        fallback_candidates(classifications={2: "integration_failure"})
    assert fallback_candidates(classifications={2: "capacity_oom"}) == (1,)


def test_nccl_failure_is_never_classified_as_capacity():
    assert classify_failure("NCCL watchdog timed out") == "integration_failure"
    assert classify_failure("torch.OutOfMemoryError: CUDA out of memory") == "capacity_oom"
```

- [ ] **Step 2: Run tests and verify the module is absent**

Run: `PYTHONPATH="$PWD/transformers/src:$PWD" ./venv/ilvr-venv/bin/pytest tests/multinode/test_capacity.py -q`

Expected: module import failure.

- [ ] **Step 3: Implement the bounded controller**

Use exact statuses `success`, `capacity_oom`, and `integration_failure`. Recognize capacity only from `torch.OutOfMemoryError`, `CUDA out of memory`, or allocator failure text containing CUDA allocation; override to integration failure if logs contain NCCL, collective mismatch, nonfinite loss/gradient, checkpoint, SSH, Docker, or rendezvous failures.

- [ ] **Step 4: Implement the wrapper and evidence**

The wrapper invokes `python -m src.multinode capacity-stage1`. It runs candidates sequentially with fresh child run IDs, stops at the first OOM or after 8 succeeds, and writes one JSON report below `$result_root/{run_id}/capacity.json`; `--result-root` defaults to `/root/workspace/ILVR/results`. It does not retry integration failures.

- [ ] **Step 5: Run tests and syntax**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD" ./venv/ilvr-venv/bin/pytest tests/multinode/test_capacity.py -q
bash -n run_multinode_stage1_batch_probe.sh
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/multinode.py src/multinode_capacity.py \
  run_multinode_stage1_batch_probe.sh tests/multinode/test_capacity.py
git commit -m "feat: probe four-GPU Stage 1 capacity from batch two"
```

---

### Task 10: Build the 18-case tiny lifecycle matrix

**Files:**
- Modify: `tests/stage1/zero_compat_diagnostic.py`
- Modify: `tests/stage1/test_distributed_smoke.py`
- Modify: `tests/stage2/integration/tiny_stage2_train.py`
- Modify: `tests/stage2/integration/test_distributed_contract.py`
- Create: `run_multinode_lifecycle_matrix.sh`
- Create: `tests/multinode/test_lifecycle_matrix.py`
- Create: `src/multinode_results.py`
- Create: `tests/multinode/test_result_validation.py`
- Modify: `src/multinode.py`

**Interfaces:**
- Consumes: the Stage 1 and Stage 2 tiny harnesses and fixed topology coordinator
- Produces: `build_lifecycle_matrix()` with exactly 18 unique cases, harness options for two initial steps plus one resumed step, and one final evidence validator

- [ ] **Step 1: Write failing matrix tests**

```python
def test_matrix_contains_exactly_eighteen_unique_cases():
    cases = build_lifecycle_matrix()
    assert len(cases) == len(set(cases)) == 18
    assert {case.stage for case in cases} == {"stage1", "stage2"}
    assert {case.method for case in cases} == {"full", "lora", "dora"}
    assert {case.zero_stage for case in cases} == {1, 2, 3}
    assert {(case.initial_steps, case.resumed_steps) for case in cases} == {(2, 3)}
```

Add exact parser tests for the two harnesses:

```python
def test_stage1_tiny_parser_accepts_requested_lifecycle_steps():
    args = build_zero_compat_parser().parse_args([
        "--zero-stage", "3", "--method", "dora",
        "--deepspeed", "configs/deepspeed_zero3.json",
        "--experiment-id", "stage1-zero3-dora",
        "--initial-steps", "2", "--resumed-steps", "3",
    ])
    assert (args.initial_steps, args.resumed_steps) == (2, 3)


def test_stage2_tiny_parser_preserves_old_defaults_and_accepts_override():
    defaults = build_tiny_stage2_parser().parse_args([
        "--experiment-id", "stage2-defaults",
    ])
    requested = build_tiny_stage2_parser().parse_args([
        "--experiment-id", "stage2-requested",
        "--initial-steps", "2", "--resumed-steps", "3",
    ])
    assert (defaults.initial_steps, defaults.resumed_steps) == (1, 2)
    assert (requested.initial_steps, requested.resumed_steps) == (2, 3)
```

Add the result-validator contract:

```python
def test_validate_results_requires_matrix_eval_and_all_capacity_cells(tmp_path):
    report = validate_results(
        write_matrix_witnesses(tmp_path, count=18),
        write_evaluation_witness(tmp_path, source_indices=range(4)),
        write_capacity_witnesses(tmp_path, stage1_count=9, stage2_count=9),
    )
    assert report.matrix_cases == 18
    assert report.evaluation_samples == 4
    assert len(report.stage1_recommendations) == 9
    assert set(report.stage2_recommendations.values()) == {1}


def test_validate_results_rejects_missing_capacity_cell(tmp_path):
    with pytest.raises(ValueError, match="nine Stage 1 capacity cells"):
        validate_results(
            write_matrix_witnesses(tmp_path, count=18),
            write_evaluation_witness(tmp_path, source_indices=range(4)),
            write_capacity_witnesses(tmp_path, stage1_count=8, stage2_count=9),
        )
```

- [ ] **Step 2: Run tests and verify matrix/harness options are missing**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" ./venv/ilvr-venv/bin/pytest \
  tests/multinode/test_lifecycle_matrix.py \
  tests/multinode/test_result_validation.py \
  tests/stage1/test_distributed_smoke.py \
  tests/stage2/integration/test_distributed_contract.py -q
```

Expected: new tests fail for missing matrix and parser options.

- [ ] **Step 3: Parameterize the existing harnesses**

Add positive integer `--initial-steps` and `--resumed-steps`; require `resumed_steps == initial_steps + 1`. Replace hardcoded checkpoint/global step assertions with these values. Preserve current defaults 1 and 2 so existing two-rank tests do not change meaning.

- [ ] **Step 4: Implement matrix planning and wrapper**

`run_multinode_lifecycle_matrix.sh` calls `python -m src.multinode lifecycle-matrix`. Iterate stages, methods, then ZeRO stages in deterministic order. Each case receives a fresh run ID and explicit port, executes once, collects checkpoint shards, and validates its witness before the next case. Stop on the first failure without retry. Accept `--result-root`, default it to `/root/workspace/ILVR/results`, and create all 18 case witnesses below `$result_root/$run_id/`. Only after all 18 pass, exclusively create `$result_root/$run_id/matrix-complete.json` with the case identities, ports, witness paths, and exit codes.

- [ ] **Step 5: Implement the single final result validator**

In `src/multinode_results.py`, define the immutable fields below and implement `validate_results(matrix_root: Path, evaluation_root: Path, capacity_root: Path) -> ValidationReport`:

```python
@dataclass(frozen=True)
class ValidationReport:
    matrix_cases: int
    evaluation_samples: int
    stage1_recommendations: Mapping[str, int]
    stage2_recommendations: Mapping[str, int]
```

Require all 18 `(stage, method, zero_stage)` witnesses, four unique evaluation shard indices and source indices, nine Stage 1 capacity reports whose `attempts[0].batch_per_gpu` is 2 and whose recommendation comes from `recommended_batch_per_gpu`, and nine Stage 2 witnesses with batch/GPU 1. Reject non-finite loss or gradient, wrong world size, incomplete ranks, missing numbered step-2 checkpoint/resume step 3, mismatched checkpoint identity/method artifact, and any worker error. Add `python -m src.multinode validate-results` with required `--matrix-root`, `--evaluation-root`, `--capacity-root`, and `--output-root` flags. The output root must be an absent report bundle directory published by one atomic no-replace rename and contain `validation-manifest.json` plus `validation-summary.md`; JSON contains the complete report and Markdown contains matching 3×3 Stage 1/2 recommendation tables.

- [ ] **Step 6: Run focused matrix tests and dry run**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" ./venv/ilvr-venv/bin/pytest \
  tests/multinode/test_lifecycle_matrix.py \
  tests/multinode/test_result_validation.py \
  tests/stage1/test_distributed_smoke.py \
  tests/stage2/integration/test_distributed_contract.py -q
bash -n run_multinode_lifecycle_matrix.sh
./run_multinode_lifecycle_matrix.sh --run-id four-gpu-matrix-dry \
  --first-port 29700 --dry-run
```

Expected: dry-run prints exactly 18 plans, ports 29700-29717, and no SSH call.

- [ ] **Step 7: Commit**

```bash
git add tests/stage1/zero_compat_diagnostic.py \
  tests/stage1/test_distributed_smoke.py \
  tests/stage2/integration/tiny_stage2_train.py \
  tests/stage2/integration/test_distributed_contract.py \
  run_multinode_lifecycle_matrix.sh tests/multinode/test_lifecycle_matrix.py \
  src/multinode.py src/multinode_results.py \
  tests/multinode/test_result_validation.py
git commit -m "test: define four-rank ZeRO lifecycle matrix"
```

---

### Task 11: Document operation and run one focused local verification

**Files:**
- Create: `docs/multinode-training.md`
- Modify: `README.md`
- Verify: `tests/multinode/`

**Interfaces:**
- Consumes: all operator-facing launchers
- Produces: exact preflight, Stage 1/2, resume, evaluation, capacity, and failure-cleanup commands

- [ ] **Step 1: Write the operator document**

Document the fixed host/container/root/ranks/interface, one-time provisioning boundary, explicit port requirement, ZeRO selection, no-overwrite behavior, remote-rank-first flow, and these executable examples using shell-generated run IDs:

```bash
run_id="stage1-lora-zero3-$(date -u +%Y%m%d-%H%M%S)"
./run_multinode_stage1_training.sh \
  --run-id "$run_id" --zero-stage 3 --main-process-port 29801 \
  --method lora \
  --model /root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct \
  --data-path /root/workspace/ILVR/data/TRAIN.jsonl \
  --image-root /root/workspace/ILVR/data/images_comt \
  --micro-batch 2 --save-strategy steps --save-steps 10 --max-steps 10
```

Use these remaining examples with distinct ports:

```bash
run_id="stage2-dora-zero2-$(date -u +%Y%m%d-%H%M%S)"
STAGE1_CHECKPOINT=/root/workspace/ILVR/checkpoints/comt_ckpt \
DATA_PATH=/root/workspace/ILVR/data/TRAIN.jsonl \
IMAGE_ROOT=/root/workspace/ILVR/data/images_comt \
TUNING_METHOD=dora \
./run_multinode_stage2_training.sh \
  --run-id "$run_id" --zero-stage 2 --main-process-port 29802 \
  --max_steps 10 --save_steps 10

STAGE1_CHECKPOINT=/root/workspace/ILVR/checkpoints/comt_ckpt \
DATA_PATH=/root/workspace/ILVR/data/TRAIN.jsonl \
IMAGE_ROOT=/root/workspace/ILVR/data/images_comt \
RESUME_CHECKPOINT="/root/workspace/ILVR/checkpoints/$run_id/checkpoint-10" \
TUNING_METHOD=dora \
./run_multinode_stage2_training.sh \
  --run-id "$run_id" --attempt-id resume-step10-to11 \
  --zero-stage 2 --main-process-port 29803 \
  --max_steps 11 --save_steps 11

prepared_model_dir="/root/workspace/ILVR/checkpoints/evaluation-staging/$run_id/prepared-checkpoint-10"
./venv/ilvr-venv/bin/python -m src.multinode prepare-inference \
  --stage stage2 --tuning-method dora --zero-stage 2 \
  --run-id "$run_id" --main-process-port 29802 \
  --checkpoint-step 10 --destination "$prepared_model_dir"

eval_id="eval-four-worker-$(date -u +%Y%m%d-%H%M%S)"
./run_multinode_evaluation.sh \
  --run-id "$eval_id" \
  --model-dir "$prepared_model_dir" \
  --test-data-path /root/workspace/ILVR/data/TEST.jsonl \
  --task-name zebra-cot \
  --output-dir "/root/workspace/ILVR/results/$eval_id"

matrix_id="tiny-matrix-$(date -u +%Y%m%d-%H%M%S)"
./run_multinode_lifecycle_matrix.sh \
  --run-id "$matrix_id" --first-port 29804

capacity_id="stage1-capacity-lora-zero3-$(date -u +%Y%m%d-%H%M%S)"
./run_multinode_stage1_batch_probe.sh \
  --run-id "$capacity_id" --zero-stage 3 --tuning-method lora \
  --main-process-port 29822 \
  --main-process-port 29823 \
  --main-process-port 29824 \
  --main-process-port 29825 \
  --model /root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct \
  --data-path /root/workspace/ILVR/data/TRAIN.jsonl \
  --image-root /root/workspace/ILVR/data/images_comt \
  --save-strategy steps --save-steps 1 --max-steps 1
```

The matrix owns all 18 consecutive ports `29804..29821`, not only its first
two ports. Add the exact `tmux list-sessions`, ownership JSON inspection, and
`python -m src.multinode stop-owned --run-id RUN_ID --machine-rank RANK`
cleanup commands; state that manual `kill`, `pkill`, and session-name-only
termination are forbidden. The final validator example must pass one absent
bundle below a fresh parent and identify both files inside it:

```bash
validation_id="validation-four-gpu-$(date -u +%Y%m%d-%H%M%S)"
validation_parent="/root/workspace/ILVR/results/$validation_id"
test ! -e "$validation_parent"
./venv/ilvr-venv/bin/python -m src.multinode validate-results \
  --matrix-root "/root/workspace/ILVR/results/$matrix_id" \
  --evaluation-root "/root/workspace/ILVR/results/$eval_id" \
  --capacity-root "$capacity_root" \
  --output-root "$validation_parent/validation-report"
test -f "$validation_parent/validation-report/validation-manifest.json"
test -f "$validation_parent/validation-report/validation-summary.md"
```

- [ ] **Step 2: Run the focused implementation suite once**

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src:$PWD/tests/stage2" \
  ./venv/ilvr-venv/bin/pytest \
  tests/multinode \
  tests/stage1/test_launcher.py \
  tests/stage1/test_distributed_smoke.py \
  tests/stage2/test_launcher.py \
  tests/stage2/test_eval_integration.py \
  tests/stage2/integration/test_distributed_contract.py \
  tests/test_artifact_paths.py \
  tests/test_deepspeed_runtime.py -q
bash -n run_stage1_training.sh run_stage2_training.sh \
  run_multinode_stage1_training.sh run_multinode_stage2_training.sh \
  run_multinode_evaluation.sh run_multinode_stage1_batch_probe.sh \
  run_multinode_lifecycle_matrix.sh
```

Expected: all selected tests and shell syntax checks pass.

- [ ] **Step 3: Commit docs**

```bash
git add README.md docs/multinode-training.md
git commit -m "docs: explain ILVR four-GPU multi-node operation"
```

---

### Task 12: Provision `llm_kd` once and execute hardware validation

**Files:**
- Create during execution: `/root/workspace/ILVR/results/<hardware-id>/`
- Create during execution: `/root/workspace/ILVR/logs/<hardware-id>/`
- Create during execution: `/root/workspace/ILVR/checkpoints/<hardware-id>-*/`
- Update after execution: `docs/multinode-training.md`

**Interfaces:**
- Consumes: a clean committed implementation head, local venv, tiny harnesses, published `comt_ckpt`, Qwen2.5-VL-3B, and selected CoMT rows/images
- Produces: remote matching checkout/venv, 18 lifecycle witnesses, four-worker evaluation output, Stage 1 capacity evidence, and batch recommendations

- [ ] **Step 1: Verify no training/evaluation process or remote target conflicts**

```bash
ps -eo pid,etimes,args | rg '/root/workspace/ILVR|src/main.py|eval.py' || true
ssh -o BatchMode=yes -o ConnectTimeout=8 korea@172.16.11.41 \
  "docker exec llm_kd bash -lc 'ps -eo pid,etimes,args | grep -E \"/root/workspace/ILVR|src/main.py|eval.py\" | grep -v grep || true; test ! -e /root/workspace/ILVR'"
```

Expected: no active ILVR job and remote repository target absent. If the target exists by execution time, stop and validate ownership/commit instead of overwriting it.

- [ ] **Step 2: Create and transfer one committed Git bundle**

```bash
implementation_commit=$(git rev-parse HEAD)
git diff --exit-code
git diff --cached --exit-code
git bundle create /tmp/ilvr-multinode.bundle HEAD
scp /tmp/ilvr-multinode.bundle korea@172.16.11.41:/home/korea/Workspace/hjh/ilvr-multinode.bundle
ssh korea@172.16.11.41 \
  "docker exec llm_kd git clone /root/ilvr-multinode.bundle /root/workspace/ILVR"
ssh korea@172.16.11.41 \
  "docker exec llm_kd git -C /root/workspace/ILVR checkout --detach $implementation_commit"
```

Expected: remote `git rev-parse HEAD` equals `$implementation_commit`.

- [ ] **Step 3: Provision the validated venv and repository storage links once**

```bash
rsync -a --ignore-existing /root/workspace/ILVR/venv/ilvr-venv/ \
  korea@172.16.11.41:/home/korea/Workspace/hjh/workspace/ILVR/venv/ilvr-venv/
ssh korea@172.16.11.41 "docker exec llm_kd bash -lc '
  mkdir -p /raid/workspace/ILVR/checkpoints /root/workspace/ILVR/results /root/workspace/ILVR/logs
  test ! -e /root/workspace/ILVR/checkpoints
  ln -s /raid/workspace/ILVR/checkpoints /root/workspace/ILVR/checkpoints
  /root/workspace/ILVR/venv/ilvr-venv/bin/python -c \"import torch, deepspeed, accelerate; assert torch.__version__ == \\\"2.7.1+cu128\\\"; assert torch.cuda.device_count() == 2\"
'"
```

Expected: one matching remote venv and writable artifact roots. This is provisioning, not launcher-time synchronization.

- [ ] **Step 4: Copy only required real-model evaluation and capacity inputs once**

```bash
rsync -a --ignore-existing /root/workspace/ILVR/checkpoints/comt_ckpt/ \
  korea@172.16.11.41:/home/korea/Workspace/hjh/workspace/ILVR/checkpoints/comt_ckpt/
rsync -a --ignore-existing /root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct/ \
  korea@172.16.11.41:/home/korea/Workspace/hjh/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct/
rsync -a --ignore-existing /root/workspace/ILVR/data/TEST.jsonl \
  /root/workspace/ILVR/data/TRAIN.jsonl \
  korea@172.16.11.41:/home/korea/Workspace/hjh/workspace/ILVR/data/
```

Materialize four evaluation rows, 32 capacity rows, and the exact referenced-image list:

```bash
./venv/ilvr-venv/bin/python - <<'PY'
import json
from pathlib import Path

data_root = Path("/root/workspace/ILVR/data")
jobs = (
    (data_root / "TEST.jsonl", data_root / "multinode-eval-subset.jsonl", 4),
    (data_root / "TRAIN.jsonl", data_root / "multinode-capacity-subset.jsonl", 32),
)
image_paths = set()
for source, destination, count in jobs:
    if destination.exists():
        raise FileExistsError(destination)
    rows = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if len(rows) == count:
                break
            row = json.loads(line)
            rows.append(row)
            image_paths.update(row.get("image_input", ()))
            for item in row.get("sequence_plan", ()):
                helper = item.get("helper_image")
                if helper:
                    image_paths.add(helper)
    if len(rows) != count:
        raise ValueError(f"{source} contains only {len(rows)} rows")
    with destination.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

image_list = Path("/tmp/ilvr-input-images.files")
with image_list.open("x", encoding="utf-8") as handle:
    for relative in sorted(image_paths):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or path.parts[0] != "images_comt":
            raise ValueError(f"unsafe image path: {relative}")
        if not (data_root / path).is_file():
            raise FileNotFoundError(data_root / path)
        handle.write(path.as_posix() + "\n")
PY

rsync -a --ignore-existing \
  /root/workspace/ILVR/data/multinode-eval-subset.jsonl \
  /root/workspace/ILVR/data/multinode-capacity-subset.jsonl \
  korea@172.16.11.41:/home/korea/Workspace/hjh/workspace/ILVR/data/
rsync -a --ignore-existing --files-from=/tmp/ilvr-input-images.files \
  /root/workspace/ILVR/data/ \
  korea@172.16.11.41:/home/korea/Workspace/hjh/workspace/ILVR/data/

./venv/ilvr-venv/bin/python - <<'PY'
import json
from pathlib import Path
root = Path("/root/workspace/ILVR/data")
row = json.loads((root / "multinode-eval-subset.jsonl").open().readline())
assert all((root / image).is_file() for image in row["image_input"])
PY
ssh korea@172.16.11.41 "docker exec llm_kd /root/workspace/ILVR/venv/ilvr-venv/bin/python -c '
import json
from pathlib import Path
root = Path(\"/root/workspace/ILVR/data\")
row = json.loads((root / \"multinode-eval-subset.jsonl\").open().readline())
assert all((root / image).is_file() for image in row[\"image_input\"])
'"
```

Expected: both nodes can parse the same first sample and resolve every referenced `image_input` file.

- [ ] **Step 5: Run preflight and the 18 tiny lifecycles detached**

```bash
hardware_id="multinode-four-gpu-$(date -u +%Y%m%d-%H%M%S)"
hardware_result_root="/root/workspace/ILVR/results/$hardware_id"
hardware_log_root="/root/workspace/ILVR/logs/$hardware_id"
mkdir "$hardware_result_root" "$hardware_log_root"
matrix_id="${hardware_id}-matrix"
tmux new-session -d -s "$matrix_id" -- bash -lc \
  "cd /root/workspace/ILVR && ./run_multinode_lifecycle_matrix.sh \
   --run-id '$matrix_id' --first-port 29900 \
   --result-root '$hardware_result_root' \
   >'$hardware_log_root/$matrix_id.stdout.log' \
   2>'$hardware_log_root/$matrix_id.stderr.log'"
while tmux has-session -t "$matrix_id" 2>/dev/null; do
  sleep 5
done
test -f "$hardware_result_root/$matrix_id/matrix-complete.json"
```

Expected: 18 valid witnesses, each reporting world size 4, two initial steps, resumed global step 3, finite losses, correct ZeRO/method identity, and ranks 0-3.

- [ ] **Step 6: Train and prepare one real Stage 2 evaluation artifact, then evaluate**

```bash
eval_model_run_id="${hardware_id}-stage2-eval-full-zero1"
STAGE1_CHECKPOINT=/root/workspace/ILVR/checkpoints/comt_ckpt \
DATA_PATH=/root/workspace/ILVR/data/multinode-capacity-subset.jsonl \
IMAGE_ROOT=/root/workspace/ILVR/data/images_comt \
TUNING_METHOD=full \
./run_multinode_stage2_training.sh \
  --run-id "$eval_model_run_id" \
  --zero-stage 1 \
  --main-process-port 29918 \
  --max_steps 1 \
  --save_steps 1

prepared_model_dir="/root/workspace/ILVR/checkpoints/evaluation-staging/$eval_model_run_id/prepared-eval-model"
./venv/ilvr-venv/bin/python -m src.multinode prepare-inference \
  --stage stage2 \
  --tuning-method full \
  --zero-stage 1 \
  --run-id "$eval_model_run_id" \
  --main-process-port 29918 \
  --checkpoint-step 1 \
  --destination "$prepared_model_dir"

eval_id="${hardware_id}-eval"
./run_multinode_evaluation.sh \
  --run-id "$eval_id" \
  --model-dir "$prepared_model_dir" \
  --test-data-path /root/workspace/ILVR/data/multinode-eval-subset.jsonl \
  --task-name zebra-cot \
  --output-dir "$hardware_result_root/$eval_id" \
  --max-new-tokens 64
```

Expected: the Stage 2 source witness proves checkpoint 1 with finite loss and
gradient; the same prepared inference-only artifact and identity manifest
validate at `$prepared_model_dir` on both node-local filesystems before any
evaluator starts; then shard indices 0-3 cover every source index exactly once
and produce one merged JSONL in source order plus one summary JSON.

- [ ] **Step 7: Probe Stage 1 capacity and confirm Stage 2 batch 1**

Run the nine Stage 1 capacity probes. Each probe owns four distinct ports and uses fresh child run IDs for candidates 2, 4, and 8; candidate 1 is permitted only by the OOM rule and uses the fourth port:

```bash
capacity_evidence_root="$hardware_result_root/capacity-evidence"
capacity_root="$capacity_evidence_root/capacity"
mkdir "$capacity_evidence_root" "$capacity_root"
methods=(full lora dora)
zero_stages=(1 2 3)
case_index=0
for method in "${methods[@]}"; do
  for zero_stage in "${zero_stages[@]}"; do
    run_id="${hardware_id}-stage1-capacity-${method}-zero${zero_stage}"
    port_base=$((30000 + case_index * 4))
    ./run_multinode_stage1_batch_probe.sh \
      --run-id "$run_id" \
      --result-root "$capacity_root" \
      --zero-stage "$zero_stage" \
      --tuning-method "$method" \
      --main-process-port "$port_base" \
      --main-process-port "$((port_base + 1))" \
      --main-process-port "$((port_base + 2))" \
      --main-process-port "$((port_base + 3))" \
      --model /root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct \
      --data-path /root/workspace/ILVR/data/multinode-capacity-subset.jsonl \
      --image-root /root/workspace/ILVR/data/images_comt \
      --save-strategy steps \
      --save-steps 1 \
      --max-steps 1
    case_index=$((case_index + 1))
  done
done
```

Expected: exactly nine `capacity.json` reports; every report records `attempts[0].batch_per_gpu == 2`, a non-null `recommended_batch_per_gpu` backed by a successful child witness, and at most one first CUDA capacity OOM.

Run the nine Stage 2 one-step confirmations at the fixed batch/GPU 1:

```bash
stage2_root="$capacity_evidence_root/stage2-batch1"
mkdir "$stage2_root"
case_index=0
for method in "${methods[@]}"; do
  for zero_stage in "${zero_stages[@]}"; do
    run_id="${hardware_id}-stage2-batch1-${method}-zero${zero_stage}"
    STAGE1_CHECKPOINT=/root/workspace/ILVR/checkpoints/comt_ckpt \
    DATA_PATH=/root/workspace/ILVR/data/multinode-capacity-subset.jsonl \
    IMAGE_ROOT=/root/workspace/ILVR/data/images_comt \
    TUNING_METHOD="$method" \
    ./run_multinode_stage2_training.sh \
      --run-id "$run_id" \
      --result-root "$stage2_root" \
      --zero-stage "$zero_stage" \
      --main-process-port "$((30100 + case_index))" \
      --max_steps 1 \
      --save_steps 1
    case_index=$((case_index + 1))
  done
done
```

Expected: nine `initial-train-witness.json` files with `batch_per_gpu == 1`, world size 4, ranks 0-3, finite loss, and one optimizer step. Do not test Stage 2 batch 2 because its launcher contract fixes batch/GPU 1.

- [ ] **Step 8: Validate once and publish the report**

Run the single validator and require exclusive report creation:

```bash
validation_parent="/root/workspace/ILVR/results/${hardware_id}-validation"
test ! -e "$validation_parent"
report_bundle="$validation_parent/validation-report"
./venv/ilvr-venv/bin/python -m src.multinode validate-results \
  --matrix-root "$hardware_result_root/$matrix_id" \
  --evaluation-root "$hardware_result_root/$eval_id" \
  --capacity-root "$capacity_evidence_root" \
  --output-root "$report_bundle"
if tmux list-sessions -F '#{session_name}' 2>/dev/null | \
  rg -q "^ilvr-${hardware_id}-"; then
  echo 'owned ILVR tmux session remains' >&2
  exit 1
fi
```

Expected: validator exits 0 after reading 18 matrix witnesses, four evaluation sidecars, nine Stage 1 capacity reports, and nine Stage 2 batch-1 witnesses. The absent `$report_bundle` is published once and contains both `validation-manifest.json` and `validation-summary.md`. Update `docs/multinode-training.md` by copying the measured 3×3 recommendation tables from `$report_bundle/validation-summary.md`, and report exact checkpoint, result, and log paths.

- [ ] **Step 9: Commit only the final documentation update**

```bash
git add docs/multinode-training.md
git commit -m "docs: report four-GPU multi-node validation"
```
