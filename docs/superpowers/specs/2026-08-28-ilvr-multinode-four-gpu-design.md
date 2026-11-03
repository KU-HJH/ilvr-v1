# ILVR 2-Node 4-GPU 학습·평가 설계

## 배경

ILVR의 Stage 1과 Stage 2는 단일 노드의 두 GPU에서 Full, LoRA, DoRA와
DeepSpeed ZeRO-1, ZeRO-2, ZeRO-3 조합을 실행하는 경로를 가진다. 이번 변경은
trainer, model, data, loss, checkpoint identity 계약을 다시 만들지 않고 기존
경로를 두 노드의 총 네 GPU로 확장한다.

두 실행 노드는 각각 NVIDIA H100 NVL 두 장과 `bond0.11` 인터페이스를 가진다.
원격 실행은 반드시 `korea@172.16.11.41`의 `llm_kd` Docker container에서
이루어진다. 두 노드의 `/root/workspace/ILVR`과 artifact 저장소는 같은 문자열
경로를 사용하지만 물리적으로 공유되는 filesystem은 아니다.

## 목표

1. Stage 1과 Stage 2의 Full, LoRA, DoRA가 하나의 4-rank job에서
   ZeRO-1, ZeRO-2, ZeRO-3을 명시적으로 선택할 수 있다.
2. 원격 GPU 두 장과 로컬 GPU 두 장을 GPU당 한 process로 사용한다.
3. 기존 Stage별 launcher와 training entrypoint를 유지하고 얇은 multi-node
   coordinator만 추가한다.
4. 각 노드에 분리 저장된 run-owned checkpoint shard를 안전하게 모으고 같은
   identity로 resume할 수 있다.
5. 평가 dataset을 네 독립 worker로 분할하고 source 순서를 보존한 하나의
   결과로 병합한다.
6. 실제 4-GPU tiny save/resume lifecycle 18개와 평가 shard/merge를 검증한다.

## 비목표

- 범용 cluster scheduler, retry framework, dashboard 또는 provenance database를
  만들지 않는다.
- launcher 실행 때 repository, venv, dataset 또는 pretrained model 전체를
  자동 동기화하지 않는다.
- 평가 sample을 cross-node DDP collective로 묶지 않는다.
- 서로 다른 ZeRO stage, world size, ILVR stage 또는 tuning method 사이의
  optimizer checkpoint 변환을 지원하지 않는다.
- GPU를 사용 중이라는 이유만으로 다른 process를 종료하지 않는다.
- 기존 trainer, model architecture, loss 또는 metric 정의를 multi-node 때문에
  변경하지 않는다.

## 고정 topology

| 항목 | 값 |
|---|---|
| 원격 SSH host | `korea@172.16.11.41` |
| 원격 container | `llm_kd` |
| 양쪽 repository root | `/root/workspace/ILVR` |
| 양쪽 Python environment | `/root/workspace/ILVR/venv/ilvr-venv` |
| rendezvous/master address | `172.16.11.41` |
| socket interface | `bond0.11` |
| 원격 machine rank | `0` |
| 로컬 machine rank | `1` |
| machines | `2` |
| GPUs per machine | `2` |
| cluster-wide processes/world size | `4` |
| process placement | GPU당 process 하나 |

`MASTER_ADDR`, `GLOO_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME`은 표의 값을 양쪽에
명시적으로 전달한다. `MASTER_PORT`는 각 실행에서 `--main-process-port`로
명시하며 두 노드에서 사용 가능해야 한다. IP 주소나 interface를 자동 추측하지
않는다.

## Repository와 환경 준비

Multi-node launcher와 분리된 일회성 준비 절차가 원격
`/root/workspace/ILVR` checkout과 `venv/ilvr-venv`를 만든다. 원격 checkout은
실행할 local tracked commit과 같아야 한다. venv는 repository `AGENTS.md`의
Python 3.11, PyTorch 2.7.1+cu128, DeepSpeed 0.18.0, Accelerate 1.10.1 계약을
만족해야 한다.

양쪽 `checkpoints/`는 각 노드의 `/raid/workspace/ILVR/checkpoints/`를 가리키는
repository symlink로 준비한다. `results/`와 `logs/`는 각 repository 아래의
쓰기 가능한 directory로 준비한다. dataset, pretrained model, Stage 1 handoff
checkpoint 등 실행 입력은 launcher가 시작되기 전에 같은 논리 경로에 존재해야
한다. preflight는 이를 검사하지만 누락된 대형 asset을 복사하지 않는다.

## 구성 요소

### `src/multinode.py`

고정 topology와 다음 동작을 소유한다.

- local/remote preflight와 dry-run plan 생성
- SSH, `docker exec llm_kd`, `tmux` command 구성
- run ownership 기록 및 rank 0 준비 확인
- 두 node job 상태 수집과 run-owned 종료
- checkpoint manifest 검증과 제한된 artifact 전달
- 평가 worker assignment, shard 검증 및 최종 병합

이 module은 trainer 또는 DeepSpeed 내부 구현을 감싸지 않는다. 외부 command는
array로 구성하며 shell-evaluated user input을 만들지 않는다.

### Stage별 multi-node launcher

다음 root launcher를 추가한다.

```text
run_multinode_stage1_training.sh
run_multinode_stage2_training.sh
run_multinode_evaluation.sh
```

Stage 1 launcher는 기존 `run_stage1_training.sh` 인자를 유지하면서
`--zero-stage 1|2|3`, `--run-id`, `--main-process-port`, `--dry-run`을 받는다.
Stage 2 launcher는 기존 `STAGE1_CHECKPOINT`, `DATA_PATH`, `IMAGE_ROOT`,
`OUTPUT_DIR`, `TUNING_METHOD` 환경변수 계약을 유지하고 같은 multi-node 옵션을
받는다. `--zero-stage`는 다음 파일에 일대일 대응한다.

```text
1 -> configs/deepspeed_zero1.json
2 -> configs/deepspeed_zero2.json
3 -> configs/deepspeed_zero3.json
```

사용자가 coordinator가 소유하는 rank, world-size, master address, interface,
DeepSpeed config 또는 artifact identity를 stage launcher passthrough로 다시
지정하면 시작 전에 거부한다.

기존 Stage launcher에는 공통 distributed 환경변수 입력을 추가한다.

```text
ILVR_NUM_MACHINES=2
ILVR_NUM_PROCESSES=4
ILVR_MACHINE_RANK=0|1
ILVR_MAIN_PROCESS_IP=172.16.11.41
ILVR_MAIN_PROCESS_PORT=<explicit port>
ILVR_SOCKET_IFNAME=bond0.11
```

각 launcher는 이를 Accelerate의 `--num_machines`, cluster-wide
`--num_processes`, `--machine_rank`, `--main_process_ip`,
`--main_process_port`로 전달한다. distributed 환경변수가 없으면 기존 단일
노드 두 GPU 동작을 유지한다.

## 학습 실행 흐름

1. local coordinator가 입력, topology, commit, port, artifact 경로를 양쪽에서
   검사하고 하나의 immutable launch plan을 만든다.
2. plan의 작은 JSON 사본을 원격 run log directory에 보낸다.
3. coordinator가 원격 `llm_kd`의 전용 `tmux` session에서 machine rank 0을
   시작한다.
4. 원격 process의 PID, start time, command digest, run ID, repository commit이
   포함된 mode `0600` ownership record가 생성될 때까지 bounded condition
   polling을 수행한다.
5. ownership과 rendezvous 준비가 확인되면 local 전용 `tmux` session에서
   machine rank 1을 시작한다.
6. 네 rank가 기존 Stage trainer를 실행하고 각 노드의 동일 논리 artifact
   경로에 자기 DeepSpeed shard를 기록한다.
7. 두 node job이 모두 성공한 뒤 checkpoint를 canonical owner로 수집하고
   completion manifest를 한 번 검증한다.

원격 rank 0 시작 실패 시 local rank 1은 시작하지 않는다. 한 node가 학습 중
실패하면 coordinator는 상대 node의 ownership record가 이번 run과 정확히
일치할 때만 해당 PID에 종료 신호를 보낸다.

## ZeRO와 checkpoint identity

Stage 1/2, Full/LoRA/DoRA, ZeRO-1/2/3의 모든 조합은 `world_size=4`로 실행한다.
DeepSpeed config validation은 CUDA 초기화 전에 선택한 stage, bf16, batch,
offload와 world-size 계약을 검사한다.

번호가 붙은 checkpoint와 final artifact는 다음 identity를 기록한다.

- ILVR stage와 tuning method
- ZeRO stage와 DeepSpeed config digest
- `world_size=4`, `num_machines=2`, node rank mapping
- optimizer, scheduler, global step와 RNG state
- Stage 1 EMA 또는 Stage 2 sampler/PEFT sidecar
- repository commit과 run ID

identity가 다른 checkpoint resume은 process group 생성 전에 거부한다.
Stage 1 capacity probe는 micro-batch/GPU `2`, `4`, `8` 순으로 시작하고 `2`가
CUDA allocator의 capacity OOM으로 실패한 경우에만 `1`을 검사한다. Stage 2는
현재 계약대로 micro-batch/GPU `1`을 유지한다.

## 비공유 filesystem artifact 전달

각 node는 같은 `/root/workspace/ILVR/checkpoints/<run-id>` 문자열 경로를 쓰지만
filesystem은 공유하지 않는다. 원격 rank 0 node를 성공한 run의 canonical
artifact owner로 지정한다.

1. 각 node는 자기 파일의 상대 경로, byte size, rank ownership을 작은 manifest로
   기록한다. 전체 weight 또는 checkpoint hash는 계산하지 않는다.
2. 양쪽 completion marker와 manifest가 유효한 경우에만 local rank shard를
   원격 canonical directory로 전송한다.
3. 전송은 이번 run directory에만 제한하고 기존 destination file을 덮어쓰지
   않는다. 동일 상대 경로의 작은 공통 metadata는 size와 identity가 다르면
   실패한다.
4. 수집 후 manifest의 rank 0~3 shard가 정확히 한 번 존재하는지 검증한다.
5. resume 전에는 원격 canonical copy의 필요한 local-rank shard와 작은 공통
   metadata만 local run directory에 배치한다.
6. 최종 inference artifact는 원격에서 완성·검증한 뒤 local node의 fresh
   evaluation directory로 한 번 복제한다.

source tree, venv, dataset 또는 pretrained model은 이 전달 경로에 포함하지
않는다. artifact transfer 실패는 학습 성공으로 숨기지 않으며 원본 node-local
shard를 삭제하지 않는다.

## 4-worker 평가

평가는 DDP collective가 아니라 네 독립 process로 실행한다. `eval.py`와
`src/evaluate_deepseed.py`의 기존 batch-size 1 inference를 유지하고 다음
입력만 추가한다.

```text
--num-shards 4
--shard-index 0|1|2|3
```

각 worker는 원본 JSONL의 zero-based source index가
`source_index % 4 == shard_index`인 sample만 처리한다. 원격 GPU 0/1은 shard
0/1, local GPU 0/1은 shard 2/3을 담당한다. worker output에는 source index,
sample identity, prediction과 기존 metric 입력을 기록한다.

모든 worker가 exit 0일 때 원격 JSONL 두 개만 local result directory로
전송한다. merger는 다음을 한 번 검사한다.

- 모든 expected source index가 정확히 한 번 존재함
- 중복, 누락, 범위 밖 index가 없음
- shard identity, model identity, dataset stat이 동일함
- 각 line이 유효한 JSON object이고 기존 평가 필드를 가짐

검증 후 source index 순서로 하나의 JSONL을 만들고 기존 metric 계산을 한 번
실행한다. worker 실패 또는 불완전 shard가 있으면 partial result를 최종 결과로
게시하지 않는다.

## Preflight와 안전 규칙

실제 process를 시작하기 전에 다음을 모두 검사한다.

- bounded noninteractive SSH와 실행 중인 `llm_kd`
- 양쪽 repository, venv Python, Accelerate, DeepSpeed, torchrun
- 동일 tracked Git commit과 clean tracked worktree
- 양쪽 H100 GPU 정확히 두 장과 `bond0.11`
- master address와 명시한 port의 사용 가능 여부
- 선택한 Stage, method, ZeRO config와 `world_size=4`
- model, dataset, image root, resume/handoff checkpoint 입력
- fresh run ID, tmux session, checkpoint/result/log directory
- 양쪽 filesystem의 충분한 쓰기 권한

preflight는 package 설치, remote source 수정, 대형 asset 복사, 전체 repository
hash, 전체 dataset scan 또는 checkpoint weight hash를 수행하지 않는다.

ownership record 없이 PID를 종료하지 않는다. `pkill`, process 이름, GPU 사용,
경로 유사성 또는 확인되지 않은 remote PID를 cleanup 근거로 사용하지 않는다.
run ID, PID, start time, command digest와 repository commit이 모두 일치할 때만
run-owned signal을 보낸다.

## 오류 처리

- 잘못된 topology, input, config, artifact identity는 GPU 초기화 전에 실패한다.
- 원격 rank 0 준비가 bounded timeout 안에 확인되지 않으면 local rank 1을
  시작하지 않는다.
- NCCL, collective mismatch, nonfinite loss/gradient, checkpoint 누락,
  artifact transfer 실패는 integration failure이다.
- batch 감소는 CUDA allocator가 직접 보고한 capacity OOM에만 허용한다.
- 자동 retry하지 않는다. failure log를 보존하고 root cause에 대한 최소 수정과
  causal regression test 하나를 추가한 뒤 fresh run ID로 다시 실행한다.
- 실패한 run의 node-local checkpoint shard와 log는 자동 삭제하지 않는다.

## 검증 전략

1. fake SSH, Docker, tmux, process와 artifact transfer command를 사용한 단위
   테스트로 topology와 ownership 경계를 검증한다. 실제 SSH, GPU, remote 수정은
   단위 테스트에서 수행하지 않는다.
2. `--help`, shell syntax와 `--dry-run`에서 두 machine, 네 process, rank 0/1,
   master address, port, interface, 선택한 ZeRO config를 확인한다.
3. path traversal, duplicate run ID, dirty tracked commit, GPU 비대칭, busy port,
   foreign PID와 incomplete shard를 거부하는 focused test를 실행한다.
4. 실제 H100 네 장에서 tiny model로 다음 18조합을 실행한다.

   ```text
   Stage 1/2 x Full/LoRA/DoRA x ZeRO-1/2/3
   ```

   각 조합은 2 optimizer step, numbered checkpoint 저장, same-identity 1 step
   resume을 수행한다. finite loss/gradient, optimizer step, global step, rank
   completion, checkpoint identity와 method별 artifact를 확인한다.
5. 작은 TEST subset을 네 worker로 평가하고 shard 0~3의 완전성, 순서 보존 병합,
   metric output을 확인한다.
6. 대표적인 실제 모델 설정의 Stage 1 capacity probe를 micro-batch/GPU 2부터
   실행하고 Stage 2는 micro-batch/GPU 1의 짧은 실행을 확인한다.

한 producer가 필요한 witness를 만들면 같은 조합의 broad test나 장시간 smoke를
반복하지 않는다.

## 완료 기준

1. 두 node가 고정 topology와 같은 commit/environment로 preflight를 통과한다.
2. Stage 1/2와 Full/LoRA/DoRA가 ZeRO-1/2/3을 4-rank job에서 선택할 수 있다.
3. 18개 tiny save/resume lifecycle이 hang 없이 완료된다.
4. checkpoint collection 후 rank 0~3 shard와 identity가 완전하며 같은 identity
   resume이 성공한다.
5. cross-ZeRO, cross-world-size, cross-stage, cross-method resume이 명확히
   거부된다.
6. 4-worker 평가가 모든 sample을 정확히 한 번 처리하고 source 순서로 병합한다.
7. Stage 1 batch probe의 첫 실제 후보는 micro-batch/GPU 2이고 Stage 2는 1이다.
8. checkpoint, result, log는 각 node의 `/root/workspace/ILVR` 아래 run별 고유
   directory를 사용한다.
9. 실행 중 source tree, venv, dataset 또는 pretrained model을 자동 동기화하지
   않는다.
10. 외부 process를 종료하거나 기존 artifact를 덮어쓰지 않는다.
