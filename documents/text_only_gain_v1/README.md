# Text-only Gain v1 — 새 workspace용 참고 자료

상위 명세서 `text_only_gain_v1.md`와 이 폴더를 함께 옮긴다. 모든 문서 링크는 상대 경로다.

## 읽는 순서와 역할

| 경로 | 용도 |
| --- | --- |
| [학습 설정](config/training.json) | 새 launcher에 명시적으로 전달할 기준값; 실행 manifest가 아님 |
| [마스킹 설정](config/prefix-conditioning.json) | KL 예산 기반 text-only mask 설정 |
| [평가 프로토콜](config/evaluation.json) | 자체 final checkpoint의 두 GPU 평가 기준 |
| [기준 성능](reference/evaluation_summary.json), [비교표](reference/comparison.json) | 기존 측정값; 새 환경 검증 결과가 아님 |
| `reference/checkpoint_metadata/` | checkpoint 필드·손실 계약을 설명하는 자료; weight와 optimizer state는 없음 |
| `source/` | 핵심 연산, 모델 연결, launcher와 테스트의 참고 구현 |
| `data/` | TRAIN/TEST JSONL; 이미지 파일은 별도 준비 |
| [파일 목록](manifest.json) | 이 묶음 내부의 경로·크기·내용 digest |

## 새 프로젝트에 통합하기

새 프로젝트 루트를 `W`라고 부른다. `source/`의 내용을 W의 대응 위치에 통합하고 `config/`와 `data/`도 W 아래에 둔다. `source/configs/deepspeed_zero1.json`은 `W/configs/`에 대응한다. 자료 폴더를 실행 루트로 사용하는 것이 아니다.

- Base model/tokenizer/processor: `W/models/Qwen2.5-VL-3B-Instruct/` 또는 별도로 지정한 경로.
- 원본·helper image: `W/data/images_comt/`. JSONL의 `image_input` 및 `sequence_plan` 내 helper-image 경로는 `image_root=W/data`를 기준으로 해석한다.
- 출력: `W/checkpoints/<run-id>/`, `W/results/<run-id>/`, `W/logs/<run-id>/`.
- 첨부 launcher를 쓴다면 가상환경은 `W/venv/ilvr-venv/`에 둔다. 다른 launcher에서는 사용할 interpreter를 명시한다.

Python artifact 경로는 설치된 `src/artifact_paths.py`의 위치를 기준으로 W를 계산하고, shell launcher의 log root도 script 위치를 기준으로 계산한다. 특정 서버 절대 경로를 설정하지 않는다. 모델 계산 코드는 유지하고 이 두 파일의 경로 규칙만 workspace 기준으로 조정했다.

## 실행 코드의 포함 범위

이 묶음은 완전한 설치·실행 checkout이 아니다. 최소한 다음 연결 부분은 호환되는 프로젝트 코드와 통합해야 한다.

- Entrypoint와 인자 처리: `src/main.py`, 필요 시 benchmark launcher 처리.
- Dataset/template 및 분산 runtime: `task_deepseed`, `deepspeed_runtime`.
- Stage 1 보조 모듈: `stage1.peft`, `stage1.token_rows`, `stage1.iier`와 그 의존성.
- 모델 수정의 보조 모듈: latent-rollout, vision SDPA, prefix-cache, sidecar 관련 코드.
- 평가 경로에서 import하는 Stage 2 checkpoint/inference/PEFT 모듈.
- 호환되는 Transformers/PEFT/DeepSpeed/PyTorch, tokenizer/processor, pretrained weight, 실제 이미지.

따라서 stock Transformers에 이 파일 일부만 덮어쓴 뒤 바로 학습하는 안내로 해석하지 않는다. 첨부 테스트 역시 통합된 import 환경을 전제로 한다. 참고 소스가 기존 성능을 낸 전체 실행 소스와 동일하다는 인증은 제공하지 않는다.

## Launcher에 전달할 설정

첨부 `run_stage1_training.sh`의 일반 기본값은 명세와 다르므로 아래 값을 명시한다. `config/training.json`은 이 값들의 기준이며 shell이 자동으로 읽지 않는다.

| 설정 | 값 |
| --- | --- |
| `--method`, `--epochs`, `--learning-rate` | `dora`, `3`, `5e-5` |
| `--micro-batch`, `--gradient-accumulation`, `--seed` | `1`, `4`, `42` |
| `--deepspeed-config`, `--attention-implementation` | `W/configs/deepspeed_zero1.json`, `sdpa` |
| `--vision-gain-alpha`, `--vision-gain-execution`, `--vision-gain-branch-mode` | `0.001`, `packed`, `independent` |
| `--prefix-conditioning-config` | `W/config/prefix-conditioning.json` |
| `--text-kl-beta`, `--text-kl-warmup-ratio`, `--clean-kl-beta` | `0.01`, `0.05`, `0` |
| `--ema-setting`, `--preference-mode` | `moving`, `none` |
| `--max-seq-length`, `--task`, `--save-strategy`, `--final-save` | `32768`, `zebra-cot`, `epoch`, 사용 |
| `--model`, `--data-path`, `--image-root`, `--cache-dir` | 새 W의 모델, `W/data/TRAIN.jsonl`, `W/data`, `W/cache` |
| `--output-dir`, `--log-file` | `W/checkpoints/<run-id>`, `W/logs/<run-id>/train.log` |
| `STAGE1_NUM_PROCESSES`, `CUDA_VISIBLE_DEVICES` | `2`, 사용할 두 GPU ID |

EMA decay·CE/alignment weight·LR warmup 등 shell에 직접 노출되지 않은 값도 `config/training.json`과 일치하는지 entrypoint에서 확인한다. 학습 시작 전에 자체 final checkpoint 저장에서 평가까지 이어지는 handoff를 준비한다. 평가는 400개를 실제로 두 GPU에 나누며, 기존 latent-generation 외 추가 teacher/reference pass를 넣지 않는다.

## 평가 worker의 실행 위치와 seed

첨부 `eval.py`는 `--image-root` 인자가 없고 상대 이미지 경로를 **현재 작업 디렉터리**에서 찾는다. 각 worker를 `cwd=W/data`로 실행하거나 입력 이미지 경로를 새 서버의 절대 경로로 변환한다. 학습 launcher에 전달한 `--image-root`는 이 평가기에 전달되지 않는다. `PYTHONPATH`에는 통합된 `W`, `W/src`, `W/transformers/src`를 포함한다.

또한 평가 entrypoint는 RNG를 초기화하지 않는다. 각 GPU worker의 호출자가 `eval.main()` 전에 `random.seed(42)`, `numpy.random.seed(42)`, `torch.manual_seed(42)`를 실행한다. Seed 숫자를 설정 파일에 적는 것만으로 적용되지 않는다. 원본 index의 짝수/홀수에 따라 200개씩 나누고, 병합 때 원본 index를 복원한다. `--artifact_stage stage1`을 명시한다. 이 실행 조건은 [evaluation.json](config/evaluation.json)에 함께 기록했다.

## 검증 범위

이식 자료의 링크, 설정, 경로 독립성, Python/shell 문법을 확인한다. 이 검사는 새 서버의 GPU 호환성이나 학습·평가 성능을 확인한 것이 아니다. 기존 수치와 학습 메커니즘을 보존하면서 실행 위치에 의존하는 설명과 과거 운영 기록을 제거했다.

## 전체 파일

- [config/evaluation.json](config/evaluation.json)
- [config/prefix-conditioning.json](config/prefix-conditioning.json)
- [config/training.json](config/training.json)
- [data/TEST.jsonl](data/TEST.jsonl)
- [data/TRAIN.jsonl](data/TRAIN.jsonl)
- [reference/checkpoint_metadata/adapter_config.json](reference/checkpoint_metadata/adapter_config.json)
- [reference/checkpoint_metadata/deepspeed-runtime.json](reference/checkpoint_metadata/deepspeed-runtime.json)
- [reference/checkpoint_metadata/stage1_clean_kl.json](reference/checkpoint_metadata/stage1_clean_kl.json)
- [reference/checkpoint_metadata/stage1_peft_metadata.json](reference/checkpoint_metadata/stage1_peft_metadata.json)
- [reference/checkpoint_metadata/stage1_preference.json](reference/checkpoint_metadata/stage1_preference.json)
- [reference/checkpoint_metadata/stage1_text_kl.json](reference/checkpoint_metadata/stage1_text_kl.json)
- [reference/checkpoint_metadata/stage1_vision_gain.json](reference/checkpoint_metadata/stage1_vision_gain.json)
- [reference/comparison.json](reference/comparison.json)
- [reference/evaluation_summary.json](reference/evaluation_summary.json)
- [source/configs/deepspeed_zero1.json](source/configs/deepspeed_zero1.json)
- [source/docs/stage1-text-kl.md](source/docs/stage1-text-kl.md)
- [source/eval.py](source/eval.py)
- [source/run_stage1_training.sh](source/run_stage1_training.sh)
- [source/src/artifact_paths.py](source/src/artifact_paths.py)
- [source/src/peft_common.py](source/src/peft_common.py)
- [source/src/stage1/checkpointing.py](source/src/stage1/checkpointing.py)
- [source/src/stage1/clean_kl.py](source/src/stage1/clean_kl.py)
- [source/src/stage1/composition.py](source/src/stage1/composition.py)
- [source/src/stage1/data.py](source/src/stage1/data.py)
- [source/src/stage1/ema.py](source/src/stage1/ema.py)
- [source/src/stage1/helper_attention_proxy.py](source/src/stage1/helper_attention_proxy.py)
- [source/src/stage1/prefix_mask_accounting.py](source/src/stage1/prefix_mask_accounting.py)
- [source/src/stage1/prefix_mask_config.py](source/src/stage1/prefix_mask_config.py)
- [source/src/stage1/prefix_mask_metadata.py](source/src/stage1/prefix_mask_metadata.py)
- [source/src/stage1/prefix_mask_plan.py](source/src/stage1/prefix_mask_plan.py)
- [source/src/stage1/prefix_mask_reference.py](source/src/stage1/prefix_mask_reference.py)
- [source/src/stage1/targets.py](source/src/stage1/targets.py)
- [source/src/stage1/text_kl.py](source/src/stage1/text_kl.py)
- [source/src/stage1/trainer.py](source/src/stage1/trainer.py)
- [source/src/stage1/types.py](source/src/stage1/types.py)
- [source/src/stage1/vision_gain.py](source/src/stage1/vision_gain.py)
- [source/src/stage1/vision_gain_execution.py](source/src/stage1/vision_gain_execution.py)
- [source/src/stage1/vision_gain_packed_execution.py](source/src/stage1/vision_gain_packed_execution.py)
- [source/src/stage1/vision_gain_update.py](source/src/stage1/vision_gain_update.py)
- [source/src/utils_deepseed.py](source/src/utils_deepseed.py)
- [source/tests/stage1/prefix_unit/conftest.py](source/tests/stage1/prefix_unit/conftest.py)
- [source/tests/stage1/prefix_unit/test_selection.py](source/tests/stage1/prefix_unit/test_selection.py)
- [source/tests/stage1/prefix_unit/test_text_kl.py](source/tests/stage1/prefix_unit/test_text_kl.py)
- [source/tests/stage1/prefix_unit/test_text_kl_cuda_dropout.py](source/tests/stage1/prefix_unit/test_text_kl_cuda_dropout.py)
- [source/tests/stage1/test_vision_gain_projection.py](source/tests/stage1/test_vision_gain_projection.py)
- [source/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py](source/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)
- [source/transformers/src/transformers/models/qwen2_5_vl/modular_qwen2_5_vl.py](source/transformers/src/transformers/models/qwen2_5_vl/modular_qwen2_5_vl.py)
- [source/transformers/src/transformers/models/qwen2_5_vl/stage1_prefix_conditioning.py](source/transformers/src/transformers/models/qwen2_5_vl/stage1_prefix_conditioning.py)
- [source/transformers/src/transformers/models/qwen2_5_vl/stage1_prefix_trace.py](source/transformers/src/transformers/models/qwen2_5_vl/stage1_prefix_trace.py)
- [source/transformers/src/transformers/models/qwen2_5_vl/stage1_vision_gain.py](source/transformers/src/transformers/models/qwen2_5_vl/stage1_vision_gain.py)
- [source/transformers/src/transformers/models/qwen2_5_vl/stage1_vision_gain_packed.py](source/transformers/src/transformers/models/qwen2_5_vl/stage1_vision_gain_packed.py)
