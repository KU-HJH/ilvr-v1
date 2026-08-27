# Stage 1 학습 산출물 기반 평가 가이드

이 문서는 이 worktree에서 완료한 Qwen2.5-VL-3B Stage 1 Full/DoRA 3-epoch
학습 산출물을 이용해 후속 모델 품질을 평가하는 절차를 설명한다.

## 1. 먼저 알아야 할 점

현재 저장된 두 산출물은 **Stage 1 모델**이다.

```text
outputs/stage1-timed/full-3epoch
outputs/stage1-timed/dora-3epoch
```

현재 `eval.py`는 Stage 2 artifact만 받는다. 따라서 위 Stage 1 경로를
`eval.py --model_dir`에 직접 전달하면 안 된다. Stage 1은 helper image에서
teacher target을 만드는 학습 단계이고, 독립적인 Stage 1 생성 controller는
현재 구현되어 있지 않다.

평가 목적은 다음처럼 구분한다.

1. Stage 1 학습 실행·산출물 검증은 기존 sealed evidence를 확인한다.
2. downstream task accuracy는 Stage 1 산출물에서 Stage 2를 학습한 뒤
   Stage 2 checkpoint를 `eval.py`로 평가한다.

Stage 1의 train loss만으로 CoMT task accuracy나 논문 재현 성능을 주장하면
안 된다.

## 2. 현재 Stage 1 산출물

| 항목 | Full | DoRA |
| --- | --- | --- |
| 경로 | `outputs/stage1-timed/full-3epoch` | `outputs/stage1-timed/dora-3epoch` |
| 학습 | 3 epochs, 648 optimizer steps | 3 epochs, 648 optimizer steps |
| 샘플 처리 수 | 10,350 | 10,350 |
| effective global batch | 16 | 16 |
| learning rate | `1e-5` | `1e-5` |
| 형식 | Full model shards | DoRA adapter + sparse latent rows |

Full artifact의 핵심 파일은 다음과 같다.

```text
config.json
model.safetensors.index.json
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
tokenizer/processor files
```

DoRA artifact의 핵심 파일은 다음과 같다.

```text
adapter_config.json
adapter_model.safetensors
ilvr_token_rows.safetensors
stage1_peft_metadata.json
tokenizer/processor files
```

DoRA artifact는 base model을 포함하지 않으며 merge된 모델도 아니다.
원본 base는 `Qwen/Qwen2.5-VL-3B-Instruct`이다.

기존 학습 증거는 다음 파일에 있다.

```text
outputs/stage1-timed/full-3epoch/evidence.json
outputs/stage1-timed/dora-3epoch/evidence.json
outputs/stage1-timed/comparison.json
outputs/stage1-timed/final-report.json
```

이미 검증된 모델 tree를 다시 hash하거나 `validate-timed-run`을 반복하지
말고 위 evidence를 재사용한다.

## 3. 공통 환경 설정

모든 명령은 이 worktree의 virtual environment를 사용한다.

```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
VENV="$REPO_ROOT/venv/ilvr-venv"

FULL_STAGE1="$REPO_ROOT/outputs/stage1-timed/full-3epoch"
DORA_STAGE1="$REPO_ROOT/outputs/stage1-timed/dora-3epoch"
BASE_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"

# TRAIN.jsonl, TEST.jsonl, images_comt/가 들어 있는 디렉터리
DATA_ROOT=/path/to/ILVR/data
CACHE_DIR="$REPO_ROOT/cache"

export PYTHONPATH="$REPO_ROOT/transformers/src:$REPO_ROOT/src"
export TOKENIZERS_PARALLELISM=false
```

현재 worktree에는 데이터 디렉터리가 없을 수 있다. 이 경우 실제 데이터가
있는 main checkout 또는 별도 데이터 경로를 `DATA_ROOT`로 지정한다.

가벼운 파일 존재 확인은 다음 정도로 제한한다.

```bash
test -f "$FULL_STAGE1/config.json"
test -f "$FULL_STAGE1/model.safetensors.index.json"

test -f "$DORA_STAGE1/adapter_config.json"
test -f "$DORA_STAGE1/adapter_model.safetensors"
test -f "$DORA_STAGE1/ilvr_token_rows.safetensors"
test -f "$DORA_STAGE1/stage1_peft_metadata.json"

test -f "$DATA_ROOT/TRAIN.jsonl"
test -f "$DATA_ROOT/TEST.jsonl"
test -d "$DATA_ROOT/images_comt"
```

## 4. Stage 2 학습 전 smoke

본 학습 전에 Full과 DoRA를 각각 별도 출력 경로에서 `max_steps=1`로
확인한다. smoke 출력 경로를 본 학습 경로로 재사용하지 않는다.

### 4.1 Full Stage 1 → Full Stage 2 smoke

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$PYTHONPATH" \
"$VENV/bin/python" "$REPO_ROOT/src/main.py" \
  --stage stage2 \
  --load_model_path "$FULL_STAGE1" \
  --tuning_method full \
  --data_path "$DATA_ROOT/TRAIN.jsonl" \
  --image_root "$DATA_ROOT" \
  --save_model_path "$REPO_ROOT/outputs/stage2-smoke/from-full" \
  --log_file "$REPO_ROOT/outputs/stage2-smoke/from-full.log" \
  --latent_size 8 \
  --supervise_latent_boundaries \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --attention_implementation flash_attention_2 \
  --max_seq_length_train 8192 \
  --max_steps 1
```

### 4.2 DoRA Stage 1 → DoRA Stage 2 smoke

Stage 1 DoRA는 base와 adapter 경로를 분리해서 전달해야 한다.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$PYTHONPATH" \
"$VENV/bin/python" "$REPO_ROOT/src/main.py" \
  --stage stage2 \
  --load_model_path "$BASE_MODEL" \
  --stage1_adapter_path "$DORA_STAGE1" \
  --tuning_method dora \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --data_path "$DATA_ROOT/TRAIN.jsonl" \
  --image_root "$DATA_ROOT" \
  --save_model_path "$REPO_ROOT/outputs/stage2-smoke/from-dora" \
  --log_file "$REPO_ROOT/outputs/stage2-smoke/from-dora.log" \
  --latent_size 8 \
  --supervise_latent_boundaries \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --attention_implementation flash_attention_2 \
  --max_seq_length_train 8192 \
  --max_steps 1
```

다음 사항을 확인한다.

- Stage 2가 `batch_size=1`로 시작되는가
- original input image만 visual input으로 들어가는가
- helper image tensor 관련 오류가 없는가
- latent block 개수와 latent norm diagnostics가 유한한가
- Full 경로는 full artifact로, DoRA 경로는 base + adapter로 로드되는가
- Stage 1 optimizer/scheduler/global step이 Stage 2로 전달되지 않는가

## 5. Stage 2 본 학습

실제 task accuracy 비교를 위해서는 Full과 DoRA에 같은 데이터 split,
`latent_size`, boundary supervision, seed, generation 설정을 사용한다.

아래 예시는 두 GPU와 ZeRO-3를 사용한다. Stage 2는 rank-local batch가
항상 `1`이며 effective global batch는
`world_size × 1 × gradient_accumulation_steps`이다.

### 5.1 Full Stage 1 → Full Stage 2

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH="$PYTHONPATH" \
"$VENV/bin/torchrun" --nproc_per_node=2 \
  "$REPO_ROOT/src/main.py" \
  --stage stage2 \
  --load_model_path "$FULL_STAGE1" \
  --tuning_method full \
  --data_path "$DATA_ROOT/TRAIN.jsonl" \
  --image_root "$DATA_ROOT" \
  --save_model_path "$REPO_ROOT/outputs/stage2-eval/full" \
  --log_file "$REPO_ROOT/outputs/stage2-eval/full.log" \
  --latent_size 8 \
  --supervise_latent_boundaries \
  --epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --attention_implementation flash_attention_2 \
  --gradient_checkpointing \
  --max_seq_length_train 8192 \
  --deepspeed_config "$REPO_ROOT/configs/config_stage3.json"
```

### 5.2 DoRA Stage 1 → DoRA Stage 2

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH="$PYTHONPATH" \
"$VENV/bin/torchrun" --nproc_per_node=2 \
  "$REPO_ROOT/src/main.py" \
  --stage stage2 \
  --load_model_path "$BASE_MODEL" \
  --stage1_adapter_path "$DORA_STAGE1" \
  --tuning_method dora \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --data_path "$DATA_ROOT/TRAIN.jsonl" \
  --image_root "$DATA_ROOT" \
  --save_model_path "$REPO_ROOT/outputs/stage2-eval/dora" \
  --log_file "$REPO_ROOT/outputs/stage2-eval/dora.log" \
  --latent_size 8 \
  --supervise_latent_boundaries \
  --epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --attention_implementation flash_attention_2 \
  --gradient_checkpointing \
  --max_seq_length_train 8192 \
  --deepspeed_config "$REPO_ROOT/configs/config_stage3.json"
```

DoRA 경로는 adapter를 merge하지 않는다. Stage 1에서 학습한 DoRA와
sparse latent-token rows를 이어받되 Stage 2 optimizer, scheduler,
global step은 새로 시작한다.

위 3-epoch Stage 2 설정은 평가용 실험 제안이며, 현재 저장된 sealed
Stage 1 3-epoch 결과와는 별개의 새 GPU 실험이다. 실행 전에 예상 비용과
출력 경로를 확인한다.

## 6. 평가 데이터 준비

`eval.py`의 일반 CoMT exact-match 경로는 샘플마다 다음 필드를 사용한다.

```json
{
  "text_input": "질문 또는 지시문",
  "image_input": "images_comt/.../input.png",
  "original_final_answer": "정답"
}
```

`image_input`은 문자열 또는 문자열 배열을 허용한다. helper image는 평가
입력으로 사용하지 않는다.

현재 evaluator의 상대 이미지 경로 기준점은 실행 당시 working directory다.
따라서 아래 평가 명령은 `DATA_ROOT`에서 실행한다. 또는 TEST JSONL의
`image_input`을 절대 경로로 변환한다.

먼저 10개 샘플로 smoke split을 만든다.

```bash
mkdir -p "$REPO_ROOT/outputs/evaluation"
head -n 10 "$DATA_ROOT/TEST.jsonl" \
  > "$REPO_ROOT/outputs/evaluation/TEST-10.jsonl"
```

원본 TEST 파일은 수정하지 않는다.

## 7. Stage 2 checkpoint 평가

재현 가능한 비교의 기본 generation 설정은 greedy decoding이다.

```text
temperature=0.0
top_p=1.0
max_new_tokens=4096
batch_size=1
```

답변이 짧은 데이터라면 `max_new_tokens`를 줄일 수 있지만 Full/DoRA에
동일한 값을 사용한다.

### 7.1 Full Stage 2 평가

Full Stage 2 artifact는 `--base_model_dir` 없이 로드한다.

```bash
FULL_STAGE2="$REPO_ROOT/outputs/stage2-eval/full"
FULL_RESULT="$REPO_ROOT/outputs/evaluation/full-results.jsonl"

(
  cd "$DATA_ROOT"
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$REPO_ROOT/transformers/src:$REPO_ROOT/src" \
  "$VENV/bin/python" "$REPO_ROOT/eval.py" \
    --model_dir "$FULL_STAGE2" \
    --test_data_path "$DATA_ROOT/TEST.jsonl" \
    --task_name comt \
    --output_json_path "$FULL_RESULT" \
    --cache_dir "$CACHE_DIR" \
    --max_new_tokens 4096 \
    --temperature 0.0 \
    --top_p 1.0
)
```

### 7.2 Stage 1 DoRA를 이어받은 Stage 2 DoRA 평가

Stage 2 DoRA artifact는 원본 Qwen base와 함께 로드한다. Stage 1 DoRA
artifact를 `--base_model_dir`로 전달하지 않는다.

```bash
DORA_STAGE2="$REPO_ROOT/outputs/stage2-eval/dora"
DORA_RESULT="$REPO_ROOT/outputs/evaluation/dora-results.jsonl"

(
  cd "$DATA_ROOT"
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$REPO_ROOT/transformers/src:$REPO_ROOT/src" \
  "$VENV/bin/python" "$REPO_ROOT/eval.py" \
    --model_dir "$DORA_STAGE2" \
    --base_model_dir "$BASE_MODEL" \
    --test_data_path "$DATA_ROOT/TEST.jsonl" \
    --task_name comt \
    --output_json_path "$DORA_RESULT" \
    --cache_dir "$CACHE_DIR" \
    --max_new_tokens 4096 \
    --temperature 0.0 \
    --top_p 1.0
)
```

Stage 1 Full checkpoint에서 새 Stage 2 LoRA/DoRA를 학습했다면 해당
Stage 2 PEFT 평가의 `--base_model_dir`는 `$FULL_STAGE1`이어야 한다.

### 7.3 Smoke 평가

전체 TEST보다 먼저 위 명령의 `--test_data_path`를 다음 경로로 바꿔
10개 샘플을 평가한다.

```text
$REPO_ROOT/outputs/evaluation/TEST-10.jsonl
```

다음을 모두 확인한 후 전체 TEST를 실행한다.

- artifact와 processor 로드 성공
- latent token ID 불일치 없음
- 각 latent block에서 정확히 `K=8` 상태 생성
- original image 로드 성공
- 결과 JSONL의 마지막 줄에 summary 존재
- error row가 0개

## 8. 결과 확인

`eval.py`는 샘플별 결과를 JSONL로 기록하고 마지막 줄에 summary를
추가한다.

```bash
tail -n 1 "$FULL_RESULT"
tail -n 1 "$DORA_RESULT"
```

summary 예시는 다음과 같다.

```json
{
  "summary": true,
  "num_samples": 100,
  "average_inference_time_sec": 1.23,
  "total_inference_time_sec": 123.0,
  "accuracy": 0.45,
  "success": 45
}
```

정확도만 보지 말고 error row도 별도로 확인한다.

```bash
"$VENV/bin/python" - "$FULL_RESULT" "$DORA_RESULT" <<'PY'
import json
import sys
from pathlib import Path

for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    summary = rows[-1]
    errors = [row for row in rows[:-1] if "error" in row]
    print(
        f"{path}: accuracy={summary['accuracy']:.6f}, "
        f"success={summary['success']}/{summary['num_samples']}, "
        f"errors={len(errors)}, "
        f"avg_sec={summary['average_inference_time_sec']:.6f}"
    )
PY
```

현재 evaluator는 실패한 샘플도 전체 분모에 포함한다. 또한
`average_inference_time_sec`도 전체 샘플 수를 분모로 사용하므로 error가
있다면 속도 수치를 그대로 비교하지 않는다.

## 9. Full/DoRA 비교 원칙

공정한 비교를 위해 다음을 고정한다.

- 동일한 Stage 2 train/test split
- 동일한 `latent_size=8`
- 동일한 `supervise_latent_boundaries=true`
- 동일한 Stage 2 epoch, learning rate, seed
- 동일한 `max_new_tokens`, temperature, top-p
- 동일한 평가 GPU와 attention backend
- 동일한 answer normalization과 task name
- error row가 0개인 실행만 성능 비교에 사용

다음 수치는 구분해서 기록한다.

1. task accuracy
2. 성공 수 / 전체 샘플 수
3. error 수
4. 평균 generation 시간
5. Stage 2 trainable parameter 수
6. Stage 2 artifact 크기
7. Stage 2 학습 시간

`Full Stage 1 → Full Stage 2`와
`DoRA Stage 1 → DoRA Stage 2` 비교는 전체 전략 비교다. Stage 1 방식의
효과만 분리한 ablation은 아니다. Stage 2 tuning 방식까지 통제하려면
두 경로 모두 Stage 2 DoRA를 사용한 별도 실험이 필요하다.

## 10. 자주 발생하는 오류

### `Stage 2 artifact` 관련 오류

Stage 1 경로를 `eval.py --model_dir`에 직접 넣었는지 확인한다. 평가에는
Stage 2 artifact가 필요하다.

### DoRA에서 `base_model_dir is required`

Stage 2 DoRA 평가 명령에 원본 base를 지정한다.

```text
--base_model_dir Qwen/Qwen2.5-VL-3B-Instruct
```

Stage 1 Full에서 시작한 Stage 2 PEFT라면 Stage 1 Full artifact를 지정한다.

### latent token ID mismatch

Stage 2 adapter 디렉터리의 tokenizer/processor를 사용해야 한다. 토큰을
평가 시점에 다시 추가하거나 다른 tokenizer로 교체하지 않는다.

### image not found

`eval.py`는 `--image_root` 인자를 받지 않는다. `DATA_ROOT`에서 실행하거나
TEST JSONL의 `image_input`을 절대 경로로 바꾼다.

### CUDA OOM

평가는 batch size 1이지만 `max_new_tokens=4096`과 긴 입력 때문에 메모리가
커질 수 있다. 우선 `max_new_tokens`를 줄여 smoke하고, 다른 GPU process가
없는지 확인한다. 수학적 latent recurrence나 `latent_size`를 임의로 바꿔
OOM을 숨기지 않는다.

### FlashAttention 초기화 오류

평가 loader는 FlashAttention 초기화가 불가능한 것으로 판정될 때 eager로
fallback한다. 다른 runtime/model 오류까지 fallback으로 숨기지는 않는다.

## 11. 현재 한계

- Stage 1 artifact를 직접 task-generation 평가하는 controller는 없다.
- 실제 paper-level Stage 2 CoMT 재현 학습은 아직 실행하지 않았다.
- Stage 2 inference는 batch size 1만 지원한다.
- differentiable/cached KV-cache 최적화는 없다.
- adapter merge, stacking/fusion, QLoRA는 지원하지 않는다.
- `eval.py`의 일반 CoMT 분기는 exact match이며 논문의 모든 세부 metric을
  자동으로 재현하는 종합 evaluator는 아니다.

## 12. 평가 완료 체크리스트

- [ ] 올바른 Stage 1 Full/DoRA source를 사용했다.
- [ ] Stage 2 smoke training을 별도 경로에서 통과했다.
- [ ] Stage 2 artifact 형식을 확인했다.
- [ ] 10-sample evaluation에서 error가 없었다.
- [ ] Full/DoRA가 동일한 TEST와 generation 설정을 사용했다.
- [ ] 전체 평가 JSONL과 마지막 summary를 보존했다.
- [ ] error row 수를 별도로 기록했다.
- [ ] 학습/평가 GPU, commit, seed, hyperparameters를 기록했다.
- [ ] 결과를 paper-level reproduction으로 과장하지 않았다.

