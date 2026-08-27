# Stage 1 학습 산출물에서 Stage 2 학습 시작하기

이 문서는 이 worktree에서 3 epochs 학습한 Qwen2.5-VL-3B Stage 1
Full/DoRA 산출물을 사용해 Stage 2를 새로 시작하고, 중단된 Stage 2를
재개하는 절차를 설명한다.

평가 방법은 같은 디렉터리의 `guideline-stage1.md`를 참고한다. 이 문서는
Stage 2 학습 시작과 checkpoint lifecycle에 집중한다.

## 1. 핵심 개념

Stage 2는 Stage 1 모델의 optimizer 상태를 이어서 학습하는 과정이 아니다.

```text
Stage 1 weights / adapter / tokenizer / processor / latent metadata
                            ↓
                   fresh Stage 2 run
                            ↓
          new optimizer + new scheduler + global step 0
```

반면 이미 진행 중이던 Stage 2 numbered checkpoint를 재개할 때는 해당
Stage 2 optimizer, scheduler, RNG, global step을 정상적으로 복원한다.

```text
Stage 1 → fresh Stage 2       : optimizer/scheduler/global step reset
Stage 2 checkpoint → Stage 2 : optimizer/scheduler/RNG/global step resume
```

이 둘을 혼동하지 않는다.

## 2. 사용할 Stage 1 산출물

현재 완료된 Stage 1 산출물은 다음과 같다.

| 방식 | 경로 | 형식 |
| --- | --- | --- |
| Full | `outputs/stage1-timed/full-3epoch` | Full Qwen model + processor |
| DoRA | `outputs/stage1-timed/dora-3epoch` | DoRA adapter + sparse latent rows + processor |

공통 학습 조건은 다음과 같다.

```text
model: Qwen/Qwen2.5-VL-3B-Instruct
epochs: 3
optimizer steps: 648
samples processed: 10,350
learning rate: 1e-5
latent_size: 8
world size: 2
micro batch: 4/rank
gradient accumulation: 2
effective global batch: 16
```

Stage 1 DoRA artifact는 원본 base model을 포함하거나 merge하지 않는다.

## 3. 지원하는 Stage 1 → Stage 2 조합

| Stage 1 source | Stage 2 tuning | 지원 여부 | 비고 |
| --- | --- | --- | --- |
| Full | Full | 지원 | Full Stage 1 전체 weight에서 시작 |
| Full | LoRA | 지원 | Full Stage 1을 frozen base로 사용 |
| Full | DoRA | 지원 | Full Stage 1을 frozen base로 사용 |
| DoRA | DoRA | 지원 | 기존 DoRA와 sparse latent rows를 계속 학습 |
| DoRA | Full | 미지원 | implicit merge/변환을 허용하지 않음 |
| DoRA | LoRA | 미지원 | adapter stacking을 허용하지 않음 |

Stage 1 DoRA → Stage 2 DoRA에서는 다음 값이 Stage 1 metadata와 정확히
일치해야 한다.

```text
latent_size=8
r=32
alpha=64
dropout=0.05
targets=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

## 4. Stage 2 학습 의미

CoMT Stage 1 trajectory의 helper-image 위치는 Stage 2에서 다음 scaffold로
변환된다.

```text
text A
<latent_start>
<latent_pad> × K
<latent_end>
text B
```

여기서 `K=latent_size`이며 초기 설정은 `8`이다. `<latent_pad>` embedding을
추론 표현으로 쓰는 것이 아니라 앞 prefix의 final `last_hidden_state`로
각 latent slot을 순서대로 교체한다.

Stage 2의 중요한 실행 조건은 다음과 같다.

- rank-local/per-device batch size는 반드시 `1`
- 한 샘플 안의 여러 latent block은 모두 지원
- visual encoder 전체 동결
- original/user image는 batch당 한 번만 encoding
- helper image는 Stage 2 visual input으로 사용하지 않음
- latent slot은 항상 CE mask
- `supervise_latent_boundaries=true`이면 start/end 둘 다 CE 대상
- `false`이면 start/end 둘 다 CE mask
- latent recurrence는 differentiable하며 `detach`하지 않음
- initial backend는 no-cache recurrence

## 5. 공통 환경

```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
VENV="$REPO_ROOT/venv/ilvr-venv"

FULL_STAGE1="$REPO_ROOT/outputs/stage1-timed/full-3epoch"
DORA_STAGE1="$REPO_ROOT/outputs/stage1-timed/dora-3epoch"
BASE_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"

# TRAIN.jsonl과 images_comt/가 있는 실제 데이터 디렉터리
DATA_ROOT=/path/to/ILVR/data
CACHE_DIR="$REPO_ROOT/cache"

export PYTHONPATH="$REPO_ROOT/transformers/src:$REPO_ROOT/src"
export TOKENIZERS_PARALLELISM=false

# main.py가 logging을 초기화하기 전에 log parent가 존재해야 한다.
mkdir -p \
  "$REPO_ROOT/outputs/stage2-smoke" \
  "$REPO_ROOT/outputs/stage2-training"
```

이 repository에서는 반드시 `./venv/ilvr-venv`를 사용한다. system Python과
legacy `./ilvr-venv`를 사용하지 않는다.

현재 worktree에 `data/`가 없다면 main checkout이나 별도 데이터 경로를
`DATA_ROOT`로 지정한다.

### 5.1 가벼운 사전 확인

이미 검증된 3B model tree를 다시 hash하지 않고 필요한 파일 존재만
확인한다.

```bash
test -f "$FULL_STAGE1/config.json"
test -f "$FULL_STAGE1/model.safetensors.index.json"

test -f "$DORA_STAGE1/adapter_config.json"
test -f "$DORA_STAGE1/adapter_model.safetensors"
test -f "$DORA_STAGE1/ilvr_token_rows.safetensors"
test -f "$DORA_STAGE1/stage1_peft_metadata.json"

test -f "$DATA_ROOT/TRAIN.jsonl"
test -d "$DATA_ROOT/images_comt"
test -x "$VENV/bin/python"
test -x "$VENV/bin/torchrun"
```

기존 Stage 1 evidence는 다음 경로에서 재사용한다.

```text
outputs/stage1-timed/full-3epoch/evidence.json
outputs/stage1-timed/dora-3epoch/evidence.json
outputs/stage1-timed/final-report.json
```

## 6. 권장 초기 Stage 2 설정

아래 값은 현재 구현의 안정적인 출발점이다. paper-level 최적 설정을
검증했다는 의미는 아니다.

```text
latent_size=8
supervise_latent_boundaries=true
learning_rate=1e-5
per_device_train_batch_size=1
gradient_accumulation_steps=8
attention_implementation=flash_attention_2
gradient_checkpointing=true
max_seq_length_train=8192
seed=42
```

두 GPU에서 `gradient_accumulation_steps=8`이면 effective global batch는
다음과 같다.

```text
2 ranks × 1 sample/rank × 8 accumulation = 16
```

Stage 2는 `B>1/rank`를 지원하지 않으므로 batch를 키우지 않는다.

## 7. 먼저 단일 GPU smoke 실행

본 학습과 다른 출력 경로에서 각 방식당 `max_steps=1`을 실행한다. smoke가
성공해도 paper reproduction이나 본 학습 성공을 의미하지 않는다.

### 7.1 Full Stage 1 → Full Stage 2 smoke

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$PYTHONPATH" \
"$VENV/bin/python" "$REPO_ROOT/src/main.py" \
  --stage stage2 \
  --load_model_path "$FULL_STAGE1" \
  --tuning_method full \
  --data_path "$DATA_ROOT/TRAIN.jsonl" \
  --image_root "$DATA_ROOT" \
  --cache_dir "$CACHE_DIR" \
  --save_model_path "$REPO_ROOT/outputs/stage2-smoke/full" \
  --log_file "$REPO_ROOT/outputs/stage2-smoke/full.log" \
  --latent_size 8 \
  --supervise_latent_boundaries \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --attention_implementation flash_attention_2 \
  --max_seq_length_train 8192 \
  --seed 42 \
  --max_steps 1
```

### 7.2 Stage 1 Full → Stage 2 DoRA smoke

Stage 1 Full checkpoint를 base로 새 Stage 2 DoRA를 만드는 경우다.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$PYTHONPATH" \
"$VENV/bin/python" "$REPO_ROOT/src/main.py" \
  --stage stage2 \
  --load_model_path "$FULL_STAGE1" \
  --tuning_method dora \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --data_path "$DATA_ROOT/TRAIN.jsonl" \
  --image_root "$DATA_ROOT" \
  --cache_dir "$CACHE_DIR" \
  --save_model_path "$REPO_ROOT/outputs/stage2-smoke/full-to-dora" \
  --log_file "$REPO_ROOT/outputs/stage2-smoke/full-to-dora.log" \
  --latent_size 8 \
  --supervise_latent_boundaries \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --attention_implementation flash_attention_2 \
  --max_seq_length_train 8192 \
  --seed 42 \
  --max_steps 1
```

### 7.3 Stage 1 DoRA → Stage 2 DoRA continuation smoke

`--load_model_path`에는 원본 Qwen base를, `--stage1_adapter_path`에는
Stage 1 DoRA artifact를 전달한다.

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
  --cache_dir "$CACHE_DIR" \
  --save_model_path "$REPO_ROOT/outputs/stage2-smoke/dora-continuation" \
  --log_file "$REPO_ROOT/outputs/stage2-smoke/dora-continuation.log" \
  --latent_size 8 \
  --supervise_latent_boundaries \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --attention_implementation flash_attention_2 \
  --max_seq_length_train 8192 \
  --seed 42 \
  --max_steps 1
```

### 7.4 smoke 확인 항목

- Stage 1 source와 tokenizer/processor가 정상 로드되는가
- config와 tokenizer의 latent token ID가 일치하는가
- Stage 2 config에 `stage="stage2"`, `latent_size=8`이 기록되는가
- Full/DoRA trainable parameter 범위가 의도와 일치하는가
- visual parameter가 모두 frozen인가
- helper image가 Stage 2 visual input으로 전달되지 않는가
- loss와 latent norm이 finite인가
- filtered sample 수와 비율이 예상 범위인가
- `B=1` 오류가 발생하지 않는가

smoke 출력 디렉터리를 본 학습 또는 resume source로 사용하지 않는다.

## 8. 두 GPU Stage 2 본 학습

`run_stage2_training.sh`는 내부적으로 일반 Python process 하나를 실행한다.
두 GPU data parallel 학습은 `torchrun --nproc_per_node=2`로
`src/main.py`를 직접 실행한다.

아래 명령은 3 epochs 예시다. Stage 2 epoch 수는 Stage 1이 3 epochs였다는
이유만으로 자동 결정되는 값이 아니므로 실제 실험 전에 확정한다.

### 8.1 Full Stage 1 → Full Stage 2

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
  --cache_dir "$CACHE_DIR" \
  --save_model_path "$REPO_ROOT/outputs/stage2-training/full" \
  --log_file "$REPO_ROOT/outputs/stage2-training/full.log" \
  --latent_size 8 \
  --supervise_latent_boundaries \
  --epochs 3 \
  --save_steps 100 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --attention_implementation flash_attention_2 \
  --gradient_checkpointing \
  --max_seq_length_train 8192 \
  --deepspeed_config "$REPO_ROOT/configs/config_stage3.json" \
  --seed 42
```

### 8.2 Full Stage 1 → Stage 2 DoRA

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH="$PYTHONPATH" \
"$VENV/bin/torchrun" --nproc_per_node=2 \
  "$REPO_ROOT/src/main.py" \
  --stage stage2 \
  --load_model_path "$FULL_STAGE1" \
  --tuning_method dora \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --data_path "$DATA_ROOT/TRAIN.jsonl" \
  --image_root "$DATA_ROOT" \
  --cache_dir "$CACHE_DIR" \
  --save_model_path "$REPO_ROOT/outputs/stage2-training/full-to-dora" \
  --log_file "$REPO_ROOT/outputs/stage2-training/full-to-dora.log" \
  --latent_size 8 \
  --supervise_latent_boundaries \
  --epochs 3 \
  --save_steps 100 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --attention_implementation flash_attention_2 \
  --gradient_checkpointing \
  --max_seq_length_train 8192 \
  --deepspeed_config "$REPO_ROOT/configs/config_stage3.json" \
  --seed 42
```

### 8.3 Stage 1 DoRA → Stage 2 DoRA continuation

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
  --cache_dir "$CACHE_DIR" \
  --save_model_path "$REPO_ROOT/outputs/stage2-training/dora-continuation" \
  --log_file "$REPO_ROOT/outputs/stage2-training/dora-continuation.log" \
  --latent_size 8 \
  --supervise_latent_boundaries \
  --epochs 3 \
  --save_steps 100 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --attention_implementation flash_attention_2 \
  --gradient_checkpointing \
  --max_seq_length_train 8192 \
  --deepspeed_config "$REPO_ROOT/configs/config_stage3.json" \
  --seed 42
```

Stage 1 DoRA continuation은 Stage 1 optimizer, scheduler, EMA teacher,
global step을 가져오지 않는다. 학습된 DoRA parameter와 sparse latent-token
row만 Stage 2 초기값으로 사용한다.

## 9. Stage 2 checkpoint 형식

### 9.1 Full Stage 2

Full final artifact에는 일반적으로 다음이 포함된다.

```text
config.json
generation_config.json
model*.safetensors
tokenizer/processor files
```

### 9.2 LoRA/DoRA Stage 2

PEFT final artifact 또는 numbered checkpoint에는 최소 다음 파일이 필요하다.

```text
adapter_config.json
adapter_model.safetensors
stage2_peft_metadata.json
```

Stage 1 DoRA에서 계속된 sparse-token 방식은 다음 파일도 필요하다.

```text
ilvr_token_rows.safetensors
```

### 9.3 resumable numbered checkpoint

`checkpoint-N`에는 model/adapter 파일 외에 다음 상태가 필요하다.

```text
trainer_state.json
RNG state
optimizer state
scheduler state
DeepSpeed global_step*/ state when ZeRO-3 is enabled
```

final artifact는 inference/handoff용일 수 있으며 Trainer 상태가 없으면
resume source가 아니다. resume에는 완전한 numbered checkpoint를 사용한다.

## 10. 중단된 Stage 2 재개

resume 시 다음을 유지한다.

- 동일한 `save_model_path`
- 동일한 world size와 DeepSpeed backend
- 동일한 `latent_size`
- 동일한 boundary supervision
- 동일한 tuning method와 adapter topology
- 동일한 base source 종류

`--stage1_adapter_path`와 `--resume_from_checkpoint_path`는 상호 배타적이다.
Stage 1 DoRA continuation으로 만들어진 Stage 2 DoRA를 재개할 때도
`--stage1_adapter_path`를 다시 전달하지 않는다. Stage 2 checkpoint가
continuation provenance를 보유한다.

resume 명령의 `--epochs`는 checkpoint 이후에 추가로 실행할 epoch 수가
아니라 run 전체의 목표 epoch 수다. 예를 들어 1.4 epoch에서 중단된 run을
`--epochs 3`으로 재개하면 전체 3 epochs 지점까지 진행한다.

### 10.1 Full Stage 2 resume

```bash
STAGE2_FULL_CHECKPOINT="$REPO_ROOT/outputs/stage2-training/full/checkpoint-100"

CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH="$PYTHONPATH" \
"$VENV/bin/torchrun" --nproc_per_node=2 \
  "$REPO_ROOT/src/main.py" \
  --stage stage2 \
  --load_model_path "$FULL_STAGE1" \
  --resume_from_checkpoint_path "$STAGE2_FULL_CHECKPOINT" \
  --tuning_method full \
  --data_path "$DATA_ROOT/TRAIN.jsonl" \
  --image_root "$DATA_ROOT" \
  --cache_dir "$CACHE_DIR" \
  --save_model_path "$REPO_ROOT/outputs/stage2-training/full" \
  --log_file "$REPO_ROOT/outputs/stage2-training/full-resume.log" \
  --latent_size 8 \
  --supervise_latent_boundaries \
  --epochs 3 \
  --save_steps 100 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --attention_implementation flash_attention_2 \
  --gradient_checkpointing \
  --max_seq_length_train 8192 \
  --deepspeed_config "$REPO_ROOT/configs/config_stage3.json" \
  --seed 42
```

### 10.2 Full Stage 1 기반 Stage 2 DoRA resume

```bash
STAGE2_DORA_CHECKPOINT="$REPO_ROOT/outputs/stage2-training/full-to-dora/checkpoint-100"

CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH="$PYTHONPATH" \
"$VENV/bin/torchrun" --nproc_per_node=2 \
  "$REPO_ROOT/src/main.py" \
  --stage stage2 \
  --load_model_path "$FULL_STAGE1" \
  --resume_from_checkpoint_path "$STAGE2_DORA_CHECKPOINT" \
  --data_path "$DATA_ROOT/TRAIN.jsonl" \
  --image_root "$DATA_ROOT" \
  --cache_dir "$CACHE_DIR" \
  --save_model_path "$REPO_ROOT/outputs/stage2-training/full-to-dora" \
  --log_file "$REPO_ROOT/outputs/stage2-training/full-to-dora-resume.log" \
  --latent_size 8 \
  --supervise_latent_boundaries \
  --epochs 3 \
  --save_steps 100 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --attention_implementation flash_attention_2 \
  --gradient_checkpointing \
  --max_seq_length_train 8192 \
  --deepspeed_config "$REPO_ROOT/configs/config_stage3.json" \
  --seed 42
```

PEFT resume에서 `--tuning_method`과 `--lora_*`를 생략하면 checkpoint의
`stage2_peft_metadata.json` 값을 사용한다. 명시할 경우 method, rank,
alpha, dropout, target modules가 모두 정확히 일치해야 한다.

### 10.3 Stage 1 DoRA continuation 기반 Stage 2 DoRA resume

```bash
STAGE2_CONTINUED_DORA_CHECKPOINT="$REPO_ROOT/outputs/stage2-training/dora-continuation/checkpoint-100"

CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH="$PYTHONPATH" \
"$VENV/bin/torchrun" --nproc_per_node=2 \
  "$REPO_ROOT/src/main.py" \
  --stage stage2 \
  --load_model_path "$BASE_MODEL" \
  --resume_from_checkpoint_path "$STAGE2_CONTINUED_DORA_CHECKPOINT" \
  --data_path "$DATA_ROOT/TRAIN.jsonl" \
  --image_root "$DATA_ROOT" \
  --cache_dir "$CACHE_DIR" \
  --save_model_path "$REPO_ROOT/outputs/stage2-training/dora-continuation" \
  --log_file "$REPO_ROOT/outputs/stage2-training/dora-continuation-resume.log" \
  --latent_size 8 \
  --supervise_latent_boundaries \
  --epochs 3 \
  --save_steps 100 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --attention_implementation flash_attention_2 \
  --gradient_checkpointing \
  --max_seq_length_train 8192 \
  --deepspeed_config "$REPO_ROOT/configs/config_stage3.json" \
  --seed 42
```

이 경우 base는 계속 원본 `Qwen/Qwen2.5-VL-3B-Instruct`다. Stage 1 DoRA
artifact나 Stage 2 adapter를 `--load_model_path`로 전달하지 않는다.

## 11. 로그와 초기 실행에서 확인할 항목

Stage 2를 시작한 직후 다음 값을 기록한다.

```text
git commit
Stage 1 source path and kind
Stage 2 tuning method
latent_size
supervise_latent_boundaries
filtered sample count and percentage
world size
per-rank batch size
gradient accumulation
effective global batch
learning rate
attention backend
gradient checkpointing
DeepSpeed config
trainable parameter count
visual requires_grad state
```

초기 수십 step에서는 다음 diagnostics를 관찰한다.

- CE loss가 finite인가
- latent block count가 데이터 구조와 일치하는가
- latent state L2 norm이 finite인가
- 연속 step의 cosine similarity가 NaN이 아닌가
- early latent state까지 gradient가 도달하는가
- visual parameter gradient가 없는가

diagnostics는 관찰용이며 새로운 loss objective로 사용하지 않는다.

## 12. 데이터 필터링

Stage 2는 block-aware truncation을 하지 않는다.

```text
full preprocessing
        ↓
sequence length measurement
        ↓
max_seq_length_train 초과 sample 필터링
```

다음을 기록한다.

- 원본 sample 수
- 유지된 sample 수
- 필터링된 sample 수와 비율
- 길이 분포
- latent block 개수 분포

예상보다 많은 sample이 제거되면 즉시 max length를 늘리거나 데이터를
잘라내지 말고 원인을 조사한다. helper image가 Stage 2 input에 남았는지,
chat template가 중복 적용됐는지, latent block 수가 비정상인지 먼저 본다.

## 13. 오류 대응

### `Stage 1 PEFT-only base is not supported`

Stage 1 DoRA artifact를 `--load_model_path`로 전달한 경우다.

```text
--load_model_path Qwen/Qwen2.5-VL-3B-Instruct
--stage1_adapter_path outputs/stage1-timed/dora-3epoch
```

처럼 base와 adapter를 분리한다.

### `Stage 1 DoRA cannot initialize Stage 2 full tuning`

지원하지 않는 DoRA→Full 조합이다. DoRA→DoRA를 사용하거나 Full Stage 1
artifact에서 Stage 2 Full을 새로 시작한다.

### `Stage 1 adapter initialization and Stage 2 resume are mutually exclusive`

fresh continuation과 resume 인자를 함께 전달했다. 최초 Stage 2 시작에는
`--stage1_adapter_path`, 이후 재개에는 `--resume_from_checkpoint_path`만
사용한다.

### latent size 또는 token ID mismatch

Stage 1 DoRA continuation은 `latent_size=8`과 기존 tokenizer/processor를
보존해야 한다. 평가나 학습 시작 시 latent token을 다시 추가하지 않는다.

### malformed latent block

다음 구조를 확인한다.

```text
#latent_start == #latent_end
각 block의 latent_pad 수 == K
nested boundary 없음
incomplete block 없음
```

collator가 해당 sample을 명확한 오류로 중단시키는 것이 정상이다.

### `B>1` 오류

Stage 2의 `per_device_train_batch_size`는 반드시 `1`이다. 전체 batch를
늘리려면 GPU rank 수 또는 gradient accumulation을 조정한다.

### CUDA OOM

다음 순서로 원인을 확인한다.

1. 다른 GPU process와 실제 free memory 확인
2. sequence length와 latent block 수 확인
3. visual encoder가 recurrent loop 안에서 반복 호출되는지 확인
4. gradient checkpointing과 FlashAttention 초기화 확인
5. Full 대신 DoRA 실험이 목적에 부합하는지 검토

latent recurrence를 `detach`하거나 latent 수를 몰래 바꾸거나 language model
구간을 `torch.no_grad()`로 감싸 OOM을 숨기지 않는다.

### DeepSpeed resume 오류

생성 당시와 같은 world size, ZeRO config, optimizer/scheduler ownership을
사용한다. checkpoint 내부의 개별 파일을 복사해 새 checkpoint처럼 만들지
않는다.

### train/inference latent mismatch

동일한 model, prefix, position IDs, latent size, rollout 설정인지 확인한다.
generic `GenerationMixin`에 별도 latent 로직을 추가해 우회하지 않는다.

## 14. 성능 실험 순서

권장 순서는 다음과 같다.

1. 단일 GPU, `max_steps=1` smoke
2. 두 GPU, 3~10 optimizer-step distributed smoke
3. checkpoint 저장
4. 같은 backend로 다음 optimizer step까지 resume smoke
5. filtered data 통계 확인
6. 본 학습 예상 시간과 저장 공간 확인
7. 본 학습 1회 실행
8. 완료 artifact와 small evidence 기록
9. `guideline-stage1.md`에 따라 평가

성공한 고비용 smoke나 본 학습을 evidence 생성만을 위해 반복하지 않는다.

## 15. 현재 한계

- Stage 2 rank-local batch size는 `1`만 지원한다.
- arbitrary `B>1` latent scheduling은 없다.
- differentiable KV-cache optimization은 없다.
- Stage 2 paper-level CoMT reproduction은 아직 실행하지 않았다.
- Stage 2 3B Full/DoRA 본 학습 시간은 아직 실측하지 않았다.
- QLoRA, adapter merge, stacking/fusion은 지원하지 않는다.
- adaptive latent length와 vision tower fine-tuning은 지원하지 않는다.

## 16. 시작 전 체크리스트

- [ ] Full 또는 DoRA Stage 1 source를 정확히 선택했다.
- [ ] DoRA continuation에서 원본 base와 adapter를 분리했다.
- [ ] 지원되지 않는 DoRA→Full/LoRA 조합을 사용하지 않았다.
- [ ] `latent_size=8`을 확인했다.
- [ ] boundary supervision 값을 기록했다.
- [ ] `per_device_train_batch_size=1`이다.
- [ ] effective global batch를 계산했다.
- [ ] Stage 1과 Stage 2 output directory를 분리했다.
- [ ] smoke와 본 학습 output directory를 분리했다.
- [ ] 단일 GPU 1-step smoke를 통과했다.
- [ ] 두 GPU save/resume smoke를 통과했다.
- [ ] filtered sample 수와 비율을 확인했다.
- [ ] visual encoder가 frozen인지 확인했다.
- [ ] Stage 2 optimizer/scheduler가 fresh하게 시작하는지 확인했다.
- [ ] resume에는 완전한 Stage 2 numbered checkpoint를 사용했다.
- [ ] 로그에 commit, source, batch, LR, seed, backend를 기록했다.
- [ ] 고비용 실행을 중복하지 않도록 출력/evidence 경로를 확정했다.
