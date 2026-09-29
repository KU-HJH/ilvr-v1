#!/usr/bin/env bash
# Run the coefficient/margin follow-up arms (lambda_vis 0.3, m_V 0) sequentially.
# Records per-arm status, the measured time cap (first 10 updates x 212 x 3) and
# results/<run-id>/run_config.json. A failed arm is recorded and the next arm
# still runs (pre-registration stop rules). Time-cap breaches are flagged in the
# status file for the operator; the coordinator itself is not killed here.
set -uo pipefail

repo=/root/workspace/ILVR
status_dir=${STATUS_DIR:?set STATUS_DIR}
port_base=${PORT_BASE:-29910}
remote=korea@10.1.102.41
mkdir -p "$status_dir"
arms=("$@")
[[ ${#arms[@]} -gt 0 ]] || arms=(visiononly-l03m0 hient-independent-l03m0 hient-cumulative-l03m0)

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$status_dir/status.log"; }

remote_progress() {  # prints "<done> <elapsed_seconds>" from rank-0 tqdm, or nothing
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$remote" \
    "docker exec llm_kd bash -lc 'tr \"\\r\" \"\\n\" < /root/workspace/ILVR/logs/multinode/$1/initial/stderr-rank0.log 2>/dev/null | grep -E \"[0-9]+/212 \\[\" | tail -1'" 2>/dev/null |
    sed -nE 's/.* ([0-9]+)\/212 \[([0-9:]+)<.*/\1 \2/p' |
    awk '{n=split($2,t,":"); s=0; for(i=1;i<=n;i++) s=s*60+t[i]; print $1, s}'
}

index=0
for arm in "${arms[@]}"; do
  run_id=stage2-$arm-gb16-s212-$(date -u +%Y%m%d-%H%M%S)
  port=$((port_base + index)); index=$((index + 1))
  log "START arm=$arm run_id=$run_id port=$port commit=$(git -C "$repo" rev-parse --short HEAD)"
  launcher=$repo/scripts/stage2/launch_vision_margin_arm.sh
  [[ $arm == hient-* ]] && launcher=$repo/scripts/stage2/launch_highentropy_vision_arm.sh
  "$launcher" "$arm" --main-process-port "$port" \
    --run-id "$run_id" --launch > "$status_dir/$run_id-coordinator.log" 2>&1 &
  coordinator=$!
  cap=""
  while kill -0 "$coordinator" 2>/dev/null; do
    sleep 60
    read -r done elapsed < <(remote_progress "$run_id") || true
    [[ -n ${done:-} ]] || continue
    if [[ -z $cap && $done -ge 10 ]]; then
      cap=$(( elapsed * 212 * 3 / done ))
      log "TIME_CAP arm=$arm run_id=$run_id updates=$done elapsed=${elapsed}s cap_seconds=$cap"
    fi
    if [[ -n $cap && $elapsed -gt $cap ]]; then
      log "TIME_CAP_EXCEEDED arm=$arm run_id=$run_id elapsed=${elapsed}s cap=$cap updates=$done"
    fi
    printf '%s %s %s/212 %ss\n' "$(date -u +%FT%TZ)" "$run_id" "$done" "$elapsed" > "$status_dir/progress.txt"
    unset done elapsed
  done
  wait "$coordinator"; code=$?
  if [[ $code -eq 0 ]]; then outcome=COMPLETED; else outcome=FAILED; fi
  log "END arm=$arm run_id=$run_id exit=$code outcome=$outcome"
  if [[ -d $repo/results/$run_id ]]; then
    python3 - "$repo/results/$run_id/run_config.json" "$arm" "$run_id" "${cap:-0}" "$outcome" <<'PY'
import json, sys
path, arm, run_id, cap, outcome = sys.argv[1:]
hient = arm.startswith("hient-")
json.dump({"arm": arm, "run_id": run_id, "outcome": outcome, "seed": 42, "learning_rate": 5e-05,
           "global_batch_size": 16, "max_updates": 212, "lambda_dyn": 0.0, "lambda_vis": 0.3, "m_V": 0.0,
           "vision_token_reduction": "token_loss_mean" if hient else "score_mean",
           "vision_token_selection": "step0_entropy" if hient else "post_window",
           "vision_mask_scope": "cumulative" if "cumulative" in arm else "independent",
           "post_window": 1, "pre_window": 1, "m_execution_mode": "packed_suffix",
           "stage1_adapter": "checkpoints/stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210/checkpoint-431",
           "vision_attention_implementation": "flash_attention_2",
           "stage2_latent_activation_checkpointing": True, "time_cap_seconds": int(cap)},
          open(path, "x"), indent=2)
PY
  fi
done
log "ALL_DONE"
