from src.multinode_runtime import validated_stage2_environment


def test_coordinator_keeps_until_and_latent_checkpoint_options():
    env = validated_stage2_environment(
        {"STAGE1_CHECKPOINT": "/models/stage1", "DATA_PATH": "/data/train.jsonl",
         "IMAGE_ROOT": "/data/images", "STEP_PREFERENCE_TARGET_SCOPE": "until_next_latent",
         "STAGE2_LATENT_ACTIVATION_CHECKPOINTING": "true"},
        output_dir="/outputs/stage2", log_file="/logs/stage2.log",
        deepspeed_config="/configs/zero.json")
    assert env["STEP_PREFERENCE_TARGET_SCOPE"] == "until_next_latent"
    assert env["STAGE2_LATENT_ACTIVATION_CHECKPOINTING"] == "true"
