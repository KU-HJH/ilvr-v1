import importlib
import importlib.util
import json
import pytest
from pathlib import Path
from types import SimpleNamespace as N
from tests.step_preference.test_packed_capacity_gate import capacity_fixture, inputs, write_adapter


def reporter():
    name = "scripts.publish_dora_ablation"
    assert importlib.util.find_spec(name) is not None, "ablation reporter missing"
    return importlib.import_module(name)


def command(run_id, triplet):
    return ["accelerate", "launch", "src/main.py", "--save_model_path", "/checkpoints/" + run_id,
        "--log_file", "/logs/" + run_id + "/train.log", "--stage2_length_manifest",
        "/results/" + run_id + "/stage2-length-manifest.json", "--ce_weight", str(triplet[0]),
        "--step_dpo_weight", str(triplet[1]), "--step_entropy_weight", str(triplet[2])]


def test_normalization_replaces_only_six_values_and_rejects_noise():
    m = reporter(); ids = [f"step-pref-dora-{arm}-test" for arm in ("ce", "dpo", "entropy")]
    normalized = [m.normalize_command(json.dumps(command(r, t)), r, t, ids)
        for r, t in zip(ids, [(1, 0, 0), (0, 1, 0), (0, 0, 1)])]
    assert normalized[0] == normalized[1] == normalized[2]
    for raw in ["noise\n" + json.dumps(command(ids[0], (1,0,0))), "{}",
        json.dumps(command(ids[0], (1,0,0)) + [ids[0]]),
        json.dumps(command(ids[0], (1,0,0)) + ["--ce_weight", "1"])]:
        with pytest.raises(ValueError): m.normalize_command(raw, ids[0], (1,0,0), ids)


def test_phase_reporter_rejects_duplicate_epoch_options():
    with pytest.raises(ValueError, match="exactly one --epochs"):
        reporter().option_value(["--epochs", "1", "--epochs", "2"], "--epochs")


def test_competition_ranks_preserve_ties_and_markdown_is_deterministic():
    m = reporter(); runs = [dict(arm=a, coverage_accuracy=s, correct=int(s*400),
        valid_gold_accuracy=s, invalid_gold_count=0, average_inference_time_sec=1.0)
        for a,s in [("entropy", .5), ("ce", .75), ("dpo", .75)]]
    ranked = m.rank_runs(runs)
    assert [(r["arm"], r["rank"]) for r in ranked] == [("ce",1), ("dpo",1), ("entropy",3)]
    markdown = m.comparison_markdown(ranked)
    assert "capacity-admitted subset" in markdown and "400" in markdown
    assert markdown == m.comparison_markdown(ranked)


def test_evaluation_validator_rejects_short_or_error_output(tmp_path):
    m = reporter(); output=tmp_path / "predictions.jsonl"; source=tmp_path / "TEST.jsonl"
    output.write_text('{"summary":true}\n'); source.write_text('{}\n' * 400)
    with pytest.raises(ValueError): m.validate_evaluation(output, source)


def evaluation_fixture(tmp_path,invalid=False):
    from step_preference.capacity_evidence import canonical
    source=tmp_path/"TEST.jsonl"; predictions=tmp_path/"predictions.jsonl"
    rows=[]
    for index in range(400):
        rows.append(dict(index=index,task_name="comt",image_input=[],prompt_identity="p"*64,prompt_token_length=2,
            generated_suffix_token_ids=[21,22],raw_output="Final answer: 42",assistant_only="Final answer: 42",
            prediction="Final answer: 42",extracted_final_answer="42",prediction_parse_status="parsed",prediction_parse_detail=None,
            prediction_normalized="42",gold_final_answer=None if invalid else "42",gold_parse_status="invalid_gold" if invalid else "parsed",
            gold_parse_detail="missing" if invalid else None,gold_normalized="" if invalid else "42",
            invalid_gold_reason="missing" if invalid else None,latent_start_count=1,latent_block_count=1,stop_reason="eos",
            match=not invalid,inference_time_sec=1.25))
    summary=dict(summary=True,num_samples=400,average_inference_time_sec=1.25,total_inference_time_sec=500.,
        accuracy=0. if invalid else 1.,success=0 if invalid else 400,correct=0 if invalid else 400,source_count=400,
        valid_gold_count=0 if invalid else 400,invalid_gold_count=400 if invalid else 0,
        invalid_gold=[dict(index=i,reason="missing") for i in range(400)] if invalid else [],
        coverage_accuracy=0. if invalid else 1.,valid_gold_accuracy=0. if invalid else 1.)
    source.write_bytes(b"".join(canonical(dict(original_final_answer=None if invalid else "42",image_input=[])) for _ in rows))
    predictions.write_bytes(b"".join(canonical(r) for r in [*rows,summary]))
    return source,predictions,rows,summary


@pytest.mark.parametrize("invalid",[False,True])
def test_400_row_evaluation_recomputes_zero_valid_gold_and_rejects_mutation(tmp_path,invalid):
    m=reporter(); source,predictions,rows,summary=evaluation_fixture(tmp_path,invalid)
    actual=m.validate_evaluation(predictions,source)
    assert actual.correct == (0 if invalid else 400)
    assert actual.valid_gold_accuracy == (0. if invalid else 1.)
    from step_preference.capacity_evidence import canonical
    rows[7]["match"]=not rows[7]["match"]
    predictions.write_bytes(b"".join(canonical(r) for r in [*rows,summary]))
    with pytest.raises(ValueError,match="scores"):m.validate_evaluation(predictions,source)


@pytest.mark.parametrize("mutation",["unknown","index","nonfinite","bool","error","summary"])
def test_evaluation_exact_schema_fails_closed_per_bound_field(tmp_path,mutation):
    m=reporter(); source,predictions,rows,summary=evaluation_fixture(tmp_path)
    if mutation == "unknown":rows[0]["extra"]=0
    if mutation == "index":rows[1]["index"]=0
    if mutation == "nonfinite":rows[0]["inference_time_sec"]=float("nan")
    if mutation == "bool":rows[0]["latent_start_count"]=True
    if mutation == "error":rows[0]["error"]="inference failure"
    if mutation == "summary":summary["correct"]=399
    predictions.write_bytes(b"".join((json.dumps(r,sort_keys=True,separators=(",",":"))+"\n").encode() for r in [*rows,summary]))
    with pytest.raises(ValueError):m.validate_evaluation(predictions,source)


def test_evaluation_digest_is_bound_to_decoded_bytes(tmp_path,monkeypatch):
    from step_preference.capacity_evidence import digest
    m=reporter();source,predictions,_,_=evaluation_fixture(tmp_path)
    original=predictions.read_bytes();reader=m.read_bytes
    def read(path,*args):
        raw=reader(path,*args)
        if Path(path) == predictions:predictions.write_bytes(b"changed after descriptor read")
        return raw
    monkeypatch.setattr(m,"read_bytes",read)
    summary,sha=m.validate_evaluation(predictions,source,with_identity=True)
    assert summary.correct == 400 and sha == digest(original)


def full_command(root,identifier,triplet,summary,identity):
    paths={name:root/name/identifier for name in ("checkpoints","results","logs")}
    options={"--save_model_path":paths["checkpoints"],"--log_file":paths["logs"]/"train.log",
        "--stage2_length_manifest":paths["results"]/"stage2-length-manifest.json","--stage2_length_manifest_sha256":"0"*64,
        "--data_path":identity.retained_data_path,"--image_root":identity.image_root_path,
        "--step_negative_path":identity.negative_path,"--step_negative_manifest":identity.negative_manifest_path,
        "--step_packed_admission_manifest":summary.admission_manifest_path,"--step_packed_admission_manifest_sha256":summary.admission_manifest_sha256,
        "--step_packed_admission_verification":summary.verification_path,"--step_packed_admission_verification_sha256":summary.verification_sha256,
        "--load_model_path":identity.checkpoint_path,"--stage2a_checkpoint_path":identity.checkpoint_path,
        "--step_preference_mode":"stage2b","--step_preference_objective":"legacy_dpo","--step_max_packed_sequence_length":"4608","--tuning_method":"dora",
        "--lora_r":"32","--lora_alpha":"64","--lora_dropout":"0.05","--seed":"42",
        "--per_device_train_batch_size":"1","--gradient_accumulation_steps":"2","--dataloader_num_workers":"0",
        "--attention_implementation":"sdpa","--vision_attention_implementation":"flash_attention_2",
        "--max_seq_length_train":"8192","--epochs":"2","--warm_up_steps":"10","--save_steps":"612",
        "--stage":"stage2","--latent_size":"8","--learning_rate":"5e-5","--step_dpo_beta":"0.1","--step_entropy_margin":"0.1",
        "--lora_target_modules":"q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "--deepspeed_config":root/"configs/deepspeed_zero2.json",
        "--ce_weight":triplet[0],"--step_dpo_weight":triplet[1],"--step_entropy_weight":triplet[2]}
    return [str(root/"venv/ilvr-venv/bin/accelerate"),"launch","--num_processes","2",str(root/"src/main.py"),
        *[token for key,value in options.items() for token in (key,str(value))],
        "--step_preference_enabled","--step_preference_packing","--gradient_checkpointing","--supervise_latent_boundaries","--stage2_launcher_log_capture"]


@pytest.fixture
def ablation_fixture(capacity_fixture,monkeypatch,tmp_path):
    from scripts import validate_packed_capacity_gate as gate
    from step_preference.capacity_evidence import publish_json
    m=reporter(); summary=gate.build_summary(capacity_fixture)
    sha=publish_json(capacity_fixture.output_path,summary,gate.PackedCapacitySummary)
    args=N(capacity_summary=capacity_fixture.output_path,capacity_summary_sha256=sha,
        comparison_dir=str(tmp_path/"results/step-pref-dora-comparison-test"))
    for name in ("results","logs"):(tmp_path/name).mkdir()
    _,manifest=m.capacity_and_manifest(args)
    for arm,triplet in m.ARMS.items():
        identifier=f"step-pref-dora-{arm}-test"
        setattr(args,arm+"_run_id",identifier)
        setattr(args,arm+"_command_json",json.dumps(full_command(tmp_path,identifier,triplet,summary,manifest.identity)))
    return args,summary,manifest


def test_real_launcher_phase_budget_is_rejected_at_freeze_and_reopen(ablation_fixture,tmp_path,monkeypatch):
    import os
    import shutil
    import subprocess
    import main as training_main
    from utils_deepseed import parse_args
    from step_preference.capacity_evidence import canonical

    m=reporter();args,summary,manifest=ablation_fixture
    repository=Path(__file__).resolve().parents[2]
    launcher=tmp_path/"run_stage2_training.sh"
    shutil.copyfile(repository/launcher.name,launcher)
    for name in ("src","transformers","venv"):
        (tmp_path/name).symlink_to(repository/name,target_is_directory=True)
    (tmp_path/"configs").mkdir()
    shutil.copyfile(repository/"configs/deepspeed_zero2.json",tmp_path/"configs/deepspeed_zero2.json")
    identity=manifest.identity
    environment={"PATH":os.environ["PATH"],"STAGE2_DRY_RUN":"true","STAGE2_DRY_RUN_FORMAT":"json",
        "STAGE2_TOTAL_STEPS":"4","STAGE1_CHECKPOINT":identity.checkpoint_path,
        "STAGE2A_CHECKPOINT_PATH":identity.checkpoint_path,"DATA_PATH":identity.retained_data_path,
        "IMAGE_ROOT":identity.image_root_path,"STEP_NEGATIVE_PATH":identity.negative_path,
        "STEP_NEGATIVE_MANIFEST":identity.negative_manifest_path,"STEP_PREFERENCE_ENABLED":"true",
        "STEP_PREFERENCE_MODE":"stage2b","STEP_PREFERENCE_OBJECTIVE":"legacy_dpo","STEP_PREFERENCE_PACKING":"true",
        "STEP_PACKED_ADMISSION_MANIFEST":summary.admission_manifest_path,
        "STEP_PACKED_ADMISSION_MANIFEST_SHA256":summary.admission_manifest_sha256,
        "STEP_PACKED_ADMISSION_VERIFICATION":summary.verification_path,
        "STEP_PACKED_ADMISSION_VERIFICATION_SHA256":summary.verification_sha256,
        "TUNING_METHOD":"dora","LEARNING_RATE":"5e-5","GRADIENT_ACCUMULATION_STEPS":"2",
        "GRADIENT_CHECKPOINTING":"true","STAGE2_NUM_PROCESSES":"2",
        "DEEPSPEED_CONFIG":str(tmp_path/"configs/deepspeed_zero2.json")}
    # SFTConfig alone is doubled: the launcher, parser, phase split and main builder execute.
    monkeypatch.setattr(training_main,"SFTConfig",lambda **kwargs:N(**kwargs))
    for arm,triplet in m.ARMS.items():
        identifier=getattr(args,arm+"_run_id")
        environment.update(OUTPUT_DIR=str(tmp_path/"checkpoints"/identifier),
            LOG_FILE=str(tmp_path/"logs"/identifier/"train.log"),
            CE_WEIGHT=str(triplet[0]),STEP_DPO_WEIGHT=str(triplet[1]),STEP_ENTROPY_WEIGHT=str(triplet[2]))
        result=subprocess.run(["bash",str(launcher),"--epochs","2","--warm_up_steps","10","--save_steps","612"],
            env=environment,check=True,capture_output=True,text=True)
        command=json.loads(result.stdout)
        epoch_index=command.index("--epochs")
        assert command[epoch_index+1] == "1"
        del command[epoch_index:epoch_index+2]
        assert command.count("--epochs") == 1
        parsed=parse_args(command[command.index(str(tmp_path/"src/main.py"))+1:])
        effective=training_main.build_stage2_training_arguments(parsed)
        assert parsed.step_preference_config.phase_steps == effective.max_steps == 2
        assert effective.num_train_epochs == 2
        setattr(args,arm+"_command_json",json.dumps(command))
    with pytest.raises(ValueError,match="step|phase"):
        m.freeze_launches(args,tmp_path)
    assert not Path(args.comparison_dir).exists()
    for arm in m.ARMS:
        tokens=json.loads(getattr(args,arm+"_command_json"))
        index=tokens.index("--stage2_total_steps");del tokens[index:index+2]
        setattr(args,arm+"_command_json",json.dumps(tokens))
    m.freeze_launches(args,tmp_path)
    originals={path:path.read_bytes() for path in Path(args.comparison_dir).glob("launch-config-*.json")}
    for override in (["--stage2_total_steps","4"],["--stage2_total_steps=4"],["--stage2_total=4"],
                     ["--max_steps=2"],["--max_step=2"]):
        for path,raw in originals.items():
            payload=json.loads(raw);payload["normalized_command"].extend(override)
            path.write_bytes(canonical(payload))
        with pytest.raises(ValueError,match="step|phase"):
            m.read_configs(args,tmp_path,summary,manifest)


def materialize_arms(args,summary,manifest,configs,root,monkeypatch):
    from step_preference.capacity_evidence import canonical,digest,LANGUAGE_CLASS,VISION_CLASS
    from tests.step_preference.test_packed_capacity_evidence import runtime_payload
    m=reporter();source,predictions,_,_=evaluation_fixture(root)
    args.test_data=str(source);args.test_data_sha256=digest(source.read_bytes())
    monkeypatch.setattr(m,"TEST_SHA",args.test_data_sha256)
    witness=json.loads(Path(summary.distributed_witnesses[0].training_runtime_identity_path).read_bytes())
    for config in configs:
        paths=m.arm_paths(root,config.run_id,config.arm);write_adapter(paths["checkpoints"])
        evaluation=paths["results"]/"evaluation";evaluation.mkdir(parents=True)
        length=paths["results"]/"stage2-length-manifest.json"
        length.write_bytes(Path(witness["training_length_manifest_path"]).read_bytes())
        runtime=runtime_payload(root)
        runtime.update(run_id=config.run_id,git_commit=manifest.identity.git_commit,
            coefficient_triplet=m.ARMS[config.arm],objective_metric_name="loss/"+config.arm,
            training_length_manifest_path=str(length),training_length_manifest_sha256=digest(length.read_bytes()),
            admission_manifest_sha256=summary.admission_manifest_sha256,verification_sha256=summary.verification_sha256)
        (paths["checkpoints"]/"stage2_training_runtime_identity.json").write_bytes(canonical(runtime))
        (evaluation/"predictions.jsonl").write_bytes(predictions.read_bytes())
        (evaluation/"evaluation-runtime-identity.json").write_bytes(canonical(dict(schema_version=1,
            kind="stage2-evaluation-runtime-identity",git_commit=manifest.identity.git_commit,tracked_worktree_clean=True,
            model_dir=str(paths["checkpoints"]),language_attention_class=LANGUAGE_CLASS,language_attention_layer_count=2,
            vision_attention_class=VISION_CLASS,vision_attention_block_count=2)))


def test_comparison_cannot_publish_from_swapped_runtime_sampler_claims(ablation_fixture,tmp_path,monkeypatch):
    from dataclasses import replace
    from stage2.data import summarize_stage2_lengths
    from stage2.length_manifest import read_stage2_length_manifest
    from step_preference.capacity_evidence import canonical,digest,TrainingRuntimeIdentity
    m=reporter();args,summary,manifest=ablation_fixture
    configs=m.freeze_launches(args,tmp_path)
    materialize_arms(args,summary,manifest,configs,tmp_path,monkeypatch)
    authenticated_hashes=[]
    for index,config in enumerate(configs):
        paths=m.arm_paths(tmp_path,config.run_id,config.arm)
        length=paths["results"]/"stage2-length-manifest.json"
        original=read_stage2_length_manifest(length)
        lengths=(original.lengths[0]+index,*original.lengths[1:])
        length.write_bytes(canonical(replace(original,lengths=lengths,stats=summarize_stage2_lengths(
            lengths=lengths,latent_block_counts=original.latent_block_counts,max_length=8192))))
        runtime_path=paths["checkpoints"]/"stage2_training_runtime_identity.json"
        runtime=json.loads(runtime_path.read_bytes());runtime["training_length_manifest_sha256"]=digest(length.read_bytes())
        runtime_path.write_bytes(canonical(runtime));authenticated_hashes.append(runtime["training_length_manifest_sha256"])
    assert len(set(authenticated_hashes)) == 3
    reader=m.read_json;swaps=[]
    def swap_before_reread(path,cls,*arguments,**kwargs):
        if cls is TrainingRuntimeIdentity:
            payload=json.loads(Path(path).read_bytes())
            payload["training_length_manifest_sha256"]="f"*64
            Path(path).write_bytes(canonical(payload));swaps.append(Path(path))
        return reader(path,cls,*arguments,**kwargs)
    monkeypatch.setattr(m,"read_json",swap_before_reread)
    monkeypatch.setattr(m,"__file__",str(tmp_path/"scripts/publish_dora_ablation.py"))
    flags=["--comparison-dir",args.comparison_dir,"--capacity-summary",args.capacity_summary,
        "--capacity-summary-sha256",args.capacity_summary_sha256,"--test-data",args.test_data,
        "--test-data-sha256",args.test_data_sha256]
    assert m.main(["publish-comparison",*flags]) == 2
    assert swaps and not (Path(args.comparison_dir)/"comparison.json").exists()
    assert not (Path(args.comparison_dir)/"comparison.md").exists()


@pytest.mark.parametrize("format",["indented","reordered"])
def test_capacity_and_arm_consumers_reject_noncanonical_sampler_with_matching_hash(
        ablation_fixture,capacity_fixture,tmp_path,monkeypatch,format):
    from step_preference.capacity_evidence import canonical,digest
    from scripts import validate_packed_capacity_gate as gate
    m=reporter();args,summary,manifest=ablation_fixture
    configs=m.freeze_launches(args,tmp_path)
    materialize_arms(args,summary,manifest,configs,tmp_path,monkeypatch)
    def rewrite_runtime(path):
        runtime=json.loads(path.read_bytes());length=Path(runtime["training_length_manifest_path"])
        payload=json.loads(length.read_bytes())
        raw=(json.dumps(payload,indent=2)+"\n").encode() if format == "indented" else (
            json.dumps(dict(reversed(list(payload.items()))),separators=(",",":"))+"\n").encode()
        length.write_bytes(raw);runtime["training_length_manifest_sha256"]=digest(raw)
        path.write_bytes(canonical(runtime))
    ce=next(config for config in configs if config.arm == "ce")
    rewrite_runtime(tmp_path/"checkpoints"/ce.run_id/"stage2_training_runtime_identity.json")
    args.arm="ce";args.phase="training"
    with pytest.raises(ValueError,match="canonical"):
        m.validate_arm(args,tmp_path,summary,manifest,{config.arm:config for config in configs})
    witness=Path(capacity_fixture.distributed_evidence[0]);evidence=json.loads(witness.read_bytes())
    runtime=Path(evidence["training_runtime_identity_path"]);rewrite_runtime(runtime)
    evidence["training_runtime_identity_sha256"]=digest(runtime.read_bytes());witness.write_bytes(canonical(evidence))
    with pytest.raises(ValueError,match="canonical"):
        gate.build_summary(capacity_fixture)


def test_three_arm_artifacts_freeze_publish_verify_and_transitive_mutations(ablation_fixture,tmp_path,monkeypatch,capsys):
    from step_preference.capacity_evidence import canonical,read_bytes,digest,TrainingRuntimeIdentity,EvaluationRuntimeIdentity,LANGUAGE_CLASS,VISION_CLASS
    from tests.step_preference.test_packed_capacity_evidence import runtime_payload
    m=reporter();args,summary,manifest=ablation_fixture
    configs=m.freeze_launches(args,tmp_path)
    assert {p.name for p in Path(args.comparison_dir).iterdir()} == {f"launch-config-{a}.json" for a in m.ARMS}
    for config in configs:
        assert all(not p.exists() for p in m.arm_paths(tmp_path,config.run_id,config.arm).values())
    source,predictions,_,_=evaluation_fixture(tmp_path)
    args.test_data=str(source);args.test_data_sha256=digest(read_bytes(source))
    # The fixed production TEST digest is not fabricated; this fixture isolates its hash boundary.
    monkeypatch.setattr(m,"TEST_SHA",args.test_data_sha256)
    witness_runtime=json.loads(Path(summary.distributed_witnesses[0].training_runtime_identity_path).read_bytes())
    for config in configs:
        paths=m.arm_paths(tmp_path,config.run_id,config.arm);write_adapter(paths["checkpoints"])
        evaluation_dir=paths["results"]/"evaluation";evaluation_dir.mkdir(parents=True)
        length=paths["results"]/"stage2-length-manifest.json"
        length.write_bytes(Path(witness_runtime["training_length_manifest_path"]).read_bytes())
        runtime=runtime_payload(tmp_path);runtime.update(run_id=config.run_id,git_commit=manifest.identity.git_commit,
            coefficient_triplet=m.ARMS[config.arm],objective_metric_name="loss/"+config.arm,
            training_length_manifest_path=str(length),training_length_manifest_sha256=digest(read_bytes(length)),
            admission_manifest_sha256=summary.admission_manifest_sha256,verification_sha256=summary.verification_sha256)
        (paths["checkpoints"]/"stage2_training_runtime_identity.json").write_bytes(canonical(runtime))
        (evaluation_dir/"predictions.jsonl").write_bytes(predictions.read_bytes())
        (evaluation_dir/"evaluation-runtime-identity.json").write_bytes(canonical(dict(schema_version=1,kind="stage2-evaluation-runtime-identity",
            git_commit=manifest.identity.git_commit,tracked_worktree_clean=True,model_dir=str(paths["checkpoints"]),
            language_attention_class=LANGUAGE_CLASS,language_attention_layer_count=2,vision_attention_class=VISION_CLASS,vision_attention_block_count=2)))
    directory,comparison,markdown=m.build_comparison(args,tmp_path)
    assert [run.rank for run in comparison.runs] == [1,1,1]
    monkeypatch.setattr(m,"__file__",str(tmp_path/"scripts/publish_dora_ablation.py"))
    flags=["--comparison-dir",args.comparison_dir,"--capacity-summary",args.capacity_summary,"--capacity-summary-sha256",args.capacity_summary_sha256,
        "--test-data",args.test_data,"--test-data-sha256",args.test_data_sha256]
    assert m.main(["publish-comparison",*flags]) == 0
    assert m.main(["verify-comparison",*flags]) == 0
    assert "VERIFIED_DORA_ABLATION arms=3 test_rows=400" in capsys.readouterr().out
    assert (directory/"comparison.md").read_text() == markdown
    paths=m.arm_paths(tmp_path,configs[0].run_id,configs[0].arm)
    for path in [directory/"launch-config-ce.json",paths["checkpoints"]/"adapter_model.safetensors",
                 paths["checkpoints"]/"stage2_training_runtime_identity.json",paths["results"]/"evaluation/predictions.jsonl",
                 paths["results"]/"evaluation/evaluation-runtime-identity.json",directory/"comparison.md"]:
        original=path.read_bytes();path.write_bytes(original+b" ")
        assert m.main(["verify-comparison",*flags]) == 2
        path.write_bytes(original)
    # Rewriting all normalized commands consistently still cannot change a frozen protocol setting.
    for path in directory.glob("launch-config-*.json"):
        payload=json.loads(path.read_bytes());tokens=payload["normalized_command"]
        tokens[tokens.index("--learning_rate")+1]="0.1";path.write_bytes(canonical(payload))
    with pytest.raises(ValueError,match="learning_rate"):m.read_configs(args,tmp_path,summary,manifest)


def test_preflight_is_silent_nonwriting_and_checks_each_legal_phase(ablation_fixture,tmp_path,monkeypatch,capsys):
    from step_preference.capacity_evidence import read_json,PackedGpuSnapshot
    m=reporter();args,summary,manifest=ablation_fixture
    observed=[];snapshot=read_json(summary.gpu_snapshot_path,PackedGpuSnapshot)
    monkeypatch.setattr(m,"capture_gpu_snapshot",lambda:snapshot)
    monkeypatch.setattr(m,"require_idle",lambda ranks:observed.append(ranks))
    for phase,arm in [("freeze",None),("training","ce"),("evaluation","entropy")]:
        args.phase=phase;args.arm=arm;m.preflight(args,tmp_path)
    assert observed == [(0,1),(0,1),(0,)]
    assert capsys.readouterr().out == "" and not Path(args.comparison_dir).exists()
    args.phase="freeze";args.arm="ce"
    with pytest.raises(ValueError,match="phase/arm"):m.preflight(args,tmp_path)
    args.phase="training";args.arm=None
    with pytest.raises(ValueError,match="phase/arm"):m.preflight(args,tmp_path)
    args.arm="ce"
    (tmp_path/"tracked").write_text("dirty\n")
    with pytest.raises(ValueError,match="tracked"):m.preflight(args,tmp_path)
    (tmp_path/"tracked").write_text("clean\n")
    monkeypatch.setattr(m,"git_identity",lambda:"0"*40)
    with pytest.raises(ValueError,match="commit"):m.preflight(args,tmp_path)
    monkeypatch.setattr(m,"git_identity",lambda:manifest.identity.git_commit)
    def busy(ranks):raise ValueError("capacity GPU has a compute application")
    monkeypatch.setattr(m,"require_idle",busy)
    with pytest.raises(ValueError,match="compute application"):m.preflight(args,tmp_path)
    assert not Path(args.comparison_dir).exists()
