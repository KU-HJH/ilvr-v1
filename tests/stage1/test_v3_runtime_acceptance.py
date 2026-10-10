"""Synthetic records exercise the validator, never stand in for runtime evidence."""
import importlib
from copy import deepcopy
import pytest


def api():return importlib.import_module('tests.stage1.v3_runtime_acceptance')


def test_rank_records_require_runtime_evidence():
    for records in ([],[{'global_rank':i,'success':True} for i in range(4)]):
        with pytest.raises(ValueError):api().validate_v3_rank_records(records,commit='a'*40)


def test_witness_validator_roundtrip_and_causal_negative_fields():
    records=synthetic_validator_fixture()
    api().validate_v3_rank_records(records,commit='a'*40)
    mutations=[lambda r:r.pop(),lambda r:r[1].update(global_rank=0),
        lambda r:r[0].update(hostname=r[2]['hostname']),lambda r:r[0].update(world_size=2),
        lambda r:r[0].update(commit='b'*40),lambda r:r[0].update(repository='/tmp/wrong'),
        lambda r:r[0]['contract']['config'].update(gain_alpha=1.),
        lambda r:r[1]['baseline']['events'][0].update(aux_vjp_arities=[1,1,1]),
        lambda r:r[1]['baseline']['events'][0].update(core_forwards=['F']),
        lambda r:r[1]['baseline']['events'][0].update(sites=[]),
        lambda r:r[1]['baseline']['boundaries'][0].update(master_error_ratio=float('nan')),
        lambda r:r[1]['baseline']['boundaries'][0].update(moment_residual=1.),
        lambda r:r[1]['resumed']['events'][0].update(occurrence=99),
        lambda r:r[1]['resumed']['state'].update(rng='b'*64),
        lambda r:r[1]['skip'].update(applied=True),
        lambda r:r[1]['skip'].update(overflow_calls=0),
        lambda r:r[1]['faults'].update(post_apply_checkpoint_blocked=False)]
    for change in mutations:
        bad=deepcopy(records);change(bad)
        with pytest.raises(ValueError):api().validate_v3_rank_records(bad,commit='a'*40)


def test_acceptance_rejects_unowned_or_single_node_without_cuda(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setenv('WORLD_SIZE','1')
    with pytest.raises(ValueError,match='four-rank'):api().run_acceptance(SimpleNamespace(commit='a'*40))


def synthetic_validator_fixture():
    from dataclasses import asdict
    from stage1.v3_config import Stage1V3Config
    from stage1.v3_resume import V3_FRESH_PROFILE,FORMULAS
    profile={**V3_FRESH_PROFILE,'world_size':4,'effective_global_batch':16,'zero_stage':1,'bf16':True,'frozen_teacher_sha256':'c'*64}
    contract=dict(schema_version=1,variant='v3',config=Stage1V3Config().to_dict(),formula_ids=FORMULAS,
        schedule=dict(retained_ids_sha256='d'*64,sampler_identity={'world_size':4},microsteps_per_rank=6,
            accumulation_steps=4,updates_per_epoch=2,planned_steps=2,remainder_microsteps=2),
        planned_steps=2,profile=profile,frozen_teacher_sha256='c'*64,boundary='complete')
    def event(rank,i):
        graphs=['F'] if rank==0 else ['F','gain_clean','packed_replay']
        sites=[dict(graph=g,unit=['decoder',j],site=f'layer{j}.q_proj',phase='original',backward=None) for g in graphs for j in range(2)]
        inventory=[(x['graph'],x['unit'],x['site']) for x in sites]
        sites += [dict(graph='F' if role=='ordinary' else 'packed_replay',unit=['decoder',0],site='layer0.q_proj',phase='recompute',backward=role)
                  for role in (['ordinary'] if rank==0 else ['joint_r_kl','rv','ordinary'])]
        return dict(occurrence=rank+4*i,core_forwards=['F'] if rank==0 else ['F','gain_clean','reference','packed_replay'],
            aux_vjp_arities=[] if rank==0 else [2,1],helper_builds=1,ordinary_backwards=1,lifecycle='CLOSED',sites=sites,inventory=inventory)
    def boundary(n):return dict(completed_microsteps=n,applied=True,global_blocks=24 if n==4 else 12,
        master_error_ratio=0.,sync_residual=0.,moment_residual=0.,movement=.0001)
    state={k:'e'*64 for k in ('masters','optimizer','scheduler','rng','teacher')}
    roots=dict(local='/local/ILVR',remote='/remote/ILVR',host='/host/ILVR')
    return [dict(schema_version=1,kind='stage1-v3-runtime-acceptance',commit='a'*40,
        packages={k:'test-version' for k in ('torch','transformers','peft','trl','accelerate','deepspeed')},
        roots=roots,repository=roots['remote' if rank<2 else 'local'],hostname=f'node{rank//2}',
        global_rank=rank,machine_rank=rank//2,local_rank=rank%2,world_size=4,zero_stage=1,contract=deepcopy(contract),teacher_before='e'*64,
        baseline=dict(events=[event(rank,i) for i in range(6)],boundaries=[boundary(4),boundary(6)],state=deepcopy(state)),
        interrupted=dict(events=[event(rank,i) for i in range(4)],boundaries=[boundary(4)],state=deepcopy(state)),
        resumed=dict(events=[event(rank,i) for i in range(4,6)],boundaries=[boundary(6)],state=deepcopy(state),restored_state=deepcopy(state)),
        skip=dict(cause='injected_overflow_decision',overflow_calls=1,applied=False,gain_steps_delta=0,master_residual=0.,moment_residual=0.,sync_residual=0.),
        faults=dict(cause='injected',nonfinite_checkpoint_blocked=True,post_apply_checkpoint_blocked=True)) for rank in range(4)]


def test_later_fault_attempts_cannot_change_successful_trajectory_record():
    from types import SimpleNamespace
    trainer=SimpleNamespace(events=[{'occurrence':4}],boundaries=[{'applied':True}],restored_state={'rng':'saved'})
    record=api().trajectory_record(trainer,{'masters':[__import__('torch').tensor([1.])]},'teacher',resume=True)
    trainer.events.append({'occurrence':9});trainer.boundaries[0]['applied']=False
    assert record['events']==[{'occurrence':4}] and record['boundaries']==[{'applied':True}]


def test_acceptance_fixture_retains_images_and_three_auxiliary_blocks(monkeypatch):
    from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds as Stage1TokenIds
    import torch
    monkeypatch.setattr(api().dist,'get_rank',lambda:3)
    deps=api().acceptance_dependencies()
    # Tiny tokenizer adds pad/start/end at original vocabulary boundary.
    from tests.stage1.test_integration import TinyIntegrationProcessor
    processor=TinyIntegrationProcessor()
    from stage1.composition import LATENT_TOKEN_STRINGS
    processor.tokenizer.add_special_tokens({'additional_special_tokens':list(LATENT_TOKEN_STRINGS)})
    ids=Stage1TokenIds(*(processor.tokenizer.convert_tokens_to_ids(t) for t in LATENT_TOKEN_STRINGS))
    batch=deps.collator_factory(processor,ids,8,v3_enabled=True)([7])
    metadata=batch['stage1_batch_metadata']
    assert len(metadata.v3.blocks)==3 and metadata.samples[0].source_index==7
    assert [b.start for b in metadata.v3.blocks]==[17,29,41]
    assert batch['input_ids'][0,:4].tolist()==[80]*4
    for b in metadata.v3.blocks:assert tuple(batch['input_ids'][0,list(b.target_positions)].tolist())==b.target_ids
