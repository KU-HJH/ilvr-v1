"""CUDA numerical witnesses; host-only coverage and launch contracts."""
import json
from types import SimpleNamespace
import pytest
import torch


def api():
    from scripts import collect_entropy_distribution as module
    return module


def test_cuda_full_vocab_source_and_alignment():
    # Breaks if collection uses top-k entropy, mismatched identities or signed gap.
    m = api()
    ids = torch.tensor([[7, 8, 9]], device='cuda')
    span = SimpleNamespace(step_id=0, target_text_start=1, target_text_end=3)
    clean = torch.tensor([1., 2.], device='cuda')
    row = {'name': 'step-0-visual-off', 'predictor_positions': (0, 1),
           'target_ids': ids[0, 1:], 'entropies': torch.tensor([3., .5], device='cuda'),
           'dil_tokens': {'negative': torch.tensor([99., 99.], device='cuda')}}
    got = m.paired_tokens(ids, [span], [clean], {(0, 'visual'): row})
    assert got == [{'step': 0, 'target_position': 1, 'predictor_position': 0, 'token_id': 8, 'Hc': 1., 'Hv': 3., 'G': 2.},
                   {'step': 0, 'target_position': 2, 'predictor_position': 1, 'token_id': 9, 'Hc': 2., 'Hv': .5, 'G': 1.5}]
    row['target_ids'] = torch.tensor([8, 10], device='cuda')
    with pytest.raises(ValueError, match='alignment'):
        m.paired_tokens(ids, [span], [clean], {(0, 'visual'): row})
    row['target_ids'] = ids[0, 1:]
    row['entropies'][0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        m.paired_tokens(ids, [span], [clean], {(0, 'visual'): row})


def sample(index, hc, hv):
    return {'retained_index': index, 'original_index': index + 10, 'sample_id': str(index),
            'tokens': [{'step': 0, 'target_position': j+1, 'predictor_position': j, 'token_id': j+5,
                        'Hc': c, 'Hv': v, 'G': abs(v-c)} for j, (c,v) in enumerate(zip(hc,hv))]}


def test_cuda_sample_local_ties_ratios_intersections_and_weighting():
    # Hand-ranked: C={sample0 token0, sample1 token0}; D={sample0 token1, sample1 token0}.
    samples = [sample(0, [10., 9., 1., 0., 0.], [10., 5., 3., 0., 0.]), sample(1, [2.], [3.]), sample(2, [], [])]
    got, annotated, paired = api().analyze_samples(samples, device='cuda')
    assert [[t['C'] for t in s['tokens']] for s in annotated] == [[True,False,False,False,False],[True],[]]
    assert [[t['D'] for t in s['tokens']] for s in annotated] == [[False,True,False,False,False],[True],[]]
    assert got['groups']['ALL']['count'] == 6
    assert got['groups']['C']['mean_Hc'] == 6.
    assert got['groups']['D']['mean_G'] == 2.5
    assert got['groups']['C']['relative_gap_percent'] == pytest.approx(100*.5/6)
    assert got['groups']['D']['mean_signed_gap'] == -1.5
    assert got['groups']['D']['proportion_Hv_gt_Hc'] == .5
    assert got['groups']['C&D']['count'] == 1
    assert got['groups']['C-D']['mean_G'] == 0.
    assert got['groups']['D-C']['mean_G'] == 4.
    assert got['differences']['meanG_D_minus_C'] == 2.
    assert got['paired_sample_differences']['G']['mean'] == 2.
    assert paired[2]['meanG_D_minus_C'] is None
    tie = [sample(0, [5.]*6, [6.]*6)]
    _, tied, _ = api().analyze_samples(tie, device='cuda')
    assert [t['C'] for t in tied[0]['tokens']] == [True,True,False,False,False,False]
    empty, _, _ = api().analyze_samples([sample(0, [0.], [0.])], device='cuda')
    assert empty['groups']['C-D']['mean_Hc'] is None
    assert empty['groups']['ALL']['relative_gap_percent'] is None


def test_host_exact_shards_and_reject_missing_duplicate_records():
    m = api()
    assert [m.shard_indices(7, rank, 4) for rank in range(4)] == [[0,4], [1,5], [2,6], [3]]
    records = [sample(0, [], []), sample(1, [], [])]
    m.validate_samples(records, [10,11])
    with pytest.raises(ValueError, match='coverage'):
        m.validate_samples(records[:1], [10,11])
    with pytest.raises(ValueError, match='duplicate'):
        m.validate_samples([records[0],records[0]], [10,11])
    tokens = sample(0, [1.], [2.])
    tokens['tokens'] *= 2
    with pytest.raises(ValueError, match='duplicate'):
        m.validate_samples([tokens], [10])


def test_host_launch_profile_does_not_inherit_old_probe_loss():
    from src.multinode import build_launch_plan
    from src.multinode_runtime import build_dil_full_model_probe_commands
    m = api()
    manifest = {'base': m.BASE, 'initializer': '/root/workspace/ILVR/.worktrees/dil-both-softplus-20261001/checkpoints/'+m.INITIALIZER,
                'data_path': 'data/TRAIN.jsonl', 'image_root': 'data', 'length_manifest': '/tmp/manifest', 'length_manifest_sha256': 'a'*64}
    argv = m.native_arguments(manifest, '/root/workspace/ILVR/.worktrees/dil-both-softplus-20261001', 'test-run')
    from utils_deepseed import parse_args
    config = parse_args(argv).step_preference_config
    assert (config.beta,config.lambda_ce,config.preference_margin,config.preference_loss_type) == (1.,.1,0.,'dil_bce_rf_topk_both_lower')
    plan = build_launch_plan(stage='stage2',tuning_method='dora',zero_stage=2,run_id='entropy-host-fixture',main_process_port=29884)
    commands = build_dil_full_model_probe_commands(plan,commit='a'*40,manifest='/tmp/manifest',manifest_sha256='b'*64,diagnostic='entropy-distribution')
    assert len(commands.node_specs) == 2
    assert all('collect_entropy_distribution.py' in ' '.join(spec.command) for spec in commands.node_specs)


def test_host_smoke_includes_longest_retained_sample_without_padding():
    m = api()
    # Longest raw sample is excluded; longest retained belongs to rank 2.
    retained = [0,1,2,3,4,5,6]
    lengths = [10,11,12,13,14,15,100,999]
    shards = [m.smoke_indices(retained,lengths,rank) for rank in range(4)]
    assert shards == [[0],[1],[2,6],[3]]
    assert len({i for shard in shards for i in shard}) == 5


def test_cuda_long_samples_do_not_silently_replace_paired_sample_average(tmp_path):
    m = api()
    samples = [sample(0,[10.,9.,1.,0.,0.,0.],[10.,5.,3.,0.,0.,0.]),sample(1,[2.],[3.])]
    summary,annotated,paired = m.analyze_samples(samples,device='cuda')
    assert summary['differences']['meanG_D_minus_C'] == pytest.approx(2/3)
    assert summary['paired_sample_differences']['G']['mean'] == .5
    assert summary['differences']['meanHc_D_minus_C'] == -3.
    assert summary['paired_sample_differences']['Hc']['mean'] == -2.25
    assert summary['groups']['D']['mean_G'] >= summary['groups']['C']['mean_G']
    assert summary['groups']['C']['mean_Hc'] >= summary['groups']['D']['mean_Hc']
    m.write_analysis(summary,annotated,paired,tmp_path)
    saved = json.loads((tmp_path/'summary.json').read_text())
    assert saved['groups']['ALL']['count'] == 7
    assert len((tmp_path/'paired-tokens.csv').read_text().splitlines()) == 8
    assert len((tmp_path/'paired-samples.jsonl').read_text().splitlines()) == 2
    assert 'relative_gap_percent' in (tmp_path/'magnitude-table.csv').read_text()
    for name in ('Hc-histogram-cdf','G-histogram-cdf','Hc-versus-G-density'):
        assert (tmp_path/f'{name}.png').read_bytes().startswith(b'\x89PNG')
        assert (tmp_path/f'{name}.pdf').read_bytes().startswith(b'%PDF')
    with pytest.raises(FileExistsError):
        m.write_analysis(summary,annotated,paired,tmp_path)


def test_host_length_identity_ignores_only_image_mount_paths():
    m = api()
    left = {'images':[{'path':'/raid/image.png','sha256':'a'*64}],'lengths':[1,2],'retained_indices':[0,1]}
    right = {**left,'images':[{'path':'/root/image.png','sha256':'a'*64}]}
    assert m.length_identity(left) == m.length_identity(right)
    right['images'][0]['sha256'] = 'b'*64
    assert m.length_identity(left) != m.length_identity(right)


def test_cuda_duplicate_steps_fail_before_publishing_records():
    m = api()
    ids = torch.tensor([[1,2]],device='cuda')
    span = SimpleNamespace(step_id=0,target_text_start=1,target_text_end=2)
    ent = torch.tensor([.5],device='cuda')
    row = {'name':'step-0-visual-off','predictor_positions':(0,), 'target_ids':ids[0,1:], 'entropies':ent}
    with pytest.raises(ValueError,match='duplicate'):
        m.paired_tokens(ids,[span,span],[ent,ent],{(0,'visual'):row})


def test_cuda_worker_observer_survives_parent_metric_path_without_training_stats(tmp_path, monkeypatch):
    # Breaks on the actual worker's empty reducer metrics reaching strict parent accumulation.
    # Replace large checkpoint loading/Trainer initialization only; retain the worker's
    # real local subclass, native compute_loss/context dispatch, decoder and sidecar.
    m = api()
    from step_preference.shared_prefix_measurement import tiny_measurement_case
    from step_preference.equivalence import _trainer
    from tests.step_preference.test_dil_config import _bce_config
    from stage2.trainer import Stage2Trainer
    import main
    import stage2.data
    import deepspeed_runtime
    from scripts import probe_dil_full_model
    with torch.device('cuda:0'):
        model, batch = tiny_measurement_case(steps=2,total_target_tokens=6,device='cuda:0')
    batch['step_preference'] = {'steps':tuple({'span':span} for span in batch['reasoning_step_spans'])}
    model.config.supervise_latent_boundaries = True
    model.to(dtype=torch.bfloat16)
    before = {name: p.detach().clone() for name,p in model.named_parameters()}
    config = _bce_config(preference_loss_type='dil_bce_rf_topk_both_lower',preference_topk=7,
                        beta=1.,lambda_ce=.1,target_scope='until_next_latent',
                        preference_entropy_top_percent=20.,sidecar_attention_backend='dense')
    created = []
    class InitializedTinyTrainer(Stage2Trainer):
        def __init__(self, **kwargs):
            self.__dict__.update(_trainer(config).__dict__)
            self.model = kwargs['model']
            self.optimizer = self.deepspeed = None
            self.accelerator = SimpleNamespace(device=torch.device('cuda:0'),process_index=2,num_processes=4)
            created.append(self)
        def _prepare_inputs(self, inputs):
            return inputs
    rows = [{'sample_id':f'case-{i}'} for i in range(3379)]
    examples = [SimpleNamespace(sample_id=row['sample_id']) for row in rows]
    def collate(selected):
        return {**batch,'sample_ids':(selected[0].sample_id,)}
    components = SimpleNamespace(model=model,trainer_class=InitializedTinyTrainer,train_dataset=examples,
        data_collator=collate,processor=None,peft_metadata=None,step_preference_config=config)
    length = tmp_path/'length.json'
    length.write_text(json.dumps({'lengths':[1]*3379,'retained_indices':list(range(3379))}))
    manifest = {'base':m.BASE,'initializer':str(tmp_path/'checkpoints'/m.INITIALIZER),
        'data_path':str(tmp_path/'data/TRAIN.jsonl'),'image_root':str(tmp_path/'data'),
        'length_manifest':str(length),'length_manifest_sha256':m.sha256(length),'retained_indices':list(range(3379))}
    monkeypatch.setattr(m,'load_manifest',lambda _path,_root:manifest)
    monkeypatch.setattr(probe_dil_full_model,'bootstrap',lambda *_args:None)
    monkeypatch.setattr(main,'build_stage2_training_components',lambda _args:components)
    monkeypatch.setattr(main,'build_stage2_training_arguments',lambda _args:None)
    monkeypatch.setattr(deepspeed_runtime,'bind_deepspeed_runtime_profile',lambda *_args,**_kwargs:None)
    monkeypatch.setattr(stage2.data,'load_jsonl_examples',lambda _path:rows)
    monkeypatch.setenv('LOCAL_RANK','0'); monkeypatch.setenv('RANK','2'); monkeypatch.setenv('ILVR_MACHINE_RANK','1')
    head_logits, outputs = [], []
    head_hook = model.lm_head.register_forward_hook(lambda _m,_a,out:head_logits.append(out.detach()))
    output_hook = model.register_forward_hook(lambda _m,_a,out:outputs.append(out))
    try:
        m.worker(SimpleNamespace(repository=str(tmp_path),manifest=length,manifest_sha256=m.sha256(length),
                                 run_id='diagnostic-causal-test',smoke=True))
    finally:
        head_hook.remove(); output_hook.remove()
    output = tmp_path/'results/diagnostic-causal-test'
    records = [json.loads(line) for line in (output/'entropy-rank-2.jsonl').read_text().splitlines()]
    assert len(records) == 1 and records[0]['retained_index'] == 2
    assert len(records[0]['tokens']) == 6
    clean_lp = outputs[0].stage2_selected_logits[0].float().log_softmax(-1)
    off_lp = head_logits[-1][0].float().log_softmax(-1)
    clean_entropy,off_entropy = -(clean_lp.exp()*clean_lp).sum(-1),-(off_lp.exp()*off_lp).sum(-1)
    torch.testing.assert_close(torch.tensor([t['Hc'] for t in records[0]['tokens']],device='cuda'),clean_entropy)
    torch.testing.assert_close(torch.tensor([t['Hv'] for t in records[0]['tokens']],device='cuda'),off_entropy)
    assert created[0]._step_preference_metric_totals == {}  # no fabricated composed-loss metrics
    witness = json.loads((output/'entropy-witness-rank-2.json').read_text())
    assert witness['eval_no_grad_frozen'] and witness['weights_version_unchanged']
    assert all(not module.training for module in model.modules())
    for name,p in model.named_parameters():
        assert not p.requires_grad and p.grad is None
        torch.testing.assert_close(p,before[name],atol=0,rtol=0)
