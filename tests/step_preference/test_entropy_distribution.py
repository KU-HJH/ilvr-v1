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
