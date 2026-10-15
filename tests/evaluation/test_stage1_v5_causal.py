"""Catches sign/weighting, invalid pairing, subset drift, and unsafe replay."""
import json
from pathlib import Path

import pytest
import torch

from evaluation import stage1_v5_causal as causal


def test_metrics_use_off_minus_clean_and_equal_block_weight():
    # Block 1 has one harmed token, block 2 three unchanged tokens.
    clean = torch.tensor([[2., 0.], [0., 0.], [0., 0.], [0., 0.]])
    off = torch.zeros_like(clean)
    result = causal.block_metrics(clean, off, [0, 0, 0, 0], [1, 3, 0], chunk_size=2)
    assert result['sample']['delta_ce'] == pytest.approx(0.2831096)
    assert result['token_weighted']['delta_ce'] == pytest.approx(0.1415548)
    assert result['empty_blocks'] == 1
    assert result['sample']['kl_clean_off'] > 0
    assert result['blocks'][0]['entropy_clean_top50'] < result['blocks'][0]['entropy_off_top50']
    assert causal.block_metrics(clean[:0], off[:0], [], [0])['exclusion'] == 'no_eligible_targets'


def row(i, match, off=False, gold='parsed'):
    return dict(index=i, model_identity='model-A', generation_profile={'max_new_tokens': 4096},
                source_sha256='abc', prompt_identity=f'p{i}', gold_parse_status=gold,
                gold_normalized='yes', prediction_parse_status='parsed', match=match, prediction_normalized='yes' if match else 'no',
                generated_suffix_token_ids=[1, 2], latent_start_count=0, latent_block_count=0,
                stop_reason='eos', disable_latent_generation=off)


def test_pairing_has_complete_coverage_excludes_invalid_and_rejects_identity():
    on = [row(0, True), row(1, False), row(2, False, gold='invalid_gold')]
    off = [row(0, False, True), row(1, True, True), row(2, False, True, 'invalid_gold')]
    result = causal.pair_generation(on, off, [0, 1, 2], seed=42)
    assert result['on_correct_off_wrong'] == result['on_wrong_off_correct'] == 1
    assert result['valid_gold_count'] == result['inferential_pairs'] == 2
    assert result['legacy_on_accuracy'] == pytest.approx(1/3)
    assert result['valid_gold_on_accuracy'] == .5
    assert result['excluded_invalid_gold'] == [2]
    assert result['effect_off_minus_on'] == 0
    with pytest.raises(ValueError, match='coverage'):
        causal.pair_generation(on, off[:2], [0, 1, 2])
    off[0]['model_identity'] = 'wrong'
    with pytest.raises(ValueError, match='identity'):
        causal.pair_generation(on, off, [0, 1, 2])


def test_spot_baseline_requires_exact_tokens_prompt_and_scoring():
    saved = [row(5, True), row(9, False)]
    spot = [dict(saved[0], index=0), dict(saved[1], index=1)]
    causal.verify_baseline(saved, spot, [5, 9])
    spot[0]['generated_suffix_token_ids'] = [3]
    with pytest.raises(ValueError, match='generated_suffix_token_ids'):
        causal.verify_baseline(saved, spot, [5, 9])


def test_selection_verifies_source_hash_and_keeps_original_indices(tmp_path):
    source = tmp_path/'source.jsonl'
    source.write_text(''.join(json.dumps({'id': i})+'\n' for i in range(100)))
    selected, manifest = causal.select_rows(source, count=64, seed=42)
    assert len(selected) == len(set(manifest['source_indices'])) == 64
    assert manifest['source_indices'] == sorted(manifest['source_indices'])
    assert [r['id'] for r in selected] == manifest['source_indices']
    fixed = tmp_path/'indices.json'
    fixed.write_text('[3, 12]')
    rows, fixed_manifest = causal.select_rows(source, indices_path=fixed)
    assert [r['id'] for r in rows] == [3, 12]
    with pytest.raises(ValueError, match='sha256'):
        causal.select_rows(source, count=64, expected_sha256='wrong')


def test_eval_binding_rejects_grad_and_training_and_dropout():
    model = torch.nn.Linear(2, 2).eval()
    with pytest.raises(RuntimeError, match='no_grad'):
        with causal.analysis_execution(model):
            pass
    with torch.no_grad():
        model.train()
        with pytest.raises(RuntimeError, match='eval'):
            with causal.analysis_execution(model):
                pass
        model.eval()
        with causal.analysis_execution(model) as execution:
            with pytest.raises(RuntimeError, match='dropout'):
                execution.dropout_mask()


def test_real_native_replay_matches_unmasked_and_literal_body_mask():
    from tests.stage1.v5.test_replay import fixture, realize, IDS, POSITIONS
    from stage1.v5_metadata import V5BatchMetadata, V5Block
    student, _, _, _, live, helper, _, _ = fixture(0.)
    student.eval()
    qwen, embeds = realize(student, live, helper)
    metadata = V5BatchMetadata((V5Block(0, 7, 6, 9, (10, 11, 12), (31, 32, 33), (3,)),))
    with torch.no_grad():
        clean, off, drift = causal.replay_selected(student,
            dict(inputs_embeds=embeds, position_ids=POSITIONS, use_cache=False, return_dict=True),
            metadata, check_unmasked=True, atol=1e-5, rtol=1e-4)
        visible = torch.ones(26, 26, dtype=torch.bool).tril()
        visible[9:, 7:9] = False
        bias = torch.zeros(26, 26).masked_fill(~visible, -torch.inf)[None, None]
        literal = qwen.model(inputs_embeds=embeds, position_ids=POSITIONS,
                             attention_mask=bias, use_cache=False, return_dict=True)
        expected = qwen.lm_head(literal.last_hidden_state[0, [9, 10, 11]])
    torch.testing.assert_close(off, expected, atol=1e-5, rtol=1e-4)
    assert drift['passed'] and drift['query_keys'] == [[0, 9], [0, 10], [0, 11]]
    assert not torch.allclose(clean, off, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize('disable', [False, True])
def test_eval_flag_and_explicit_stage1_controller_forwarding(disable, monkeypatch):
    from tests.stage2.test_eval_integration import (
        _import_entry_point, RecordingStage1Model, RecordingController, FakeProcessor)
    import sys
    module = _import_entry_point('eval', monkeypatch)
    argv = ['eval.py', '--model_dir', 'model', '--test_data_path', 'data',
            '--task_name', 'zebra-cot', '--runtime_identity_path', 'runtime']
    if disable:
        argv.append('--disable-latent-generation')
    monkeypatch.setattr(sys, 'argv', argv)
    assert module.get_eval_args().disable_latent_generation is disable
    controller = RecordingController()
    monkeypatch.setattr(module, 'ContinuousLatentController', lambda: controller)
    result = module.run_one_example(RecordingStage1Model(), FakeProcessor(),
        {'text_input': 'question'}, {'max_new_tokens': 5}, disable_latent_generation=disable)
    assert controller.calls[0]['disable_latent_generation'] is disable
    assert result['extracted_final_answer'] == '42'


def test_stage2_rejects_stage1_only_off_flag(monkeypatch):
    from tests.stage2.test_eval_integration import _import_entry_point, ModelWhoseGenerateRaises, FakeProcessor
    module = _import_entry_point('eval', monkeypatch)
    with pytest.raises(ValueError, match='Stage 1'):
        module.run_one_example(ModelWhoseGenerateRaises(), FakeProcessor(),
            {'text_input': 'question'}, {}, disable_latent_generation=True)


def test_cli_materializes_only_fixed_subset_with_source_map(tmp_path):
    import subprocess
    import sys
    source = tmp_path/'source.jsonl'
    source.write_text(''.join(json.dumps({'id': i, 'image_input': f'i{i}.png'})+'\n' for i in range(5)))
    indices = tmp_path/'indices.json'
    indices.write_text('[1, 4]')
    output = tmp_path/'subset.jsonl'
    manifest = tmp_path/'selection.json'
    result = subprocess.run([sys.executable, 'scripts/analyze_stage1_v5_causal.py', 'select',
        '--source', str(source), '--indices', str(indices), '--output', str(output),
        '--manifest', str(manifest), '--absolute-images'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert [r['id'] for r in map(json.loads, output.read_text().splitlines())] == [1, 4]
    assert json.loads(manifest.read_text())['source_indices'] == [1, 4]
    assert json.loads(output.read_text().splitlines()[0])['image_input'] == str(tmp_path/'i1.png')


def test_voff_aggregation_preserves_samples_and_token_weighting():
    a = causal.block_metrics(torch.tensor([[2., 0.]]), torch.zeros(1, 2), [0], [1])
    b = causal.block_metrics(torch.zeros(3, 2), torch.zeros(3, 2), [0, 0, 0], [3])
    result = causal.summarize_voff([dict(a, source_index=1), dict(b, source_index=4)], [1, 4])
    assert result['sample_mean']['delta_ce'] == pytest.approx(.2831096)
    assert result['token_weighted']['delta_ce'] == pytest.approx(.1415548)
    assert result['positive_delta_token_fraction'] == .25
    assert result['positive_delta_block_fraction'] == .5
    assert result['paired_sample_metrics']['1']['delta_ce'] == pytest.approx(.5662192)
    with pytest.raises(ValueError, match='coverage'):
        causal.summarize_voff([dict(a, source_index=1)], [1, 4])


def test_unmasked_drift_includes_gold_ce_and_kl_at_temperature_one():
    from tests.stage1.v5.test_replay import fixture, realize, POSITIONS
    from stage1.v5_metadata import V5BatchMetadata, V5Block
    student, _, _, _, live, helper, _, _ = fixture(0.)
    student.eval()
    _, embeds = realize(student, live, helper)
    metadata = V5BatchMetadata((V5Block(0, 7, 6, 9, (10, 11), (31, 32), (3,)),))
    with torch.no_grad():
        _, _, drift = causal.replay_selected(student,
            dict(inputs_embeds=embeds, position_ids=POSITIONS), metadata,
            check_unmasked=True, atol=1e-5, rtol=1e-4)
    assert abs(drift['no_op_sample']['delta_ce']) < 1e-5
    assert abs(drift['no_op_sample']['kl_clean_off']) < 1e-6


def test_native_collator_helper_preparation_and_replay_are_eval_only():
    from tests.stage1.v5.test_metadata import SourceProcessor, source_example, source_collator
    from tests.stage1.v5.test_replay import fixture, digest_state
    from stage1.ema import Stage1EmaTeacher
    from peft_common import resolve_qwen2_5_vl_model
    student, _, _, _, _, _, _, _ = fixture(0.)
    qwen = resolve_qwen2_5_vl_model(student)
    # Real tiny model token layout follows this processor's actual image/latent IDs.
    qwen.config.image_token_id = 96
    qwen.config.vision_start_token_id = 95
    qwen.config.latent_token_id = 90
    qwen.config.latent_size = 3
    student.eval()
    processor = SourceProcessor()
    batch = source_collator(enabled=True, processor=processor)([source_example(['before', None, 'after'])])
    for key in ('pixel_values', 'pixel_values_latent'):
        batch[key] = batch[key].expand(-1, 12).contiguous()
    teacher = Stage1EmaTeacher(student, decay=1., method='dora')
    initial = digest_state(teacher.model)
    saved = dict(coverage_p=.9, image_pool_k=2, helper_group_L=2, textual_states='previous')
    with torch.no_grad():
        prepared, metadata = causal.prepare_analysis_inputs(teacher, processor, batch, saved, 'cpu')
        assert prepared['latent_hidden_states'].shape == (2, 8)
        clean, off, _ = causal.replay_selected(student, prepared, metadata,
            check_unmasked=True, atol=1e-5, rtol=1e-4)
    assert clean.shape == off.shape == (5, 128)
    assert digest_state(teacher.model) == initial
    assert all(p.grad is None for p in student.parameters())


def test_generation_summary_authenticates_saved_source_before_reuse(tmp_path):
    from types import SimpleNamespace
    from scripts.analyze_stage1_v5_causal import generation_summary
    baseline = [row(3, True), row(12, False)]
    spot = [dict(r, index=i) for i, r in enumerate(baseline)]
    off = [dict(r, index=i, disable_latent_generation=True, source_sha256='subset')
           for i, r in enumerate(baseline)]
    source = tmp_path/'source.jsonl'
    source.write_text('{}\n')
    values = dict(on=baseline, spot=spot, off=off,
        manifest=dict(source_indices=[3, 12], source_sha256='wrong', subset_sha256='subset'),
        baseline_manifest=dict(successful=True, model_identity='model-A',
            generation_config={'max_new_tokens': 4096}, dataset_identity={'path': str(source)}))
    paths = {}
    for key, value in values.items():
        path = tmp_path/(key+'.json')
        path.write_text(''.join(json.dumps(r)+'\n' for r in value) if isinstance(value, list) else json.dumps(value))
        paths[key] = str(path)
    args = SimpleNamespace(**paths, output=str(tmp_path/'out.json'), seed=42)
    with pytest.raises(ValueError, match='source.*identity'):
        generation_summary(args)
