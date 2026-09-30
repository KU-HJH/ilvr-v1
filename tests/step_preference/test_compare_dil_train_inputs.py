"""Negative controls catch each material real-input witness mutation."""
import copy
import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).with_name('compare_dil_train_inputs.py')


def diagnostic():
    assert MODULE_PATH.exists(), 'real-input diagnostic is missing'
    spec = importlib.util.spec_from_file_location('compare_dil_train_inputs', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_diagnostic_detects_each_material_change():
    module = diagnostic()
    source = {'input_ids': [1, 2], 'labels': [-100, 2], 'positions': [0, 1],
              'target': {'start': 1, 'end': 2, 'ids': [2]}, 'pixels': 'original',
              'visibility': {'body_key': False, 'image_key': True}}
    for field, replacement in [('input_ids', [1, 3]), ('labels', [-100, -100]),
                               ('positions', [1, 1]), ('target', {'start': 1, 'end': 3, 'ids': [2]}),
                               ('pixels', 'altered'),
                               ('visibility', {'body_key': True, 'image_key': True}),
                               ('visibility', {'body_key': False, 'image_key': False})]:
        altered = copy.deepcopy(source)
        altered[field] = replacement
        differences = module.differing_fields(source, altered)
        assert differences and all(path.startswith(field) for path in differences)
    assert module.differing_fields(source, copy.deepcopy(source)) == []


def test_tensor_hash_detects_bytes_dtype_shape_and_target_endpoint():
    import torch
    module = diagnostic()
    original = torch.tensor([1, 2], dtype=torch.long)
    expected = module.tensor_record(original)
    for changed in (torch.tensor([1, 3]), original.int(), original.view(1, 2), original[:1]):
        assert module.differing_fields(expected, module.tensor_record(changed))
    pixels = torch.tensor([0.25, 0.5], dtype=torch.float32)
    altered = pixels.clone()
    altered.view(torch.uint8)[0] ^= 1
    assert module.differing_fields(module.tensor_record(pixels), module.tensor_record(altered))


def test_actual_visibility_detects_body_exposure_and_original_image_blocking():
    import torch
    import torch.nn.attention.flex_attention as flex
    from step_preference.sidecar import build_context_sidecar_plan, build_sidecar_flex_block_mask
    from step_preference.spans import ReasoningStepSpan
    module = diagnostic()
    ids = torch.arange(13).view(1, -1)
    spans = (ReasoningStepSpan(0, 0, 1, 2, 4, 5, 7, 0),
             ReasoningStepSpan(1, 5, 7, 8, 10, 11, 13, 1))
    plan = build_context_sidecar_plan(clean_input_ids=ids, attention_mask=torch.ones_like(ids),
                                     spans=spans, visual_only=True)
    saved = flex.create_block_mask
    flex.create_block_mask = lambda fn, **kw: fn
    try:
        callback = build_sidecar_flex_block_mask(13, plan.groups[0], 'cpu')
    finally:
        flex.create_block_mask = saved
    # These literal visible keys name image/prefix, own replay, and future/foreign exclusions.
    actual = callback(0, 0, torch.arange(4)[:, None], torch.arange(17)[None, :])
    assert [row.nonzero().flatten().tolist() for row in actual] == [
        [0, 1, 13], [0, 1, 13, 14], [0, 1, 2, 3, 4, 5, 6, 7, 15],
        [0, 1, 2, 3, 4, 5, 6, 7, 15, 16]]
    expected = module.visibility_record(callback, 4, 17, chunk=1)
    assert expected['edges_examined'] == 68
    assert expected['padded_edges_examined'] == 16316
    assert expected['padded_visible_edges'] == 0
    assert module.visibility_record(callback, 4, 17, chunk=3) == expected
    # Every mutation changes one actual emitted edge, even between chunk boundaries.
    for q, k, value in [(0, 2, True), (0, 0, False), (0, 4, True),
                        (0, 15, True), (0, 14, True), (4, 0, True)]:
        def altered(b, h, query, key):
            values = callback(b, h, query, key)
            return torch.where((query == q) & (key == k), value, values)
        assert module.differing_fields(expected, module.visibility_record(altered, 4, 17, chunk=3))


def test_pre_latent_text_join_crossing_is_reported_as_unsupported():
    import pytest
    import torch
    from step_preference.spans import build_assistant_plan, map_reasoning_step_spans
    # Qwen tokenization merges hello across the he/llo plan join. A literal
    # token offset [0,5] cannot represent the immediate previous text [2,5].
    plan = [{'type': 'text', 'content': 'he'}, {'type': 'text', 'content': 'llo'},
            {'type': 'latent', 'helper_image': 'helper.png'},
            {'type': 'text', 'content': 'reply'}]
    text, char_spans, steps = build_assistant_plan(plan, 8, assistant_plan_rendering='concatenated-v1')
    offsets = [(0, 5)]
    cursor = 5
    for token in ['<|latent_start|>'] + ['<|latent_pad|>'] * 8 + ['<|latent_end|>', 'reply']:
        offsets.append((cursor, cursor + len(token)))
        cursor += len(token)
    ids = torch.tensor([[100, 2] + [1] * 8 + [3, 101]])
    with pytest.raises(ValueError, match='exact contiguous token interval'):
        map_reasoning_step_spans(rendered_text=text, assistant_text=text,
                                 plan_char_spans=char_spans, reasoning_steps=steps,
                                 offset_mapping=[offsets], input_ids=ids, latent_pad_token_id=1,
                                 preference_target_scope='until_next_latent')


def test_comparison_rejects_skipped_duplicate_or_reordered_rows(tmp_path):
    import json
    import pytest
    from types import SimpleNamespace
    module = diagnostic()
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'retained_indices': [0, 1]}))
    row0 = {'index': 0, 'visibility': {'edges_examined': 1}}
    row1 = {'index': 1, 'visibility': {'edges_examined': 2}}
    (tmp_path / 'source-rows.jsonl').write_text(json.dumps(row0) + '\n' + json.dumps(row1) + '\n')
    args = SimpleNamespace(output=str(tmp_path), manifest=str(manifest), limit=0)
    for invalid in [[row0], [row0, row0], [row1, row0]]:
        (tmp_path / 'port-rows.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in invalid))
        with pytest.raises(ValueError, match='coverage'):
            module.compare(args)
