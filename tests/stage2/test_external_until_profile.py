"""External-until profile contracts, independent of external runtime imports."""
import copy
import importlib.util
from types import SimpleNamespace

import pytest
import torch
from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import BatchSampler

from stage2 import sampler as sampling


def test_grouping_two_stream_survives_four_actual_shards_and_real_tail(tmp_path):
    # Catches grouping by actual world4, zero-loss replacement and dropped tail.
    assert hasattr(sampling, 'ExternalUntilBalancedSampler'), 'external grouping2 profile missing'
    data = [SimpleNamespace(latent_block_count=1)] * 3379
    sampler = sampling.ExternalUntilBalancedSampler(data, num_replicas=4, seed=42,
        gradient_accumulation_steps=4)
    # Independent literal algorithm: shuffle within the only bucket, repeat its
    # first real index, then permute pairs with the same seeded generator.
    generator = torch.Generator().manual_seed(42)
    source = torch.randperm(3379, generator=generator).tolist()
    source.append(source[0])
    pairs = [source[i:i+2] for i in range(0, 3380, 2)]
    order = torch.randperm(1690, generator=generator).tolist()
    expected = [index for position in order for index in pairs[position]]
    shards = [list(BatchSamplerShard(BatchSampler(sampler, 1, False), num_processes=4,
        process_index=rank, even_batches=False)) for rank in range(4)]
    actual = [batch[0] for row in zip(*shards) for batch in row]
    assert actual == expected
    assert len(actual) == 3380 and len(set(actual)) == 3379
    assert len([actual[i:i+16] for i in range(0, len(actual), 16)]) == 212
    assert len(actual[-4:]) == 4 and all(index >= 0 for index in actual)
    sampling.write_stage2_sampler_sidecar(tmp_path, sampler.identity, global_step=0)
    assert sampling.read_stage2_sampler_sidecar(tmp_path) == sampler.identity
    assert sampler.identity.to_dict()['reference_grouping'] == 2
    assert sampler.identity.to_dict()['padding_weight'] == 1


def test_profile_clean_and_suffix_use_differentiable_dynamic_cache_without_checkpoint(monkeypatch):
    # Catches routing the new profile through tuple-capture or detached prefixes.
    from step_preference.shared_prefix_measurement import tiny_measurement_case
    from step_preference.equivalence import _trainer
    from step_preference.config import StepPreferenceConfig
    from transformers.cache_utils import DynamicCache
    model, batch = tiny_measurement_case(steps=2, total_target_tokens=4, device='cpu')
    model.config.stage2_execution_profile = 'latest_external_until'
    batch['step_preference']['steps'] = tuple({'span': s} for s in batch['reasoning_step_spans'])
    config = StepPreferenceConfig(enabled=True, mode='stage2b', stage2a_checkpoint_path='tiny',
        sidecar_attention_backend='dense', beta=.5, lambda_preference=1.,
        visual_only_context_preference=True, target_scope='until_next_latent')
    trainer = _trainer(config)
    trainer.stage2_execution_profile = 'latest_external_until'
    calls = []
    def record(module, args, kwargs, output):
        calls.append((kwargs, output))
    hook = model.model.register_forward_hook(record, with_kwargs=True)
    loss = trainer.compute_loss(model, copy.deepcopy(batch))
    loss.backward()
    hook.remove()
    cached = [(kw, out) for kw, out in calls if kw.get('use_cache')]
    assert len(cached) == 2, 'clean and one visual suffix must use DynamicCache'
    clean, suffix = cached
    assert isinstance(clean[1].past_key_values, DynamicCache)
    assert isinstance(suffix[0]['past_key_values'], DynamicCache)
    assert suffix[0]['past_key_values'] is not clean[1].past_key_values
    assert clean[1].past_key_values.key_cache[0].grad_fn is not None
    assert clean[1].past_key_values.get_seq_length() == batch['input_ids'].shape[1]
    assert suffix[1].past_key_values.get_seq_length() > clean[1].past_key_values.get_seq_length()
    assert len(calls) == len([x for x in calls if not x[0].get('stage2_prefix_key_values')])


def test_dora_scope_matches_original_through_optimizer_update_and_releases_norms():
    # Catches stale detached norm reuse after optimizer mutation or gradient loss.
    assert importlib.util.find_spec('stage2.dora') is not None, 'external DoRA scope missing'
    from stage2.dora import dora_microbatch_scope
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora.dora import DoraLinearLayer
    from contextlib import nullcontext
    torch.manual_seed(17)
    original = get_peft_model(torch.nn.Sequential(torch.nn.Linear(8, 8)),
        LoraConfig(r=2, lora_alpha=4, lora_dropout=.05, use_dora=True, target_modules=['0'])).train()
    with torch.no_grad():
        for name, p in original.named_parameters():
            if 'lora_B' in name:
                p.normal_(std=.05)
    aligned = copy.deepcopy(original)
    for _ in range(2):
        gradients = []
        for model in (original, aligned):
            model.zero_grad()
            torch.manual_seed(42)
            with dora_microbatch_scope(model) if model is aligned else nullcontext():
                x = torch.randn(2, 8)
                loss = model(model(x)).square().mean()
                loss.backward()
            gradients.append({n: p.grad.clone() for n,p in model.named_parameters() if p.grad is not None})
            torch.optim.SGD(model.parameters(), lr=.1).step()
        for name in gradients[0]:
            torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=1e-7, rtol=2e-5)
        assert all('forward' not in m.__dict__ for m in aligned.modules() if type(m) is DoraLinearLayer)


def test_cli_profile_resolves_approved_settings_without_changing_legacy():
    # Catches accepting the profile flag while leaving stale sourced objectives.
    from utils_deepseed import parse_args
    args = parse_args(['--stage', 'stage2', '--stage2_execution_profile', 'latest_external_until'])
    config = args.step_preference_config
    assert args.stage2_sampler_policy == 'external_until_balanced'
    assert config.target_scope == 'until_next_latent' and config.visual_only_context_preference
    assert config.beta == .5 and config.lambda_preference == 1. and config.lambda_entropy == 0.
    assert args.max_steps == 212 and args.stage2_total_steps == 212
    assert args.gradient_accumulation_steps == 4 and args.learning_rate == 5e-5
    assert not args.gradient_checkpointing and args.stage2_latent_activation_checkpointing
    assert not config.visual_off_suppress_latent_boundaries
    assert parse_args([]).stage2_execution_profile == 'legacy'


def test_profile_flex_compiles_dynamic_true_without_mutating_legacy_wrapper(monkeypatch):
    # Compilation is the boundary; don't run Triton on CPU to inspect this option.
    from transformers.integrations import flex_attention as integration
    requests = []
    monkeypatch.setattr(torch, 'compile', lambda fn, **kw: requests.append(kw) or fn)
    assert hasattr(integration, 'ExternalUntilFlexAttention'), 'profile compiler missing'
    monkeypatch.setattr(integration.ExternalUntilFlexAttention, "_instance", None)
    monkeypatch.setattr(integration.ExternalUntilFlexAttention, "_is_flex_compiled", False)
    integration.ExternalUntilFlexAttention()()
    assert requests == [{'dynamic': True}]
    monkeypatch.setattr(integration.WrappedFlexAttention, "_instance", None)
    monkeypatch.setattr(integration.WrappedFlexAttention, "_is_flex_compiled", False)
    integration.WrappedFlexAttention()()
    assert requests[-1] == {'dynamic': False}


def test_external_flash_vision_rotary_routes_profile_only(monkeypatch):
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as qwen
    calls = []
    assert hasattr(qwen, 'apply_rotary_pos_emb_flashatt'), 'external vision helper missing'
    # CPU cannot execute Flash kernels: preserve real QKV/rotary-input generation,
    # witness which rotary implementation receives it at the kernel boundary.
    monkeypatch.setattr(qwen, 'apply_rotary_pos_emb_flashatt',
        lambda q,k,c,s: (calls.append('external') or q, k))
    monkeypatch.setattr(qwen, 'apply_rotary_pos_emb_vision',
        lambda q,k,c,s: (calls.append('legacy') or q, k))
    monkeypatch.setattr(qwen, 'flash_attn_varlen_func', lambda q,k,v,*a: v)
    module = qwen.Qwen2_5_VLVisionFlashAttention2(8, num_heads=2)
    kwargs = dict(cu_seqlens=torch.tensor([0,2], dtype=torch.int32),
        position_embeddings=(torch.ones(2,4), torch.zeros(2,4)))
    module(torch.randn(2,8), **kwargs)
    module.stage2_execution_profile = 'latest_external_until'
    module(torch.randn(2,8), **kwargs)
    assert calls == ['legacy','external']


def test_external_trainer_real_tail_uses_ceil_and_configured_accumulation(tmp_path):
    # Real inherited training_step's division is not replaced by a test formula.
    from tests.stage2.test_reference_random_sampler import cpu_trainer
    trainer = cpu_trainer(tmp_path, 4, policy='external_until_balanced')
    values = trainer.set_initial_training_values(trainer.args, list(range(845)), 16)
    assert values[1] == 212 and values[6] == 212


def test_worker_forwards_execution_profile_in_resolved_command():
    from tests.stage2.test_launcher import run_launcher, dry_run_tokens, option_value
    tokens = dry_run_tokens(run_launcher(env_overrides={'STAGE2_EXECUTION_PROFILE':'latest_external_until'}))
    assert option_value(tokens, '--stage2_execution_profile') == 'latest_external_until'


from tests.stage2.test_peft_inference import sparse_inference_artifact


def test_saved_profile_survives_actual_peft_inference_loading(sparse_inference_artifact, tmp_path):
    # Catches save-only metadata that the original-base adapter loader discards.
    import shutil
    from stage2.peft import load_stage2_model_for_inference, resolve_stage2_qwen_model
    artifact = sparse_inference_artifact
    target = tmp_path / 'profile-adapter'
    shutil.copytree(artifact.stage2_dir, target)
    config = resolve_stage2_qwen_model(artifact.trained).config
    config = copy.deepcopy(config)
    config.stage2_execution_profile = 'latest_external_until'
    config.vision_config.stage2_execution_profile = 'latest_external_until'
    config.save_pretrained(target)
    model = load_stage2_model_for_inference(artifact.original_base_dir, target, artifact.processor)
    qwen = resolve_stage2_qwen_model(model)
    assert getattr(qwen.config, 'stage2_execution_profile', 'legacy') == 'latest_external_until'
    assert all(getattr(block.attn, 'stage2_execution_profile', 'legacy') == 'latest_external_until'
        for block in qwen.visual.blocks)


def test_explicit_profile_overrides_stale_adapter_target_environment():
    from utils_deepseed import parse_args
    args = parse_args(['--stage', 'stage2', '--stage2_execution_profile', 'latest_external_until',
        '--lora_target_modules', 'q_proj'])
    assert set(args.lora_target_modules.split(',')) == {
        'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}


@pytest.mark.parametrize('decoder_backend', ['sdpa', 'eager'])
def test_external_dynamic_cache_flex_suffix_matches_dense_scores_and_gradients(monkeypatch, decoder_backend):
    # Catches passing a BlockMask into ordinary Tensor slicing after cache append.
    # Only the CUDA compiler boundary is replaced: execute PyTorch's real Flex
    # operator on CPU, retaining the production planner, mask, cache, and backward.
    from dataclasses import replace
    from torch.nn.attention.flex_attention import BlockMask, flex_attention
    from transformers.integrations import flex_attention as integration
    from step_preference.shared_prefix_measurement import tiny_measurement_case
    from step_preference.equivalence import _trainer
    from step_preference.config import StepPreferenceConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    template, batch = tiny_measurement_case(steps=2, total_target_tokens=4, device='cpu')
    config = copy.deepcopy(template.config)
    config._attn_implementation = decoder_backend
    config.stage2_execution_profile = 'latest_external_until'
    model = Qwen2_5_VLForConditionalGeneration(config).float().train()
    model.load_state_dict(template.state_dict())
    model.visual.requires_grad_(False)
    reference = copy.deepcopy(model)
    batch['step_preference']['steps'] = tuple({'span': s} for s in batch['reasoning_step_spans'])
    config = StepPreferenceConfig(enabled=True, mode='stage2b', stage2a_checkpoint_path='tiny',
        sidecar_attention_backend='flex', beta=.5, lambda_preference=1.,
        visual_only_context_preference=True, target_scope='until_next_latent')
    observed = []
    def cpu_flex(q, k, v, **kwargs):
        assert isinstance(kwargs['block_mask'], BlockMask)
        assert kwargs.pop('external_until') is True
        observed.append((q.shape[-2], k.shape[-2]))
        return flex_attention(q, k, v, **kwargs)
    monkeypatch.setattr(integration, 'compile_friendly_flex_attention', cpu_flex)
    losses = []
    for current, backend in ((model, 'flex'), (reference, 'dense')):
        trainer = _trainer(replace(config, sidecar_attention_backend=backend))
        trainer.stage2_execution_profile = 'latest_external_until'
        loss = trainer.compute_loss(current, copy.deepcopy(batch))
        loss.backward()
        losses.append(loss.detach())
    assert observed and all(k == batch['input_ids'].shape[1] + q for q,k in observed)
    torch.testing.assert_close(losses[0], losses[1], atol=1e-6, rtol=1e-5)
    for (name, actual), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
        assert (actual.grad is None) == (expected.grad is None), name
        if actual.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, atol=1e-6, rtol=1e-4, msg=name)
