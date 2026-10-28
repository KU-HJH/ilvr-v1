"""Actual teacher queries/features must stay clean and no extra encode may occur."""
from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from peft_common import resolve_qwen2_5_vl_model
from stage1.query import Stage1QueryConfig
from tests.stage1.history_fixtures import make_history_case
from tests.stage1.test_history_targets import RecordingBuilder
from tests.stage1.helpers import make_stage1_collator, prepared_example

MODES=('cross_helper_reverse_topk_independent','cross_helper_reverse_topk_cumulative')


def owned_case(singleton=False):
    case=make_history_case()
    sample=case.metadata.samples[0]
    if singleton:
        case.inputs['input_ids']=torch.cat((case.inputs['input_ids'][:,:19],case.inputs['input_ids'][:,-1:]),1)
        case.inputs['attention_mask']=torch.ones_like(case.inputs['input_ids'])
        case.inputs['pixel_values_latent']=case.inputs['pixel_values_latent'][:4]
        case.inputs['image_grid_thw_latent']=case.inputs['image_grid_thw_latent'][:1]
        sample=replace(sample,latent_spans=sample.latent_spans[:1],helper_to_block=(0,),
            helper_grid_range=(0,1),helper_pixel_range=(0,4),unpadded_length=20)
    sample=replace(sample,helper_image_ids=tuple(f'helper:{i}' for i in sample.helper_to_block),original_image_ids=('original',))
    return replace(case,metadata=replace(case.metadata,samples=(sample,),sequence_length=sample.unpadded_length))


def run(case,query_mode,voff_mode=None,validate=True):
    qwen=resolve_qwen2_5_vl_model(case.teacher)
    visuals=[]; prefixes=[]
    handles=[qwen.visual.register_forward_hook(lambda m,a,o:visuals.append(o.detach().clone())),
             qwen.model.register_forward_pre_hook(lambda m,a:prefixes.append(1))]
    builder=RecordingBuilder(case.tokenizer,query_config=Stage1QueryConfig(mode=query_mode),
        coverage_p=.1,helper_group_L=0,validate=validate)
    try:
        kwargs={} if voff_mode is None else {'voff_mode':voff_mode}
        targets=builder.build(case.teacher,dict(case.inputs),case.metadata,**kwargs)
    finally:
        for h in handles:h.remove()
    return targets,builder,visuals,prefixes


@pytest.mark.parametrize('query_mode',['legacy','adaptive'])
@pytest.mark.parametrize('singleton',[False,True])
def test_both_modes_share_exact_clean_queries_features_and_teacher_call_counts(query_mode,singleton):
    case=owned_case(singleton)
    clean,positive,base_visual,base_prefix=run(deepcopy(case),query_mode)
    assert clean.rejected_blocks == ()
    previous=None
    for mode in MODES:
        targets,builder,visuals,prefixes=run(deepcopy(case),query_mode,mode)
        torch.testing.assert_close(targets.latent_values,clean.latent_values,rtol=0,atol=0)
        assert targets.selected_counts==clean.selected_counts
        assert len(visuals)==len(base_visual)==2 and len(prefixes)==len(base_prefix)
        assert len(targets.rejected_blocks)==len(case.metadata.samples[0].latent_spans)
        for i,b in enumerate(targets.rejected_blocks):
            torch.testing.assert_close(builder.queries[i],positive.queries[i],rtol=0,atol=0)
            span=case.metadata.samples[0].latent_spans[i]
            assert b.positions==span.latent_positions and b.values.shape==(2,8)
            assert not b.values.requires_grad and b.batch_index==0 and b.block_index==i
            assert b.query_provenance==f'clean_teacher_{query_mode}'
            assert b.donor_kind==('original' if singleton else 'helper')
            source=visuals[0] if singleton else visuals[1][(1-i)*4:(2-i)*4]
            query=positive.queries[i].float()
            scores=(source.float()*query).sum(-1)/(source.float().norm(dim=-1).clamp_min(1e-8)*query.norm().clamp_min(1e-8))
            indices=scores.argsort(stable=True)[:2]
            torch.testing.assert_close(b.values,source[indices],rtol=0,atol=0)
            if previous is not None:torch.testing.assert_close(b.values,previous.rejected_blocks[i].values,rtol=0,atol=0)
        assert targets.selected_counts==((1,),) if singleton else targets.selected_counts==((1,1),)
        assert int(targets.diagnostics['rejected_original_blocks'])==(1 if singleton else 0)
        assert int(targets.diagnostics['rejected_helper_blocks'])==(0 if singleton else 2)
        previous=targets


def test_legacy_fallback_rejects_missing_prepared_embeddings_even_without_debug(monkeypatch):
    case=owned_case(True)
    original=case.teacher.forward
    def missing(**kwargs):
        output=original(**kwargs);output.inputs_embeds=None;return output
    monkeypatch.setattr(case.teacher,'forward',missing)
    with pytest.raises(ValueError,match='projected original'):
        run(case,'legacy',MODES[0],validate=False)


def test_collator_carries_shared_object_and_normalized_path_identity(tmp_path):
    collator=make_stage1_collator()
    example=prepared_example(source_index=0,original_images=1,helper_images=2,blocks=2)
    same=example.helper_image_paths[0]
    shared=replace(example,helper_image_paths=(same,same))
    batch=collator([shared]); sample=batch['stage1_batch_metadata'].samples[0]
    assert sample.helper_image_ids[0]==sample.helper_image_ids[1]
    assert sample.original_image_ids[0]!=sample.helper_image_ids[0]
    path=tmp_path/'image.png';same.save(path)
    alias=tmp_path/'alias.png';alias.symlink_to(path)
    sample2=collator._build_metadata([replace(example,helper_image_paths=(str(path),str(alias)))],
        {'input_ids':batch['input_ids']},[sample.latent_spans],[sample.unpadded_length],
        [1],[2],[1],[2]).samples[0]
    assert sample2.helper_image_ids[0]==sample2.helper_image_ids[1]
