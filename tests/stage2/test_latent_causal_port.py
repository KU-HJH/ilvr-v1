"""Focused CUDA witnesses for the existing native causal evaluator port."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from tests.stage2.helpers import tiny_stage2_qwen


def prepared_samples():
    from stage2.latent_causal_intervention import Block
    samples=[]
    for name,prefix in [('a',1),('b',2)]:
        ids=torch.tensor([[prefix]+sum(([91]+[90]*8+[92,10+i,20+i] for i in range(3)),[])])
        blocks=[Block(i,tuple(range(2+12*i,10+12*i)),11+12*i,13+12*i) for i in range(3)]
        samples.append(SimpleNamespace(sample_id=name,task='selection',image_digests=[name],
            blocks=blocks,latent_blocks=blocks,model_inputs=dict(input_ids=ids,attention_mask=torch.ones_like(ids))))
    return samples


def frozen_plan(samples):
    from stage2.latent_causal_intervention import select_donors
    from stage2.latent_intervention_suite import shuffle_mapping
    descriptors=[dict(sample_id=s.sample_id,task=s.task,image_digests=s.image_digests,
        blocks={b.block_index:b.K for b in s.blocks}) for s in samples]
    mapping=select_donors(descriptors,42)
    assert mapping==select_donors(list(reversed(descriptors)),42)
    return mapping,{s.sample_id:shuffle_mapping(s.sample_id,[0,1,2],42) for s in samples}


def cuda_model(device='cuda:0'):
    assert torch.cuda.device_count()>=2, 'required two visible GPUs unavailable'
    torch.manual_seed(7)
    model=tiny_stage2_qwen(attn_implementation='sdpa')
    model.config.latent_size=8
    return model.to(device).float().eval().requires_grad_(False)


@pytest.mark.gpu
def test_native_capture_shift_independent_conditions_and_publication(tmp_path):
    # A one-token shift, recurrence re-run, or change outside intended slots breaks this witness.
    from stage2.latent_causal_intervention import make_replay, score_replay, verify_identity
    from stage2.latent_causal_runner import capture_sample
    from stage2.latent_intervention_suite import CONDITIONS, intervention_replay, evaluate_suite, publish_suite
    model=cuda_model(); prepared=prepared_samples(); mapping,permutations=frozen_plan(prepared)
    collected=[capture_sample(model,s,include_top1=True) for s in prepared]
    sample=collected[0]; block=sample.blocks[1]
    clean=make_replay(sample,block)
    assert clean.predictor_positions.tolist()==[22,23]
    assert clean.targets.tolist()==[11,21]
    with torch.inference_mode():
        replay_scores=score_replay(model,clean)
        verify_identity(sample.native_logps[1],replay_scores,.001,.0001)
        observed=[]
        hook=model.model.register_forward_pre_hook(lambda m,a,k:observed.append({
            n:k[n].clone() for n in ('inputs_embeds','attention_mask','position_ids')}),with_kwargs=True)
        try:
            zero,_=intervention_replay(sample,block,'mask_zero')
            masked,_=intervention_replay(sample,block,'mask_attention')
            score_replay(model,zero); score_replay(model,masked)
        finally:
            hook.remove()
        for call in observed:
            assert torch.equal(call['position_ids'],clean.position_ids.to('cuda:0'))
        assert torch.count_nonzero(observed[0]['inputs_embeds'][0,14:22])==0
        assert torch.equal(observed[0]['inputs_embeds'][0,:14],clean.embeddings[0,:14].to('cuda:0'))
        assert observed[1]['attention_mask'][0,14:22].tolist()==[0]*8
        assert observed[1]['attention_mask'][0,[13,22]].tolist()==[1,1]
        assert torch.equal(make_replay(sample,block).embeddings,clean.embeddings)
        rows,witness=evaluate_suite(model,collected,mapping,permutations,.001,.0001,91,92)
    summary=publish_suite(tmp_path,rows,dict(seed=42),witness)
    assert len(rows)==42 and witness['identity_checked_blocks']==6
    assert summary['conditions']['cross']['valid_blocks']==6
    assert summary['conditions']['step_prev']['valid_blocks']==4
    assert summary['conditions']['step_next']['valid_blocks']==4
    assert set(r['condition'] for r in rows)==set(CONDITIONS)
    assert json.loads((tmp_path/'summary.json').read_text())['status']=='complete'
    with pytest.raises(ValueError,match='validation'):
        publish_suite(tmp_path/'failure',rows,dict(seed=42),dict(witness,passed=False))
    assert not (tmp_path/'failure'/'summary.json').exists()


@pytest.mark.gpu
def test_two_cuda_streams_match_serial_and_failure_stops_publication(tmp_path):
    # Wrong device context, missing cross-shard bank or missed worker failure breaks real execution.
    from stage2.latent_causal_runner import capture_sample, run_parallel_suite, replicate_qwen_for_device
    from stage2.latent_intervention_suite import evaluate_suite
    model=cuda_model(); replica=replicate_qwen_for_device(model,'cuda:1')
    prepared=prepared_samples(); mapping,permutations=frozen_plan(prepared)
    collected=[capture_sample(model,s,include_top1=True) for s in prepared]
    expected,witness=evaluate_suite(model,collected,mapping,permutations,.001,.0001,91,92)
    actual,parallel_witness=run_parallel_suite((model,replica),('cuda:0','cuda:1'),prepared,
        mapping,permutations,.001,.0001,91,92)
    assert actual==expected and parallel_witness==witness
    replica.train()
    with pytest.raises(ValueError,match='eval mode'):
        run_parallel_suite((model,replica),('cuda:0','cuda:1'),prepared,mapping,permutations,.001,.0001,91,92)
    assert not (tmp_path/'summary.json').exists()


@pytest.mark.gpu
def test_identity_rejects_cancelling_errors_on_cuda():
    from stage2.latent_causal_intervention import verify_identity
    with pytest.raises(ValueError,match='identity'):
        verify_identity(torch.tensor([-1.,-2.],device='cuda'),torch.tensor([-.5,-2.5],device='cuda'),.001,.0001)


def test_paired_comparison_uses_common_support_and_rejects_changed_plan():
    # Unpaired CI subtraction or pooling condition-specific supports gives the wrong literal means.
    from stage2.latent_causal_comparison import compare_runs
    metadata=dict(status='complete',seed=42,K=8,selected_sample_indices=[0,1],
        donor_mapping=[dict(sample_id='a',block_index=0,donor_sample_id='b')],shuffle_permutations={'a':{'0':1,'1':0}},
        sample_metadata=[dict(sample_id='a',task='selection',image_digests=['a'],blocks={'0':8,'1':8})])
    def row(s,b,c,d,status='ok'):
        return dict(sample_id=s,block_index=b,condition=c,status=status,delta_nll=d,K=8,
            target_token_count=1,target_ids=[10],target_logps=[-1],clean_target_logps=[-1])
    left=[row('a',0,'mask_zero',1),row('a',1,'mask_zero',3),row('b',0,'mask_zero',10),
        row('a',0,'cross',2),row('a',1,'cross',0,'skip'),row('b',0,'cross',4)]
    right=[dict(r,delta_nll=r['delta_nll']+2) for r in left]
    result=compare_runs(left,right,metadata,metadata)
    assert result['conditions']['mask_zero']['mean_delta_nll']==2
    assert result['conditions']['cross']['valid_blocks']==2
    assert result['common_conditions']['mask_zero']['valid_blocks']==2
    assert result['common_conditions']['mask_zero']['mean_delta_nll']==2
    assert result['conditions']['mask_zero']['delta_nll_95ci']==[2,2]
    with pytest.raises(ValueError,match='donor_mapping'):
        compare_runs(left,right,metadata,dict(metadata,donor_mapping=[]))
    with pytest.raises(ValueError,match='target'):
        compare_runs(left,[dict(r,target_ids=[11]) for r in right],metadata,metadata)


@pytest.mark.gpu
def test_real_mixed_vision_native_capture_and_replay():
    # Wrong real vision backend/dtype or loss of original image/mRoPE input breaks this witness.
    from tests.stage2.test_attention_backend import _load_tiny_mixed_model
    from stage2.latent_causal_intervention import Block,make_replay,score_replay,verify_identity
    from stage2.latent_causal_runner import capture_sample,apply_vision_dtype,validate_analysis_runtime
    model,_,_,_=_load_tiny_mixed_model()
    model.config.stage='stage2'; model.config.latent_size=8
    model.config.latent_start_token_id=91; model.config.latent_pad_token_id=90; model.config.latent_end_token_id=92
    model=model.to('cuda:0').float().eval().requires_grad_(False)
    apply_vision_dtype(model,'bfloat16'); validate_analysis_runtime(model)
    ids=torch.tensor([[1,81,80,80,80,80,2,91]+[90]*8+[92,10,11]])
    sample=SimpleNamespace(sample_id='image',task='selection',image_digests=['original'],
        blocks=[Block(0,tuple(range(8,16)),17,19)],
        model_inputs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),
            pixel_values=torch.linspace(-1,1,48).reshape(4,12),image_grid_thw=torch.tensor([[1,2,2]])))
    received=[]
    hook=model.visual.register_forward_pre_hook(lambda m,a,k:received.append(a[0].dtype),with_kwargs=True)
    try:
        collected=capture_sample(model,sample,include_top1=True)
        with torch.inference_mode():
            clean=score_replay(model,make_replay(collected,collected.blocks[0]))
        verify_identity(collected.native_logps[0],clean,.001,.0001)
    finally:
        hook.remove()
    assert received==[torch.bfloat16]
    assert collected.embeddings.dtype==torch.float32
    assert not torch.equal(collected.position_ids[0],collected.position_ids[1])
    model.visual.float()
    with pytest.raises(ValueError,match='BF16'):
        validate_analysis_runtime(model)


def test_source_preparation_reads_original_only_and_preserves_processor_bounds(tmp_path):
    from PIL import Image
    from stage2.latent_causal_runner import prepare_sample,image_digest
    from tests.stage2.test_data import RecordingProcessor,TOKENS
    class Tokenizer:
        pad_token_id=0
        def __call__(self,text,**kwargs):
            return {'input_ids':torch.tensor([[31,32]] if text=='<|im_start|>assistant' else [[7]])}
    Image.new('RGB',(2,2),(1,2,3)).save(tmp_path/'original.png')
    Image.new('RGB',(2,2),(1,2,3)).save(tmp_path/'duplicate.bmp')
    ids=torch.tensor([[1,31,32,40,91]+[90]*8+[92,41,42,43,7]])
    processor=RecordingProcessor(ids);processor.tokenizer=Tokenizer()
    row=dict(task='selection',image_input=['original.png'],text_input='question',
        sequence_plan=[dict(type='text',content='before'),dict(type='latent',helper_image='absent.png'),
                       dict(type='text',content='reasoning')])
    sample,skips=prepare_sample(row,processor,TOKENS,8,tmp_path,32768,{})
    assert skips==[] and sample.blocks[0].latent_positions==tuple(range(5,13))
    assert (sample.blocks[0].target_start,sample.blocks[0].target_end)==(14,17)
    assert len(processor.calls[0]['images'][0])==1
    assert image_digest(tmp_path/'original.png',{})==image_digest(tmp_path/'duplicate.bmp',{})
    sample,skips=prepare_sample(row,processor,TOKENS,8,tmp_path,10,{})
    assert sample is None and skips[0]['skip_reason']=='context_length_exceeded'


def test_paired_comparison_rejects_missing_population_and_publishes_host_cli(tmp_path,monkeypatch,capsys):
    import sys
    from stage2.latent_causal_comparison import compare_runs,main
    metadata=dict(status='complete',seed=42,K=8,selected_sample_indices=[0],donor_mapping=[],
        shuffle_permutations={},sample_metadata=[],data_sha256='same')
    rows=[dict(sample_id='a',block_index=0,condition='cross',status='ok',delta_nll=1.,K=8,
        target_token_count=1,target_ids=[10])]
    with pytest.raises(ValueError,match='evaluation units'):
        compare_runs(rows,[],metadata,metadata)
    with pytest.raises(ValueError,match='data_sha256'):
        compare_runs(rows,rows,metadata,dict(metadata,data_sha256='different'))
    # Use the actual worktree results tree: the CLI enforces storage ownership.
    root=__import__('pathlib').Path(__file__).resolve().parents[2]/'results'/'dil-latent-causal-workflow'/'test-paired-cli'
    import tempfile
    with tempfile.TemporaryDirectory(dir=root.parent,prefix='test-paired-') as temporary:
        root=__import__('pathlib').Path(temporary)
        for name in ('high','gap'):
            directory=root/name;directory.mkdir()
            (directory/'summary.json').write_text(json.dumps(metadata))
            (directory/'per_block.jsonl').write_text(json.dumps(rows[0])+'\n')
        monkeypatch.setattr(sys,'argv',['compare','--high-entropy',str(root/'high'),
            '--entropy-gap',str(root/'gap'),'--output-dir',str(root/'out')])
        main()
        assert json.loads((root/'out'/'summary.json').read_text())['conditions']['cross']['mean_delta_nll']==0.


@pytest.mark.gpu
def test_cuda_replica_preserves_dora_sparse_rows_and_whole_block_shuffle():
    # Sharing adapter/row storage or shuffling vectors within a block changes real replay inputs.
    from peft import LoraConfig,get_peft_model
    from stage1.token_rows import attach_sparse_output_token_rows
    from stage2.peft import resolve_stage2_qwen_model
    from stage2.latent_causal_runner import capture_sample,replicate_qwen_for_device
    from stage2.latent_intervention_suite import intervention_replay
    from tests.stage2.helpers import activate_stage2_adapter_deltas
    wrapped=get_peft_model(cuda_model(),LoraConfig(r=2,lora_alpha=4,use_dora=True,
        target_modules=['model.layers.0.self_attn.q_proj'],task_type='CAUSAL_LM'))
    activate_stage2_adapter_deltas(wrapped)
    wrapped.peft_config['default'].trainable_token_indices={'model.embed_tokens':[90,91,92]}
    rows=attach_sparse_output_token_rows(wrapped,(90,91,92))
    with torch.no_grad():
        rows.output_delta.fill_(.03)
    wrapped.eval().requires_grad_(False)
    source=resolve_stage2_qwen_model(wrapped);replica=replicate_qwen_for_device(source,'cuda:1')
    assert source.lm_head.output_delta.data_ptr()!=replica.lm_head.output_delta.data_ptr()
    prepared=prepared_samples()[0]
    a=capture_sample(source,prepared,include_top1=True);b=capture_sample(replica,prepared,include_top1=True)
    assert torch.equal(a.embeddings,b.embeddings)
    assert torch.equal(a.native_logps[0],b.native_logps[0])
    permutation={0:1,1:2,2:0}
    replay,info=intervention_replay(a,a.blocks[1],'shuffle',permutation=permutation)
    assert torch.equal(replay.embeddings[0,2:10],a.bank[1])
    assert torch.equal(replay.embeddings[0,14:22],a.bank[2])
    assert info['changed_embedding_positions']==list(range(2,10))+list(range(14,22))
    assert a.embeddings.shape[1]==37


def test_explicit_invalid_sample_id_is_rejected_before_preparation():
    # Falling back to a digest for malformed explicit IDs changes the original donor identity contract.
    from stage2.latent_causal_runner import prepare_sample
    for value in ('', ' ', 12):
        with pytest.raises(ValueError,match='sample_id'):
            prepare_sample(dict(sample_id=value),None,None,8,None,32768,{})
