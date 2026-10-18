"""Bounded read-only pinned-source vs restored GPU witness; ROOT executes after review.

Example (at most three source indices, <=512 tokens per side/sample):
  python -m evaluation.original_witness --model-dir CHECKPOINT --base-model-dir BASE \
      --test-data-path data/TEST.jsonl --indices 0,1,2 --output-dir results/RUN

No model is saved or merged. Parameter hashes stream small CPU chunks; decoder
inputs are observed without changing outputs. Use an external 600-second timeout
so the controller owns the aggregate GPU budget, including model loading.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time


def validate_budget(indices, max_new_tokens):
    selected=[int(value) for value in indices.split(',')]
    if len(selected)!=len(set(selected)) or any(index<0 for index in selected):
        raise ValueError('witness indices must be distinct nonnegative source indices')
    if not 1<=len(selected)<=3 or not 1<=max_new_tokens<=512:
        raise ValueError('witness budget is at most 3 distinct samples and 512 tokens each')
    return selected


def tensor_identity(tensor):
    import torch
    contiguous=tensor.detach().contiguous()
    digest=hashlib.sha256()
    flat=contiguous.reshape(-1)
    for offset in range(0,flat.numel(),1024*1024):
        chunk=flat[offset:offset+1024*1024].cpu().contiguous().view(torch.uint8)
        digest.update(chunk.numpy().tobytes())
    result=dict(shape=list(tensor.shape),dtype=str(tensor.dtype),sha256=digest.hexdigest())
    if tensor.numel()<=32:
        result['values']=tensor.detach().cpu().tolist()
    return result


def parameter_identity(model):
    """Hash actual unmerged base, DoRA, and sparse-row parameters without full-model copies."""
    digest=hashlib.sha256()
    count=0
    for name,parameter in model.named_parameters():
        value=dict(name=name,**tensor_identity(parameter))
        digest.update(json.dumps(value,sort_keys=True).encode())
        count+=parameter.numel()
    return dict(sha256=digest.hexdigest(),parameter_count=count)


def call_owners(model):
    import inspect
    return {name:dict(module=getattr(type(model),name).__module__,
        file=inspect.getsourcefile(getattr(type(model),name)),
        line=getattr(type(model),name).__code__.co_firstlineno)
        for name in ('generate','_sample','forward','prepare_inputs_for_generation',
                     '_get_initial_cache_position','_update_model_kwargs_for_generation')}


class ProcessorWitness:
    def __init__(self,processor):
        self.processor=processor
        self.tokenizer=processor.tokenizer
        self.calls=[]
    def __getattr__(self,name):
        return getattr(self.processor,name)
    def __call__(self,**kwargs):
        import torch
        result=self.processor(**kwargs)
        self.calls.append(dict(text=kwargs.get('text'),inputs={key:tensor_identity(value)
            for key,value in result.items() if torch.is_tensor(value)}))
        return result


def _decoder_record(_module,_args,kwargs):
    import torch
    value={key:tensor_identity(kwargs[key]) for key in
        ('inputs_embeds','position_ids','attention_mask','cache_position') if torch.is_tensor(kwargs.get(key))}
    cache=kwargs.get('past_key_values')
    value['past_length']=cache.get_seq_length() if cache is not None else None
    value['use_cache']=kwargs.get('use_cache')
    return value


def load_for_mode(mode,model_dir,base_model_dir,cache_dir):
    from transformers import AutoProcessor
    from stage1.checkpointing import load_stage1_peft_artifact
    if mode=='restored':
        from eval import load_evaluation_artifacts
        return load_evaluation_artifacts(model_dir=model_dir,base_model_dir=base_model_dir,
            cache_dir=cache_dir,evaluation_protocol='original')
    from evaluation.original_oracle import source_evaluation_functions
    original=source_evaluation_functions()
    processor=AutoProcessor.from_pretrained(model_dir,cache_dir=cache_dir)
    if (Path(model_dir)/'adapter_config.json').is_file():
        if base_model_dir is None:
            raise ValueError('source PEFT oracle requires the original base model')
        base=original['load_model'](base_model_dir,cache_dir)
        # Representation-only extension: preserve unmerged DoRA and sparse rows.
        model=load_stage1_peft_artifact(base,processor,Path(model_dir),
            requested_base_model_reference=base_model_dir,is_trainable=False)
        model.eval()
    else:
        model=original['load_model'](model_dir,cache_dir)
    return model,processor


def run_mode(mode,*,model_dir,base_model_dir,dataset_path,indices,cap,cache_dir,deadline):
    import torch
    from peft_common import resolve_qwen2_5_vl_model
    from multinode_evaluation import resolve_evaluation_image_paths
    from evaluation.original_ilvr import execution_identity,config_resolution
    from evaluation.original_oracle import source_evaluation_functions
    torch.manual_seed(42)
    model,processor=load_for_mode(mode,model_dir,base_model_dir,cache_dir)
    qwen=resolve_qwen2_5_vl_model(model)
    kwargs=dict(max_new_tokens=cap,temperature=0.,top_p=1.,do_sample=False)
    metadata=dict(parameters=parameter_identity(model),call_owners=call_owners(qwen),
        execution_identity=execution_identity(qwen,'original',kwargs),
        config_resolution=config_resolution(model_dir,base_model_dir,qwen))
    original=source_evaluation_functions(dataset_root=dataset_path.parent)
    rows=[json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    observations=[]
    for index in indices:
        if time.monotonic()>=deadline:
            raise TimeoutError('aggregate witness time budget exhausted')
        sample=dict(rows[index])
        sample['image_input']=resolve_evaluation_image_paths(sample.get('image_input',[]),dataset_path.parent)
        traced_processor=ProcessorWitness(processor)
        calls=[]
        hook=qwen.model.register_forward_pre_hook(lambda module,args,kw:calls.append(_decoder_record(module,args,kw)),with_kwargs=True)
        sequences=[]
        generate=model.generate
        def capture_generate(**kw):
            result=generate(**kw)
            sequences.append(result.detach().cpu().tolist())
            return result
        model.generate=capture_generate
        try:
            if mode=='source':
                result=original['run_one_example'](model,traced_processor,sample,kwargs)
            else:
                from eval import run_one_example
                result=run_one_example(model,traced_processor,sample,kwargs,evaluation_protocol='original')
        finally:
            model.generate=generate
            hook.remove()
        prompt_length=traced_processor.calls[0]['inputs']['input_ids']['shape'][1]
        suffix=sequences[0][0][prompt_length:]
        eos=qwen.generation_config.eos_token_id
        eos=[eos] if isinstance(eos,int) else eos or []
        extracted=result['extracted_final_answer']
        gold=sample.get('original_final_answer','')
        observations.append(dict(index=index,prompt=traced_processor.calls,
            generated_suffix_token_ids=suffix,assistant_only=result['assistant_only'],
            extracted_final_answer=extracted,raw_output=result['raw_output'],
            stop_reason='eos' if suffix and suffix[-1] in eos else 'max_new_tokens',
            match=original['normalize_for_match'](extracted)==original['normalize_for_match'](gold),
            decoder_calls=calls))
    del model,qwen
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dict(metadata=metadata,observations=observations)


def compare(source,restored):
    """Return observed differences; equality alone does not prove block coverage."""
    differences=[]
    for field in ('parameters','execution_identity'):
        if source['metadata'][field]!=restored['metadata'][field]:
            differences.append('metadata.'+field)
    for index,(left,right) in enumerate(zip(source['observations'],restored['observations'])):
        for field in left:
            if left[field]!=right[field]:
                differences.append(f'sample[{index}].{field}')
    if len(source['observations'])!=len(restored['observations']):
        differences.append('sample_count')
    return differences


def _covered_source_indices(payload):
    latent_size = payload['metadata']['execution_identity'].get('model_config', {}).get('latent_size')
    if type(latent_size) is not int or latent_size <= 0:
        return []
    covered = []
    for row in payload['observations']:
        tokens = row['generated_suffix_token_ids']
        for position, token in enumerate(tokens):
            if (token == 151666 and position + latent_size + 2 < len(tokens)
                    and tokens[position + 1:position + latent_size + 1] == [151665] * latent_size
                    and tokens[position + latent_size + 1] == 151667):
                covered.append(row['index'])
                break
    return covered


def comparison_verdict(source, restored):
    """Separate exact observed equality from a covered natural latent transition."""
    differences = compare(source, restored)
    coverage = dict(source=_covered_source_indices(source), restored=_covered_source_indices(restored))
    covered = bool(set(coverage['source']) & set(coverage['restored']))
    verdict = 'mismatch' if differences else 'pass' if covered else 'inconclusive'
    token_equality = [
        (row['index'], row['generated_suffix_token_ids']) for row in source['observations']
    ] == [(row['index'], row['generated_suffix_token_ids']) for row in restored['observations']]
    return dict(verdict=verdict, parity=None if verdict == 'inconclusive' else verdict == 'pass',
        token_equality=token_equality, observed_trace_equality=not differences,
        first_nail_passed=verdict == 'pass', latent_transition_coverage=coverage, differences=differences)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',required=True)
    parser.add_argument('--base-model-dir')
    parser.add_argument('--test-data-path',required=True,type=Path)
    parser.add_argument('--indices',required=True)
    parser.add_argument('--max-new-tokens',type=int,default=512)
    parser.add_argument('--cache-dir',default='data/cache')
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args(argv)
    indices=validate_budget(args.indices,args.max_new_tokens)
    root=Path(__file__).resolve().parents[2]
    args.output_dir.resolve().relative_to((root/'results').resolve())
    if args.output_dir.exists():
        raise ValueError('witness output already exists; use a fresh run ID')
    args.output_dir.mkdir(parents=True)
    deadline=time.monotonic()+600
    outputs={}
    for mode in ('source','restored'):
        outputs[mode]=run_mode(mode,model_dir=args.model_dir,base_model_dir=args.base_model_dir,
            dataset_path=args.test_data_path.resolve(strict=True),indices=indices,cap=args.max_new_tokens,
            cache_dir=args.cache_dir,deadline=deadline)
        (args.output_dir/f'{mode}.json').write_text(json.dumps(outputs[mode],sort_keys=True,allow_nan=False)+'\n')
    report=comparison_verdict(outputs['source'],outputs['restored'])
    report.update(source_indices=indices, max_new_tokens=args.max_new_tokens,
        scope='bounded witness only; not full TEST400 parity')
    (args.output_dir/'comparison.json').write_text(json.dumps(report,sort_keys=True)+'\n')
    print(json.dumps(report,sort_keys=True))
    return {'pass':0, 'mismatch':1, 'inconclusive':2}[report['verdict']]


if __name__=='__main__':
    raise SystemExit(main())
