import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import pytest
import yaml

from visulogic_fixtures import make_case, make_source, pin_source

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / 'configs/eval/visulogic_vlmeval_rules_v1.yaml'


def args(case):
    return ['--benchmark','visulogic','--profile','visulogic_vlmeval_rules_v1',
            '--source-root',str(case['source']),'--output-dir',str(case['output'])]


def test_prepare_dispatch_recipe_and_unwanted_comt_paths(monkeypatch):
    from scripts import prepare_eval
    from evaluation.io import parse_config
    case = make_source(); pin_source(monkeypatch, case)
    monkeypatch.setattr(prepare_eval,'verify_source_reference',lambda *_: pytest.fail('VisuLogic entered VSP verification'))
    assert prepare_eval.main(args(case),repo_root=case['root']) == 0
    raw = yaml.safe_load(RECIPE.read_text()); raw['dataset_root'] = str(case['output'])
    config = parse_config(raw,case['root'])
    assert config.max_steps is None and config.categories == ('Position','Quantity','Style')
    assert config.benchmark == 'visulogic' and config.profile == 'visulogic_vlmeval_rules_v1'
    rejected = make_source()
    assert prepare_eval.main(args(rejected)+['--image-root',str(rejected['source'])],repo_root=rejected['root']) == 2
    assert not rejected['output'].exists()


def test_visulogic_generation_rejected_before_model_load(monkeypatch):
    from scripts import run_eval
    case = make_case(1)
    case['config']['run_metadata']['generation_config'] = dict(max_new_tokens=8,max_sequence_length=32768,
        repetition_penalty=1.05,temperature=0.0,top_p=1.0,do_sample=False)
    path=case['data']/'config.yaml';path.write_text(yaml.safe_dump(case['config']))
    calls=[];monkeypatch.setattr(run_eval,'load_runtime',lambda: calls.append(True))
    output=ROOT/'results/evaluation'/('p4-run-rejection-'+uuid.uuid4().hex)
    assert run_eval.main(['--manifest',str(case['manifest']),'--config',str(path),'--model-dir','checkpoints/not-needed',
        '--model-identity','synthetic:visulogic','--output-dir',str(output)],repo_root=ROOT) == 2
    assert not calls and not output.exists()


CHILD = r'''
import importlib.abc
import hashlib,json,sys
from pathlib import Path
import yaml
blocked={'torch','transformers','Reference','eval','evaluation.scoring','evaluation.runtime','evaluation.continuous_latent'}
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname in blocked or fullname.split('.')[0] in blocked: raise AssertionError(fullname)
sys.meta_path.insert(0,Guard())
from evaluation import visulogic_prepare as prepare_module
from evaluation.schema import RunMetadata,run_fingerprint
from scripts import prepare_eval,score_eval
root=Path.cwd();source=root/'data/evaluation/source';output=root/'data/evaluation'/sys.argv[1]
prepare_module.SOURCE_SHA256=hashlib.sha256((source/'data.jsonl').read_bytes()).hexdigest()
prepare_module.ARCHIVE_SHA256=hashlib.sha256((source/'images.zip').read_bytes()).hexdigest()
assert prepare_eval.main(['--benchmark','visulogic','--profile','visulogic_vlmeval_rules_v1',
    '--source-root',str(source),'--output-dir',str(output)])==0
manifest=output/'manifest.jsonl';digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
rows=[json.loads(x) for x in manifest.read_text().splitlines()]
expected_categories=['Position']*136+['Quantity']*353+['Style']*90
expected_ids=[f'visulogic/{category}/{i:05}' for i,category in enumerate(expected_categories)]
assert [r['id'] for r in rows]==expected_ids
source_rows=[json.loads(x) for x in (source/'data.jsonl').read_text().splitlines()]
for i,row in enumerate(rows):
    image=f'images/{i:05}.png'
    assert row['model_input']==dict(text=source_rows[i]['question'],images=[image],choices=['first','second','third','fourth'])
    assert row['target']==dict(index=1,label='B',content='second')
    assert row['provenance']==dict(dataset_id='VisuLogic/VisuLogic',revision='3e483f8cca0bc766ecc5f70299f2b8a34b5035ca',
        config='default',split='pqs579-v1',source_id=f'{i:05}',asset_sha256={image:sys.argv[2]})
    assert hashlib.sha256((output/image).read_bytes()).hexdigest()==sys.argv[2]
recipe=yaml.safe_load((root/'recipe.yaml').read_text())
recipe['dataset_root']=str(output)
recipe['run_metadata']=dict(model_identity='synthetic:visulogic',prompt_template_identity='synthetic:question',generation_config={})
config=output/'score-config.yaml';config.write_text(yaml.safe_dump(recipe))
fingerprint=run_fingerprint(RunMetadata(**recipe['run_metadata']))
inputs=root/'results/evaluation'/(sys.argv[1]+'-inputs');inputs.mkdir()
predictions=inputs/'predictions.jsonl'
# Literal B responses for this explicitly B-labelled synthetic fixture, never copied from targets.
pred=[dict(id=sid,manifest_sha256=digest,run_fingerprint=fingerprint,response='B',status='ok') for sid in expected_ids]
predictions.write_text(''.join(json.dumps(x)+'\n' for x in pred))
score=root/'results/evaluation'/(sys.argv[1]+'-score')
common=['--manifest',str(manifest),'--predictions',str(predictions),'--config',str(config)]
assert score_eval.main(common+['--output-dir',str(score)])==0
summary=json.loads((score/'summary.json').read_text())
assert summary['N_expected']==summary['observed']==summary['correct']==579 and summary['accuracy']==1.0
assert summary['manifest_sha256']==digest and summary['run_fingerprint']==fingerprint
assert summary['categories']=={c:dict(count=n,observed=n,correct=n,complete=True,accuracy=1.0) for c,n in [('Position',136),('Quantity',353),('Style',90)]}
predictions.write_text(''.join(json.dumps(x)+'\n' for x in pred[:-1]))
missing=root/'results/evaluation'/(sys.argv[1]+'-missing')
assert score_eval.main(common+['--output-dir',str(missing)])==1
partial=json.loads((missing/'summary.json').read_text())
assert partial['N_expected']==579 and partial['observed']==578 and partial['accuracy'] is None and not partial['complete']
assert not blocked.intersection(sys.modules)
origins={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items() if (name.startswith('evaluation') or name.startswith('scripts')) and getattr(module,'__file__',None)}
assert all(Path(p).is_relative_to(root) for p in origins.values())
assert not (root/'Reference').exists() and not (root/'eval.py').exists()
print(json.dumps(dict(complete=summary,partial=partial,module_origins=origins,model_run=False)))
'''


def test_copied_root_prepare_score_and_incomplete_without_reference():
    case=make_source();root=case['root']
    files=['src/evaluation/'+name for name in ['__init__.py','schema.py','io.py','registry.py','report.py',
           'vsp_prepare.py','visulogic_prepare.py','benchmarks/__init__.py','benchmarks/vsp.py','benchmarks/visulogic.py']]
    files+=['scripts/'+name for name in ['__init__.py','prepare_eval.py','score_eval.py']]
    for name in files:
        dst=root/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dst)
    shutil.copyfile(RECIPE,root/'recipe.yaml')
    logdir=ROOT/'logs/evaluation'/(case['run_id']+'-child');logdir.mkdir(parents=True)
    env=dict(os.environ,PYTHONPATH='src:.',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
    result=subprocess.run([sys.executable,'-c',CHILD,case['run_id'],hashlib.sha256(case['png']).hexdigest()],cwd=root,env=env,capture_output=True,text=True)
    (logdir/'integration.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0, result.stdout+result.stderr
    evidence=json.loads(result.stdout.splitlines()[-1])
    assert evidence['complete']['correct']==579 and evidence['partial']['accuracy'] is None
