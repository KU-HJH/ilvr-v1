import hashlib
import json
import zipfile
import pytest

from evaluation.schema import ContractError
from visulogic_fixtures import make_source, pin_source, write_rows


def prepare(case):
    from evaluation.visulogic_prepare import prepare_visulogic
    return prepare_visulogic(case['source'], case['output'], repo_root=case['root'])


def test_pinned_prepare_preserves_579_rows_choices_gold_and_image_bytes(monkeypatch):
    from evaluation.io import parse_config, read_manifest
    case = make_source(); pin_source(monkeypatch, case)
    before = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in case['source'].iterdir()}
    summary = prepare(case)
    assert summary['N_expected'] == 579
    assert summary['categories'] == dict(Position=136, Quantity=353, Style=90)
    assert summary['unselected_categories'] == {'Spatial Reasoning':231,'Attribute Reasoning':82,'Other':108}
    raw = [json.loads(line) for line in (case['output']/'manifest.jsonl').read_text().splitlines()]
    expected_categories = ['Position']*136 + ['Quantity']*353 + ['Style']*90
    for i, (row, category) in enumerate(zip(raw, expected_categories)):
        sid = f'{i:05}'
        assert row['id'] == f'visulogic/{category}/{sid}'
        assert row['model_input'] == dict(text=case['rows'][i]['question'], images=[f'images/{sid}.png'],
                                          choices=['first','second','third','fourth'])
        assert row['target'] == dict(index=1,label='B',content='second')
        assert row['provenance'] == dict(dataset_id='VisuLogic/VisuLogic',revision='3e483f8cca0bc766ecc5f70299f2b8a34b5035ca',
            config='default',split='pqs579-v1',source_id=sid,asset_sha256={f'images/{sid}.png':hashlib.sha256(case['png']).hexdigest()})
        assert (case['output']/f'images/{sid}.png').read_bytes() == case['png']
    assert len(list((case['output']/'images').iterdir())) == 579
    config = parse_config(dict(schema_version=1,benchmark='visulogic',profile='visulogic_vlmeval_rules_v1',
        categories=['Position','Quantity','Style'],dataset_root=summary['dataset_root'],
        run_metadata=dict(model_identity='synthetic:test',prompt_template_identity='synthetic:question',generation_config={})),case['root'])
    parsed = read_manifest(case['output']/'manifest.jsonl',config,case['root'])
    assert len(parsed.samples) == 579 and parsed.manifest_sha256 == summary['manifest_sha256']
    assert before == {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in case['source'].iterdir()}
    with pytest.raises(ContractError): prepare(case)


@pytest.mark.parametrize('defect', ['revision','source_hash','archive_hash','duplicate_id','category','count','label',
    'choice_order','choice_empty','extra_field','path_escape','duplicate_image','missing_image','corrupt_image','outside_data'])
def test_bad_source_fails_before_output(monkeypatch,defect):
    case = make_source()
    if defect == 'revision':
        p=case['source']/'dataset-info.json';p.write_text(p.read_text().replace('3e483f8cca0bc766ecc5f70299f2b8a34b5035ca','unknown'))
    elif defect in ('source_hash','archive_hash'):
        pin_source(monkeypatch,case)
        p=case['source']/('data.jsonl' if defect=='source_hash' else 'images.zip');p.write_bytes(p.read_bytes()+b' ')
    elif defect in ('missing_image','corrupt_image'):
        with zipfile.ZipFile(case['source']/'images.zip','w') as z:
            for row in case['rows']:
                if row['id']=='00000' and defect=='missing_image':continue
                z.writestr(row['image_path'], b'broken' if row['id']=='00000' else case['png'])
    elif defect == 'outside_data':case['output']=case['root']/'forbidden'
    else:
        row=case['rows'][0]
        if defect=='duplicate_id': row['id']=case['rows'][1]['id']
        if defect=='category': row['tag']='Unknown'
        if defect=='count': case['rows'].pop()
        if defect=='label': row['label']='AB'
        if defect=='choice_order': row['question']=row['question'].replace('B: second','E: second')
        if defect=='choice_empty': row['question']=row['question'].replace('B: second','B: ')
        if defect=='extra_field': row['rationale']='SECRET GOLD'
        if defect=='path_escape': row['image_path']='../outside.png'
        if defect=='duplicate_image': row['image_path']=case['rows'][1]['image_path']
        write_rows(case['source']/'data.jsonl',case['rows'])
    if defect not in ('source_hash','archive_hash'):pin_source(monkeypatch,case)
    with pytest.raises(ContractError): prepare(case)
    assert not case['output'].exists()


def test_png_named_source_can_contain_decodable_jpeg_bytes(monkeypatch):
    # Real pinned archive contains JPEG-encoded members with .png names.
    from io import BytesIO
    from PIL import Image
    case=make_source()
    buffer=BytesIO();Image.new('RGB',(2,2),(17,34,51)).save(buffer,format='JPEG')
    jpeg=buffer.getvalue()
    with zipfile.ZipFile(case['source']/'images.zip','w') as archive:
        for row in case['rows']:archive.writestr(row['image_path'],jpeg)
    pin_source(monkeypatch,case)
    assert prepare(case)['N_expected']==579
    assert (case['output']/'images/00000.png').read_bytes()==jpeg
