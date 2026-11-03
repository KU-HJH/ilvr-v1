"""Synthetic VisuLogic data; fixtures never depend on downloaded datasets."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import uuid

from evaluation.schema import RunMetadata, run_fingerprint

CATEGORIES = ['Position', 'Quantity', 'Style']


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def make_case(count=3):
    root = Path(__file__).resolve().parents[2]
    run_id = 'p4-fixture-' + uuid.uuid4().hex
    data = root / 'data/evaluation' / run_id
    results = root / 'results/evaluation' / run_id
    data.mkdir(parents=True)
    results.mkdir(parents=True)
    metadata = RunMetadata('synthetic:visulogic', 'synthetic:question', {})
    rows = [dict(schema_version=1, id=f'visulogic/{CATEGORIES[i % 3]}/s{i}',
                 benchmark='visulogic', subtask=CATEGORIES[i % 3],
                 model_input=dict(text='Synthetic question', images=[], choices=['first', 'second', 'third', 'fourth']),
                 target=dict(index=1, label='B', content='second'),
                 provenance=dict(dataset_id='synthetic:visulogic', revision='fixture', config='default',
                                 split='fixture', source_id=f's{i}', asset_sha256={})) for i in range(count)]
    manifest = data / 'manifest.jsonl'
    write_rows(manifest, rows)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    predictions = [dict(id=r['id'], manifest_sha256=digest, run_fingerprint=run_fingerprint(metadata),
                        response='B', status='ok') for r in rows]
    pred = results / 'predictions.jsonl'
    write_rows(pred, predictions)
    config = dict(schema_version=1, benchmark='visulogic', profile='visulogic_vlmeval_rules_v1',
                  categories=CATEGORIES, dataset_root=str(data), run_metadata=asdict(metadata))
    return dict(root=root, data=data, results=results, manifest=manifest, predictions=pred,
                rows=rows, prediction_rows=predictions, config=config, run_id=run_id)


def make_source():
    import io
    import zipfile
    from PIL import Image
    parent = Path(__file__).resolve().parents[2]
    run_id = 'p4-source-fixture-' + uuid.uuid4().hex
    base = parent / 'data/evaluation' / run_id
    root = base / 'repo'
    root.mkdir(parents=True)
    (base / 'repo-data/evaluation').mkdir(parents=True)
    (root / 'data').symlink_to(base / 'repo-data', target_is_directory=True)
    for name in ('results', 'logs', 'checkpoints'):
        (root / name).symlink_to(parent / name, target_is_directory=True)
    source = root / 'data/evaluation/source'
    source.mkdir()
    image = io.BytesIO()
    Image.new('RGB', (2, 2), (17, 34, 51)).save(image, format='PNG')
    png = image.getvalue()
    tags = [('Positional Reasoning',136), ('Quantitative Reasoning',353), ('Stylistic Reasoning',90),
            ('Spatial Reasoning',231), ('Attribute Reasoning',82), ('Other',108)]
    rows = []
    for tag, count in tags:
        for _ in range(count):
            sid = f'{len(rows):05}'
            rows.append(dict(id=sid, image_path=f'images/{sid}.png', label='B', tag=tag,
                             question=f'Synthetic question {sid}\nA: first\nB: second\nC: third\nD: fourth'))
    write_rows(source / 'data.jsonl', rows)
    (source / 'dataset-info.json').write_text(json.dumps(dict(id='VisuLogic/VisuLogic', sha='3e483f8cca0bc766ecc5f70299f2b8a34b5035ca')))
    with zipfile.ZipFile(source / 'images.zip', 'w') as archive:
        for row in rows: archive.writestr(row['image_path'], png)
    return dict(root=root, source=source, output=root / 'data/evaluation' / run_id, rows=rows, png=png, run_id=run_id)


def pin_source(monkeypatch, case):
    from evaluation import visulogic_prepare as module
    for name, filename in [('SOURCE_SHA256','data.jsonl'),('ARCHIVE_SHA256','images.zip')]:
        monkeypatch.setattr(module, name, hashlib.sha256((case['source']/filename).read_bytes()).hexdigest())
