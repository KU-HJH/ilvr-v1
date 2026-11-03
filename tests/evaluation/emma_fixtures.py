"""Independent synthetic EMMA fixtures."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import uuid

from evaluation.schema import RunMetadata, run_fingerprint

CATEGORIES = ['Chemistry', 'Coding', 'Math', 'Physics']


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def make_case(count=4):
    root = Path(__file__).resolve().parents[2]
    run_id = 'p5-emma-fixture-' + uuid.uuid4().hex
    data = root / 'data/evaluation' / run_id
    results = root / 'results/evaluation' / run_id
    data.mkdir(parents=True)
    results.mkdir(parents=True)
    metadata = RunMetadata('synthetic:emma', 'synthetic:question', {})
    rows = []
    for i in range(count):
        subject = CATEGORIES[i % len(CATEGORIES)]
        pid = f'{subject.lower()}_{i}'
        rows.append(dict(schema_version=1, id=pid, benchmark='emma', subtask=subject,
                         model_input=dict(text='Synthetic question', images=[],
                                          choices=['first', 'second', 'third', 'fourth']),
                         target=dict(question_type='multiple_choice', answer='b', index=1,
                                     content='second', category='synthetic', task='reasoning'),
                         provenance=dict(dataset_id='synthetic:emma', revision='fixture', config=subject,
                                         split='test', source_id=pid, asset_sha256={})))
    manifest = data / 'manifest.jsonl'
    write_rows(manifest, rows)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    prediction_rows = [dict(id=row['id'], manifest_sha256=digest,
                            run_fingerprint=run_fingerprint(metadata), response='B', status='ok')
                       for row in rows]
    predictions = results / 'predictions.jsonl'
    write_rows(predictions, prediction_rows)
    config = dict(schema_version=1, benchmark='emma', profile='emma_safe_rules_v1',
                  categories=CATEGORIES, dataset_root=str(data), run_metadata=asdict(metadata))
    return dict(root=root, data=data, results=results, manifest=manifest, predictions=predictions,
                config=config, rows=rows, prediction_rows=prediction_rows, run_id=run_id)
