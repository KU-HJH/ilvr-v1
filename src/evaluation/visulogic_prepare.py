"""Prepare the pinned Position/Quantity/Style subset without rewriting questions."""
from collections import Counter
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

from PIL import Image

from evaluation.benchmarks.visulogic import LABELS, validate_target
from evaluation.io import exact_keys, read_jsonl_bytes, reject_constant, unique_object
from evaluation.schema import ContractError, McqTarget, ModelInput, Provenance, Sample, canonical_json

DATASET_ID = 'VisuLogic/VisuLogic'
SOURCE_REVISION = '3e483f8cca0bc766ecc5f70299f2b8a34b5035ca'
SOURCE_SHA256 = '969ff4407a433f7282d52635dc4f30bdeacd6ac0e4ea31cf604b85313c8562c0'
ARCHIVE_SHA256 = '86157164e4ee61296ed923dca61c115f08ba0633ca7c973029dbec3e0edf555b'
CATEGORY_MAP = {'Positional Reasoning':'Position', 'Quantitative Reasoning':'Quantity', 'Stylistic Reasoning':'Style'}
SOURCE_COUNTS = {'Positional Reasoning':136, 'Quantitative Reasoning':353, 'Stylistic Reasoning':90,
                 'Spatial Reasoning':231, 'Attribute Reasoning':82, 'Other':108}


def _inside(path: Path, parent: Path) -> Path:
    try:
        result = path.resolve()
        result.relative_to(parent)
    except (OSError, ValueError) as exc:
        raise ContractError('path must remain inside its data/source root') from exc
    return result


def _read(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ContractError('cannot read source input') from exc


def _choices(question: str) -> tuple[str, ...]:
    matches = list(re.finditer(r'(?m)^([ABCD]):[ \t]*', question))
    if tuple(m.group(1) for m in matches) != LABELS:
        raise ContractError('question must contain one ordered A/B/C/D option block')
    choices = tuple(question[m.end():matches[i+1].start() if i < 3 else len(question)].strip()
                    for i, m in enumerate(matches))
    if not all(choices):
        raise ContractError('question options must be nonempty')
    return choices


def prepare_visulogic(source_root: Path, output: Path, *, repo_root: Path) -> dict:
    data = (Path(repo_root).resolve() / 'data').resolve()
    source_root = _inside(Path(source_root), data)
    output = _inside(Path(output), data)
    if output.exists() or not source_root.is_dir():
        raise ContractError('source must exist and output must be fresh')
    metadata_bytes = _read(_inside(source_root / 'dataset-info.json', source_root))
    try:
        metadata = json.loads(metadata_bytes, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (UnicodeError, ValueError) as exc:
        raise ContractError('invalid source metadata') from exc
    if not isinstance(metadata, dict) or metadata.get('id') != DATASET_ID or metadata.get('sha') != SOURCE_REVISION:
        raise ContractError('source identity/revision must match pinned VisuLogic')
    payload = _read(_inside(source_root / 'data.jsonl', source_root))
    archive_path = _inside(source_root / 'images.zip', source_root)
    archive_bytes = _read(archive_path)
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
        raise ContractError('VisuLogic source SHA-256 mismatch')
    if hashlib.sha256(archive_bytes).hexdigest() != ARCHIVE_SHA256:
        raise ContractError('VisuLogic archive SHA-256 mismatch')
    rows = read_jsonl_bytes(payload)
    ids, paths = set(), set()
    counts = Counter()
    for row in rows:
        exact_keys(row, ('id','image_path','question','label','tag'))
        if any(not isinstance(row[k], str) or not row[k].strip() for k in row):
            raise ContractError('source fields must be nonempty strings')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', row['id']) or row['id'] in ids:
            raise ContractError('source IDs must be safe and unique')
        image = PurePosixPath(row['image_path'])
        if (len(image.parts) != 2 or image.parts[0] != 'images' or image.suffix != '.png'
                or image.as_posix() != row['image_path'] or '\\' in row['image_path']
                or row['image_path'] in paths):
            raise ContractError('source images must be unique images/<file>.png paths')
        if row['tag'] not in SOURCE_COUNTS or row['label'] not in LABELS:
            raise ContractError('unknown source category or label')
        ids.add(row['id']); paths.add(row['image_path']); counts[row['tag']] += 1
    if counts != SOURCE_COUNTS:
        raise ContractError('source must have the pinned 1000-row category population')

    samples, assets, selection = [], {}, []
    try:
        with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ContractError('archive contains duplicate names')
            for line, row in enumerate(rows, 1):
                if row['tag'] not in CATEGORY_MAP:
                    continue
                category = CATEGORY_MAP[row['tag']]
                choices = _choices(row['question'])
                target = McqTarget(LABELS.index(row['label']), row['label'], choices[LABELS.index(row['label'])])
                image = row['image_path']
                info = archive.getinfo(image)
                if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                    raise ContractError('selected image must be a regular archive file')
                blob = archive.read(info)
                with Image.open(BytesIO(blob)) as decoded:
                    decoded.load()
                model_input = ModelInput(row['question'], (image,), choices)
                validate_target(target, model_input)
                sample = Sample(1, f"visulogic/{category}/{row['id']}", 'visulogic', category,
                                model_input, target, Provenance(DATASET_ID, SOURCE_REVISION, 'default',
                                'pqs579-v1', row['id'], {image:hashlib.sha256(blob).hexdigest()}))
                samples.append(asdict(sample)); assets[image] = blob
                selection.append(dict(source_line=line, source_id=row['id'], source_tag=row['tag'], category=category))
    except ContractError:
        raise
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, RuntimeError) as exc:
        raise ContractError('invalid archive or selected image') from exc

    manifest = b''.join(canonical_json(row) + b'\n' for row in samples)
    summary = dict(N_expected=len(samples), categories=dict(Counter(row['subtask'] for row in samples)),
                   unselected_categories={k:v for k,v in counts.items() if k not in CATEGORY_MAP},
                   excluded_selected_rows=0, dataset_id=DATASET_ID, revision=SOURCE_REVISION,
                   dataset_profile='visulogic-pqs579-v1', config='default', split='pqs579-v1',
                   dataset_root=(Path('data') / output.relative_to(data)).as_posix(),
                   source_sha256=SOURCE_SHA256, archive_sha256=ARCHIVE_SHA256,
                   manifest_sha256=hashlib.sha256(manifest).hexdigest(),
                   protocol_kind='modified', paper_protocol_verified=False, ood_verified='unknown')
    # All source rows, selected options and image bytes passed preflight above.
    output.mkdir(parents=True, exist_ok=False)
    (output / 'images').mkdir()
    for name, blob in assets.items(): (output / name).write_bytes(blob)
    (output / 'manifest.jsonl').write_bytes(manifest)
    source_record = dict(schema_version=1, **summary,
                         source_root=(Path('data') / source_root.relative_to(data)).as_posix())
    (output / 'SOURCE.json').write_bytes(canonical_json(source_record) + b'\n')
    (output / 'source-selection.json').write_bytes(canonical_json(dict(schema_version=1, rows=selection,
        unselected_categories=summary['unselected_categories'], source_sha256=SOURCE_SHA256)) + b'\n')
    return summary
