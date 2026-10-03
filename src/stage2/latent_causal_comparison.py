"""Offline paired model differences on identical sample/block support."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from .latent_causal_intervention import _atomic_write_text, summarize


def compare_runs(high_rows, gap_rows, high_metadata, gap_metadata):
    """Entropy-gap minus high-entropy, bootstrap samples with fixed interventions."""
    for field in ('seed','K','selected_sample_indices','donor_mapping',
                  'shuffle_permutations','sample_metadata','data_sha256'):
        if high_metadata.get(field)!=gap_metadata.get(field):
            raise ValueError(f'paired run plan differs: {field}')
    if high_metadata.get('status')!='complete' or gap_metadata.get('status')!='complete':
        raise ValueError('paired comparison requires two completed runs')
    def indexed(rows):
        result={}
        for row in rows:
            key=(row['sample_id'],row['block_index'],row['condition'])
            if key in result:
                raise ValueError('duplicate paired evaluation unit')
            result[key]=row
        return result
    high,gap=indexed(high_rows),indexed(gap_rows)
    if set(high)!=set(gap):
        raise ValueError('paired runs have different evaluation units')
    grouped=defaultdict(list)
    exclusions=defaultdict(lambda:defaultdict(int))
    for key,a in high.items():
        b=gap[key]; condition=key[2]
        if a['status']!='ok' or b['status']!='ok':
            exclusions[condition][f"high:{a['status']};gap:{b['status']}"]+=1
            continue
        for field in ('K','target_token_count','target_ids','donor_sample_id',
                      'source_block_index','permutation','changed_embedding_positions','masked_positions'):
            if a.get(field)!=b.get(field):
                raise ValueError(f'paired target/intervention mismatch: {field}')
        delta=b['delta_nll']-a['delta_nll']
        if not math.isfinite(delta):
            raise ValueError('nonfinite paired delta')
        grouped[condition].append(dict(sample_id=key[0],block_index=key[1],status='ok',delta_nll=delta))
    conditions=sorted({k[2] for k in high})
    common=set.intersection(*(set((r['sample_id'],r['block_index']) for r in grouped[c])
                             for c in conditions)) if conditions else set()
    seed=high_metadata['seed']
    return dict(status='complete',difference='entropy_gap minus high_entropy',units='nat/token',
        interpretation='Positive intervention delta harms GT prediction; paired positive means larger harm in entropy_gap. '
                       'This is not final-answer accuracy or proof of visual reasoning.',
        conditions={c:dict(summarize(grouped[c],seed),excluded_pairs=dict(exclusions[c])) for c in conditions},
        common_conditions={c:summarize([r for r in grouped[c]
            if (r['sample_id'],r['block_index']) in common],seed) for c in conditions},
        common_support_blocks=len(common),bootstrap_pairing='subtract aligned block deltas, then average per sample and bootstrap samples',
        bootstrap_replicates=2000,bootstrap_seed=seed,donors_and_shuffle_fixed=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--high-entropy',required=True,help='Completed causal result directory')
    parser.add_argument('--entropy-gap',required=True,help='Completed causal result directory')
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2]
    paths=[Path(p).resolve() for p in (args.high_entropy,args.entropy_gap,args.output_dir)]
    if any(not p.is_relative_to((repo/'results').resolve()) for p in paths):
        raise ValueError('comparison inputs and output must be inside this worktree results tree')
    high,gap,output=paths
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('paired output directory is not empty')
    def read(path):
        metadata=json.loads((path/'summary.json').read_text())
        rows=[json.loads(line) for line in (path/'per_block.jsonl').read_text().splitlines()]
        return rows,metadata
    a,ma=read(high); b,mb=read(gap)
    result=compare_runs(a,b,ma,mb)
    result.update(high_entropy=str(high),entropy_gap=str(gap))
    _atomic_write_text(output/'summary.json',json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
