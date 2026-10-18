# Stage1-v5 GPU optimization final verification record

Reviewed source: `3e51bf79b6a89fd3c8763a63b682d33e85526bcb`.
Independent whole-change reviewer: `/root/gpu_util_final_review`, final PASS,
no Critical/Important/Minor findings. Review traced the integrated actual
Trainer/native attention/selector/replay/checkpoint/dropout/resume paths and
checked complete underlying evidence. No further implementation changes.
This controller record does not change runtime or test sources.

The root announced verification-before-completion and
finishing-a-development-branch. Finishing Step1's full project suite is also
the selected verification command for the overlapping test-pass claim. It
was launched once on the reviewed source, with an unused checkpoint basetemp,
explicit real-training smoke opt-ins disabled and two visible idle H100 NVLs.
Resource precheck tool transcript d42fd6 showed 0MiB/0percent and no apps on
both devices. The source worktree was clean at launch.

```bash
env CUDA_VISIBLE_DEVICES=0,1 ILVR_REAL_GPU_TESTS=0 RUN_SHARED_PREFIX_DDP=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transformers/src:src:. ./venv/ilvr-venv/bin/python -m pytest -q --basetemp=checkpoints/stage1-v5-gpu-utilization-20261018-tests/final-full-suite > logs/stage1-v5-gpu-utilization-20261018/final-full-suite.log 2>&1
```

Full-suite result: exit0, **5272 passed, 11 skipped, 1 warning in1286.34s**
(21min26s). Root read all82lines/6544bytes of the complete output. The sole
warning is Torch cpp_extension's unset TORCH_CUDA_ARCH_LIST notice in the
existing CPU-Adam mapping test. No test failed. The full suite ran once and
is reused for finishing Step1's overlapping test-pass claim; no repeated
suite or additional model-training smoke was launched.

Full log SHA256:
`a9e9181e4ae2508b406bd83b58d06c9638bc1f51b415280f7092fbd0317a0c7d`.
Artifact-check log SHA256:
`b9a9b7fbaa996ab3a70d869ea640709eac89a98981467803dd9ff7dada137bed`.
Root additionally ran
`git diff --exit-code 3e51bf79b6a89fd3c8763a63b682d33e85526bcb -- src scripts tests transformers/src`,
exit0, after tests. Only controller evidence records differ from the reviewed
source; no runtime/test changes require another test cycle.

The distinct artifact/measurement claim was checked once by the following
read-only program; it does not rerun benchmarks or tests. Exit0, complete
output read from `logs/stage1-v5-gpu-utilization-20261018/final-artifact-check.log`.
It matched all twelve runtime hashes and eight immutable hashes, all four
complete report/stdout pairs, 640 raw timing values and their medians, 2772
ordered-key rows, paired order, transfer/memory arithmetic, seven-case
correctness flags, restored long boundary, current benchmark/fixture bytes,
predeclared tolerances and unchanged original failure digests. CUDA-default
attention remains close rather than bitwise. No production BF16 or distributed
training performance is established.

```bash
./venv/ilvr-venv/bin/python - <<'PY' > logs/stage1-v5-gpu-utilization-20261018/final-artifact-check.log 2>&1
import hashlib,json,statistics,subprocess
from pathlib import Path
root=Path.cwd()
run=root/'results/stage1-v5-gpu-utilization-20261018'
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
evidence=run/'task4-completion-evidence.json'
assert digest(evidence)=='0d0dce514d24014bb2683c91a61097e651c979443a233c87d60bd2079aa20c39'
e=json.loads(evidence.read_text())
for name,path in e['runtime_identity']['source_paths'].items():
    assert digest(path)==e['runtime_identity']['source_sha256'][name],name
repair=json.loads((run/'task3-numerical-repair-evidence.json').read_text())
for path,sha in repair['immutable_sha256'].items():
    assert digest(root/path)==sha,path
operators=('attention','selection','replay_mask','dropout')
flags=('attention_close','joint_selected_keys_exact_equal','selected_keys_exact_equal','mask_exact_equal','dropout_exact_equal','dropout_global_rng_unchanged')
rows=times=0
for m in e['measurements']:
    assert m['exit_code']==0
    report=root/m['report']['path']
    log=root/m['full_stdout_log']['path']
    for item in (m['report'],m['full_stdout_log']):
        p=root/item['path']
        assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256'],p
    assert report.read_bytes()==log.read_bytes()
    j=json.loads(report.read_text())
    assert j['warmup']==3 and j['repetitions']==20
    assert j['attention_atol']==1e-7 and j['attention_rtol']==1e-5
    assert all(j[f] is True for f in flags)
    assert j['operator_profile']=={'id':'stage1_v5_vectorized_ops_v1','attention_query_chunk_max':32,'attention_score_probability_target_bytes':33554432}
    identity=j['source_identity']
    for name,path in identity['candidate_paths'].items():
        assert digest(path)==identity['candidate_sha256'][name],name
    assert digest(root/'scripts/benchmark_stage1_v5_attention_selection.py')==identity['benchmark_sha256']
    assert digest(root/'tests/stage1/v5/gpu_utilization_fixtures.py')==identity['fixtures_sha256']
    for op in operators:
        assert j[op+'_paired_order']==[['before','after'] if i%2==0 else ['after','before'] for i in range(20)]
        for side in ('before','after'):
            d=j[op+'_'+side]
            assert len(d['milliseconds'])==20 and all(t>0 for t in d['milliseconds'])
            assert statistics.median(d['milliseconds'])==d['median_ms']
            times+=len(d['milliseconds'])
            mem=d['memory']
            if 'extra_peak_allocated_bytes' in mem:
                assert mem['baseline_allocated_bytes']+mem['extra_peak_allocated_bytes']==mem['peak_allocated_bytes']
            counts=d['operations']
            assert sum(x['bytes'] for x in counts['tolist_transfers'])==counts['tolist_bytes']
            assert sum(x['elements'] for x in counts['tolist_transfers'])==counts['tolist_elements']
    assert len(j['joint_selection_cases'])==7
    for c in j['joint_selection_cases'].values():
        assert c['exact'] and c['selector_on_identical_after_rows_exact']
        for row in c['keys']:
            assert row['before']==row['after']
            rows+=1
    if j['geometry_profile']=='long':
        boundary=next(x for x in j['joint_selection_cases']['cumulative_equal']['keys'] if x['query']==[1,8125])
        assert boundary['before']==boundary['after']==[289,1235]
    print(report.name,'PASS','max_attention_error=',j['attention_max_absolute_error'])
assert times==640 and rows==2772,(times,rows)
assert digest(run/'task4-evidence.json')=='be033964bab41b85ae1b2e82470cb5c8c3fcae5448894be4e4aea1ec970a0875'
assert digest(run/'task4-cpu-long-discrete-failure.json')=='cdb66d95c382c13e71db07958d855a2f2bbed248af8ea4b84190858d3ba4a27e'
print('PASS: 12 runtime hashes, 8 immutable hashes, 4 complete report/stdout pairs, 640 raw timings, 2772 ordered-key rows, original failure digests unchanged')
print('verified_revision',subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
PY
```

An exploratory schema inspection before the actual verifier exited1 because
it treated evidence.measurements as a dictionary instead of its actual list.
This changed no code/artifact/test or verification criterion; the above
single substantive artifact check uses the recorded list and passed.

## Requirements and finishing disposition

The four approved operator tasks and actual runtime identity/admission are
implemented, independently specification/quality reviewed, and separately
whole-change reviewed. The full project command includes the original long
selection counterexample, frozen comparisons and native loss/gradient/
checkpoint/replay integration. Operator performance is supported only by the
four preserved paired reports and this current-source artifact check. The
two regressions and memory increases remain explicitly disclosed; production
BF16, future-input discrete equality and actual two-node/four-rank throughput
remain unverified. Losses, Teacher/reference, sampler/rank assignment and
existing checkpoints retain their reviewed scope; no fresh training ran.

Finishing Steps2/3 ran only after the green full-suite result. Observed Git
dir `/root/workspace/ILVR/.git/worktrees/ILVR3` differs from common dir
`/root/workspace/ILVR/.git`; symbolic-ref has no branch, so HEAD is detached.
`git merge-base HEAD main` gives baseline65643f08cb6e61fbbb200de825d57086f5c7a127.
This is a host-managed worktree: preserve it, with no merge, push, PR or
deletion performed. Present the required reduced three-option handoff to the
human: push as new branch/PR, keep as-is, or discard. Integration choice is
pending; implementation and verification are complete.

After the suite, root removed only29 test-generated .pyc files in four newly
untracked src __pycache__ directories, first asserting every entry was a
regular .pyc file. No product source, checkpoints, user files, worktree or
skill records were deleted. The only committed post-review changes are
controller workflow/plan/verification evidence.

