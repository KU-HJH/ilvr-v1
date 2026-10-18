The `original` Stage 1 evaluation protocol restores ILVR commit
`13a67ed2ee975d6ca56b4e9eba18d68b352b166b`. It uses private, mechanically pinned
Qwen modeling and GenerationMixin snapshots; it does not change global Transformer
classes or the Stage 2/continuous implementation. Source hashes, integration-only
adaptations, shared support hashes, and retained notices are recorded in
`original-provenance.json`.

Actual `eval.py` CLI and multinode workers resolve `auto` to `original` for Stage 1
and `continuous` for Stage 2. Programmatic `run_one_example`, `load_model`, and
`load_evaluation_artifacts` retain their existing `continuous` default. Stage 1 OFF,
KV-cache, and static-cache diagnostics require explicit `continuous`. Stage 2's
entrypoint accepts `auto`/`continuous`; it rejects `original`.

Original generation inherits the loaded generation config, including repetition
penalty, EOS IDs and cache policy. An explicit repetition argument overrides that
field. `native` per-token max-new-token/EOS termination preserves incomplete latent
blocks. Continuous limit policies are rejected for `original`.
`max_sequence_length` is a preflight resource guard: prompt length plus the requested
native token cap must fit, and it does not become a second generation stopping rule.
The automatic Stage 1 recipe selects original/native and inherits repetition.

PEFT plus sparse input/output rows is an artifact-representation extension. It
restores the same unmerged DoRA/row parameters with the existing validated loader.
As in original Transformers' adapter redirect, model/generation configuration comes
from the original base. A saved training config is recorded separately; its
`use_cache=False` is neither relabelled as residue nor silently overwritten. Full
artifacts retain their own resolved configs. The original source's cache-off latent
assertion is preserved and covered by a CPU test.

Primary original scoring uses full raw decode, original assistant extraction,
original answer parsing/normalization, and the all-source denominator, including
legacy empty/missing-gold matches. Corrected parser fields are nested as
`corrected_scoring_diagnostic`; they do not replace the primary metric.
Historical unlabelled manifests remain readable as continuous. New manifests carry
protocol, resolved config and concrete attention classes; merging different or
unknown execution identities fails.

`python -m evaluation.original_witness --help` describes the independent bounded
source-vs-restored probe. Its source side reconstructs and hash-checks pristine
source, executes separate original model/sampler classes and original eval functions,
and uses the same unmerged artifact loader. It records actual parameter hashes,
method owners, prompt tensors, decoder inputs, exact suffix/answer/stop/match, config
resolution and backend classes. It saves metadata only. ROOT must execute it after
review with an external 600-second timeout, at most three distinct source indices,
and at most 512 generated tokens per side/sample. CPU contracts and source hashing
alone do not establish parity for trained GPU evaluation or TEST400.
