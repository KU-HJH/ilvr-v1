# Task 6 report — non-shared checkpoint shard collection

## Status

Implemented and committed as `781ca82 feat: collect run-owned DeepSpeed checkpoint shards`.

## Result

- Added lstat-only `ArtifactEntry` and `NodeArtifactManifest` construction and
  strict JSON decoding. Symlinks, non-regular files, traversal/control
  characters, foreign process ranks, invalid completion markers, incompatible
  identities, and incomplete global ranks are rejected without opening or
  hashing checkpoint contents.
- Added canonical DeepSpeed shard-rank inference for optimizer/model state and
  rank RNG filenames. Collection requires exactly process ranks 0 through 3,
  one node manifest per fixed node, one run ID, and one strict runtime identity.
- Added small-common-metadata classification. Equal-sized JSON/control state is
  accepted only under an identical runtime identity; weight-like duplicates
  remain ambiguous and fail even when byte sizes match.
- Added exact NUL-delimited rsync file lists and fixed argv-safe push/pull
  commands using `--archive --ignore-existing --files-from --from0`, fixed
  noninteractive SSH, canonical local/remote checkpoint subtrees, and no
  checksum/content reads.
- Added destination precondition validation. Existing ranked/weight
  destinations fail; only explicitly validated, equal-sized small common
  metadata can preexist. `--ignore-existing` remains defense in depth.
- Added resume identity validation for stage, tuning method, ZeRO stage/config
  digest, world/rank mapping, global step, optimizer, scheduler, RNG ranks,
  Stage sidecar, commit, and run ID. Resume selection contains only ranks 2/3
  plus required small common metadata, and tests prove identity validation runs
  before destination inspection or transfer.
- Added fresh inference-copy preparation that validates the remote manifest,
  rejects existing/symlink destinations, and transfers only manifest-listed
  paths.
- Reused Task 5's trusted dirfd/no-follow/atomic-no-replace publication path via
  `publish_small_file()` for JSON manifests and rsync file lists.
- Integrated shard collection after both owned node exits are zero and before
  the success/incomplete witness decision. Transfer failures propagate and no
  source tree is deleted or moved.

## TDD evidence

1. Initial RED: `tests/multinode/test_artifacts.py` failed collection with
   `ModuleNotFoundError: No module named 'src.multinode_artifacts'`.
2. Manifest/transfer GREEN: `30 passed in 0.04s`.
3. Coordinator/publication RED: imports failed for missing preparation and
   publication interfaces; the concrete subprocess-runner test then failed
   because `SubprocessCoordinatorRunner` lacked `read_artifact_manifest`.
4. Coordinator/publication GREEN: `35 passed in 0.06s`, followed by the bounded
   node-CLI and rsync-list test passing.
5. CLI RED: both metadata-only subcommands failed as invalid CLI choices.
6. CLI GREEN: `2 passed in 0.23s`.
7. Security self-review RED: three tests exposed equal-size weight acceptance
   and missing NUL mode; two further tests exposed untrusted manifest paths.
8. Security GREEN: the focused causal tests passed, then the full requested
   suite passed.

## Verification

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/transformers/src:$PWD" \
  ./venv/ilvr-venv/bin/pytest \
  tests/multinode/test_artifacts.py tests/multinode/test_coordinator.py -q

83 passed in 0.54s
```

`git diff --check` and `git diff --cached --check` were clean before commit.
The three modified/created Python modules also passed repository-venv
`py_compile`.

## Self-review and concerns

- All checkpoint inspection is metadata-only; JSON/control metadata is the only
  payload read or digested.
- Source/destination roots are fixed to the run-owned canonical checkpoint
  subtree, and rsync commands contain no delete/move/source/data/venv sync.
- Tests make no real SSH, rsync, Docker, tmux, process-group, CUDA, or GPU calls.
- Remaining external concern: real OpenSSH/rsync/container filesystem behavior
  still requires the planned hardware integration run; unit tests deliberately
  use fake command runners.
- Pre-existing untracked `__pycache__` state was preserved and not committed.

## Fix round 1/5 — strict checkpoint contract and production transfers

Committed as `03625d5 fix: enforce complete multinode checkpoints`.

### Review findings addressed

- Replaced rank-set-only acceptance with an exact checkpoint contract:
  `checkpoint-N` and `global_stepN` must match the strict identity, each rank
  0–3 has exactly one accepted `bf16_`, `fp16_`, or unprefixed optimizer shard
  and one top-level RNG state, and model state follows the explicit replicated
  or partitioned layout. Scheduler, trainer state, runtime metadata, and the
  stage/method sidecars are mandatory.
- Replaced extension-based metadata selection with a stage/method/ZeRO-keyed
  allowlist. Inference uses a separate top-level inference-only allowlist and
  excludes optimizer, model-state, RNG, scheduler, trainer, runtime, and
  training-sidecar files.
- Made all identity/manifest scalar and collection types exact. Booleans no
  longer satisfy integer fields; JSON rank-map keys and values, completion,
  owner ranks, run ID, checkpoint relative path, and canonical directory are
  bound before a transfer.
- Rebuilt the remote rank-0 canonical manifest after rsync and compared the
  exact expected path/size/process-rank union before returning collection
  success or permitting witness publication.
- Added production resume and inference transfer methods on the subprocess
  coordinator. Resume validates the remote canonical manifest and full identity
  before any training process starts, reserves a fresh local rank-2/3 copy,
  executes an exact NUL files list, and validates the transferred metadata.
  Inference preparation is callable before Task 8 evaluation and securely
  reserves, rechecks, transfers, and validates only inference allowlisted files.
- Scoped rsync sources to the exact numbered checkpoint directory. Fixed
  logical checkpoint-root links are mapped to their physical trusted storage
  root for dirfd reservation and no-overwrite validation; arbitrary symlink
  ancestors remain rejected. Original source trees are never moved or deleted.

### Fix-round TDD evidence

1. RED: the new strict-type, prefix/layout, incomplete-checkpoint,
   post-rsync, inference allowlist/race, and production-transfer tests initially
   failed at import and then produced `27 failed, 36 passed` while the old weak
   fixtures and helper-only implementation remained.
2. GREEN: causal subsets proved all three optimizer prefixes, RNG-only and
   partial-state rejection, step mismatch, exact types, explicit metadata and
   inference selection, destination reservation/race detection, exact
   post-rsync comparison, and concrete fake-runner resume/inference rsync lists.
3. Final focused verification:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/transformers/src:$PWD" \
  ./venv/ilvr-venv/bin/pytest -q \
  tests/multinode/test_artifacts.py tests/multinode/test_coordinator.py

115 passed in 0.73s
```

`git diff --check` was clean. No real SSH, rsync, Docker, tmux, CUDA, GPU, or
process-group operation was used; the concrete runner tests inspect commands
and materialize only temporary fake transfer results. The remaining external
concern is the planned hardware validation of the fixed SSH/container and
logical-to-physical checkpoint storage mapping. Pre-existing untracked
`__pycache__` state remains uncommitted.

## Fix round 2/5 — observed identity and loadable same-path artifacts

### Review findings addressed

- Removed the caller-selectable model-state layout. The coordinator derives
  replicated state for ZeRO-1/2 and partitioned state for ZeRO-3, while each
  node independently verifies that observed DeepSpeed filenames agree.
- Bound checkpoint identity to producer evidence: numbered `checkpoint-N`,
  `latest`, `trainer_state.json`, the strict DeepSpeed runtime sidecar, strict
  Stage 1/2 PEFT metadata, or the unique full-checkpoint Stage sidecar plus the
  existing full serialization classifier. Stage 2 full checkpoints also pass
  the strict sampler parser and agree on step and world size. Caller JSON cannot
  relabel malformed or conflicting producer metadata.
- Bound manifest run ID to the no-follow canonical ancestor. The fixed
  checkpoint-root symlink remains supported, while symlinked run/checkpoint
  descendants and non-immediate resume paths are rejected.
- Changed resume redistribution to the same immediate logical
  `<checkpoint-root>/<run-id>/checkpoint-N` path on both node namespaces. A
  fresh local manifest proves retained same-run paths and sizes; only missing
  local ranks 2/3, explicit control metadata, and loader-required
  full/LoRA/DoRA/config/processor files are pulled without overwrite. Both
  node-local manifests are revalidated before any training process starts.
- Added complete inference artifact filtering and validation. PEFT copies carry
  strict stage metadata, adapter config/weights, Stage 1 or conditional Stage 2
  sparse rows, and present config/processor files. Full copies accept one
  supported unsharded family or a parsed index with the exact complete sibling
  shard set; partial and mixed serializations fail. Training state is excluded.
- Added an executable `prepare-inference` coordinator CLI and a bounded remote
  `artifact-inference-manifest` node CLI. Help is local and side-effect free;
  the production runner rebuilds, identity-binds, reserves, transfers, and
  verifies the filtered manifest before an evaluator can be created.

### Fix-round TDD evidence

1. RED: layout derivation initially failed at import; the new observed-sidecar,
   inference-completeness, same-path resume, and CLI tests then produced five
   passing and three failing causal cases while the round-1 implementation still
   relied on caller layout and helper-only transfer paths.
2. RED: after the first integration pass, the complete artifact suite exposed
   ten fixture/integration incompatibilities; focused repair reduced that to
   two concrete fake-runner mismatches and then `80 passed`.
3. RED: added independent run-ancestor, logical resume, filtered-manifest,
   latest/trainer, DeepSpeed runtime, ZeRO-layout, strict full-index, and Stage 2
   sampler counterexamples. Four of eight identity/path cases failed before the
   production bindings; the malformed sampler causal test also failed with the
   old caller-labeled stage path.
4. GREEN: all causal subsets passed after binding the manifest run ancestor,
   retaining the logical root symlink, validating filtered remote identity, and
   using strict producer parsers/classifiers. No checkpoint or weight contents
   are read or hashed.

### Verification and self-review

The final focused suite, CLI help check, and diff checks are recorded immediately
before the round-2 commit. Tests use fake command runners and temporary files;
no real SSH, rsync, Docker, tmux, process group, CUDA, or GPU operation is used.
Original remote/local checkpoint trees are never deleted or moved, including on
failure. The remaining external concern is hardware validation of the fixed
host/container mapping and rsync behavior. Pre-existing untracked `__pycache__`
state remains preserved and uncommitted.

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src:$PWD/transformers/src:$PWD" \
  ./venv/ilvr-venv/bin/pytest -q \
  tests/multinode/test_artifacts.py tests/multinode/test_coordinator.py

133 passed in 6.59s
```

`python -m src.multinode prepare-inference --help` and `git diff --check`
both passed in the same final verification command.

## Fix round 3/5 — mandatory loader artifacts and pinned inspection

### Review findings addressed

- Made the complete loader asset family mandatory for collection, resume, and
  inference: `config.json`, `preprocessor_config.json`,
  `tokenizer_config.json`, and either `tokenizer.json` or the paired
  `vocab.json`/`merges.txt` serialization. A weights-only artifact is rejected.
- Covered all six Stage 1/Stage 2 × Full/LoRA/DoRA rows with tiny fixtures. The
  production validator mirrors the existing classifiers using exact filenames,
  complete full serialization/index sibling sets, strict small PEFT metadata,
  and conditional Stage 2 sparse-row markers without reading weight payloads.
  The tests separately invoke the real Stage classifiers on the tiny fixtures
  and prove the same Full/PEFT classification for every row.
- Required Stage 1 PEFT adapter config, safetensors adapter, strict Stage 1
  metadata, and sparse token rows. Stage 2 sparse rows are present exactly when
  its strict metadata declares `sparse_adapter`; unsupported legacy adapter and
  mixed/unknown weight markers fail closed.
- Reused the same completeness selection for inference while excluding
  training state. Full inference still requires one supported standalone family
  or a parsed index whose exact sibling shard set is complete.
- Closed the checkpoint-directory swap window. Canonical root/run/checkpoint
  traversal opens no-follow dirfds, then one pinned checkpoint descriptor stays
  authoritative for recursive enumeration and all bounded sidecar/config reads.
  The node inference CLI filters the already validated manifest and does not
  reopen the checkpoint pathname after releasing that descriptor.
- Added a successful fake `prepare-inference` CLI dispatch test proving the
  concrete runner call, canonical checkpoint path, strict identity, and fresh
  destination are wired without any remote or hardware action.

### Fix-round TDD evidence

1. RED: the initial six-row loader tests failed to import the missing
   `validate_loader_artifact`; after the interface was added, mandatory marker
   cases reached `18 passed, 90 deselected`.
2. RED: the causal pathname replacement test changed the observed runtime
   digest from the opened checkpoint's value to the attacker's replacement
   value. GREEN: pinned no-follow inspection remained bound to the opened inode,
   followed by `109 passed` for the artifact suite at that checkpoint.
3. RED: weights-only full inference and a mismatched config stage were accepted.
   GREEN: the bounded config/processor/tokenizer contract and stage binding
   rejected both.
4. RED: two legacy `adapter_model.bin` counterexamples passed the metadata-only
   PEFT mirror. GREEN: both Stage mirrors now reject unsupported/unknown markers.
5. GREEN: a `Path.read_bytes` guard rejects every `.pt`, `.pth`,
   `.safetensors`, and `.bin` read while a strict sparse Stage 2 PEFT artifact
   validates successfully; actual tensor decoding occurs only in the separate
   tiny-fixture classifier characterization test.

### Final verification and self-review

```text
PYTHONDONTWRITEBYTECODE=1 ./venv/ilvr-venv/bin/pytest -q \
  tests/multinode/test_artifacts.py tests/multinode/test_coordinator.py

159 passed in 6.68s
```

`PYTHONDONTWRITEBYTECODE=1 ./venv/ilvr-venv/bin/python -m src.multinode
prepare-inference --help` returned zero and listed the bounded stage/method/ZeRO,
run, port, step, and destination arguments. `git diff --check` was clean.

Self-review confirmed that production inspection reads only bounded JSON/text
identity metadata and stats/names; it never opens model, adapter, optimizer,
RNG, or sparse-row payloads. Tests use only fake runners and temporary tiny
fixtures: no SSH, rsync, Docker, tmux, process group, CUDA, or GPU operation was
performed. Source checkpoint trees are not moved or deleted. The fixed-host
OpenSSH/rsync and real Stage loader behavior remain intentionally deferred to
the planned hardware integration run. Pre-existing untracked `__pycache__`
state was preserved and remains uncommitted.
