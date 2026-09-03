# Final Stage 1 → Stage 2 Provenance Manifest

## Capture boundary

Understood as: publish a self-contained runtime and documentation lineage for
the consolidated checkout, with every included runtime path tied to source and
final Git blobs, and grade local evidence without upgrading historical reports.

This manifest was captured on 2026-09-03. It does not claim that historical
scores were rerun or that a complete Stage 2 evaluation artifact was recovered.

## Fixed source identities and patches

| Input | Identity |
|---|---|
| Main code baseline | `85e74e97de3773375ff037690511153c8d1f40cc` |
| Pinned Stage 2 source | `b0ce19b086cd46961fa9dcc12aa3844f659ff0b0` |
| Stage 1/Stage 2 merge base | `82d10d8d9ae26a288da42df18b0c5c655efb04ff` |
| Stage 1 textual tests patch SHA-256 | `511796ae812d6df0a07495fbbaa95d6cd7efc30e6d20da5f3bae51c05dbb3a89` |
| Stage 1 textual production patch SHA-256 | `92611aeaafe873f7fb10b6a9dba439157334ab2ca289e0e497dd3ee6d70b6ec6` |
| Stage 2 no-cache tests patch SHA-256 | `86601e93f202a3679b87e59e8a115b2dfd9306640743df3d8a80b288d8f4cbe1` |
| Stage 2 no-cache production patch SHA-256 | `72aa6d04660f2fbf0371cb203dfd622649a46a7378e8807e0027d79b25fe745f` |
| Stage 2 shared tests patch SHA-256 | `44ae09c3181751fea2ccaeeccf3e316094b93e050c53def92f92bb91b7f0799b` |
| Stage 2 shared production patch SHA-256 | `018c1a43e78136b7f36728c0d5ee0a0a183269c7a7cb5cf50b7f6b891747d505` |
| Approved uncommitted three-file PEFT patch SHA-256 | `0729ecd05ed8950a7c1c8d2ca74b2ba64a48172484e4a7e0ce3570cabe636557` |
| Worktree path-fix commit | `3893ec71c3559917f726248cae81980cec161a76` |
| Tracked Stage 2 PEFT selector commit | `caaf264a86b78277bf764ba240cd5197d3d40623` |
| Tracked Stage 2 PEFT selector patch SHA-256 | `efcadcaf30ea9c3a116bd6ae9653dd1e53024959f95682ee27f1860750313905` |

### PEFT provenance correction

Task 4 disproved the earlier premise that main already subsumed every pinned
Stage 2 PEFT selector fix. The tracked commit
`caaf264a86b78277bf764ba240cd5197d3d40623`, an ancestor of the pinned Stage 2
source, rejects unmatched compact selectors. Its exact two-file patch was
selectively ported into `src/stage2/peft.py` and
`tests/stage2/test_peft.py`; those rows use the `stage2-peft-port` source class.
That tracked port is distinct from the approved, uncommitted three-file dirty
PEFT patch, which was not applied wholesale.

## Runtime blob manifest

The inclusion command produced 109 unique runtime paths. The table contains 144
rows because a path is repeated whenever multiple source classes contributed.
Every source blob was resolved with `git rev-parse <commit>:<path>` (or the
captured working-tree blob for the dirty patch), and every final blob with
`git hash-object <path>`.

Source classes mean:

- `main-baseline`: retained behavior from the fixed main baseline.
- `stage1-textual-port`: the pinned Stage 1 textual-history contribution.
- `stage2-no-cache-port`: pinned Stage 2 no-cache/manifest lineage; on the two
  dirty-comparison-only files it records the separately available tracked Stage
  2 blob, not an assertion that those files implement no-cache behavior.
- `manual-three-way-merge`: one of the exact eight shared paths selectively
  merged from main and pinned Stage 2.
- `worktree-path-fix`: active-checkout path behavior from Task 5.
- `main-baseline/subsumes-dirty-patch`: captured dirty-patch input whose required
  successful-run behavior is preserved without applying the patch wholesale.
- `stage2-peft-port`: the controller-approved tracked selector validation port.

| Path | Source class | Source commit/patch | Source blob SHA-1 | Final blob SHA-1 | Reason |
|---|---|---|---|---|---|
| `configs/deepspeed_zero1.json` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `1751d1eb328fe6a400916c5210ca258728d1d00c` | `1751d1eb328fe6a400916c5210ca258728d1d00c` | Unchanged baseline runtime/support file. |
| `configs/deepspeed_zero2.json` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `d193d19ec4e24f9596a9789a5302b8a498a8b555` | `d193d19ec4e24f9596a9789a5302b8a498a8b555` | Unchanged baseline runtime/support file. |
| `configs/deepspeed_zero3.json` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `c38c80968f4770755974d5efcdb5032fe695468f` | `c38c80968f4770755974d5efcdb5032fe695468f` | Unchanged baseline runtime/support file. |
| `configs/deepspeed_zero3_stage2_peft.json` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `a5a970cf2cbcc8bae59c24944d5a8dccc748f287` | `a5a970cf2cbcc8bae59c24944d5a8dccc748f287` | Unchanged baseline runtime/support file. |
| `run_stage1_batch_probe.sh` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `48fd1cac6a07ba732b9942efd90faae359d00e2e` | `3ec0f2678deac96b9ffebad18a698df569203905` | Original main path/runtime behavior. |
| `run_stage1_batch_probe.sh` | `worktree-path-fix` | `3893ec71c3559917f726248cae81980cec161a76` | `3ec0f2678deac96b9ffebad18a698df569203905` | `3ec0f2678deac96b9ffebad18a698df569203905` | Active-checkout path derivation. |
| `run_stage1_training.sh` | `stage1-textual-port` | `ca30ab92e54e0c41fafaacc221a5120e0be8e6a5` | `e098464176d3ef6bdca5dae7738137b317f7b768` | `f6957a919224243ea3fa3994a0bed0e69be81be5` | Pinned Stage 1 textual-history contribution. |
| `run_stage1_training.sh` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `ea1130c50a2bfb511a9f0cd7c74d2d3f0b31e332` | `f6957a919224243ea3fa3994a0bed0e69be81be5` | Original main path/runtime behavior. |
| `run_stage1_training.sh` | `worktree-path-fix` | `3893ec71c3559917f726248cae81980cec161a76` | `f6957a919224243ea3fa3994a0bed0e69be81be5` | `f6957a919224243ea3fa3994a0bed0e69be81be5` | Active-checkout path derivation. |
| `run_stage1_zero_matrix.sh` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `e5b3c1e24bb5dcf79357ae20170a6341626d14f0` | `f3fb6718c088d885193bc508ea6ebf0a4ebfe405` | Original main path/runtime behavior. |
| `run_stage1_zero_matrix.sh` | `worktree-path-fix` | `3893ec71c3559917f726248cae81980cec161a76` | `f3fb6718c088d885193bc508ea6ebf0a4ebfe405` | `f3fb6718c088d885193bc508ea6ebf0a4ebfe405` | Active-checkout path derivation. |
| `run_stage2_training.sh` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `1af1e9af9449a31a097c609a418445af25b314ce` | `b7960e2e269ffe0e510579d78db536aabf4b02ec` | Later-main behavior retained in the selective merge. |
| `run_stage2_training.sh` | `manual-three-way-merge` | `b0ce19b086cd46961fa9dcc12aa3844f659ff0b0` | `00be469270a9e08e3bda3659c1bdb70063050fd3` | `b7960e2e269ffe0e510579d78db536aabf4b02ec` | Pinned Stage 2 behavior selectively merged with main. |
| `src/artifact_paths.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `c34f0f04c55dd348027f63b8ce094aedde17c90f` | `2e3b1e2e89b8dca56915bef0c29bc9b4c1eb7fe7` | Later-main behavior retained in the selective merge. |
| `src/artifact_paths.py` | `manual-three-way-merge` | `b0ce19b086cd46961fa9dcc12aa3844f659ff0b0` | `d456440bac7cdb51c1593272925a06d3f48c0899` | `2e3b1e2e89b8dca56915bef0c29bc9b4c1eb7fe7` | Pinned Stage 2 behavior selectively merged with main. |
| `src/artifact_paths.py` | `worktree-path-fix` | `3893ec71c3559917f726248cae81980cec161a76` | `2e3b1e2e89b8dca56915bef0c29bc9b4c1eb7fe7` | `2e3b1e2e89b8dca56915bef0c29bc9b4c1eb7fe7` | Active-checkout path derivation. |
| `src/main.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `d47061b1d8124d8dea9edefeb57f8a5956e4fb8a` | `d2614eb62f4a6ef81ac8a94f25f011650e56a608` | Later-main behavior retained in the selective merge. |
| `src/main.py` | `manual-three-way-merge` | `b0ce19b086cd46961fa9dcc12aa3844f659ff0b0` | `e83bdae4aa193a27eaaac69a6872949c79ad51c3` | `d2614eb62f4a6ef81ac8a94f25f011650e56a608` | Pinned Stage 2 behavior selectively merged with main. |
| `src/multinode.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `a7a1be3799424d62a23ff94bc11fd7b8a4c79b46` | `41eff0bc1aaba5c681d02a1bf4629c03de5249c5` | Original main path/runtime behavior. |
| `src/multinode.py` | `worktree-path-fix` | `3893ec71c3559917f726248cae81980cec161a76` | `41eff0bc1aaba5c681d02a1bf4629c03de5249c5` | `41eff0bc1aaba5c681d02a1bf4629c03de5249c5` | Active-checkout path derivation. |
| `src/multinode_artifacts.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `25bd200b074cb5d6ddc3b4ab24209c3eac3d80db` | `04ec0e1fd18052524f75b7dc57d3055e6ffca501` | Original main path/runtime behavior. |
| `src/multinode_artifacts.py` | `worktree-path-fix` | `3893ec71c3559917f726248cae81980cec161a76` | `04ec0e1fd18052524f75b7dc57d3055e6ffca501` | `04ec0e1fd18052524f75b7dc57d3055e6ffca501` | Active-checkout path derivation. |
| `src/multinode_capacity.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `826c87d4659e66edaad1119e08dcd1c27af0f9d0` | `f52e71b9a6bcc72c9c39890c99e739f0b0ed60f4` | Original main path/runtime behavior. |
| `src/multinode_capacity.py` | `worktree-path-fix` | `3893ec71c3559917f726248cae81980cec161a76` | `f52e71b9a6bcc72c9c39890c99e739f0b0ed60f4` | `f52e71b9a6bcc72c9c39890c99e739f0b0ed60f4` | Active-checkout path derivation. |
| `src/multinode_evidence.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `7df51fa5dd5e98934adc3c060627d07ce58c3b6e` | `471db9106f73eed27d45e80d76daaa9ebec630f6` | Original main path/runtime behavior. |
| `src/multinode_evidence.py` | `worktree-path-fix` | `3893ec71c3559917f726248cae81980cec161a76` | `471db9106f73eed27d45e80d76daaa9ebec630f6` | `471db9106f73eed27d45e80d76daaa9ebec630f6` | Active-checkout path derivation. |
| `src/multinode_results.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `1f989126c153c681e104f25a6be0601903ced211` | `12f0f5e8fc9cce372e1858b74295d98770188848` | Original main path/runtime behavior. |
| `src/multinode_results.py` | `worktree-path-fix` | `3893ec71c3559917f726248cae81980cec161a76` | `12f0f5e8fc9cce372e1858b74295d98770188848` | `12f0f5e8fc9cce372e1858b74295d98770188848` | Active-checkout path derivation. |
| `src/multinode_runtime.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `7faafd384db02520ee70c3264e8573bc00b93843` | `0dd6b46737782ad2d7c41cb3bc67c6bca5a970c7` | Original main path/runtime behavior. |
| `src/multinode_runtime.py` | `worktree-path-fix` | `3893ec71c3559917f726248cae81980cec161a76` | `0dd6b46737782ad2d7c41cb3bc67c6bca5a970c7` | `0dd6b46737782ad2d7c41cb3bc67c6bca5a970c7` | Active-checkout path derivation. |
| `src/stage1/__init__.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `183cda803ff249ae8c627d5593744370e955f159` | `183cda803ff249ae8c627d5593744370e955f159` | Unchanged baseline runtime/support file. |
| `src/stage1/benchmark.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `a9df2d8422d3f985e992f341d266372fa7c0c848` | `a9df2d8422d3f985e992f341d266372fa7c0c848` | Unchanged baseline runtime/support file. |
| `src/stage1/checkpointing.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `34a17110d9da338c46aa7ae0dfdfbaedcd7dce74` | `34a17110d9da338c46aa7ae0dfdfbaedcd7dce74` | Unchanged baseline runtime/support file. |
| `src/stage1/composition.py` | `stage1-textual-port` | `ca30ab92e54e0c41fafaacc221a5120e0be8e6a5` | `53326cd9c868137cee787ec57ef5bfe9e2df0dec` | `8de4167bc514e0644f5b4ed774441e0934f2af2b` | Pinned Stage 1 textual-history contribution. |
| `src/stage1/data.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `c4418d46618f771b57faa2e9d6a8bbcb9124a8f8` | `c4418d46618f771b57faa2e9d6a8bbcb9124a8f8` | Unchanged baseline runtime/support file. |
| `src/stage1/ema.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `24298fcdc49267c169462bdc2c229939bb208ec5` | `24298fcdc49267c169462bdc2c229939bb208ec5` | Unchanged baseline runtime/support file. |
| `src/stage1/equivalence.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `d4eefbfacacc0c85149a3ec5ce7f9d0acf11415c` | `d4eefbfacacc0c85149a3ec5ce7f9d0acf11415c` | Unchanged baseline runtime/support file. |
| `src/stage1/evaluation_diagnostics.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `76bd419138af99a47d96e41c7a59e47e9c5b61c0` | `76bd419138af99a47d96e41c7a59e47e9c5b61c0` | Unchanged baseline runtime/support file. |
| `src/stage1/peft.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `e94ee3c080b40da2d5aa12618e4b14af58a969d0` | `e94ee3c080b40da2d5aa12618e4b14af58a969d0` | Unchanged baseline runtime/support file. |
| `src/stage1/targets.py` | `stage1-textual-port` | `ca30ab92e54e0c41fafaacc221a5120e0be8e6a5` | `543d36a1fd9b36f0b48d6806f43fa717311403dd` | `c0cf88c5002a4deae2b40ef86e604387602dd116` | Pinned Stage 1 textual-history contribution. |
| `src/stage1/token_rows.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `9010a41dd4e8adf001a57d22930d0e91bf36e0ce` | `9010a41dd4e8adf001a57d22930d0e91bf36e0ce` | Unchanged baseline runtime/support file. |
| `src/stage1/trainer.py` | `stage1-textual-port` | `ca30ab92e54e0c41fafaacc221a5120e0be8e6a5` | `e793b5b2d482bb70633d43e244bf336d2b59313c` | `db41046f385dfa4d366043fbea51431ee9c91559` | Pinned Stage 1 textual-history contribution. |
| `src/stage1/training_evidence.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `70417db00bcc15fa8dbcf0406f242ba1c9444bd0` | `70417db00bcc15fa8dbcf0406f242ba1c9444bd0` | Unchanged baseline runtime/support file. |
| `src/stage1/types.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `b6b3a1a6473f9f97e37ee7a70cbb46296cb418cc` | `b6b3a1a6473f9f97e37ee7a70cbb46296cb418cc` | Unchanged baseline runtime/support file. |
| `src/stage1/zero_matrix.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `ba1c071a0b8965e8fc3d9db4a0331f0aeab66d10` | `ba1c071a0b8965e8fc3d9db4a0331f0aeab66d10` | Unchanged baseline runtime/support file. |
| `src/stage2/__init__.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `de8447130205daa54ac38773e4498e87e0cf4d32` | `de8447130205daa54ac38773e4498e87e0cf4d32` | Unchanged baseline runtime/support file. |
| `src/stage2/backend_fallback.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `2e94ada0ebd42ec4aaea1c461fb566e56650bab5` | `2e94ada0ebd42ec4aaea1c461fb566e56650bab5` | Unchanged baseline runtime/support file. |
| `src/stage2/checkpointing.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `d99f2767d86454a4c09095296998a32b5ac7ff1b` | `d99f2767d86454a4c09095296998a32b5ac7ff1b` | Main behavior retained in final runtime. |
| `src/stage2/checkpointing.py` | `stage2-no-cache-port` | `b0ce19b086cd46961fa9dcc12aa3844f659ff0b0` | `f3b9fcea067d72f6ff86cfcbf351d7da85bdb6bf` | `d99f2767d86454a4c09095296998a32b5ac7ff1b` | Tracked pinned Stage 2 lineage used for dirty-patch comparison. |
| `src/stage2/checkpointing.py` | `main-baseline/subsumes-dirty-patch` | `0729ecd05ed8950a7c1c8d2ca74b2ba64a48172484e4a7e0ce3570cabe636557` | `ff90d2bd18a6fb4484516b3aede242d5e2e22246` | `d99f2767d86454a4c09095296998a32b5ac7ff1b` | Approved uncommitted PEFT patch behavior preserved semantically; patch not applied wholesale. |
| `src/stage2/data.py` | `stage2-no-cache-port` | `b0ce19b086cd46961fa9dcc12aa3844f659ff0b0` | `62236a006081c7950d54180c5d4e8ae2bbf48a7c` | `1c5407aa8259372b3ab4da523aae6cb407cc1a44` | Pinned Stage 2 no-cache/manifest contribution. |
| `src/stage2/efficiency_probe.py` | `stage2-no-cache-port` | `b0ce19b086cd46961fa9dcc12aa3844f659ff0b0` | `79cd2294ff514b169a1f2566f57bd57a2921d253` | `79cd2294ff514b169a1f2566f57bd57a2921d253` | Pinned Stage 2 no-cache/manifest contribution. |
| `src/stage2/evidence.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `62066cbda555bc87c83637c711f8d1c45436cd01` | `62066cbda555bc87c83637c711f8d1c45436cd01` | Unchanged baseline runtime/support file. |
| `src/stage2/inference.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `dd86a8b9171801dd5bb09058252e4ec93bc0f488` | `dd86a8b9171801dd5bb09058252e4ec93bc0f488` | Unchanged baseline runtime/support file. |
| `src/stage2/length_manifest.py` | `stage2-no-cache-port` | `b0ce19b086cd46961fa9dcc12aa3844f659ff0b0` | `df44929260046b76dbacd6e9bf34039a41df4a11` | `df44929260046b76dbacd6e9bf34039a41df4a11` | Pinned Stage 2 no-cache/manifest contribution. |
| `src/stage2/peft.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `16007e22a089b064e312d0e3276c7d1c3e833b4a` | `0fb353327b037c0232349237baeb164c9aa7a15b` | Original main PEFT behavior. |
| `src/stage2/peft.py` | `stage2-peft-port` | `caaf264a86b78277bf764ba240cd5197d3d40623` | `7f5e92e43b09770a63b5faa375431f6720d56828` | `0fb353327b037c0232349237baeb164c9aa7a15b` | Tracked unmatched-selector rejection port; distinct from dirty patch. |
| `src/stage2/sampler.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `3967570d24c01e69d383b11d0a42549c769dd999` | `3967570d24c01e69d383b11d0a42549c769dd999` | Unchanged baseline runtime/support file. |
| `src/stage2/trainer.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `2a810e36322605477cdd7fc4bae3df4afa241865` | `00f6a3edbbb52cf5bc1f2182fc700ce46a54a738` | Later-main behavior retained in the selective merge. |
| `src/stage2/trainer.py` | `manual-three-way-merge` | `b0ce19b086cd46961fa9dcc12aa3844f659ff0b0` | `b41de51ad93cd8a951e28ca2c43bde868908a9ea` | `00f6a3edbbb52cf5bc1f2182fc700ce46a54a738` | Pinned Stage 2 behavior selectively merged with main. |
| `src/stage2/trainer.py` | `main-baseline/subsumes-dirty-patch` | `0729ecd05ed8950a7c1c8d2ca74b2ba64a48172484e4a7e0ce3570cabe636557` | `c0c66059a56d75010dfb35ec47bd230952bffd5e` | `00f6a3edbbb52cf5bc1f2182fc700ce46a54a738` | Approved uncommitted PEFT patch behavior preserved semantically; patch not applied wholesale. |
| `src/utils_deepseed.py` | `stage1-textual-port` | `ca30ab92e54e0c41fafaacc221a5120e0be8e6a5` | `0b8676ce1beb527b6cf60fbd5040e7b3ea8ccb84` | `c76be6d117528eb1e8f3c98f100c75e0fef08403` | Pinned Stage 1 textual-history contribution. |
| `src/utils_deepseed.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `993b37233a52caa154e63a436b78ec3126532782` | `c76be6d117528eb1e8f3c98f100c75e0fef08403` | Later-main behavior retained in the selective merge. |
| `src/utils_deepseed.py` | `manual-three-way-merge` | `b0ce19b086cd46961fa9dcc12aa3844f659ff0b0` | `66c1d14583b036907c36edf08b48ddd01e3482c3` | `c76be6d117528eb1e8f3c98f100c75e0fef08403` | Pinned Stage 2 behavior selectively merged with main. |
| `tests/multinode/__init__.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `dfaa2c71e48fb69e2e4d7fdddfdd0a5a6b06175f` | `dfaa2c71e48fb69e2e4d7fdddfdd0a5a6b06175f` | Unchanged baseline runtime/support file. |
| `tests/multinode/test_artifacts.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `9aa1b9a6297936b6a7257a92425f7929aa425651` | `c775c71299964c179cfafda97443c231723f75df` | Original main path/runtime behavior. |
| `tests/multinode/test_artifacts.py` | `worktree-path-fix` | `3893ec71c3559917f726248cae81980cec161a76` | `c775c71299964c179cfafda97443c231723f75df` | `c775c71299964c179cfafda97443c231723f75df` | Active-checkout path derivation. |
| `tests/multinode/test_capacity.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `00beecff5d23680fe7972a364b9bb43e6e986f5b` | `c6de52eb1d6a6f4f4fa1dd907725c6f24f9b7b2a` | Original main path/runtime behavior. |
| `tests/multinode/test_capacity.py` | `worktree-path-fix` | `3893ec71c3559917f726248cae81980cec161a76` | `c6de52eb1d6a6f4f4fa1dd907725c6f24f9b7b2a` | `c6de52eb1d6a6f4f4fa1dd907725c6f24f9b7b2a` | Active-checkout path derivation. |
| `tests/multinode/test_coordinator.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `39f2bdf54ad3a3f91e775e7cb2cd0829fe3b6a6f` | `c03ab75e8af3253355d8ee8953a05a5e69ac4bcd` | Original main path/runtime behavior. |
| `tests/multinode/test_coordinator.py` | `worktree-path-fix` | `3893ec71c3559917f726248cae81980cec161a76` | `c03ab75e8af3253355d8ee8953a05a5e69ac4bcd` | `c03ab75e8af3253355d8ee8953a05a5e69ac4bcd` | Active-checkout path derivation. |
| `tests/multinode/test_evaluation.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `2f8c8330bfd6dd048f986c970ed1dd72cea4b320` | `2f8c8330bfd6dd048f986c970ed1dd72cea4b320` | Unchanged baseline runtime/support file. |
| `tests/multinode/test_evaluation_coordinator.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `d5a43aece558fef3d2af50f6f56c6a7c02e8bbc4` | `d5a43aece558fef3d2af50f6f56c6a7c02e8bbc4` | Unchanged baseline runtime/support file. |
| `tests/multinode/test_lifecycle_matrix.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `701bbb54f1f390d842e34fe701e538b179d7d207` | `553d4054cb6de3ac77d09564045c83ccf1b91104` | Original main path/runtime behavior. |
| `tests/multinode/test_lifecycle_matrix.py` | `worktree-path-fix` | `3893ec71c3559917f726248cae81980cec161a76` | `553d4054cb6de3ac77d09564045c83ccf1b91104` | `553d4054cb6de3ac77d09564045c83ccf1b91104` | Active-checkout path derivation. |
| `tests/multinode/test_operator_timestamps.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `d516a9bf6f561e9444d5979b15a394c0af71fa7f` | `d516a9bf6f561e9444d5979b15a394c0af71fa7f` | Unchanged baseline runtime/support file. |
| `tests/multinode/test_preflight.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `b2aa23f7939f8b59638cd36ba8ab497313d0db78` | `98aa936073cc7d90a035c44fb2f236d93de74e76` | Original main path/runtime behavior. |
| `tests/multinode/test_preflight.py` | `worktree-path-fix` | `3893ec71c3559917f726248cae81980cec161a76` | `98aa936073cc7d90a035c44fb2f236d93de74e76` | `98aa936073cc7d90a035c44fb2f236d93de74e76` | Active-checkout path derivation. |
| `tests/multinode/test_result_validation.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `23a2f0580eec3d59c496b97c960118213cd219a6` | `5b08a75acea14b8ddaa58226fa249b31cfc3e9f2` | Original main path/runtime behavior. |
| `tests/multinode/test_result_validation.py` | `worktree-path-fix` | `3893ec71c3559917f726248cae81980cec161a76` | `5b08a75acea14b8ddaa58226fa249b31cfc3e9f2` | `5b08a75acea14b8ddaa58226fa249b31cfc3e9f2` | Active-checkout path derivation. |
| `tests/multinode/test_topology.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `11b4c7a8c40cdf5a92732ae0f30df8479f8e92c9` | `94024da6b804ca0f518598d5dd57d7f4c67e3623` | Original main path/runtime behavior. |
| `tests/multinode/test_topology.py` | `worktree-path-fix` | `3893ec71c3559917f726248cae81980cec161a76` | `94024da6b804ca0f518598d5dd57d7f4c67e3623` | `94024da6b804ca0f518598d5dd57d7f4c67e3623` | Active-checkout path derivation. |
| `tests/stage1/__init__.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `3930b7d2d24911a1a80a0d663131834d13a22084` | `3930b7d2d24911a1a80a0d663131834d13a22084` | Unchanged baseline runtime/support file. |
| `tests/stage1/helpers.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `8912c4487f57eeb726ec03b3c4149e54929bc20f` | `8912c4487f57eeb726ec03b3c4149e54929bc20f` | Unchanged baseline runtime/support file. |
| `tests/stage1/test_benchmark.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `cbfcddeb46438ddcbc1e584a38b6250032039991` | `cbfcddeb46438ddcbc1e584a38b6250032039991` | Unchanged baseline runtime/support file. |
| `tests/stage1/test_checkpointing.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `677236ccb9fa2f9cc7a1af75ec1997d5a171991a` | `677236ccb9fa2f9cc7a1af75ec1997d5a171991a` | Unchanged baseline runtime/support file. |
| `tests/stage1/test_cli.py` | `stage1-textual-port` | `ca30ab92e54e0c41fafaacc221a5120e0be8e6a5` | `3ed324fc72dc8a50c317294d8dd0b004a9a79a59` | `a677e89e133326c6aa44db4e08e7542f4fee7ffc` | Pinned Stage 1 textual-history contribution. |
| `tests/stage1/test_composition.py` | `stage1-textual-port` | `ca30ab92e54e0c41fafaacc221a5120e0be8e6a5` | `d2a1026be7a058c9712fdd4414bee1988ebbe40d` | `12ba626ba89d9c414b625ae6977df2582d207271` | Pinned Stage 1 textual-history contribution. |
| `tests/stage1/test_data.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `215eee73b01100188d6997284cb25221ab4cb585` | `215eee73b01100188d6997284cb25221ab4cb585` | Unchanged baseline runtime/support file. |
| `tests/stage1/test_distributed_smoke.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `a6d9fb297fc16265a909778f22e1dd7cde65fbd1` | `a6d9fb297fc16265a909778f22e1dd7cde65fbd1` | Unchanged baseline runtime/support file. |
| `tests/stage1/test_ema.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `5be2020c54c40b615aff0f4f7c3216d7a8f14d6f` | `5be2020c54c40b615aff0f4f7c3216d7a8f14d6f` | Unchanged baseline runtime/support file. |
| `tests/stage1/test_equivalence.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `45b59f4c51fc8f00ec38da9345e5faf218f16227` | `45b59f4c51fc8f00ec38da9345e5faf218f16227` | Unchanged baseline runtime/support file. |
| `tests/stage1/test_evaluation_diagnostics.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `3ef8d6e5a6f26bd51356d4a41f418fdefe54d6a7` | `3ef8d6e5a6f26bd51356d4a41f418fdefe54d6a7` | Unchanged baseline runtime/support file. |
| `tests/stage1/test_integration.py` | `stage1-textual-port` | `ca30ab92e54e0c41fafaacc221a5120e0be8e6a5` | `13f9a6da55c1d85c6e61b0c91616f459cf34ac64` | `03159b4ce0231aaa6b4249d6f8788bbf7f490b0d` | Pinned Stage 1 textual-history contribution. |
| `tests/stage1/test_launcher.py` | `stage1-textual-port` | `ca30ab92e54e0c41fafaacc221a5120e0be8e6a5` | `049d74e5b9495e50f99780eaf7badbb6064bda42` | `c46853df8bb785372afeba3ed1d338780f332b49` | Pinned Stage 1 textual-history contribution. |
| `tests/stage1/test_peft.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `18f3226b5d362fa6a48b5896dad29101935e0189` | `18f3226b5d362fa6a48b5896dad29101935e0189` | Unchanged baseline runtime/support file. |
| `tests/stage1/test_qwen_latent_injection.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `61b3823fb08c7dabe2b55ffddb232970d292b98d` | `61b3823fb08c7dabe2b55ffddb232970d292b98d` | Unchanged baseline runtime/support file. |
| `tests/stage1/test_qwen_vision_rotary.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `b36fa92824e68acbfeca4346206ff22516627672` | `b36fa92824e68acbfeca4346206ff22516627672` | Unchanged baseline runtime/support file. |
| `tests/stage1/test_targets.py` | `stage1-textual-port` | `ca30ab92e54e0c41fafaacc221a5120e0be8e6a5` | `b573273e55a9914577f15e0f58ef839b9dae4790` | `2ba6f8e2ed45040c2057ce9393158b52292533e3` | Pinned Stage 1 textual-history contribution. |
| `tests/stage1/test_token_rows.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `7be2fecb4f390e7747812d457c0dcfb156427853` | `7be2fecb4f390e7747812d457c0dcfb156427853` | Unchanged baseline runtime/support file. |
| `tests/stage1/test_trainer.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `a6812200ad386b87cc3f7889db2f8483dc9d0ff7` | `a6812200ad386b87cc3f7889db2f8483dc9d0ff7` | Unchanged baseline runtime/support file. |
| `tests/stage1/test_training_evidence.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `fdc526baa2e8cc25e1135aff838bb5237578e4e1` | `fdc526baa2e8cc25e1135aff838bb5237578e4e1` | Unchanged baseline runtime/support file. |
| `tests/stage1/test_zero_matrix.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `5498d525ff733a1e5156f61f8fa31df3054eaca4` | `5498d525ff733a1e5156f61f8fa31df3054eaca4` | Unchanged baseline runtime/support file. |
| `tests/stage1/zero3_gc_diagnostic.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `b45c219babaa0aa560caef54c724953183961b1e` | `b45c219babaa0aa560caef54c724953183961b1e` | Unchanged baseline runtime/support file. |
| `tests/stage1/zero_compat_diagnostic.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `218642ff13c297a33a467a1311fd97eed73762b3` | `218642ff13c297a33a467a1311fd97eed73762b3` | Unchanged baseline runtime/support file. |
| `tests/stage2/__init__.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `1031caad702fb94ee2b6d26fa54cbd5b6273b780` | `1031caad702fb94ee2b6d26fa54cbd5b6273b780` | Unchanged baseline runtime/support file. |
| `tests/stage2/conftest.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `ecf6ed6776b7752e9efb91e77d08692472341811` | `ecf6ed6776b7752e9efb91e77d08692472341811` | Unchanged baseline runtime/support file. |
| `tests/stage2/helpers.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `1d1e27ebf6adc4089d7aaa6847db2a52c4ac3363` | `1d1e27ebf6adc4089d7aaa6847db2a52c4ac3363` | Unchanged baseline runtime/support file. |
| `tests/stage2/integration/stage2_rank_skew_probe.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `b95feaa8bd7929327f532a626519fc899263eb70` | `b95feaa8bd7929327f532a626519fc899263eb70` | Unchanged baseline runtime/support file. |
| `tests/stage2/integration/test_cuda_backends.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `95b7b9b6418bcdb82584f484583a56035a0d88a4` | `95b7b9b6418bcdb82584f484583a56035a0d88a4` | Unchanged baseline runtime/support file. |
| `tests/stage2/integration/test_distributed_contract.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `01469916382c431ce51d9f6a2d1a68f8eb263f47` | `01469916382c431ce51d9f6a2d1a68f8eb263f47` | Unchanged baseline runtime/support file. |
| `tests/stage2/integration/tiny_stage2_train.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `431caf91cee1308299bb8573d8d4d8bba8fe06b9` | `431caf91cee1308299bb8573d8d4d8bba8fe06b9` | Unchanged baseline runtime/support file. |
| `tests/stage2/test_checkpointing.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `a82fe4250b6b20e3fc494039f8ebda042d099d01` | `a82fe4250b6b20e3fc494039f8ebda042d099d01` | Unchanged baseline runtime/support file. |
| `tests/stage2/test_comt_readiness.py` | `stage2-no-cache-port` | `b0ce19b086cd46961fa9dcc12aa3844f659ff0b0` | `d8ee16bf260233445665605fb0c3b664b10c5135` | `c585658ec958e16eb55cbb43fa0e73bfbfd504fb` | Pinned Stage 2 no-cache/manifest contribution. |
| `tests/stage2/test_data.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `7e76c1be3ffd9024b08c7d22662eea9c092cb443` | `7e76c1be3ffd9024b08c7d22662eea9c092cb443` | Unchanged baseline runtime/support file. |
| `tests/stage2/test_efficiency_probe.py` | `stage2-no-cache-port` | `b0ce19b086cd46961fa9dcc12aa3844f659ff0b0` | `8e78a30ce5c9bdb7d69550c5502e406e114eabe4` | `8e78a30ce5c9bdb7d69550c5502e406e114eabe4` | Pinned Stage 2 no-cache/manifest contribution. |
| `tests/stage2/test_eval_integration.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `c75f798011119d36b649aafbaf8d21143fdc142a` | `c75f798011119d36b649aafbaf8d21143fdc142a` | Unchanged baseline runtime/support file. |
| `tests/stage2/test_generation_regression.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `0aceb49bf074294a7975958704ecd99b3bd7b73b` | `0aceb49bf074294a7975958704ecd99b3bd7b73b` | Unchanged baseline runtime/support file. |
| `tests/stage2/test_inference.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `67f643930e4a72324f0aa6177fd150ea531aee66` | `67f643930e4a72324f0aa6177fd150ea531aee66` | Unchanged baseline runtime/support file. |
| `tests/stage2/test_launcher.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `a6082030b99bf65955633105a7f44a95cac77e5f` | `f497696f4fc9e058de6980a217885891d27b1c85` | Later-main behavior retained in the selective merge. |
| `tests/stage2/test_launcher.py` | `manual-three-way-merge` | `b0ce19b086cd46961fa9dcc12aa3844f659ff0b0` | `c4f375f3df4b336b96a5e98ca53584aed2af7926` | `f497696f4fc9e058de6980a217885891d27b1c85` | Pinned Stage 2 behavior selectively merged with main. |
| `tests/stage2/test_length_manifest.py` | `stage2-no-cache-port` | `b0ce19b086cd46961fa9dcc12aa3844f659ff0b0` | `f3962c52a9818b049e173adf7c611d13ff4cf0d2` | `f3962c52a9818b049e173adf7c611d13ff4cf0d2` | Pinned Stage 2 no-cache/manifest contribution. |
| `tests/stage2/test_peft.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `ad28747f4f919d39b61d77bb82b376f89245b44f` | `0928fa8382a6698e7b6e29846b83567c28dac8eb` | Original main PEFT behavior. |
| `tests/stage2/test_peft.py` | `stage2-peft-port` | `caaf264a86b78277bf764ba240cd5197d3d40623` | `41725dcaec6869d0c62bad4b25356ab21a57f549` | `0928fa8382a6698e7b6e29846b83567c28dac8eb` | Tracked unmatched-selector rejection port; distinct from dirty patch. |
| `tests/stage2/test_peft_checkpointing.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `876edf96aa7664770006a5dfba19bc31d897ae1f` | `876edf96aa7664770006a5dfba19bc31d897ae1f` | Main behavior retained in final runtime. |
| `tests/stage2/test_peft_checkpointing.py` | `stage2-no-cache-port` | `b0ce19b086cd46961fa9dcc12aa3844f659ff0b0` | `df023efefb76d83ece8e41211e45bf859703f3cd` | `876edf96aa7664770006a5dfba19bc31d897ae1f` | Tracked pinned Stage 2 lineage used for dirty-patch comparison. |
| `tests/stage2/test_peft_checkpointing.py` | `main-baseline/subsumes-dirty-patch` | `0729ecd05ed8950a7c1c8d2ca74b2ba64a48172484e4a7e0ce3570cabe636557` | `8c2d2164ad3c066465793663d358a5df3d5dcc32` | `876edf96aa7664770006a5dfba19bc31d897ae1f` | Approved uncommitted PEFT patch behavior preserved semantically; patch not applied wholesale. |
| `tests/stage2/test_peft_inference.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `9a7421b4e3c39abb1c3578975f1e4e6d310df87c` | `9a7421b4e3c39abb1c3578975f1e4e6d310df87c` | Unchanged baseline runtime/support file. |
| `tests/stage2/test_qwen_forward.py` | `stage2-no-cache-port` | `b0ce19b086cd46961fa9dcc12aa3844f659ff0b0` | `3a0fd3adfbc1734c9081e44e1b2ac01c7dcd814d` | `3a0fd3adfbc1734c9081e44e1b2ac01c7dcd814d` | Pinned Stage 2 no-cache/manifest contribution. |
| `tests/stage2/test_rollout.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `cfd93ad8e2f6a07b1721a9c7fe33841a753eb111` | `cfd93ad8e2f6a07b1721a9c7fe33841a753eb111` | Unchanged baseline runtime/support file. |
| `tests/stage2/test_routing.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `8bdc15f0ff6e71eb72732bb5c91c93d0b28799e7` | `c0ed8fd10647e2ad300165dd7ab85b1139e40f08` | Later-main behavior retained in the selective merge. |
| `tests/stage2/test_routing.py` | `manual-three-way-merge` | `b0ce19b086cd46961fa9dcc12aa3844f659ff0b0` | `446ecc06d7edd4bae849f605103bf810088b2e8d` | `c0ed8fd10647e2ad300165dd7ab85b1139e40f08` | Pinned Stage 2 behavior selectively merged with main. |
| `tests/stage2/test_sampler.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `0fe890d3df9ec1775b158f40856b2127f0eb1729` | `0fe890d3df9ec1775b158f40856b2127f0eb1729` | Unchanged baseline runtime/support file. |
| `tests/stage2/test_stage1_checkpoint_handoff.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `e9623f26a98303d7ef504ba7be7afe1cf9290647` | `e9623f26a98303d7ef504ba7be7afe1cf9290647` | Unchanged baseline runtime/support file. |
| `tests/stage2/test_structure.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `317d47eaf140c1aa2ad3ead3fa7031811885d4a0` | `317d47eaf140c1aa2ad3ead3fa7031811885d4a0` | Unchanged baseline runtime/support file. |
| `tests/test_artifact_paths.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `4f8a1cf53bb42987992b9c3138db3fae0429bd3c` | `1b9162f76ce5fb2395ebb77761ae28843e28c2a2` | Later-main behavior retained in the selective merge. |
| `tests/test_artifact_paths.py` | `manual-three-way-merge` | `b0ce19b086cd46961fa9dcc12aa3844f659ff0b0` | `56d55dacd04e7a701c7097d21f289f9a8f14277d` | `1b9162f76ce5fb2395ebb77761ae28843e28c2a2` | Pinned Stage 2 behavior selectively merged with main. |
| `tests/test_artifact_paths.py` | `worktree-path-fix` | `3893ec71c3559917f726248cae81980cec161a76` | `1b9162f76ce5fb2395ebb77761ae28843e28c2a2` | `1b9162f76ce5fb2395ebb77761ae28843e28c2a2` | Active-checkout path derivation. |
| `transformers/src/transformers/models/qwen2_5_vl/__init__.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `7a9f44a7a05fc65136552948be9acef53e5ea106` | `7a9f44a7a05fc65136552948be9acef53e5ea106` | Unchanged baseline runtime/support file. |
| `transformers/src/transformers/models/qwen2_5_vl/configuration_qwen2_5_vl.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `ed3505728a6331220a60ca71aa6e1dba7aef09a0` | `ed3505728a6331220a60ca71aa6e1dba7aef09a0` | Unchanged baseline runtime/support file. |
| `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` | `stage2-no-cache-port` | `b0ce19b086cd46961fa9dcc12aa3844f659ff0b0` | `58b5ebcd52a95bb8adcda132a0f79307f238f6bd` | `58b5ebcd52a95bb8adcda132a0f79307f238f6bd` | Pinned Stage 2 no-cache/manifest contribution. |
| `transformers/src/transformers/models/qwen2_5_vl/modular_qwen2_5_vl.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `b1622e00a36c5005b873d5dc67f553f277f21be7` | `b1622e00a36c5005b873d5dc67f553f277f21be7` | Unchanged baseline runtime/support file. |
| `transformers/src/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py` | `main-baseline` | `85e74e97de3773375ff037690511153c8d1f40cc` | `6357debbe26e2b67395411e10eeee65b64cd4400` | `6357debbe26e2b67395411e10eeee65b64cd4400` | Unchanged baseline runtime/support file. |
| `transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py` | `stage2-no-cache-port` | `b0ce19b086cd46961fa9dcc12aa3844f659ff0b0` | `c898d4e9f0605ef5b1767c86dc477045207d75bd` | `c898d4e9f0605ef5b1767c86dc477045207d75bd` | Pinned Stage 2 no-cache/manifest contribution. |

## Dirty PEFT semantic witness

The uncommitted patch source blobs captured during Task 1 were:

| Path | Captured dirty source blob | Corresponding tracked Stage 2 blob |
|---|---|---|
| `src/stage2/checkpointing.py` | `ff90d2bd18a6fb4484516b3aede242d5e2e22246` | `f3b9fcea067d72f6ff86cfcbf351d7da85bdb6bf` |
| `src/stage2/trainer.py` | `c0c66059a56d75010dfb35ec47bd230952bffd5e` | `b41de51ad93cd8a951e28ca2c43bde868908a9ea` |
| `tests/stage2/test_peft_checkpointing.py` | `8c2d2164ad3c066465793663d358a5df3d5dcc32` | `df023efefb76d83ece8e41211e45bf859703f3cd` |

Task 4's semantic witness passed these four nodes:

- `tests/stage2/test_peft.py::test_stage2_peft_accepts_optimized_target_selectors_for_multilayer_decoder`
- `tests/stage2/test_peft.py::test_stage2_peft_rejects_unmatched_compact_target_selector`
- `tests/stage2/test_peft_checkpointing.py::test_relocated_stage1_base_accepts_compact_saved_target_selectors`
- `tests/stage2/test_peft_checkpointing.py::test_stage2_trainer_peft_save_excludes_resized_base_embeddings`

The observed result was `4 passed in 8.86s`. This is a semantic preservation
witness for successful-run PEFT behavior, not evidence that the dirty patch was
copied into the final tree.

## Local and historical evidence

Evidence grades are deliberately narrow:

| Claim or asset | Grade | Source and digest | Boundary |
|---|---|---|---|
| `learning_rate=5e-5`, ZeRO-2, world size 4, and Stage 1 handoff through prepared `checkpoint-431` | `direct-local-log` | `/root/workspace/ILVR/logs/zero-stage-compatibility/stage2-dora-z2-b1-ga2-e2-lr5e5-egb8-20260902-161341-r3/train-node1.log`; SHA-256 `f21fa7bcc11d105fb1f1f60503c07a6c2e56489ae1f4e29846b24a0eaca7f677` | The local log directly records launch configuration; it does not prove a complete successful run or evaluation. |
| Stage 2 length manifest | `direct-local-log` | `/root/workspace/ILVR/results/zero-stage-compatibility/stage2-dora-z2-b1-ga2-e2-lr5e5-egb8-20260902-161341-r3/stage2-length-manifest.json`; SHA-256 `c3aacf993030b30c66f4212cddf54bf02bed10bab1269f0e35240f35d57a3c8d` | Directly available run input; not an evaluation result. |
| `checkpoint-846` inventory | `local-partial-inventory` | `/raid/workspace/ILVR/checkpoints/zero-stage-compatibility/stage2-dora-z2-b1-ga2-e2-lr5e5-egb8-20260902-161341-r3/checkpoint-846`; sorted relative-path/size inventory SHA-256 `2f6e1e2b8cc83fb361590e2c59e0dc22ed4dc60cdbed8bd5998285cdbb4b8614` | Four files only: rank-2/rank-3 optimizer shards and RNG states; 365,551,060 bytes. Not complete, resumable, or evaluation-ready. |
| Training loss `0.24126` | `historical-report` | prior operator-session report; no canonical result artifact located | Not revalidated and not promoted by the local log/inventory. |
| Evaluation `201/400 (50.25%)` | `historical-report` | prior operator-session report; no canonical result artifact located | Protocol and predictions artifact were not independently recovered. |
| Non-overlap evaluation `149/286 (52.10%)` | `historical-report` | prior operator-session report; no canonical result artifact located | Protocol and subset identity were not independently recovered. |
| Stage 1 comparison `45.0%` | `historical-report` | prior operator-session report; no canonical result artifact located | Historical comparison only. |
| Reported `+5.25pp` | `historical-report` | arithmetic cross-report difference between `50.25%` and `45.0%`; no canonical result artifact located | Observed cross-report difference only; the protocols are not proven identical. |

## Selected implemented documentation

These implemented lineages are copied or already present in this checkout:

- [docs/superpowers/specs/2026-08-24-ilvr-stage2-reimplementation-design.md](../superpowers/specs/2026-08-24-ilvr-stage2-reimplementation-design.md)
- [docs/superpowers/plans/2026-08-24-ilvr-stage2-reimplementation.md](../superpowers/plans/2026-08-24-ilvr-stage2-reimplementation.md)
- [docs/superpowers/specs/2026-08-25-ilvr-stage2-peft-design.md](../superpowers/specs/2026-08-25-ilvr-stage2-peft-design.md)
- [docs/superpowers/plans/2026-08-25-ilvr-stage2-peft.md](../superpowers/plans/2026-08-25-ilvr-stage2-peft.md)
- [docs/superpowers/specs/2026-08-26-ilvr-stage1-dora-batched-training-design.md](../superpowers/specs/2026-08-26-ilvr-stage1-dora-batched-training-design.md)
- [docs/superpowers/plans/2026-08-26-ilvr-stage1-dora-batched-training.md](../superpowers/plans/2026-08-26-ilvr-stage1-dora-batched-training.md)
- [docs/superpowers/specs/2026-08-27-zero-stage-compatibility-design.md](../superpowers/specs/2026-08-27-zero-stage-compatibility-design.md)
- [docs/superpowers/plans/2026-08-27-zero-stage-compatibility.md](../superpowers/plans/2026-08-27-zero-stage-compatibility.md)
- [docs/superpowers/specs/2026-08-28-ilvr-multinode-four-gpu-design.md](../superpowers/specs/2026-08-28-ilvr-multinode-four-gpu-design.md)
- [docs/superpowers/plans/2026-08-28-ilvr-multinode-four-gpu.md](../superpowers/plans/2026-08-28-ilvr-multinode-four-gpu.md)
- [docs/superpowers/specs/2026-08-28-stage1-stage2-zero123-lora-integration-design.md](../superpowers/specs/2026-08-28-stage1-stage2-zero123-lora-integration-design.md)
- [docs/superpowers/plans/2026-08-28-stage1-stage2-zero123-lora-integration.md](../superpowers/plans/2026-08-28-stage1-stage2-zero123-lora-integration.md)
- [docs/superpowers/specs/2026-08-30-stage1-textual-states-design.md](../superpowers/specs/2026-08-30-stage1-textual-states-design.md)
- [docs/superpowers/plans/2026-08-30-stage1-textual-states.md](../superpowers/plans/2026-08-30-stage1-textual-states.md)
- [docs/superpowers/specs/2026-09-01-stage1-batch-equivalence-design.md](../superpowers/specs/2026-09-01-stage1-batch-equivalence-design.md)
- [docs/superpowers/plans/2026-09-01-stage1-batch-equivalence-v26.md](../superpowers/plans/2026-09-01-stage1-batch-equivalence-v26.md)
- [docs/superpowers/plans/2026-09-01-stage1-evaluation-repair.md](../superpowers/plans/2026-09-01-stage1-evaluation-repair.md)
- [docs/superpowers/plans/2026-09-01-stage1-true-batch-controlled-comparison.md](../superpowers/plans/2026-09-01-stage1-true-batch-controlled-comparison.md)
- [docs/superpowers/specs/2026-09-02-stage2-efficiency-design.md](../superpowers/specs/2026-09-02-stage2-efficiency-design.md)
- [docs/superpowers/plans/2026-09-02-stage2-no-cache-efficiency.md](../superpowers/plans/2026-09-02-stage2-no-cache-efficiency.md)
- [docs/superpowers/specs/2026-09-03-stage1-stage2-final-worktree-design.md](../superpowers/specs/2026-09-03-stage1-stage2-final-worktree-design.md)
- [docs/superpowers/plans/2026-09-03-stage1-stage2-final-worktree.md](../superpowers/plans/2026-09-03-stage1-stage2-final-worktree.md)

## Excluded documentation

These source artifacts were not copied because each is `not part of the verified
final runtime`:

| Excluded source artifact | Reason |
|---|---|
| `docs/superpowers/plans/2026-09-02-stage2-segmented-kv.md` | not part of the verified final runtime |
| `documents/multi-node-stage2-training-20260920.md` | not part of the verified final runtime |
| `documents/stage-2-kv-cache-20260902.md` | not part of the verified final runtime |
| `documents/TODO-ours1.md` | not part of the verified final runtime |
| `documents/vcsd-ilvr-stage2/guideline-stage2_start.md` | not part of the verified final runtime |

The excluded artifacts are named for auditability but intentionally are not
linked through another worktree.
