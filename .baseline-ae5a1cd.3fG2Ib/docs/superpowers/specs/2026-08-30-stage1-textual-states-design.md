# Stage 1 Textual-State History Mode Design

## Approved behavior

- Expose one Stage 1 option named `--textual_states` with exactly two values:
  `all` and `previous`.
- Default to `previous`, so an existing Stage 1 command that omits the option
  retains the repository behavior.
- `previous` must reproduce the pre-change query construction from Git:
  global intent, the current block's local textual span, and the immediately
  preceding block's selected-feature mean.
- `all` must follow ILVR equation (4): global intent, the mean of every textual
  rationale hidden-state row from steps 1 through i, and the mean of every
  selected latent-rationale row from steps 1 through i-1.
- The assistant control marker is not a textual-rationale hidden state in
  `all`; the first textual span begins after the longest matching assistant
  marker. The `previous` path keeps the repository's original marker boundary.
- Keep the cumulative `all` implementation bounded: running sums and counts
  produce the two history means without retaining or concatenating an
  ever-growing history tensor.

## Runtime path

`src/utils_deepseed.py::parse_args` parses the option. Stage 1 composition
passes it to `CustomTrainerStage1`, which passes it to
`Stage1TeacherTargetBuilder`. `run_stage1_training.sh` accepts and forwards the
same `--textual_states` spelling.

## Invariants

- The teacher language-model forward remains exactly once per batch.
- Original-image visual encoding remains inside that single teacher forward.
- All helper images are encoded by the teacher visual encoder in one batched
  call.
- No student helper-image visual forward is introduced.
- The option changes only target-query history construction and adds no model,
  adapter, optimizer, or DeepSpeed-owned state.
- Full, LoRA, DoRA, and DeepSpeed ZeRO-1/2/3 continue through the same trainer
  and target-builder call path.

## Evidence required

- A third-block runtime witness must expose the ordered query components and
  prove that `all` contains the cumulative text mean and cumulative prior-latent
  mean with shape `[1, H]` for the final query.
- A `previous` runtime witness must compare against hand-derived values from the
  repository implementation, including a case where the first block selects
  two rows so cumulative and previous-only latent means cannot coincide.
- Both witnesses must assert one language forward, one original visual forward,
  and one helper visual forward.
- CLI, composition, trainer, and launcher tests must prove that the selected
  value reaches the builder, while the omitted option defaults to `previous`.
