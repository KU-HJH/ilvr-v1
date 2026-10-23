# Independent input for frozen proposal v1: problem facts only

A user requests a fresh ten-epoch Stage1-v5 experiment with DIL-BCE-ORIG,
V-off top-k conditional entropy and clean-to-V-off group-mass KL, using clean
entropy's top20% target tokens. Existing approved Stage1-v5 DIL uses equal
token weighting across the actual four-rank accumulation window. Entropy/mass
use original-block means. The selector already computes a stable per-sample
top percentage over eligible immediate-text tokens, but current training only
accepts selection=all. Existing saved contracts distinguish historical block
means from current token means and strictly bind execution/source identity.

Known current profile: original Qwen2.5-VL-3B, DoRA/frozen EMA1/ZeRO1, legacy
query, micro1/GAS4/world4/global16, CE0.1/alignment0.1/DIL1, entropy0.1/mass0.05,
Text-KL0.01, independent k50/no GT union. Existing coordinator requires two
nodes/four ranks, remote rank0 on172.16.11.41, local rank1, bond0.11/llm_kd.
Source checkout has pre-existing uncommitted network-address updates; preserve
them and historical artifacts. User asks training, not an evaluation campaign.

Read-only source anchors under /root/.codex/worktrees/1061/ILVR:
src/stage1/v5_config.py; v5_selection.py; v5_trainer.py; v5_metadata.py;
v5_resume.py; v5_objective.py; src/step_preference/entropy_selection.py.
Dataset source has3450 lines; runtime retained count/budget require fresh audit.
Current local and remote GPUs are idle; matching principal package versions
were observed, but no final compatible-code or launch preflight is claimed.

Independently identify what must be decided or evidenced to avoid running an
incorrect experiment, and the smallest correctness-preserving support route.
Do not assume a preferred implementation. Prioritize any incompatible premise
or framing error, then compatible missing considerations. Do not edit files,
run training, invoke Paperthin skills or dispatch other agents.
