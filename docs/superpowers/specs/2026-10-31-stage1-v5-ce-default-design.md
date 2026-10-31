# Stage1 V5 CE default — design commitment v1

Status: proposed; final approval pending. Native implementation method inherited. User requested CE default equal to alignment after reviewing the V5 defaults table. No training launch authorized.

## Frozen proposal v1

For fresh Stage1 V5, use CE weight 1.0 when omitted, equal to alignment's current default 1.0. This sets a constant default; explicitly changing alignment does not implicitly change CE. Preserve explicit CE values and saved resume coefficients. Keep standard Stage1, V3 and Stage2 defaults outside this scoped request. Record the V5 exception to common CE0.1 policy so a later launch does not override it with stale guidance.

Approach: set the V5-specific fallback in the existing parser after resume inheritance, before constructing its compatibility StepPreferenceConfig. The shared StepPreferenceConfig default cannot implement this scope because it also governs other variants. No dynamic coefficient coupling is needed for a request about the default. Product scope: src/utils_deepseed.py, focused parser/resume tests, and the applicable training-defaults V5 exception (including canonical launch guidance as needed). Use native implementation and one independent review under the standing workflow.

## Commitment critique

Frozen proposal v1 received root-controller hate critique:
{ "root": "A shared fallback or launch-policy CE0.1 override could silently change other variants or undo the V5 default, while resume could accidentally lose its saved CE coefficient.", "first_nail": "Check fresh V5 omitted CE resolves1 with alignment1, explicit CE remains exact, saved CE0.1 resumes as0.1 and conflicting explicit1 rejects; verify standard/Stage2 fallback stays0.1 and launch guidance explicitly scopes the new exception." }

First nail is proposed causal parser/policy validation, not yet run. No expensive or state-changing premise probe. Macrothink not triggered: scope is local and readily falsifiable. No material design uncertainty remains within the V5-only constant-default scope; user can correct scope at approval. Commitment unchanged from proposal.

## Remaining gates

Final approval of v1 before product changes, concrete task brief, native TDD implementation, independent specification/quality review, fresh required verification, then human branch handoff. Standing mandatory workflow requires approval even for this bounded default change. No product files have been changed.
