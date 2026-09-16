# Verification Repair Report

## Scope

Repaired only the verification-discovered old-mode save failure in
`test_capacity_final_adapter_save_defers_processor_but_numbered_save_does_not`.
No production code, training code, weights, optimizer state, or unrelated
baseline failures were changed.

## Diagnosis

`Stage2Trainer.__init__` always establishes `self.step_preference_config`:
when the caller supplies no config, it creates `StepPreferenceConfig()`, and it
rejects values that are not `StepPreferenceConfig`. The ordinary old-mode path
in `main.run_stage2` does not pass a Stage 2B config, so it receives that default
through the constructor. `_save` therefore legitimately consumes this required
trainer state before deciding that old mode needs no evidence-replacement
identity.

The failing test constructed the trainer with `object.__new__(Stage2Trainer)`
and omitted this constructor invariant. The regression was in that synthetic
fixture, not in the runtime save contract. The minimal correction assigns the
same default `StepPreferenceConfig()` that real old-mode initialization assigns.
The existing save test is the causal regression witness: before the fixture
correction it failed at the newly exercised identity decision; afterward it
continues through both numbered and final adapter saves and checks their
processor behavior.

## RED

Exact command:

```text
PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q tests/step_preference/test_stage2b_trainer.py::test_capacity_final_adapter_save_defers_processor_but_numbered_save_does_not
```

Full output (exit 1):

```text
F                                                                        [100%]
=================================== FAILURES ===================================
_ test_capacity_final_adapter_save_defers_processor_but_numbered_save_does_not _

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x73d2b5a52ad0>
tmp_path = PosixPath('/tmp/pytest-of-root/pytest-3528/test_capacity_final_adapter_sa0')

    def test_capacity_final_adapter_save_defers_processor_but_numbered_save_does_not(monkeypatch,tmp_path):
        import stage2.trainer as module
        from types import SimpleNamespace as N
        from step_preference.capacity_evidence import CapacitySampleRecorder
        events=[]
        class Model:
            def save_pretrained(self,target,**kw):events.append(("weights",target.name))
        model=Model();trainer=object.__new__(module.Stage2Trainer)
        trainer.model=model;trainer.accelerator=N(unwrap_model=lambda m:m)
        trainer.args=N(output_dir=str(tmp_path),save_safetensors=True)
        trainer.data_collator=CapacitySampleRecorder(lambda rows:rows,"a")
        trainer.processing_class=N(save_pretrained=lambda target:events.append(("processor",target.name)))
        monkeypatch.setattr(module,"PeftModel",Model)
        monkeypatch.setattr(module,"resolve_qwen2_5_vl_model",lambda m:N(config=N(save_pretrained=lambda target:None)))
        monkeypatch.setattr(torch,"save",lambda *a:None)
>       trainer._save(tmp_path/"checkpoint-2");trainer._save(tmp_path)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/step_preference/test_stage2b_trainer.py:133:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <stage2.trainer.Stage2Trainer object at 0x73d2b55cda10>
output_dir = PosixPath('/tmp/pytest-of-root/pytest-3528/test_capacity_final_adapter_sa0/checkpoint-2')
state_dict = None

    def _save(self, output_dir=None, state_dict=None):
        """Save Stage 2 adapters without PEFT's automatic embedding export."""

        model = self.accelerator.unwrap_model(self.model)
>       attach_evidence_replacement_identity(model, self.step_preference_config)
                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       AttributeError: 'Stage2Trainer' object has no attribute 'step_preference_config'

src/stage2/trainer.py:1608: AttributeError
=========================== short test summary info ============================
FAILED tests/step_preference/test_stage2b_trainer.py::test_capacity_final_adapter_save_defers_processor_but_numbered_save_does_not
1 failed in 8.80s
```

## Minimal correction

Added this constructor invariant to the `object.__new__` fixture:

```python
trainer.step_preference_config=module.StepPreferenceConfig()
```

No `getattr` fallback or production weakening was added.

## GREEN

Exact causal command:

```text
PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q tests/step_preference/test_stage2b_trainer.py::test_capacity_final_adapter_save_defers_processor_but_numbered_save_does_not
```

Full output (exit 0):

```text
.                                                                        [100%]
1 passed in 8.70s
```

Exact focused compatibility command:

```text
PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q tests/step_preference/test_stage2b_trainer.py tests/step_preference/test_replacement_trainer.py tests/step_preference/test_config.py
```

Full output (exit 0):

```text
........................................................................ [ 82%]
...............                                                          [100%]
87 passed in 10.96s
```

The same focused compatibility command was rerun as the fresh completion
witness. Full output (exit 0):

```text
........................................................................ [ 82%]
...............                                                          [100%]
87 passed in 10.98s
```

Exact whitespace check:

```text
git diff --check
```

Full output (exit 0): empty.

## Self-review

- The change matches real old-mode initialization instead of weakening the
  production save contract.
- The fixture retains its deliberate lightweight `object.__new__` construction
  and adds only the missing state needed by `_save`.
- The mutation check is direct: removing the config assignment reproduces the
  recorded `AttributeError`; using replacement mode would exercise a different
  opt-in contract outside this old-mode fixture.
- No unrelated dirty or untracked files are included in this repair.

## Concerns

None within the assigned regression. The broader suite failures are explicitly
outside this task and remain for the root controller's separate diagnosis and
fresh final verification.
