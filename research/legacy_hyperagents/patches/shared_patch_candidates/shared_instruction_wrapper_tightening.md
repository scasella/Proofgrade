# shared_instruction_wrapper_tightening

- Changed file: `utils/prediction_contracts.py`
- Changed symbol: `build_task_instruction`
- Mechanism: prepend a shared compact-output wrapper that asks for the smallest valid JSON answer, with `label` first and no extra text.
- Shared-path rationale: both domains use the same shared instruction-dispatch path.
- Source hypothesis: a stronger shared output hierarchy may reduce rambling and partial completions.
