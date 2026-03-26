# shared_compact_input_format

- Changed file: `utils/prediction_contracts.py`
- Changed symbol: `_format_inputs`
- Mechanism: serialize the task input JSON in a more compact form before sending it to the model.
- Shared-path rationale: both domains format their task inputs through the same helper.
- Source hypothesis: a shorter prompt may leave more room for the answer and reduce formatting drift.
