# shared_truncated_json_label_salvage

- Changed file: `utils/prediction_contracts.py`
- Changed symbol: `_extract_json_label_candidate`
- Mechanism: recover a clear label from a truncated JSON prefix.
- Shared-path rationale: both `paper_review` and `imo_grading` execute the same shared JSON-label extraction helper.
- Source hypothesis: many invalid `paper_review` outputs already contain `"label": "accept"` or `"label": "reject"` before the response is cut off.
