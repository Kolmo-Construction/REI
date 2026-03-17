# Experiment Log

| Timestamp | Change Description | File | Baseline | Candidate | Delta | Decision |
|---|---|---|---|---|---|---|
| 20260316T203938Z | eval_error: eval.py exited with non-zero code 1. | greenvest/nodes/synthesizer.py | 0.8915 | N/A | N/A | REVERT (eval error) |
| 20260316T204201Z | eval_error: eval.py exited with non-zero code 1. | greenvest/nodes/query_translator.py | 0.8915 | N/A | N/A | REVERT (eval error) |
| 20260316T205033Z | eval_error: eval.py timed out after 300s — subprocess killed. | greenvest/nodes/synthesizer.py | 0.8915 | N/A | N/A | REVERT (eval error) |
| 20260316T210432Z | Patched `_REI_PERSONA` in `greenvest/nodes/synthesizer.py`: 489 chars (was 335 c | greenvest/nodes/synthesizer.py | 0.8915 | 0.9125 | +0.0210 | KEEP |
| 20260316T211142Z | Updated phrase list: added ['budget', 'affordable', 'cheap']. Rationale: Adding  | greenvest/nodes/query_translator.py | 0.9125 | 0.8655 | -0.0470 | REVERT (delta<0.01) |
| 20260316T212503Z | Updated phrase list: added ['budget', 'affordable', 'cheap']. Rationale: Adding  | greenvest/nodes/query_translator.py | 0.9125 | 0.8700 | -0.0425 | REVERT (delta<0.01) |
