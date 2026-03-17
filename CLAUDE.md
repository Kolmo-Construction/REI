# Greenvest — Claude Code Instructions

## Operational Documentation Rule

`docs/autonomous_optimizer.md` is the authoritative operator manual for the
autonomous optimization pipeline.

**Rule: whenever you modify any of the following files, update `docs/autonomous_optimizer.md`
in the same change:**

```
eval/autonomous_optimize.py
eval/critic.py
eval/optimizer_agent.py
eval/calibrate.py
eval/inspect.py
eval/grid_search.py
eval/prod_judge.py
eval/edit_tools.py
```

Also update the doc if you **add or remove any file in `eval/`**.

Sections to keep in sync:

| What changed in code | Section to update |
|---------------------|------------------|
| Gate decision codes added/removed | Pipeline Detail → Gate |
| New/removed eval/ files | File Map |
| Threshold constants changed | Configuration Reference → Key Constants |
| CLI flags added/changed | Configuration Reference → CLI Flags |
| Observability gap resolved or new one found | Observability Gaps |
| Non-obvious design choice made | Architecture Decisions Worth Knowing |
| Usage commands changed | Quick Start |
| Header changelog | Top of doc (Changes since initial release bullet list) |

Include the doc update in the **same commit** as the code change. Do not make a
separate doc-only commit unless the user explicitly asks for one.
