# Project goal
Build a portfolio-quality ML application that is:
- small enough to finish
- reproducible
- easy to demo
- well documented for future you and collaborators

Prefer a working end-to-end baseline over a more sophisticated model that is hard to defend.

# Human-owned decisions
The human owns final decisions for:
- problem framing
- dataset choice
- label definitions
- primary evaluation metric
- model family choice
- architecture changes
- deployment tradeoffs

Before changing any of those, present tradeoffs and ask for approval.

# Working style
For any non-trivial task:
1. restate the task in plain English
2. identify the files likely to change
3. state assumptions / risks
4. define how success will be verified
5. keep the diff as small as possible

# ML-specific expectations
For changes involving training, evaluation, data processing, or inference:
- explain the hypothesis before editing
- prefer the smallest valid baseline first
- change one major experimental variable at a time unless explicitly told otherwise
- preserve reproducibility: note config, seed, dataset version, and metrics touched
- after the change, explain what I should learn from it as the human owner

# Code quality
- prefer small, reviewable diffs
- do not introduce a new framework or dependency without justification
- preserve existing patterns unless there is a clear reason to refactor
- write or update focused tests for non-trivial logic
- never leave placeholder comments instead of real code unless explicitly asked

# Documentation
When behavior, interfaces, metrics, or workflows change:
- update docs/feature-plan.md if scope changed
- append a short entry to docs/experiment-log.md if any ML behavior changed
- append a short entry to docs/decision-log.md for important tradeoffs
- update README/demo notes when needed

# Cursor workflow automation
- Daily digests: `.cursor/workflow-review/daily/` (auto-generated)
- Weekly rule review: `.cursor/workflow-review/latest.md` — apply only after **yes / no / edit**
- Monday suggestions: `.cursor/workflow-suggestions/KICKOFF.md` in Agent mode; external dirs in `.cursor/workflow-paths.json`; queue marks **viewed** after kickoff reply

# Definition of done
A task is not done unless:
- the code runs
- unit tests are added for new functionality or features
- relevant tests / checks pass or failures are clearly reported
- the change is summarized in plain English
- risks / limitations are stated
