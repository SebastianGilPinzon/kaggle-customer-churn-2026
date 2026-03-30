# Kaggle: kaggle-customer-churn-2026

## AI Instructions
- The user is a **vibe coder** — NOT a programmer or data scientist
- All technical execution is done by AI. User makes decisions only.
- Explain BEFORE executing. No jargon without explanation.
- Present OPTIONS with pros/cons for user to decide.
- Respond in the same language the user writes in.
- **Deploy multiple agents in parallel** for research, analysis, and exploration tasks (3-5+ agents with different angles).
- **Document EVERYTHING** — all findings, learnings, and technical details go in `docs/MASTER.md`. Keep it updated after every significant action. CLAUDE.md = instructions only, not documentation.

## First-Time Agent? Read BOOTUP.md
If this is your FIRST session on this competition, read `../kaggle-hq/BOOTUP.md` BEFORE anything else.
It contains the complete onboarding guide, all available tools, and lessons from past competitions.

## Session Start Protocol
**At the start of EVERY new session, the AI must:**
1. Read this file (CLAUDE.md)
2. Read `docs/MASTER.md` for full context and accumulated knowledge
3. Read last 5 entries in `experiment-log.md`
4. Read last 3 entries in `decision-log.md`
5. Read `../kaggle-hq/playbook/strategies.md` and `../kaggle-hq/playbook/mistakes.md`
6. Present a status summary and propose next action

## Available Tools (in `../kaggle-hq/scripts/`)
- `status-dashboard.py` — **RUN FIRST** — complete state of all competitions
- `submit.py` — submit + auto-sync scores to experiment-log.md and CLAUDE.md
- `post-submission-report.py` — auto-generate diff analysis between submissions
- `scrape-competition.py` — scrape Kaggle pages (JS-rendered)
- `download-data.py` — download competition data
- `test-scorer.py` — local format check + fuzzy match test + full scoring
- `test-offline.py` — verify notebook works offline (code competitions)
- `analyze-leaderboard.py` — fetch + analyze leaderboard
- `analyze-experiments.py` — parse experiment log for insights
- `track-submissions.py` — fetch all scores from Kaggle
- `batch-experiments.py` — run experiments from YAML config
**USE THESE. They exist to save time and submission slots.**

## Mandatory Behavior Rules

### Auto-Debug Loop
When code fails: (1) Read the full error traceback. (2) Diagnose root cause. (3) Fix and re-execute. (4) Repeat up to 3 times. (5) If still failing after 3 attempts, STOP and present the error to the user with your diagnosis. Do NOT ask the user to debug — attempt the fix yourself first.

### Verification-First Workflow
BEFORE implementing any change based on an assumption:
1. What assumption am I making? (write it down)
2. Can I verify it against training data or the scorer? (if yes, DO IT)
3. What breaks if this assumption is wrong?
If you cannot verify, document the assumption in `decision-log.md` and flag it to the user.

### Anti-Patterns (STOP immediately if you catch yourself doing these)
1. "I'll code now, verify later" → STOP. Verify first, then code.
2. "I assume the format is X" → STOP. Check training data or run test-scorer.py.
3. "I don't need that script" → STOP. Read what it does first.
4. "I'll document later" → STOP. Document NOW in docs/MASTER.md.
5. "This is a small change, no need to test" → STOP. Every change can break the score.

### Experiment Roadmap Protocol
After completing each batch of experiments (or when told "don't stop"), generate `NEXT_EXPERIMENTS.md`:
```
### Experiment: [name]
- Hypothesis: [what we expect and why]
- Changes: [specific files/values to change]
- Effort: [hours]
- Expected impact: [score delta, with reasoning]
- Risk: [what could go wrong]
- Priority: [1-5, based on impact/effort ratio]
```
Always maintain 5 ranked ideas. Execute in order unless user redirects.
Base priorities on: what worked (strategies.md), what failed (mistakes.md), what's untried (experiment-log.md).

### Phase Summary Gate
BEFORE moving to any new phase, you MUST update `docs/MASTER.md` with:
- What was done in this phase
- Key findings
- Decisions made
- Open questions
This is a GATE — do not proceed to the next phase without this update.

## Current Status
- **Phase:** Phase 3 — Baseline + first ensemble
- **Best CV score:** 0.9159 (experiment #2, 3-model ensemble)
- **Best LB score:** 0.91427 (experiment #2, 3-model ensemble)
- **Submissions used today:** 2/5
- **Days remaining:** 4
- **Progress:**
  - [x] Phase 1: Research & EDA
  - [x] Phase 2: Strategy
  - [x] Phase 3: Baseline
  - [ ] Phase 3B: Iterations
  - [ ] Phase 3C: Ensembling
  - [ ] Phase 4: Pre-Submission Audit
  - [ ] Phase 5: Final Submission Selection
  - [ ] Phase 6: Post-Competition Review

## User Decisions
[Decisions made by the user — see decision-log.md for full history]

## What NOT to Do
[Mistakes made in this competition — updated as we learn]

## Key Files
- `competition-context.md` — Competition details, data, metric, rules
- `experiment-log.md` — ALL experiments (submitted and failed)
- `decision-log.md` — User decisions with reasoning
- `notebooks/` — Jupyter notebooks (EDA, experiments)
- `src/` — Clean Python scripts (train, infer, features)
- `submissions/` — Submission CSV files

## Methodology Reference
Follow `../kaggle-hq/methodology/COMPETITION_METHODOLOGY.md` for all work.

### Condensed Phase Guide
1. **Phase 1 (20% of time):** Research competition + EDA. Run EDA checklist. Analyze public notebooks.
2. **Phase 2 (10%):** Define baseline, CV strategy, improvement roadmap. User approves.
3. **Phase 3 (immediate):** Build baseline, submit, get first LB score.
4. **Phase 3B (60% of time):** Iterate in batch mode (3-5 experiments per batch). Track everything.
5. **Phase 3C:** Ensemble diverse models when 3+ exist.
6. **Phase 4:** Audit for leakage, reproducibility, format.
7. **Phase 5:** User selects 2 final submissions. **Blocking approval required.**
8. **Phase 6:** Post-competition review within 48 hours.

### Autonomy Rules
- **Batch mode:** AI runs 3-5 experiments from roadmap without per-experiment approval
- **Major changes need approval:** new model architecture, new data source, CV strategy change, >4h experiments
- **Always log:** every experiment in experiment-log.md, every decision in decision-log.md
- **Auto-update:** after every submission, update Current Status section above

## Reproducibility
- **Random seed:** 42
- **Python version:** 3.11
- **Key library versions:** lightgbm, xgboost, catboost, scikit-learn, optuna
