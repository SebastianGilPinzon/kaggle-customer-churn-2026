# Session Handoff — COMPETITION CLOSED

## Last Session
- **Date:** 2026-03-31
- **Phase:** Phase 5 — Final Submission Selection (COMPLETE)
- **Duration:** ~8h across 2 sessions

## Final Score
- **Best CV:** 0.9168 (v09/v11 — without original data)
- **Best LB:** 0.91434 (v15 — blend v11+v14)
- **Top #1:** 0.91771
- **Gap to #1:** 0.00337

## Final Submissions
- **Final 1:** v15 (LB 0.91434) — blend v11(70%)+v14(30%)
- **Final 2:** v11 (LB 0.91433) — 5-seed avg, no original data

## What Was Done
1. Scraped competition, downloaded data + original Telco dataset
2. Built LGB baseline (CV=0.9151, LB=0.91409)
3. 3-model ensemble LGB+XGB+CatBoost (CV=0.9159, LB=0.91409)
4. Exhaustive feature engineering: 134 features (LB=0.91426)
5. Removed original data = best improvement (+0.0008 CV)
6. Multi-seed averaging (5 seeds)
7. Stacking with 10 models (FAILED — models too correlated)
8. Feature-set diversity: 4 sets × 3 models + hill climbing
9. Expert panel consultation (5 experts)
10. 16 total submissions across competition

## Key Lessons Learned
1. **Original data HURTS** on synthetic PS competitions — different distribution
2. **Stacking fails** when models >0.99 correlated — need genuinely diverse architectures
3. **Target encoding hurts** on synthetic data — overfits to synthetic distribution
4. **Upweighting external data** causes overfitting to external distribution
5. **CPU bottleneck was catastrophic** — 10x fewer experiments than possible with GPU
6. **Feature-set diversity > model diversity** for ensemble quality

## Root Cause of Gap to #1
- ALL training on local CPU (should have used Kaggle Notebooks or Colab Pro)
- Only 2 days of competition time
- No neural networks, no TabPFN, no Optuna tuning (too slow on CPU)
- 5 seeds instead of 50-100
- Rule added to kaggle-hq: NEVER train locally without permission
