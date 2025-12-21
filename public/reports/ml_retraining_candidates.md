# ML re-training candidates for next budget window

This note highlights the instruments where a refreshed gradient boosting model would provide the largest uplift once training
budget becomes available. The ranking mixes recent precision-gate diagnostics, fallback signal quality and the amount of labelled
examples already collected.

## USDJPY — **very high** priority
- Precision gate pressure is extreme: 64% of signals are being blocked by the precision stack and 63% specifically fail the order
  flow alignment stage, so the discretionary workflow almost never reaches execution without an ML assist.【F:reports/ml_retraining_candidates_summary.json†L72-L77】
- The feature log has *zero* labelled USDJPY snapshots, so a dedicated labelling sprint is required before training starts. Booking
  this ahead of time avoids idle GPU hours while labels are collected.【F:reports/ml_retraining_candidates_summary.json†L58-L60】

## EURUSD — **high** priority
- Labelled data already exists (245 snapshots) but only 1.6% of those observations closed profitably, so the current fallback logic
  is effectively operating blind.【F:reports/ml_retraining_candidates_summary.json†L14-L23】
- Precision blocks still hit 17% of the signal stream and order-flow alignment failures are nearly identical, showing that the
  handcrafted filters cannot adjust quickly enough to the current low-volatility regime.【F:reports/ml_retraining_candidates_summary.json†L78-L82】
- Re-training on the existing labelled dataset (supplemented with the latest features) is feasible immediately, making EURUSD a
  fast win once compute opens up.

## NVDA — **medium** priority
- Equity index coverage shows a 0.98% label hit-rate across 102 samples, so the precision playbook keeps flagging setups that never
  convert. Order-flow imbalance/pressure medians are materially higher than FX, which should give an ML model richer signals to
  learn from.【F:reports/ml_retraining_candidates_summary.json†L36-L45】
- Precision gating knocks out over a quarter of all alerts, and every block is flow-related — another sign that the deterministic
  filters are too coarse for the current tape.【F:reports/ml_retraining_candidates_summary.json†L84-L88】
- Recommendation: schedule NVDA immediately after EURUSD so the desk has a US equities model ready before the next earnings cycle.

## USOIL — **medium** priority (data gathering required)
- The raw feature log shows 191 labelled observations but *zero* profitable closes. ATR medians sit around 0.002, highlighting the
  elevated volatility regime that the fallback heuristics do not currently reward.【F:reports/ml_retraining_candidates_summary.json†L47-L56】
- Precision blocks are lighter (14%), yet every order-flow gate failure in the monitoring feed maps to insufficient flow context,
  implying that a trained model could separate genuine momentum from noise more effectively.【F:reports/ml_retraining_candidates_summary.json†L90-L94】
- The live journal has not resolved any USOIL trades lately, so prioritise manual labelling/validation before spinning up a training
  job to avoid overfitting the zero-label baseline.【F:reports/ml_retraining_candidates_summary.json†L115-L118】

## GOLD_CFD — **watch list**
- Similar to USOIL, 277 labelled records exist but none close in profit. The ATR distribution is still elevated (p75 ≈ 0.0018),
  suggesting the fallback weights remain too conservative without ML calibration.【F:reports/ml_retraining_candidates_summary.json†L25-L34】
- Precision gates show no systemic blocking, so the main bottleneck is probabilistic sizing and exit timing. Keep this on the watch
  list and pull it into scope once USDJPY/EURUSD/NVDA deliver, or if discretionary performance deteriorates further.【F:reports/ml_retraining_candidates_summary.json†L108-L114】
