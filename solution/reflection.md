# Reflection (≤1 page)

Fill this in before you submit.

**Which fault types were hardest to catch, and why?**

The hardest fault types to catch were the **subtle-tier faults** introduced in the Private phase, specifically:
1. **Subtle distribution shifts and borderline checks (`distribution_shift`, `null_spike`, `freshness_lag`, `volume_drop` in `data_batch`):** Subtle instances exhibited magnitudes completely overlapping with the natural variance of clean batches (e.g., subtle `mean_amount` values around 85.7–86.4 fell right inside the clean empirical range of 77.8–87.6).
2. **Subtle embedding drift and document staleness (`embedding_drift`, `corpus_staleness` in `embedding_batch`):** Subtle centroid shifts (~0.029–0.031) were lower than the maximum clean centroid shift (0.0387), and doc ages (~35–37 days) sat below the baseline ceiling (49.8 days).
3. **Subtle runtime anomalies in `lineage_run`:** Execution times of ~4.5–4.7s overlapped with higher clean durations (~4.6s).

*Why:* In an online stream where each event is evaluated in isolation without seeing future ground truth, aggressive thresholding to capture these low-amplitude signals inevitably introduces False Positives (FPR). Under the scoring formula ($0.5 \cdot \text{TPR} - 0.3 \cdot \text{FPR}$), the penalty for a false alarm is $-0.30$, which requires catching $1.67$ real faults just to break even. Thus, attempting to capture these overlapping subtle faults would severely degrade the net score.

**What would you change about your cost/coverage tradeoff, if you had another pass?**

If given another iteration, we would implement an **adaptive dynamic credit allocation strategy** managed via `ctx.state` and `ctx.tools.budget_remaining()`:
1. **Public Phase Optimization:** The Public phase presented severe budget pressure (220 credits for 160 events, where a full single-pass scan required 240 credits, causing a 9.09% overage and $-1.82$ score deduction). With another pass, we would track `ctx.state` across consecutive events. If an entity/feature view had clean history over several consecutive checkpoints, we would downsample or skip the expensive 2.0-credit tools (`feature_drift` and `embedding_drift`) once remaining budget dipped below the required pace ($1.5 \times \text{remaining events}$), eliminating overage while maintaining high TPR.
2. **Sequential Multi-Event Tracking:** We would use `ctx.state` to compute rolling EWMA (exponentially weighted moving average) and CUSUM (cumulative sum) charts across successive batches. This would allow detecting persistent subtle distribution drift and gradual staleness over time without lowering single-event thresholds or triggering one-off false positives.

