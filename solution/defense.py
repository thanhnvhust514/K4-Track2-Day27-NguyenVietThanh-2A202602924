"""
Your defense. Implement register(ctx) and a handler per event type.
See ../README.md for the full interface + toolkit reference, and
../RULES.md before you start.
"""
from api import Verdict
import math


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def check_data_batch(payload, ctx):
    """Pillar: checks.
    Detects volume_spike, volume_drop, null_spike, distribution_shift, freshness_lag.
    """
    batch_id = payload.get("batch_id")
    profile = ctx.tools.batch_profile(batch_id)
    if "error" in profile:
        return Verdict(alert=False, pillar="checks", reason=profile["error"])

    row_count = profile.get("row_count", 0)
    null_rate = profile.get("null_rate", {}).get("customer_id", 0.0)
    mean_amount = profile.get("mean_amount", 0.0)
    staleness_min = profile.get("staleness_min", 0.0)

    # 1. Single-dimensional checks against calibrated 3-sigma baseline
    if row_count < ctx.baseline.get("row_count_min", 435.47):
        return Verdict(alert=True, pillar="checks", reason=f"volume_drop: {row_count}")
    if row_count > ctx.baseline.get("row_count_max", 561.29):
        return Verdict(alert=True, pillar="checks", reason=f"volume_spike: {row_count}")
    if null_rate > ctx.baseline.get("null_rate_max", 0.0109):
        return Verdict(alert=True, pillar="checks", reason=f"null_spike: {null_rate}")
    if mean_amount < ctx.baseline.get("mean_amount_min", 72.76) or mean_amount > ctx.baseline.get("mean_amount_max", 90.61):
        return Verdict(alert=True, pillar="checks", reason=f"distribution_shift: {mean_amount}")
    if staleness_min > ctx.baseline.get("staleness_min_max", 8.42):
        return Verdict(alert=True, pillar="checks", reason=f"freshness_lag: {staleness_min}")

    # 2. Multi-signal Composite Distance (normalized Euclidean distance)
    # mu_rows ~ 498.38, sigma_rows ~ 20.97, mu_mean ~ 81.68, sigma_mean ~ 2.97
    mu_rows = (ctx.baseline.get("row_count_min", 435.47) + ctx.baseline.get("row_count_max", 561.29)) / 2.0
    sigma_rows = (ctx.baseline.get("row_count_max", 561.29) - ctx.baseline.get("row_count_min", 435.47)) / 6.0
    mu_mean = (ctx.baseline.get("mean_amount_min", 72.76) + ctx.baseline.get("mean_amount_max", 90.61)) / 2.0
    sigma_mean = (ctx.baseline.get("mean_amount_max", 90.61) - ctx.baseline.get("mean_amount_min", 72.76)) / 6.0

    z_rows = (row_count - mu_rows) / sigma_rows
    z_mean = (mean_amount - mu_mean) / sigma_mean
    z_null = null_rate / max(ctx.baseline.get("null_rate_max", 0.0109), 1e-6)
    z_stale = staleness_min / max(ctx.baseline.get("staleness_min_max", 8.42), 1e-6)
    norm2 = math.sqrt(z_rows ** 2 + z_mean ** 2 + z_null ** 2 + z_stale ** 2)

    # Conservatively thresholded to avoid false positives (clean median ~1.4-1.5, max ~3.06)
    if norm2 > 3.2:
        return Verdict(alert=True, pillar="checks", reason=f"composite_anomaly: norm2={norm2:.3f}")

    return Verdict(alert=False, pillar="checks")


def check_contract_checkpoint(payload, ctx):
    """Pillar: contracts.
    Detects schema_break, type_violation, and sla_violation.
    """
    contract_id = payload.get("contract_id")
    checkpoint_batch_id = payload.get("checkpoint_batch_id")
    diff = ctx.tools.contract_diff(contract_id, checkpoint_batch_id)
    if "error" in diff:
        return Verdict(alert=False, pillar="contracts", reason=diff["error"])

    violations = diff.get("violations", [])
    if violations:
        return Verdict(alert=True, pillar="contracts", reason=f"contract_violations: {violations}")

    freshness_delay = diff.get("freshness_delay_min", 0.0)
    sla_declared = payload.get("declared_sla", {}).get("freshness_min", 15)
    sla_limit = min(sla_declared, ctx.baseline.get("freshness_delay_max_min", 11.11))
    if freshness_delay > sla_limit:
        return Verdict(alert=True, pillar="contracts", reason=f"sla_violation: {freshness_delay:.2f} > {sla_limit:.2f}")

    return Verdict(alert=False, pillar="contracts")


def check_lineage_run(payload, ctx):
    """Pillar: lineage.
    Detects orphan_output, missing_upstream, and runtime_anomaly.
    """
    run_id = payload.get("run_id")
    slice_data = ctx.tools.lineage_graph_slice(run_id)
    if "error" in slice_data:
        return Verdict(alert=False, pillar="lineage", reason=slice_data["error"])

    # 1. Orphan output: downstream count == 0
    if slice_data.get("actual_downstream_count", 0) == 0:
        return Verdict(alert=True, pillar="lineage", reason="orphan_output: actual_downstream_count == 0")

    # 2. Missing upstream: stg_orders requires both raw.orders and raw.customers
    actual_upstream = slice_data.get("actual_upstream", [])
    if "raw.orders" not in actual_upstream or "raw.customers" not in actual_upstream:
        return Verdict(alert=True, pillar="lineage", reason=f"missing_upstream: {actual_upstream}")

    # 3. Runtime anomaly: duration exceeds baseline threshold
    duration_ms = slice_data.get("duration_ms", 0.0)
    if duration_ms > ctx.baseline.get("lineage_duration_ms_max", 5134.98):
        return Verdict(alert=True, pillar="lineage", reason=f"runtime_anomaly: {duration_ms:.1f}ms")

    return Verdict(alert=False, pillar="lineage")


def check_feature_materialization(payload, ctx):
    """Pillar: ai_infra.
    Detects feature_skew / training-serving drift.
    """
    feature_view = payload.get("feature_view")
    batch_id = payload.get("batch_id")
    drift = ctx.tools.feature_drift(feature_view, batch_id)
    if "error" in drift:
        return Verdict(alert=False, pillar="ai_infra", reason=drift["error"])

    # Calibrated separation threshold: clean data <= 0.472, faulty skew >= 0.970
    mean_shift_sigma = drift.get("mean_shift_sigma", 0.0)
    if mean_shift_sigma > 0.65:
        return Verdict(alert=True, pillar="ai_infra", reason=f"feature_skew: mean_shift_sigma={mean_shift_sigma:.3f} > 0.65")

    return Verdict(alert=False, pillar="ai_infra")


def check_embedding_batch(payload, ctx):
    """Pillar: ai_infra.
    Detects embedding_drift and corpus_staleness in vector search / RAG.
    """
    corpus = payload.get("corpus")
    chunk_batch_id = payload.get("chunk_batch_id")
    drift = ctx.tools.embedding_drift(corpus, chunk_batch_id)
    if "error" in drift:
        return Verdict(alert=False, pillar="ai_infra", reason=drift["error"])

    centroid_shift = drift.get("centroid_shift", 0.0)
    if centroid_shift > ctx.baseline.get("embedding_centroid_shift_max", 0.0435):
        return Verdict(alert=True, pillar="ai_infra", reason=f"embedding_drift: {centroid_shift:.4f}")

    avg_doc_age = drift.get("avg_doc_age_days", 0.0)
    if avg_doc_age > ctx.baseline.get("corpus_avg_doc_age_days_max", 49.80):
        return Verdict(alert=True, pillar="ai_infra", reason=f"corpus_staleness: {avg_doc_age:.1f} days")

    return Verdict(alert=False, pillar="ai_infra")

