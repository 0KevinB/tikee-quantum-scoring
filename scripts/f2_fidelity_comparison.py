"""Checkpoint F2 (PLAN.md §4.1): SDMetrics completo sobre GaussianCopula (producción) +
tabla comparativa de tres sintetizadores + justificación de D14 con números.

Uso: .venv/bin/python scripts/f2_fidelity_comparison.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tikee.data.fidelity import (  # noqa: E402
    compare_synthesizers,
    new_row_synthesis,
    nearest_record_distance,
    run_sdmetrics_reports,
    target_correlation_mae,
)
from tikee.data.seed_generator import generate_seed_table  # noqa: E402
from tikee.data.sdv_synthesizer import synthesize  # noqa: E402
from tikee.data.target_definition import add_target  # noqa: E402

SEED = 42
SIGMA = 6.2


def main() -> int:
    seed_df = generate_seed_table(SEED, 2000)

    print("=== Fidelidad completa: GaussianCopula (sintetizador de producción, D14) ===")
    prod_sample, prod_info = synthesize(seed_df, method="gaussian_copula", n_sample=8000, seed=SEED)
    sdm = run_sdmetrics_reports(seed_df, prod_sample)
    nrs = new_row_synthesis(seed_df, prod_sample)
    nrd = nearest_record_distance(seed_df, prod_sample)
    full_df = add_target(prod_sample, sigma=SIGMA, seed=SEED)
    corr_mae_raw = target_correlation_mae(full_df, exclude_known_exceptions=False)
    corr_mae = target_correlation_mae(full_df, exclude_known_exceptions=True)

    print(f"QualityReport global:        {sdm['quality_overall']:.4f}  (umbral > 0.85 forma, > 0.80 pares)")
    print(f"  Column Shapes:             {sdm['quality_column_shapes']:.4f}")
    print(f"  Column Pair Trends:        {sdm['quality_column_pair_trends']:.4f}")
    print(f"DiagnosticReport global:     {sdm['diagnostic_overall']:.4f}  (umbral == 1.00)")
    print(f"NewRowSynthesis:             {nrs:.4f}  (umbral > 0.95)")
    print(f"Distancia al vecino real más cercano (normalizada): {nrd}")
    print(f"MAE de las 13 correlaciones objetivo (crudo):                    {corr_mae_raw:.4f}  (umbral < 0.05)")
    print(f"MAE excluyendo la desviación documentada de F1 (12 pares):      {corr_mae:.4f}  (umbral < 0.05)")
    print("  -> el par ratio_cuota_ingreso~ingreso_mensual es una imposibilidad algebraica")
    print("     documentada en validate.py (KNOWN_CORRELATION_EXCEPTIONS), no un error de ajuste.")

    print("\n=== Comparación de tres sintetizadores (E12) ===")
    full_reference = add_target(prod_sample, sigma=SIGMA, seed=SEED)
    comparison = compare_synthesizers(
        seed_df, full_reference, methods=("gaussian_copula", "ctgan", "tvae"),
        n_sample=8000, seed=SEED, ctgan_epochs=300, time_limit_s=40 * 60,
    )
    for row in comparison["rows"]:
        print(row)

    out_dir = Path(__file__).resolve().parents[1] / "reports" / "fidelity"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "production_gaussian_copula": {
            "sdmetrics": sdm, "new_row_synthesis": nrs, "nearest_record_distance": nrd,
            "target_correlation_mae_raw_13_pairs": corr_mae_raw,
            "target_correlation_mae_excl_known_exception_12_pairs": corr_mae,
        },
        "synthesizer_comparison": comparison["rows"],
        "checkpoint_f2": {
            "quality_ok": sdm["quality_column_shapes"] > 0.85 and sdm["quality_column_pair_trends"] > 0.80,
            "diagnostic_ok": sdm["diagnostic_overall"] >= 0.999,
            "new_row_synthesis_ok": nrs > 0.95,
            "correlation_mae_ok": corr_mae < 0.05,
        },
    }
    report["checkpoint_f2"]["all_ok"] = all(report["checkpoint_f2"].values())
    (out_dir / "f2_comparison.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nGuardado en {out_dir / 'f2_comparison.json'}")

    print("\n--- Checkpoint F2 ---")
    for k, v in report["checkpoint_f2"].items():
        print(f"  {k}: {v}")

    return 0 if report["checkpoint_f2"]["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
