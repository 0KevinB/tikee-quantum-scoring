"""Checkpoint F1 (PLAN.md §4.1, innegociable): calibra sigma por bisección hasta que
el AUC de una logística con las 18 variables cae en la banda [0.72, 0.82]. Una vez
encontrado, se fija en config/config.yaml y se reutiliza en las 10 semillas (D11).

Uso: .venv/bin/python scripts/f1_calibrate_sigma.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tikee.data.seed_generator import generate_seed_table  # noqa: E402
from tikee.data.sdv_synthesizer import synthesize  # noqa: E402
from tikee.data.target_definition import add_target  # noqa: E402
from tikee.data.validate import quick_auc, validate_dataset  # noqa: E402

SEED = 42
BAND = (0.72, 0.82)
N_SEED_ROWS = 2000
N_SAMPLE = 8000


def measure_auc_for_sigma(sigma: float):
    seed_df = generate_seed_table(SEED, N_SEED_ROWS)
    synth_df, synth_info = synthesize(seed_df, method="gaussian_copula", n_sample=N_SAMPLE, seed=SEED)
    df = add_target(synth_df, sigma=sigma, seed=SEED)
    auc = quick_auc(df, seed=SEED)
    return auc, df, synth_info


def calibrate_sigma(lo: float = 0.1, hi: float = 6.0, max_iter: int = 8):
    history = []
    best = None
    for it in range(1, max_iter + 1):
        mid = (lo + hi) / 2
        auc, df, synth_info = measure_auc_for_sigma(mid)
        history.append({"iter": it, "sigma": mid, "auc": auc})
        print(f"[iter {it}] sigma={mid:.4f} -> AUC={auc:.4f} "
              f"(tasa_default={df['default'].mean():.4f}, beta0={df.attrs['beta0']:.3f}, "
              f"engine={synth_info['engine']})")

        if best is None or abs(auc - sum(BAND) / 2) < abs(best["auc"] - sum(BAND) / 2):
            best = {"sigma": mid, "auc": auc, "df": df, "synth_info": synth_info}

        if BAND[0] <= auc <= BAND[1]:
            print(f"\nBanda alcanzada en la iteración {it}.")
            return mid, auc, df, history

        if auc > BAND[1]:
            lo = mid
        else:
            hi = mid

    print(f"\nNo se alcanzó la banda en {max_iter} iteraciones; "
          f"se usa el mejor candidato (sigma={best['sigma']:.4f}, AUC={best['auc']:.4f}).")
    return best["sigma"], best["auc"], best["df"], history


def main() -> int:
    sigma, auc, df, history = calibrate_sigma()

    report = validate_dataset(df, measured_auc=auc)
    print("\n--- Checkpoint F1 ---")
    print(f"sigma calibrado: {sigma:.4f}")
    print(f"AUC medido: {auc:.4f} (banda {BAND})")
    print(f"tasa de default: {df['default'].mean():.4f}")
    for name in ["ranges", "default_rate", "hard_constraint", "protected_not_predictor", "correlations", "auc_band"]:
        print(f"  {name}: {'OK' if report[name]['ok'] else 'FALLA'}")
    print(f"\nCHECKPOINT F1: {'PASA' if report['ok'] else 'FALLA'}")

    out_dir = Path(__file__).resolve().parents[1] / "reports" / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "f1_sigma_calibration.json").write_text(json.dumps({
        "sigma": sigma,
        "auc": auc,
        "history": history,
        "report": {k: v for k, v in report.items() if k != "vif"},
        "vif": report["vif"],
        "beta0": df.attrs["beta0"],
    }, indent=2, default=str))
    print(f"\nDetalle guardado en {out_dir / 'f1_sigma_calibration.json'}")

    df.to_parquet(Path(__file__).resolve().parents[1] / "data" / "processed" / "dataset_seed42_calibration.parquet")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
