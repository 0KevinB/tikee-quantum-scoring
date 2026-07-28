"""CLI: generate | fidelity | select | train | evaluate | fairness | multiseed | report | all.

ARCHITECTURE.md §10. Los subcomandos pesados (fidelity, multiseed, fairness) invocan
los scripts ya verificados en `scripts/` en vez de reimplementar la orquestación, para
no arriesgar una segunda ruta de código sin la misma cobertura de pruebas.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tikee.config import ensure_output_dirs, load_config
from tikee.data.seed_generator import generate_seed_table
from tikee.data.sdv_synthesizer import synthesize
from tikee.data.target_definition import add_target
from tikee.data.validate import quick_auc, validate_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"


def cmd_generate(args: argparse.Namespace) -> int:
    """`tikee generate --seed N [--n FILAS]`: genera un dataset completo (semilla
    -> síntesis -> target -> validación) y lo escribe en
    `data/processed/dataset_seed{N}.parquet`. Requiere `sigma_ruido` ya
    calibrado en `config.yaml` (ver `scripts/f1_calibrate_sigma.py`).

    Returns:
        0 si el dataset pasa `validate.validate_dataset`, 1 en caso contrario.
    """
    cfg = load_config()
    ensure_output_dirs(cfg)

    sigma = cfg.sigma_ruido
    if sigma is None:
        print(
            "sigma_ruido es null en config/config.yaml. Corre primero "
            "scripts/f1_calibrate_sigma.py (checkpoint F1) y guarda el valor calibrado.",
            file=sys.stderr,
        )
        return 1

    n_seed_rows = cfg.raw["data"]["n_seed_rows"]
    n_sample = args.n or cfg.raw["data"]["n_synthetic_rows"]
    target_rate = cfg.raw["data"]["tasa_base_default"]

    seed_df = generate_seed_table(args.seed, n_seed_rows)
    synth_df, synth_info = synthesize(seed_df, method="gaussian_copula", n_sample=n_sample, seed=args.seed)
    df = add_target(synth_df, sigma=sigma, seed=args.seed, target_rate=target_rate)

    auc = quick_auc(df, seed=args.seed)
    report = validate_dataset(df, measured_auc=auc, auc_band=tuple(cfg.raw["data"]["auc_band"]))

    out_path = cfg.path("processed") / f"dataset_seed{args.seed}.parquet"
    df.to_parquet(out_path)

    print(f"Generado {out_path} ({len(df)} filas). engine={synth_info['engine']}")
    print(f"tasa_default={df['default'].mean():.4f}  AUC={auc:.4f}")
    for name in ("ranges", "default_rate", "hard_constraint", "protected_not_predictor", "correlations", "auc_band"):
        print(f"  {name}: {'OK' if report[name]['ok'] else 'FALLA'}")

    if not report["ok"]:
        print("\nvalidate.py: dataset NO pasa los chequeos duros. Abortando.", file=sys.stderr)
        return 1
    return 0


def _run_script(name: str, extra_args: list[str] | None = None) -> int:
    """Ejecuta `scripts/{name}` como subproceso con el mismo intérprete de
    Python, y propaga su código de salida."""
    cmd = [sys.executable, str(SCRIPTS / name), *(extra_args or [])]
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


def cmd_fidelity(args: argparse.Namespace) -> int:
    """`tikee fidelity`: delega en `scripts/f2_fidelity_comparison.py` (F2)."""
    return _run_script("f2_fidelity_comparison.py")


def cmd_select(args: argparse.Namespace) -> int:
    """`tikee select --level {A,B}`: delega en el script de F4 (Nivel A) o F5
    (Nivel B). `--method` se ignora hoy — cada script ya resuelve los 3-5
    solucionadores aplicables a su nivel."""
    script = "f4_run_level_a_qubo.py" if args.level == "A" else "f5_run_level_b.py"
    return _run_script(script)


def cmd_train(args: argparse.Namespace) -> int:
    """`tikee train --arm X`: no hay un entry point de un solo brazo — imprime
    cómo llamar a `run_experiment.run_seed` directamente, o usar `multiseed`."""
    print(
        f"Para entrenar un brazo individual (arm={args.arm}, seed={args.seed}) usa "
        "tikee.experiments.run_experiment.run_seed(seed, xgb_params_a, xgb_params_b) "
        "en un script propio, o corre `multiseed` que ya entrena todos los brazos.",
    )
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """`tikee evaluate`: las métricas ya se generan como parte de `multiseed`;
    este subcomando solo señala dónde encontrarlas."""
    print("Las métricas por brazo se generan como parte de `multiseed` (reports/results.json).")
    return 0


def cmd_fairness(args: argparse.Namespace) -> int:
    """`tikee fairness`: delega en `scripts/f7_fairness_audit.py` (F7)."""
    return _run_script("f7_fairness_audit.py")


def cmd_multiseed(args: argparse.Namespace) -> int:
    """`tikee multiseed`: delega en `scripts/f6_run_multiseed.py` (F3-F6). El
    flag `--seeds` se ignora hoy — el script usa las 10 semillas de
    `config.yaml`; se deja el flag para uso futuro."""
    return _run_script("f6_run_multiseed.py")


def cmd_report(args: argparse.Namespace) -> int:
    """`tikee report`: regenera `reports/RESULTS.md` (solo las tablas, no la
    narrativa) desde `reports/results.json`. `reports/INFORME.md` es manual y
    no se toca aquí."""
    import json

    results_path = REPO_ROOT / "reports" / "results.json"
    if not results_path.exists():
        print(f"Falta {results_path}. Corre `tikee multiseed` primero.", file=sys.stderr)
        return 1

    results = json.loads(results_path.read_text())
    lines = ["# RESULTS.md — Tikee Quantum-Inspired Scoring (regenerado)\n"]
    for level in ("Nivel A", "Nivel B"):
        if level not in results:
            continue
        lines.append(f"\n## {level} — AUC media ± dp a través de {len(results['seeds'])} semillas\n")
        lines.append("| Brazo | AUC media | dp | min | max |")
        lines.append("|---|---|---|---|---|")
        for arm, s in results[level]["summary"].items():
            lines.append(f"| {arm} | {s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} |")
        lines.append(f"\nFriedman: {results[level]['friedman']}\n")

    out_path = REPO_ROOT / "reports" / "RESULTS.md"
    out_path.write_text("\n".join(lines))
    print(f"Regenerado {out_path} desde {results_path}.")
    print("Nota: INFORME.md es narrativo y se mantiene escrito a mano; no se regenera aquí.")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    """`tikee all`: pipeline completo end-to-end — generate, multiseed (F3-F6),
    fairness (F7), fidelity (F2) y report — abortando en el primer paso que
    falle. Es lo que corre `make experiment`."""
    steps = [
        ("generate --seed 42", lambda: cmd_generate(argparse.Namespace(seed=42, n=None))),
        ("multiseed (F3-F6)", lambda: _run_script("f6_run_multiseed.py")),
        ("fairness (F7)", lambda: _run_script("f7_fairness_audit.py")),
        ("fidelity (F2)", lambda: _run_script("f2_fidelity_comparison.py")),
        ("report", lambda: cmd_report(args)),
    ]
    for name, fn in steps:
        print(f"\n=== {name} ===")
        rc = fn()
        if rc != 0:
            print(f"Paso '{name}' falló (rc={rc}). Abortando pipeline.", file=sys.stderr)
            return rc
    print("\nPipeline completo. Corre `streamlit run app/streamlit_app.py` (o `make app`) para ver la demo.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser con los 9 subcomandos de ARCHITECTURE.md §10."""
    parser = argparse.ArgumentParser(prog="tikee")
    sub = parser.add_subparsers(dest="command", required=True)

    p_generate = sub.add_parser("generate", help="genera dataset_seed{N}.parquet")
    p_generate.add_argument("--seed", type=int, default=42)
    p_generate.add_argument("--n", type=int, default=None)
    p_generate.set_defaults(func=cmd_generate)

    p_fidelity = sub.add_parser("fidelity", help="SDMetrics + comparación de sintetizadores (F2)")
    p_fidelity.add_argument("--synthesizer", choices=["gaussian_copula", "ctgan", "tvae", "all"], default="all")
    p_fidelity.set_defaults(func=cmd_fidelity)

    p_select = sub.add_parser("select", help="selección de variables (F4/F5)")
    p_select.add_argument("--level", choices=["A", "B"], default="A")
    p_select.add_argument("--method", default="qubo-sa")
    p_select.set_defaults(func=cmd_select)

    p_train = sub.add_parser("train", help="entrena un brazo (F3)")
    p_train.add_argument("--arm", required=True)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="evalúa métricas (F3)")
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.set_defaults(func=cmd_evaluate)

    p_fair = sub.add_parser("fairness", help="auditoría de equidad (F7)")
    p_fair.add_argument("--seed", type=int, default=42)
    p_fair.set_defaults(func=cmd_fairness)

    p_multi = sub.add_parser("multiseed", help="bucle multi-semilla (F6)")
    p_multi.add_argument("--seeds", default="42,101,202,303,404,505,606,707,808,909")
    p_multi.set_defaults(func=cmd_multiseed)

    p_report = sub.add_parser("report", help="results.json -> RESULTS.md + figuras (F8)")
    p_report.set_defaults(func=cmd_report)

    p_all = sub.add_parser("all", help="pipeline completo (F8)")
    p_all.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de `python -m tikee.cli`: parsea argv y despacha al
    handler del subcomando elegido."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
