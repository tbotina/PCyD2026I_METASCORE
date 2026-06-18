"""
Benchmark automatizado — implementaciones Python de MetaScore HPC.

Ejecuta sequential.py (T0) y multicore.py con P = 1, 2, 4, 6, cpu_count(),
calcula speedup y eficiencia para cada P, estima la fracción paralela f
con la Ley de Amdahl, e imprime la tabla lista para el informe.

Uso (desde la raíz del proyecto):
    python python/benchmark_python.py
"""
import multiprocessing as mp
import re
import subprocess
import sys


# ──────────────────────────── Configuración ─────────────────────────────────
K_CANDIDATES = 100_000
WORKER_VALUES = sorted({1, 2, 4, 6, mp.cpu_count()})   # deduplica y ordena


# ──────────────────────────── Ejecución de scripts ──────────────────────────

def run_and_parse(cmd: list) -> dict:
    """
    Ejecuta un comando y extrae Tiempo, AUC y Consistencia del stdout.

    Returns:
        dict con claves 'tiempo' (float), 'auc' (float), 'consistencia' (float).
    """
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        print(f"\nError al ejecutar: {' '.join(cmd)}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    out = result.stdout
    tiempo      = _parse_float(out, r"Tiempo\s*=\s*([\d.]+)")
    auc         = _parse_float(out, r"AUC\s*=\s*([\d.]+)")
    consistencia = _parse_float(out, r"Consistencia\s*=\s*([\d.]+)")
    return {"tiempo": tiempo, "auc": auc, "consistencia": consistencia}


def _parse_float(text: str, pattern: str) -> float | None:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


# ──────────────────────────── Cálculo de métricas ───────────────────────────

def amdahl_f(speedup: float, p: int) -> float:
    """
    Estima la fracción paralela f a partir de un punto (P, S(P)).

    Despeja f de S(P) = 1 / ((1-f) + f/P):
        f = (S - 1) / (S * (1 - 1/P))
    """
    if p <= 1 or speedup <= 1:
        return 0.0
    return (speedup - 1) / (speedup * (1 - 1 / p))


# ──────────────────────────── Punto de entrada ──────────────────────────────

def main() -> None:
    python = sys.executable
    sep    = "=" * 68

    print(sep)
    print("  Benchmark Python — MetaScore HPC")
    print(sep)

    # ── Baseline secuencial ───────────────────────────────────────────────
    print(f"\n  Ejecutando sequential.py  (K={K_CANDIDATES:,}) ...")
    seq = run_and_parse([python, "python/sequential.py", "--k", str(K_CANDIDATES)])
    t0  = seq["tiempo"]
    print(f"  → T₀ = {t0:.4f} s  |  AUC = {seq['auc']:.4f}"
          f"  |  Consistencia = {seq['consistencia']:.4f}")

    # ── Multicore con distintos P ─────────────────────────────────────────
    rows = []
    total = len(WORKER_VALUES)
    for idx, p in enumerate(WORKER_VALUES, 1):
        print(f"\n  [{idx}/{total}] multicore.py --workers {p} ...")
        r = run_and_parse([python, "python/multicore.py", "--workers", str(p), "--k", str(K_CANDIDATES)])
        tp         = r["tiempo"]
        speedup    = t0 / tp if tp else None
        efficiency = speedup / p if speedup else None
        rows.append({"p": p, "tiempo": tp, "speedup": speedup,
                     "efficiency": efficiency, "auc": r["auc"]})
        print(f"  → T({p:>2}) = {tp:.4f} s  |"
              f"  S = {speedup:.2f}×  |"
              f"  E = {efficiency:.4f}  |"
              f"  AUC = {r['auc']:.4f}")

    # ── Tabla resumen ─────────────────────────────────────────────────────
    print(f"\n\n{sep}")
    print("  TABLA DE RESULTADOS — Python Multicore vs Secuencial")
    print(sep)
    print(f"  T₀ (secuencial) = {t0:.4f} s\n")
    print(f"  {'P':>4}  {'T(P) [s]':>10}  {'Speedup':>8}  {'Eficiencia':>10}  {'AUC':>6}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*6}")
    for r in rows:
        print(f"  {r['p']:>4}  {r['tiempo']:>10.4f}  "
              f"{r['speedup']:>8.2f}  {r['efficiency']:>10.4f}  {r['auc']:>6.4f}")

    # ── Amdahl: estimación de f con el punto de mayor P ──────────────────
    best = max(rows, key=lambda r: r["p"])
    f_empirico = amdahl_f(best["speedup"], best["p"])
    f_empirico = max(0.0, min(1.0, f_empirico))
    s_max      = 1.0 / (1.0 - f_empirico) if f_empirico < 1.0 else float("inf")

    print(f"\n  Ley de Amdahl (estimada con P={best['p']}, S={best['speedup']:.2f}×):")
    print(f"    Fracción paralela  f   ≈ {f_empirico:.4f}  ({f_empirico*100:.1f} %)")
    print(f"    Fracción serial  1-f   ≈ {1-f_empirico:.4f}  ({(1-f_empirico)*100:.1f} %)")
    print(f"    Speedup máximo teórico  ≈ {s_max:.2f}×  (P → ∞)")
    print(sep)


if __name__ == "__main__":
    main()
