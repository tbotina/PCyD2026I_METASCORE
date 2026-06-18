"""
Genera las 3 gráficas comparativas del benchmark MetaScore HPC.

Uso (desde la raíz del proyecto):
    python results/generar_graficas.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv

# ──────────────────────────── Cargar datos ───────────────────────────────────

def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

PLOTS_DIR = "results/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

rows = load_csv("results/benchmark.csv")

# Convertir a float donde aplique
for r in rows:
    r["tiempo_s"]    = float(r["tiempo_s"])
    r["speedup_abs"] = float(r["speedup_abs"]) if r["speedup_abs"] != "N/A" else None
    r["speedup_rel"] = float(r["speedup_rel"]) if r["speedup_rel"] != "N/A" else None
    r["eficiencia"]  = float(r["eficiencia"])  if r["eficiencia"]  != "N/A" else None
    r["nucleos"]     = r["nucleos"]

# Paleta de colores consistente
COLORES = {
    "Python": "#3498DB",
    "OpenMP": "#E67E22",
    "MPI":    "#8E44AD",
    "CUDA":   "#E74C3C",
}

# ══════════════════════════════════════════════════════════════════════════════
# Gráfica 1 — Comparativa de tiempos (mejor tiempo por implementación)
# ══════════════════════════════════════════════════════════════════════════════

mejores = [
    ("Python\nSecuencial",   next(r["tiempo_s"] for r in rows if r["implementacion"] == "Python_Secuencial"),    COLORES["Python"]),
    ("Python\nMulticore\n(P=12)", next(r["tiempo_s"] for r in rows if r["implementacion"] == "Python_Multicore_P12"), COLORES["Python"]),
    ("C\nOpenMP\n(P=12)",    next(r["tiempo_s"] for r in rows if r["implementacion"] == "C_OpenMP_P12"),          COLORES["OpenMP"]),
    ("C\nMPI\n(P=6)",        next(r["tiempo_s"] for r in rows if r["implementacion"] == "C_MPI_P6"),              COLORES["MPI"]),
    ("CUDA\nPyCUDA",         next(r["tiempo_s"] for r in rows if r["implementacion"] == "CUDA_PyCUDA"),           COLORES["CUDA"]),
    ("CUDA C",               next(r["tiempo_s"] for r in rows if r["implementacion"] == "CUDA_C"),               COLORES["CUDA"]),
]

labels  = [m[0] for m in mejores]
tiempos = [m[1] for m in mejores]
colores = [m[2] for m in mejores]

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(labels, tiempos, color=colores, edgecolor="white", linewidth=1.2, width=0.6)

for bar, t in zip(bars, tiempos):
    if t >= 1:
        label = f"{t:.2f} s"
    elif t >= 0.001:
        label = f"{t*1000:.2f} ms"
    else:
        label = f"{t*1000:.3f} ms"
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.4,
            label, ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_yscale("log")
ax.set_ylabel("Tiempo de búsqueda (s) — escala logarítmica", fontsize=11)
ax.set_title(f"Comparativa de tiempos de ejecución\nK=100,000  |  n_items=50  |  Hardware: Ryzen 5 5500U + NVIDIA T4",
             fontsize=12, fontweight="bold")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{x:.3f}s" if x >= 0.001 else f"{x*1000:.3f}ms"))
ax.grid(True, axis="y", alpha=0.3, which="both")
ax.set_ylim(bottom=1e-4)

from matplotlib.patches import Patch
leyenda = [Patch(color=COLORES["Python"], label="Python"),
           Patch(color=COLORES["OpenMP"], label="C + OpenMP"),
           Patch(color=COLORES["MPI"],    label="C + MPI"),
           Patch(color=COLORES["CUDA"],   label="CUDA (GPU T4)")]
ax.legend(handles=leyenda, loc="upper right", framealpha=0.9)

plt.tight_layout()
out1 = f"{PLOTS_DIR}/comparativa_tiempos.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"[1/3] Guardada: {out1}")

# ══════════════════════════════════════════════════════════════════════════════
# Gráfica 2 — Speedup relativo vs P
# ══════════════════════════════════════════════════════════════════════════════

impls_speedup = [
    ("Python Multicore", "Python_Multicore", COLORES["Python"], "o--"),
    ("C + OpenMP",       "C_OpenMP",         COLORES["OpenMP"], "s-"),
    ("C + MPI",          "C_MPI",            COLORES["MPI"],    "^-"),
]

fig, ax = plt.subplots(figsize=(9, 6))

p_vals = [1, 2, 4, 6, 12]

for nombre, prefijo, color, estilo in impls_speedup:
    ps, ss = [], []
    for p in p_vals:
        key = f"{prefijo}_P{p}"
        row = next((r for r in rows if r["implementacion"] == key), None)
        if row and row["speedup_rel"] is not None:
            ps.append(p)
            ss.append(row["speedup_rel"])
    if ps:
        ax.plot(ps, ss, estilo, color=color, label=nombre,
                linewidth=2, markersize=8, markerfacecolor="white",
                markeredgewidth=2)
        for p, s in zip(ps, ss):
            ax.annotate(f"{s:.2f}×", (p, s),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color=color)

# Línea de speedup ideal
p_ideal = np.array([1, 2, 4, 6, 12])
ax.plot(p_ideal, p_ideal, "k:", linewidth=1.2, alpha=0.5, label="Ideal (S=P)")

ax.set_xlabel("Número de procesos / hilos (P)", fontsize=11)
ax.set_ylabel("Speedup relativo S(P) = T(1) / T(P)", fontsize=11)
ax.set_title("Speedup relativo vs número de procesos\nK=100,000  |  n_items=50",
             fontsize=12, fontweight="bold")
ax.set_xticks([1, 2, 4, 6, 12])
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 13)
ax.set_ylim(bottom=0)

plt.tight_layout()
out2 = f"{PLOTS_DIR}/speedup_vs_P.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"[2/3] Guardada: {out2}")

# ══════════════════════════════════════════════════════════════════════════════
# Gráfica 3 — Eficiencia vs P
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 6))

for nombre, prefijo, color, estilo in impls_speedup:
    ps, es = [], []
    for p in p_vals:
        key = f"{prefijo}_P{p}"
        row = next((r for r in rows if r["implementacion"] == key), None)
        if row and row["eficiencia"] is not None:
            ps.append(p)
            es.append(row["eficiencia"])
    if ps:
        ax.plot(ps, es, estilo, color=color, label=nombre,
                linewidth=2, markersize=8, markerfacecolor="white",
                markeredgewidth=2)
        for p, e in zip(ps, es):
            ax.annotate(f"{e:.2f}", (p, e),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color=color)

ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2,
           alpha=0.5, label="Eficiencia ideal (E=1)")
ax.axhline(0.8, color="gray", linestyle="--", linewidth=1,
           alpha=0.4, label="Umbral aceptable (E=0.8)")

ax.set_xlabel("Número de procesos / hilos (P)", fontsize=11)
ax.set_ylabel("Eficiencia E(P) = S(P) / P", fontsize=11)
ax.set_title("Eficiencia vs número de procesos\nK=100,000  |  n_items=50",
             fontsize=12, fontweight="bold")
ax.set_xticks([1, 2, 4, 6, 12])
ax.set_ylim(0, 1.15)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 13)

plt.tight_layout()
out3 = f"{PLOTS_DIR}/eficiencia_vs_P.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"[3/3] Guardada: {out3}")

print("\nTodas las graficas generadas en results/plots/")
