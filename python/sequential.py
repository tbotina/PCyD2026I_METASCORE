"""
Nivel 1 — Python Baseline: Búsqueda aleatoria secuencial.

Estrategia:
  1. Cargar A, y, T, S, F desde disco.
  2. Muestrear K vectores W sobre el símplex {W1+W2+W3=1, Wi≥0}.
  3. Para cada W: calcular P → Score = A·P → AUC(y, Score).
  4. Retornar W* = argmáx AUC.

Este módulo sirve como referencia de correctitud y línea base de tiempo
para calcular el speedup de las implementaciones paralelas.
"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

# ──────────────────────────── Configuración ─────────────────────────────────
K_CANDIDATES: int = 10_000
SEED:         int = 0
DATA_DIR:     str = "data"


# ──────────────────────────── E/S de datos ──────────────────────────────────

def load_data(data_dir: str = DATA_DIR) -> tuple:
    """
    Carga la matriz A, los perfiles T, S, F y las etiquetas y desde disco.

    Args:
        data_dir: Ruta al directorio con los archivos .npy.

    Returns:
        (A, y, T, S, F)
        · A: (10, N) float32 — matriz de contribución.
        · y: (10,)   int32   — etiquetas binarias.
        · T: (N,)    float32 — perfil taxonómico.
        · S: (N,)    float32 — perfil ecológico.
        · F: (N,)    float32 — perfil funcional.

    Raises:
        FileNotFoundError: Si falta algún archivo. Ejecutar generate_data.py primero.
    """
    base = Path(data_dir)
    required = ["matrix_A.npy", "labels.npy", "profile_T.npy", "profile_S.npy", "profile_F.npy"]
    for fname in required:
        if not (base / fname).exists():
            raise FileNotFoundError(
                f"Archivo '{fname}' no encontrado en '{data_dir}/'. "
                "Ejecute: python data/generate_data.py"
            )

    A = np.load(base / "matrix_A.npy")
    y = np.load(base / "labels.npy")
    T = np.load(base / "profile_T.npy")
    S = np.load(base / "profile_S.npy")
    F = np.load(base / "profile_F.npy")
    return A, y, T, S, F


# ──────────────────────────── Muestreo del símplex ──────────────────────────

def sample_simplex(k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Muestrea K vectores de peso uniformemente sobre el símplex estándar.

    Usa el método de las exponenciales:
        xᵢ ~ Exp(1)  →  wᵢ = xᵢ / Σxⱼ  garantiza W₁+W₂+W₃=1, Wᵢ≥0.

    Args:
        k:   Número de vectores a generar.
        rng: Generador de números aleatorios (numpy).

    Returns:
        np.ndarray: Matriz (k, 3), float32.
    """
    raw = rng.exponential(scale=1.0, size=(k, 3))
    return (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)


# ──────────────────────────── Scoring ───────────────────────────────────────

def compute_item_scores(
    W: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    F: np.ndarray,
) -> np.ndarray:
    """
    Calcula el vector de scores por ítem según la ec. 2.1 del enunciado.

        P_i = W₁·T_i + W₂·S_i + W₃·F_i

    Args:
        W: Vector de pesos (3,).
        T: Perfil taxonómico (N,).
        S: Perfil ecológico (N,).
        F: Perfil funcional (N,).

    Returns:
        np.ndarray: Scores por ítem P (N,), float32.
    """
    return W[0] * T + W[1] * S + W[2] * F


def evaluate_candidate(
    W: np.ndarray,
    A: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    F: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Evalúa el AUC para un vector de pesos dado.

        Score = A · P(W)   →   AUC(y, Score)

    Args:
        W:    Vector de pesos (3,).
        A:    Matriz de contribución (10, N).
        T, S, F: Perfiles por ítem (N,).
        y:    Etiquetas binarias (10,).

    Returns:
        float: AUC ∈ [0, 1].
    """
    P = compute_item_scores(W, T, S, F)
    sample_scores = A @ P
    return float(roc_auc_score(y, sample_scores))


# ──────────────────────────── Búsqueda aleatoria ────────────────────────────

def random_search(
    K: int,
    A: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    F: np.ndarray,
    y: np.ndarray,
    seed: int = SEED,
) -> tuple:
    """
    Búsqueda aleatoria del W* que maximiza el AUC (implementación secuencial).

    El tiempo medido cubre únicamente la búsqueda (excluye carga de datos),
    siguiendo la definición de la sección 4.1 del enunciado.

    Args:
        K:       Número de candidatos a evaluar.
        A:       Matriz de contribución (10, N).
        T, S, F: Perfiles por ítem (N,).
        y:       Etiquetas binarias (10,).
        seed:    Semilla aleatoria.

    Returns:
        (W_best, auc_best, elapsed_seconds)
        · W_best: (3,) float32 — pesos óptimos.
        · auc_best: float      — AUC máximo encontrado.
        · elapsed: float       — tiempo de búsqueda en segundos.
    """
    rng = np.random.default_rng(seed)
    candidates = sample_simplex(K, rng)

    best_auc = -1.0
    best_W   = candidates[0].copy()

    t_start = time.perf_counter()

    for W in candidates:
        auc = evaluate_candidate(W, A, T, S, F, y)
        if auc > best_auc:
            best_auc = auc
            best_W   = W.copy()

    elapsed = time.perf_counter() - t_start
    return best_W, best_auc, elapsed


# ──────────────────────────── Validación de consistencia ────────────────────

def compute_consistency(scores: np.ndarray, y: np.ndarray, theta: float | None = None) -> float:
    """
    Calcula la consistencia del scoring según la ec. 2.4 del enunciado.

        Consistencia = (TP_rate + TN_rate) / 2

    Usa como umbral θ la media de scores si no se especifica.

    Args:
        scores: Vector de scores por muestra (10,).
        y:      Etiquetas binarias (10,).
        theta:  Umbral de decisión. None → media de scores.

    Returns:
        float: Consistencia ∈ [0, 1]. Se considera satisfactoria si ≥ 0.8.
    """
    if theta is None:
        theta = float(np.mean(scores))

    n_sick    = int(y.sum())
    n_healthy = len(y) - n_sick

    tp = int(((scores > theta) & (y == 1)).sum())
    tn = int(((scores <= theta) & (y == 0)).sum())

    # División por 2: normaliza a [0, 1] (balanced accuracy = media de TPR y TNR)
    return (tp / n_sick + tn / n_healthy) / 2


# ──────────────────────────── Punto de entrada ──────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="MetaScore HPC — Nivel 1 Secuencial")
    parser.add_argument("--k", type=int, default=K_CANDIDATES,
                        help=f"Candidatos a evaluar (default: {K_CANDIDATES})")
    args = parser.parse_args()
    k = args.k

    A, y, T, S, F = load_data()

    print(f"[Secuencial]  K={k:,}  N_items={A.shape[1]}")
    W_best, auc_best, elapsed = random_search(k, A, T, S, F, y)

    P_best       = compute_item_scores(W_best, T, S, F)
    scores_best  = A @ P_best
    consistency  = compute_consistency(scores_best, y)

    print(f"  W*           = [{W_best[0]:.4f}, {W_best[1]:.4f}, {W_best[2]:.4f}]")
    print(f"  AUC          = {auc_best:.4f}")
    print(f"  Consistencia = {consistency:.4f}  {'[OK]' if consistency >= 0.8 else '[--]'}")
    print(f"  Tiempo       = {elapsed:.4f} s")


if __name__ == "__main__":
    main()
