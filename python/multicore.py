"""
Nivel 1 — Python Multicore: Búsqueda aleatoria paralela con multiprocessing.

Estrategia:
  1. Muestrear K candidatos W en el proceso principal.
  2. Repartir los K candidatos equitativamente entre cpu_count() procesos.
  3. Cada proceso hijo evalúa su lote de forma independiente (sin IPC en el bucle).
  4. El proceso principal recoge los mejores locales y selecciona el óptimo global.

La separación en lotes permite que multiprocessing.Pool.map evite overhead
de IPC por iteración: solo se comunica una vez por proceso (el resultado final).
"""
import argparse
import multiprocessing as mp
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
    Carga A, y, T, S, F desde disco.

    Returns:
        (A, y, T, S, F) — ver sequential.py para descripciones de tipos.

    Raises:
        FileNotFoundError: Si falta algún archivo .npy.
    """
    base = Path(data_dir)
    required = ["matrix_A.npy", "labels.npy", "profile_T.npy", "profile_S.npy", "profile_F.npy"]
    for fname in required:
        if not (base / fname).exists():
            raise FileNotFoundError(
                f"Archivo '{fname}' no encontrado. "
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
    Muestrea K vectores de peso sobre el símplex {W1+W2+W3=1, Wi≥0}.

    Ver sequential.py para la justificación del método de las exponenciales.
    """
    raw = rng.exponential(scale=1.0, size=(k, 3))
    return (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)


# ──────────────────────────── Worker (proceso hijo) ─────────────────────────

def _evaluate_batch(args: tuple) -> tuple:
    """
    Evalúa un lote de candidatos W y retorna el mejor (W, AUC) del lote.

    Función de nivel de módulo para ser serializable por pickle
    (requerimiento de multiprocessing en Windows/macOS con spawn).

    Args:
        args: (candidates, A, T, S, F, y)
            · candidates: (K_local, 3) float32 — vectores W del lote.
            · A:          (10, N)      float32.
            · T, S, F:    (N,)         float32.
            · y:          (10,)        int32.

    Returns:
        (W_best_local, auc_best_local)
    """
    candidates, A, T, S, F, y = args

    best_auc = -1.0
    best_W   = candidates[0].copy()

    for W in candidates:
        P            = W[0] * T + W[1] * S + W[2] * F
        sample_scores = A @ P
        auc          = float(roc_auc_score(y, sample_scores))
        if auc > best_auc:
            best_auc = auc
            best_W   = W.copy()

    return best_W, best_auc


# ──────────────────────────── Búsqueda paralela ─────────────────────────────

def random_search_parallel(
    K: int,
    A: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    F: np.ndarray,
    y: np.ndarray,
    n_workers: int | None = None,
    seed: int = SEED,
) -> tuple:
    """
    Búsqueda aleatoria paralela del W* que maximiza el AUC.

    El tiempo medido cubre únicamente la búsqueda (excluye carga de datos
    e inicialización del Pool), siguiendo la sección 4.1 del enunciado.

    Args:
        K:         Número total de candidatos.
        A:         Matriz de contribución (10, N).
        T, S, F:   Perfiles por ítem (N,).
        y:         Etiquetas binarias (10,).
        n_workers: Número de procesos. None → mp.cpu_count().
        seed:      Semilla aleatoria (para reproducibilidad).

    Returns:
        (W_best, auc_best, elapsed_seconds)
    """
    if n_workers is None:
        n_workers = mp.cpu_count()

    rng        = np.random.default_rng(seed)
    candidates = sample_simplex(K, rng)
    batches    = np.array_split(candidates, n_workers)

    # Empaquetar argumentos para cada proceso hijo
    worker_args = [(batch, A, T, S, F, y) for batch in batches]

    t_start = time.perf_counter()

    with mp.Pool(processes=n_workers) as pool:
        local_results = pool.map(_evaluate_batch, worker_args)

    elapsed = time.perf_counter() - t_start

    # Seleccionar el óptimo global entre los mejores locales
    best_W, best_auc = max(local_results, key=lambda r: r[1])
    return best_W, best_auc, elapsed


# ──────────────────────────── Punto de entrada ──────────────────────────────

def _compute_consistency(W: np.ndarray, A: np.ndarray, T: np.ndarray,
                          S: np.ndarray, F: np.ndarray, y: np.ndarray) -> float:
    """Balanced accuracy con umbral θ = media de scores. Rango [0, 1]."""
    P      = W[0] * T + W[1] * S + W[2] * F
    scores = A @ P
    theta  = float(np.mean(scores))
    tp = int(((scores > theta) & (y == 1)).sum())
    tn = int(((scores <= theta) & (y == 0)).sum())
    return (tp / int(y.sum()) + tn / int((y == 0).sum())) / 2


def main() -> None:
    parser = argparse.ArgumentParser(description="MetaScore HPC — Nivel 1 Multicore")
    parser.add_argument("--workers", type=int, default=None,
                        help="Número de procesos (default: cpu_count())")
    parser.add_argument("--k", type=int, default=K_CANDIDATES,
                        help=f"Candidatos a evaluar (default: {K_CANDIDATES})")
    args = parser.parse_args()

    A, y, T, S, F = load_data()
    n_workers      = args.workers if args.workers is not None else mp.cpu_count()
    k              = args.k

    print(f"[Multicore]  K={k:,}  workers={n_workers}  N_items={A.shape[1]}")
    W_best, auc_best, elapsed = random_search_parallel(
        k, A, T, S, F, y, n_workers=n_workers, seed=SEED
    )

    consistency = _compute_consistency(W_best, A, T, S, F, y)

    print(f"  W*           = [{W_best[0]:.4f}, {W_best[1]:.4f}, {W_best[2]:.4f}]")
    print(f"  AUC          = {auc_best:.4f}")
    print(f"  Consistencia = {consistency:.4f}  {'[OK]' if consistency >= 0.8 else '[--]'}")
    print(f"  Tiempo       = {elapsed:.4f} s")


if __name__ == "__main__":
    main()
