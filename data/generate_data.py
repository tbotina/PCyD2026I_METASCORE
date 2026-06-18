"""
Generación de datos sintéticos para el sistema de scoring metagenómico.

Genera:
  - A   ∈ R^{10×N}: matriz de contribución (filas Dirichlet → suma 1).
  - y   ∈ {0,1}^10: etiquetas (0 = sano, 1 = enfermo).
  - T   ∈ R^N:      perfil taxonómico por ítem (Dirichlet).
  - S   ∈ R^N:      perfil ecológico por ítem (Dirichlet).
  - F   ∈ {0,1}^N:  perfil funcional por ítem (presencia/ausencia de gen).

Los archivos se guardan en dos formatos:
  · .npy  — para las implementaciones Python.
  · .bin  — binario float32/int32 crudo, para C y CUDA.
  · metadata.txt — n_samples y n_items, leído por C/CUDA en tiempo de ejecución.
"""
import argparse
import os

import numpy as np

# ──────────────────────────── Constantes del dominio ────────────────────────
N_SAMPLES: int = 10
N_HEALTHY: int = 5   # filas 0–4 → y = 0
N_SICK:    int = 5   # filas 5–9 → y = 1


def generate_data(n_items: int = 50, seed: int = 42) -> tuple:
    """
    Genera A ∈ R^{10×n_items}, etiquetas y, y perfiles T, S, F ∈ R^{n_items}.

    Fiel a la especificación del enunciado (sección 6):
      · Filas 0–4 de A: muestras sanas  (y=0).
      · Filas 5–9 de A: muestras enfermas (y=1).
      · Cada fila de A es Dirichlet(1,...,1) → suma 1.

    T y S se generan también con Dirichlet(1,...,1).
    F es binario: presencia (1) o ausencia (0) de gen de interés.
    Estos tres perfiles no están en el script del enunciado pero son
    necesarios para evaluar la ecuación 2.1: P_i = W1·T_i + W2·S_i + W3·F_i.

    Args:
        n_items: Número de ítems (genomas/taxones). Default: 50.
        seed:    Semilla para reproducibilidad. Default: 42.

    Returns:
        (A, y, T, S, F)
        · A: (10, n_items) float32 — matriz de contribución.
        · y: (10,)         int32   — etiquetas binarias.
        · T: (n_items,)    float32 — perfil taxonómico.
        · S: (n_items,)    float32 — perfil ecológico.
        · F: (n_items,)    float32 — perfil funcional (binario).
    """
    rng = np.random.default_rng(seed)

    # Todas las filas usan Dirichlet(1,...,1) — fiel al enunciado
    A = rng.dirichlet(np.ones(n_items), size=N_SAMPLES).astype(np.float32)
    y = np.array([0] * N_HEALTHY + [1] * N_SICK, dtype=np.int32)

    # Perfiles por ítem: extensión necesaria para la ec. 2.1 del enunciado
    T = rng.dirichlet(np.ones(n_items)).astype(np.float32)
    S = rng.dirichlet(np.ones(n_items)).astype(np.float32)
    F = rng.integers(0, 2, size=n_items).astype(np.float32)

    return A, y, T, S, F


def save_data(
    A: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    F: np.ndarray,
    out_dir: str = "data",
) -> None:
    """
    Persiste todos los arrays en disco en dos formatos complementarios.

    Formato .npy  → implementaciones Python (NumPy nativo).
    Formato .bin  → implementaciones C y CUDA (float32/int32 crudo).
    metadata.txt  → dos enteros: n_samples n_items (leídos por C/CUDA).

    Args:
        A, y, T, S, F: Arrays generados por generate_data().
        out_dir:        Directorio de salida. Default: "data".
    """
    os.makedirs(out_dir, exist_ok=True)
    n_samples, n_items = A.shape

    # ── Formato NumPy ──────────────────────────────────────────────────────
    np.save(f"{out_dir}/matrix_A.npy", A)
    np.save(f"{out_dir}/labels.npy",   y)
    np.save(f"{out_dir}/profile_T.npy", T)
    np.save(f"{out_dir}/profile_S.npy", S)
    np.save(f"{out_dir}/profile_F.npy", F)

    # ── Formato binario crudo para C y CUDA ───────────────────────────────
    A.tofile(f"{out_dir}/matrix_A.bin")
    y.astype(np.int32).tofile(f"{out_dir}/labels.bin")
    T.tofile(f"{out_dir}/profile_T.bin")
    S.tofile(f"{out_dir}/profile_S.bin")
    F.tofile(f"{out_dir}/profile_F.bin")

    # ── Metadatos de dimensiones ───────────────────────────────────────────
    with open(f"{out_dir}/metadata.txt", "w", encoding="utf-8") as meta:
        meta.write(f"{n_samples} {n_items}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera datos sintéticos para MetaScore HPC"
    )
    parser.add_argument(
        "--n_items", type=int, default=50,
        help="Número de ítems/taxones (default: 50)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semilla aleatoria (default: 42)"
    )

    parser.add_argument(
        "--out_dir", type=str, default="data",
        help="Directorio de salida (default: data)"
    )
    args = parser.parse_args()

    A, y, T, S, F = generate_data(n_items=args.n_items, seed=args.seed)
    save_data(A, y, T, S, F, out_dir=args.out_dir)

    print(f"Datos generados en '{args.out_dir}/'")
    print(f"  A : {A.shape}  dtype={A.dtype}")
    print(f"  y : {y}")
    print(f"  T,S,F : {T.shape}  dtype={T.dtype}")


if __name__ == "__main__":
    main()
