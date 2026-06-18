"""
Nivel 3 — PyCUDA: Búsqueda aleatoria acelerada en GPU.

Wrapper Python que lanza el kernel CUDA mediante PyCUDA.
Permite integrar los resultados GPU con el pipeline de métricas Python
(benchmark.csv, gráficas de speedup) sin depender del binario C.

Estrategia idéntica a scoring_kernel.cu:
  · Un hilo CUDA evalúa un candidato W_k.
  · Todos los datos (A, T, S, F, y) se transfieren al device una única vez.
  · El kernel retorna (AUC, W) por candidato; la reducción del máximo
    se realiza en host con NumPy.

Requisitos:
  · pip install pycuda numpy scikit-learn
  · Driver NVIDIA con CUDA toolkit instalado.
"""
import time
from pathlib import Path

import numpy as np
import pycuda.autoinit          # inicializa el contexto CUDA automáticamente
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# ──────────────────────────── Configuración ─────────────────────────────────
K_CANDIDATES: int = 10_000
SEED:         int = 0
BLOCK_SIZE:   int = 256
DATA_DIR:     str = "data"

# ──────────────────────────── Código del kernel (inline) ────────────────────

_KERNEL_SRC = r"""
#define N_SAMPLES   10
#define N_HEALTHY    5
#define N_SICK       5

__device__ float lcg_rand(unsigned long long *state) {
    *state = (*state * 6364136223846793005ULL) + 1442695040888963407ULL;
    return (float)((*state >> 33) & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__device__ void sample_simplex(float *w, unsigned long long *state) {
    float x[3], sum = 0.0f;
    for (int i = 0; i < 3; i++) {
        float u = lcg_rand(state);
        if (u < 1e-10f) u = 1e-10f;
        x[i]  = -logf(u);
        sum   += x[i];
    }
    for (int i = 0; i < 3; i++) w[i] = x[i] / sum;
}

__global__ void evaluate_candidates(
    int K, int n_items, unsigned int seed,
    const float * __restrict__ A,
    const float * __restrict__ T,
    const float * __restrict__ S,
    const float * __restrict__ F,
    const int   * __restrict__ labels,
    float *auc_out,
    float *w_out)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= K) return;

    unsigned long long state =
        (unsigned long long)(seed + 1) * (unsigned long long)(gid + 1)
        * 6364136223846793005ULL;

    float w[3];
    sample_simplex(w, &state);

    float scores[N_SAMPLES];
    for (int j = 0; j < N_SAMPLES; j++) {
        float s = 0.0f;
        for (int i = 0; i < n_items; i++) {
            float P_i = w[0]*T[i] + w[1]*S[i] + w[2]*F[i];
            s += A[j * n_items + i] * P_i;
        }
        scores[j] = s;
    }

    int concordant = 0;
    for (int i = 0; i < N_SAMPLES; i++) {
        if (labels[i] != 1) continue;
        for (int j = 0; j < N_SAMPLES; j++) {
            if (labels[j] != 0) continue;
            if (scores[i] > scores[j]) concordant++;
        }
    }

    auc_out[gid]       = (float)concordant / (float)(N_HEALTHY * N_SICK);
    w_out[gid * 3]     = w[0];
    w_out[gid * 3 + 1] = w[1];
    w_out[gid * 3 + 2] = w[2];
}
"""

# ──────────────────────────── E/S de datos ──────────────────────────────────

def load_data(data_dir: str = DATA_DIR) -> tuple:
    """Carga A, y, T, S, F desde disco (.npy)."""
    base = Path(data_dir)
    A = np.load(base / "matrix_A.npy")
    y = np.load(base / "labels.npy")
    T = np.load(base / "profile_T.npy")
    S = np.load(base / "profile_S.npy")
    F = np.load(base / "profile_F.npy")
    return A, y, T, S, F

# ──────────────────────────── Búsqueda GPU ──────────────────────────────────

def random_search_cuda(
    K: int,
    A: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    F: np.ndarray,
    y: np.ndarray,
    seed: int = SEED,
    block_size: int = BLOCK_SIZE,
) -> tuple:
    """
    Búsqueda aleatoria del W* mediante el kernel CUDA.

    Transfiere A, T, S, F, y al device una sola vez y lanza el kernel
    con ceil(K / block_size) bloques.

    Args:
        K:          Número de candidatos a evaluar.
        A:          Matriz de contribución (10, N), float32.
        T, S, F:    Perfiles por ítem (N,), float32.
        y:          Etiquetas binarias (10,), int32.
        seed:       Semilla aleatoria.
        block_size: Tamaño de bloque CUDA. Default: 256.

    Returns:
        (W_best, auc_best, elapsed_seconds)
    """
    # Compilar kernel
    mod    = SourceModule(_KERNEL_SRC)
    kernel = mod.get_function("evaluate_candidates")

    n_items  = A.shape[1]
    grid     = (K + block_size - 1) // block_size

    # Buffers de salida en host (pagelocked para transferencia eficiente)
    h_auc = cuda.pagelocked_empty(K, dtype=np.float32)
    h_w   = cuda.pagelocked_empty((K, 3), dtype=np.float32)

    # Transferir datos al device
    d_A      = cuda.to_device(np.ascontiguousarray(A, dtype=np.float32))
    d_T      = cuda.to_device(np.ascontiguousarray(T, dtype=np.float32))
    d_S      = cuda.to_device(np.ascontiguousarray(S, dtype=np.float32))
    d_F      = cuda.to_device(np.ascontiguousarray(F, dtype=np.float32))
    d_labels = cuda.to_device(np.ascontiguousarray(y, dtype=np.int32))
    d_auc    = cuda.mem_alloc(K * np.dtype(np.float32).itemsize)
    d_w      = cuda.mem_alloc(K * 3 * np.dtype(np.float32).itemsize)

    t_start = time.perf_counter()

    kernel(
        np.int32(K), np.int32(n_items), np.uint32(seed),
        d_A, d_T, d_S, d_F, d_labels,
        d_auc, d_w,
        block=(block_size, 1, 1),
        grid=(grid, 1),
    )
    cuda.Context.synchronize()

    elapsed = time.perf_counter() - t_start

    # Transferir resultados al host
    cuda.memcpy_dtoh(h_auc, d_auc)
    cuda.memcpy_dtoh(h_w,   d_w)

    best_idx  = int(np.argmax(h_auc))
    best_W    = h_w[best_idx].copy()
    best_auc  = float(h_auc[best_idx])

    return best_W, best_auc, elapsed


# ──────────────────────────── Punto de entrada ──────────────────────────────

def main() -> None:
    A, y, T, S, F = load_data()

    print(f"[PyCUDA]  K={K_CANDIDATES:,}  block={BLOCK_SIZE}  N_items={A.shape[1]}")
    W_best, auc_best, elapsed = random_search_cuda(K_CANDIDATES, A, T, S, F, y)

    print(f"  W*     = [{W_best[0]:.4f}, {W_best[1]:.4f}, {W_best[2]:.4f}]")
    print(f"  AUC    = {auc_best:.4f}")
    print(f"  Tiempo = {elapsed:.4f} s")


if __name__ == "__main__":
    main()
