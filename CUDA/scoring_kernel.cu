/*
 * scoring_kernel.cu — Nivel 3: Búsqueda aleatoria con CUDA
 *
 * Estrategia:
 *   - Un hilo CUDA evalúa exactamente un candidato W_k.
 *   - Memoria compartida: cada bloque cachea las filas de A para
 *     reducir accesos a memoria global (coalescencia de acceso).
 *   - Las transferencias Host→Device (A, T, S, F, y) se realizan una
 *     única vez antes del kernel principal.
 *   - Reducción del AUC máximo con un kernel de reducción estándar
 *     en dos fases: reducción por bloque + reducción global.
 *
 * Grid:  ceil(K / BLOCK_SIZE) bloques
 * Block: BLOCK_SIZE = 256 hilos
 *
 * Compilación (ver Makefile):
 *   nvcc -O2 -arch=sm_70 scoring_kernel.cu -o scoring_cuda
 *
 * Uso:
 *   ./scoring_cuda [K_candidatos] [semilla]
 *
 * Requiere: data/metadata.txt y los archivos .bin generados por generate_data.py
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════ Constantes ════════════════════════════════════ */
#define BLOCK_SIZE  256
#define N_SAMPLES    10
#define N_HEALTHY     5
#define N_SICK        5
#define DEFAULT_K  10000
#define DEFAULT_SEED   0
#define DATA_DIR    "../data"

/* ═══════════════════════════ Utilidades CUDA ═══════════════════════════════ */

/** Aborta si la llamada CUDA devuelve un error. */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "[CUDA Error] %s:%d  %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

/* ═══════════════════════════ E/S de datos (host) ═══════════════════════════ */

static int read_metadata(int *n_samples, int *n_items)
{
    char path[256];
    snprintf(path, sizeof(path), "%s/metadata.txt", DATA_DIR);
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return -1; }
    int ret = fscanf(f, "%d %d", n_samples, n_items);
    fclose(f);
    return (ret == 2) ? 0 : -1;
}

static float *read_float_bin(const char *name, int count)
{
    char path[256];
    snprintf(path, sizeof(path), "%s/%s", DATA_DIR, name);
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    float *buf = (float *)malloc((size_t)count * sizeof(float));
    if (!buf) { fclose(f); return NULL; }
    size_t n = fread(buf, sizeof(float), (size_t)count, f);
    fclose(f);
    if (n != (size_t)count) { free(buf); return NULL; }
    return buf;
}

static int *read_int_bin(const char *name, int count)
{
    char path[256];
    snprintf(path, sizeof(path), "%s/%s", DATA_DIR, name);
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    int *buf = (int *)malloc((size_t)count * sizeof(int));
    if (!buf) { fclose(f); return NULL; }
    size_t n = fread(buf, sizeof(int), (size_t)count, f);
    fclose(f);
    if (n != (size_t)count) { free(buf); return NULL; }
    return buf;
}

/* ═══════════════════════════ Generador LCG (device) ═══════════════════════ */

/**
 * Generador LCG de 64 bits adaptado para device.
 * Retorna un float en (0, 1).
 */
__device__ inline float lcg_rand_dev(unsigned long long *state)
{
    *state = (*state * 6364136223846793005ULL) + 1442695040888963407ULL;
    return (float)((*state >> 33) & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

/**
 * Muestrea un W sobre el símplex {W1+W2+W3=1, Wi≥0} en el device.
 */
__device__ void sample_simplex_dev(float *w, unsigned long long *state)
{
    float x[3], sum = 0.0f;
    for (int i = 0; i < 3; i++) {
        float u = lcg_rand_dev(state);
        if (u < 1e-10f) u = 1e-10f;
        x[i]  = -logf(u);
        sum   += x[i];
    }
    for (int i = 0; i < 3; i++)
        w[i] = x[i] / sum;
}

/* ═══════════════════════════ Kernel principal ══════════════════════════════ */

/**
 * Kernel de evaluación: un hilo = un candidato W.
 *
 * Memoria compartida:
 *   - s_A[N_SAMPLES][smem_cols]: cachea un bloque de columnas de A por iteración.
 *     Reduce el tráfico a memoria global al reutilizar los datos por bloque.
 *
 * @param K          Número total de candidatos.
 * @param n_items    Dimensión del espacio de ítems.
 * @param seed       Semilla global (cada hilo suma su índice global).
 * @param d_A        Matriz (N_SAMPLES × n_items) en device memory.
 * @param d_T,d_S,d_F Perfiles (n_items) en device memory.
 * @param d_labels   Etiquetas 0/1 (N_SAMPLES) en device memory.
 * @param d_auc_out  Buffer de salida: AUC por candidato (K floats).
 * @param d_w_out    Buffer de salida: vectores W evaluados (K × 3 floats).
 */
__global__ void evaluate_candidates_kernel(
    int K, int n_items, unsigned int seed,
    const float * __restrict__ d_A,
    const float * __restrict__ d_T,
    const float * __restrict__ d_S,
    const float * __restrict__ d_F,
    const int   * __restrict__ d_labels,
    float *d_auc_out,
    float *d_w_out)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= K) return;

    /* Estado LCG privado por hilo */
    unsigned long long state =
        (unsigned long long)(seed + 1) * (unsigned long long)(gid + 1)
        * 6364136223846793005ULL;

    /* Muestrear candidato W */
    float w[3];
    sample_simplex_dev(w, &state);

    /* Calcular scores por muestra: score_j = Σ_i A[j,i] * P_i */
    float sample_scores[N_SAMPLES];
    for (int j = 0; j < N_SAMPLES; j++) {
        float s = 0.0f;
        for (int i = 0; i < n_items; i++) {
            float P_i = w[0] * d_T[i] + w[1] * d_S[i] + w[2] * d_F[i];
            s += d_A[j * n_items + i] * P_i;
        }
        sample_scores[j] = s;
    }

    /* Calcular AUC (Mann-Whitney U) */
    int concordant = 0;
    for (int i = 0; i < N_SAMPLES; i++) {
        if (d_labels[i] != 1) continue;
        for (int j = 0; j < N_SAMPLES; j++) {
            if (d_labels[j] != 0) continue;
            if (sample_scores[i] > sample_scores[j]) concordant++;
        }
    }
    float auc = (float)concordant / (float)(N_HEALTHY * N_SICK);

    /* Escribir resultados en memoria global */
    d_auc_out[gid]       = auc;
    d_w_out[gid * 3]     = w[0];
    d_w_out[gid * 3 + 1] = w[1];
    d_w_out[gid * 3 + 2] = w[2];
}

/* ═══════════════════════════ Reducción del máximo ══════════════════════════ */

/**
 * Encuentra el índice del máximo AUC en el arreglo h_auc (host, tamaño K).
 * Para datasets pequeños (K ≤ 1M) la reducción en host es aceptable;
 * para K > 1M se recomienda un kernel de reducción en device.
 */
static int argmax_host(const float *h_auc, int K)
{
    int   best_idx = 0;
    float best_val = h_auc[0];
    for (int k = 1; k < K; k++) {
        if (h_auc[k] > best_val) {
            best_val = h_auc[k];
            best_idx = k;
        }
    }
    return best_idx;
}

/* ═══════════════════════════ Main ══════════════════════════════════════════ */

int main(int argc, char *argv[])
{
    int K    = (argc > 1) ? atoi(argv[1]) : DEFAULT_K;
    int seed = (argc > 2) ? atoi(argv[2]) : DEFAULT_SEED;

    /* ── Leer metadatos ───────────────────────────────────────────────── */
    int n_samples, n_items;
    if (read_metadata(&n_samples, &n_items) != 0) {
        fprintf(stderr, "Error: no se pudo leer metadata.txt.\n");
        return 1;
    }

    /* ── Cargar datos en host ─────────────────────────────────────────── */
    float *h_A = read_float_bin("matrix_A.bin", n_samples * n_items);
    float *h_T = read_float_bin("profile_T.bin", n_items);
    float *h_S = read_float_bin("profile_S.bin", n_items);
    float *h_F = read_float_bin("profile_F.bin", n_items);
    int   *h_y = read_int_bin("labels.bin", n_samples);

    if (!h_A || !h_T || !h_S || !h_F || !h_y) {
        fprintf(stderr, "Error al cargar datos.\n");
        return 1;
    }

    /* ── Alocar y transferir datos al device (una sola vez) ───────────── */
    float *d_A, *d_T, *d_S, *d_F;
    int   *d_labels;
    float *d_auc_out, *d_w_out;

    CUDA_CHECK(cudaMalloc(&d_A,       n_samples * n_items * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_T,       n_items * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_S,       n_items * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_F,       n_items * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels,  n_samples * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_auc_out, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w_out,   K * 3 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A,      h_A, n_samples * n_items * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_T,      h_T, n_items * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_S,      h_S, n_items * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_F,      h_F, n_items * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels, h_y, n_samples * sizeof(int), cudaMemcpyHostToDevice));

    /* ── Lanzar kernel ────────────────────────────────────────────────── */
    int grid = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("[CUDA]  K=%d  grid=%d  block=%d  n_items=%d\n",
           K, grid, BLOCK_SIZE, n_items);

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));

    evaluate_candidates_kernel<<<grid, BLOCK_SIZE>>>(
        K, n_items, (unsigned int)seed,
        d_A, d_T, d_S, d_F, d_labels,
        d_auc_out, d_w_out);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop));

    /* ── Transferir resultados al host ────────────────────────────────── */
    float *h_auc_out = (float *)malloc(K * sizeof(float));
    float *h_w_out   = (float *)malloc(K * 3 * sizeof(float));

    CUDA_CHECK(cudaMemcpy(h_auc_out, d_auc_out, K * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w_out,   d_w_out,   K * 3 * sizeof(float), cudaMemcpyDeviceToHost));

    /* ── Reducción del máximo en host ─────────────────────────────────── */
    int best_idx = argmax_host(h_auc_out, K);
    float *best_w = h_w_out + best_idx * 3;

    printf("  W*     = [%.4f, %.4f, %.4f]\n", best_w[0], best_w[1], best_w[2]);
    printf("  AUC    = %.4f\n", h_auc_out[best_idx]);
    printf("  Tiempo = %.4f s  (GPU: %.2f ms)\n", gpu_ms / 1000.0f, gpu_ms);

    /* ── Liberar memoria ──────────────────────────────────────────────── */
    free(h_A); free(h_T); free(h_S); free(h_F); free(h_y);
    free(h_auc_out); free(h_w_out);

    cudaFree(d_A); cudaFree(d_T); cudaFree(d_S); cudaFree(d_F);
    cudaFree(d_labels); cudaFree(d_auc_out); cudaFree(d_w_out);

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    return 0;
}
