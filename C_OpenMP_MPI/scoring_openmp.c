/*
 * scoring_openmp.c — Nivel 2: Búsqueda aleatoria con OpenMP (memoria compartida)
 *
 * Estrategia:
 *   - #pragma omp parallel for reparte los K candidatos entre los hilos disponibles.
 *   - Cada hilo mantiene su mejor (W, AUC) local → sin contención en el bucle interno.
 *   - #pragma omp critical protege la actualización del óptimo global al finalizar.
 *
 * Compilación:
 *   gcc -O2 -fopenmp -lm scoring_openmp.c -o scoring_openmp
 *
 * Uso:
 *   OMP_NUM_THREADS=4 ./scoring_openmp [K_candidatos] [semilla]
 *
 * Requiere: data/metadata.txt y los archivos .bin generados por generate_data.py
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/* ═══════════════════════════ Constantes ════════════════════════════════════ */
#define N_SAMPLES     10
#define N_HEALTHY      5
#define N_SICK         5
#define DEFAULT_K  10000
#define DEFAULT_SEED   0
#define DATA_DIR     "../data"

/* ═══════════════════════════ E/S de datos ══════════════════════════════════ */

/**
 * Lee n_samples y n_items desde data/metadata.txt.
 * Retorna 0 en éxito, -1 si el archivo no existe o tiene formato incorrecto.
 */
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

/**
 * Lee un array de float32 desde un archivo binario crudo.
 * El llamador debe liberar la memoria devuelta con free().
 */
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

/**
 * Lee un array de int32 desde un archivo binario crudo.
 * El llamador debe liberar la memoria devuelta con free().
 */
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

/* ═══════════════════════════ Muestreo del símplex ══════════════════════════ */

/*
 * Generador LCG de 64 bits (thread-safe: cada hilo tiene su propio estado).
 * Retorna un valor uniforme en (0, 1).
 */
static inline float lcg_rand(unsigned long long *state)
{
    *state = (*state * 6364136223846793005ULL) + 1442695040888963407ULL;
    return (float)((*state >> 33) & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

/**
 * Muestrea un vector de pesos sobre el símplex {W1+W2+W3=1, Wi≥0}.
 * Método de las exponenciales: wi = -log(ui) / Σ(-log(uj)).
 *
 * @param w     Buffer de salida (3 floats).
 * @param state Estado del LCG del hilo actual.
 */
static void sample_simplex(float *w, unsigned long long *state)
{
    float x[3], sum = 0.0f;
    for (int i = 0; i < 3; i++) {
        float u = lcg_rand(state);
        if (u < 1e-10f) u = 1e-10f;
        x[i]  = -logf(u);
        sum   += x[i];
    }
    for (int i = 0; i < 3; i++)
        w[i] = x[i] / sum;
}

/* ═══════════════════════════ Scoring ═══════════════════════════════════════ */

/**
 * Calcula el score por ítem: P_i = W1*T_i + W2*S_i + W3*F_i.
 *
 * @param w       Vector de pesos (3).
 * @param T,S,F   Perfiles por ítem (n_items).
 * @param P       Buffer de salida (n_items).
 * @param n_items Dimensión.
 */
static void compute_item_scores(const float *w, const float *T, const float *S,
                                 const float *F, float *P, int n_items)
{
    for (int i = 0; i < n_items; i++)
        P[i] = w[0] * T[i] + w[1] * S[i] + w[2] * F[i];
}

/**
 * Calcula el score por muestra: score_j = Σ_i A[j*n_items+i] * P_i.
 *
 * @param A       Matriz (N_SAMPLES × n_items), row-major.
 * @param P       Scores por ítem (n_items).
 * @param scores  Buffer de salida (N_SAMPLES).
 * @param n_items Dimensión.
 */
static void compute_sample_scores(const float *A, const float *P,
                                   float *scores, int n_items)
{
    for (int j = 0; j < N_SAMPLES; j++) {
        float s = 0.0f;
        const float *row = A + j * n_items;
        for (int i = 0; i < n_items; i++)
            s += row[i] * P[i];
        scores[j] = s;
    }
}

/**
 * Calcula el AUC mediante el estadístico de Mann-Whitney U.
 *
 * Para N_HEALTHY = N_SICK = 5: AUC = concordantes / 25.
 * Un par es concordante si score_enfermo > score_sano.
 *
 * @param scores  Scores por muestra (N_SAMPLES).
 * @param labels  Etiquetas 0/1 (N_SAMPLES).
 * @return        AUC ∈ [0, 1].
 */
static float compute_auc(const float *scores, const int *labels)
{
    int concordant = 0;
    for (int i = 0; i < N_SAMPLES; i++) {
        if (labels[i] != 1) continue;
        for (int j = 0; j < N_SAMPLES; j++) {
            if (labels[j] != 0) continue;
            if (scores[i] > scores[j]) concordant++;
        }
    }
    return (float)concordant / (float)(N_HEALTHY * N_SICK);
}

/**
 * Calcula la consistencia (balanced accuracy) con umbral θ = media de scores.
 *
 * Consistencia = (TPR + TNR) / 2  ∈ [0, 1]. Satisfactoria si ≥ 0.8.
 *
 * @param scores  Scores por muestra (N_SAMPLES).
 * @param labels  Etiquetas 0/1 (N_SAMPLES).
 * @return        Consistencia ∈ [0, 1].
 */
static float compute_consistency(const float *scores, const int *labels)
{
    float theta = 0.0f;
    for (int i = 0; i < N_SAMPLES; i++)
        theta += scores[i];
    theta /= N_SAMPLES;

    int tp = 0, tn = 0;
    for (int i = 0; i < N_SAMPLES; i++) {
        if (labels[i] == 1 && scores[i] >  theta) tp++;
        if (labels[i] == 0 && scores[i] <= theta) tn++;
    }
    return ((float)tp / N_SICK + (float)tn / N_HEALTHY) / 2.0f;
}

/* ═══════════════════════════ Búsqueda OpenMP ═══════════════════════════════ */

/**
 * Búsqueda aleatoria paralela con OpenMP.
 *
 * Cada hilo mantiene su mejor candidato local para evitar contención
 * en el bucle interno. La sección crítica actualiza el óptimo global
 * una sola vez al terminar cada hilo.
 *
 * @param K        Número de candidatos a evaluar.
 * @param A        Matriz (N_SAMPLES × n_items), row-major.
 * @param T,S,F    Perfiles por ítem (n_items).
 * @param labels   Etiquetas 0/1 (N_SAMPLES).
 * @param n_items  Dimensión del espacio de ítems.
 * @param seed     Semilla global (cada hilo deriva la suya sumando el tid).
 * @param best_w   [out] Vector de pesos óptimo (3).
 * @param best_auc [out] AUC máximo encontrado.
 */
static void random_search_omp(int K, const float *A,
                               const float *T, const float *S, const float *F,
                               const int *labels, int n_items,
                               unsigned int seed,
                               float *best_w, float *best_auc)
{
    *best_auc   = -1.0f;
    best_w[0]   = best_w[1] = best_w[2] = 1.0f / 3.0f;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        /* Estado LCG privado: derivado de la semilla global + tid */
        unsigned long long state =
            (unsigned long long)(seed + 1) * (unsigned long long)(tid + 1)
            * 6364136223846793005ULL;

        float *P_local     = (float *)malloc((size_t)n_items * sizeof(float));
        float  scores_local[N_SAMPLES];
        float  local_best_w[3]  = {1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f};
        float  local_best_auc   = -1.0f;

        #pragma omp for schedule(dynamic, 64)
        for (int k = 0; k < K; k++) {
            float w[3];
            sample_simplex(w, &state);
            compute_item_scores(w, T, S, F, P_local, n_items);
            compute_sample_scores(A, P_local, scores_local, n_items);

            float auc = compute_auc(scores_local, labels);
            if (auc > local_best_auc) {
                local_best_auc = auc;
                local_best_w[0] = w[0];
                local_best_w[1] = w[1];
                local_best_w[2] = w[2];
            }
        }

        /* Reducción del máximo global: una entrada por hilo, mínima contención */
        #pragma omp critical
        {
            if (local_best_auc > *best_auc) {
                *best_auc = local_best_auc;
                best_w[0] = local_best_w[0];
                best_w[1] = local_best_w[1];
                best_w[2] = local_best_w[2];
            }
        }

        free(P_local);
    }
}

/* ═══════════════════════════ Main ══════════════════════════════════════════ */

int main(int argc, char *argv[])
{
    int K    = (argc > 1) ? atoi(argv[1]) : DEFAULT_K;
    int seed = (argc > 2) ? atoi(argv[2]) : DEFAULT_SEED;

    /* ── Leer dimensiones ─────────────────────────────────────────────── */
    int n_samples, n_items;
    if (read_metadata(&n_samples, &n_items) != 0) {
        fprintf(stderr, "Error: no se pudo leer metadata.txt. "
                "Ejecute: python data/generate_data.py\n");
        return 1;
    }

    /* ── Cargar datos ─────────────────────────────────────────────────── */
    float *A = read_float_bin("matrix_A.bin", n_samples * n_items);
    float *T = read_float_bin("profile_T.bin", n_items);
    float *S = read_float_bin("profile_S.bin", n_items);
    float *F = read_float_bin("profile_F.bin", n_items);
    int   *y = read_int_bin("labels.bin", n_samples);

    if (!A || !T || !S || !F || !y) {
        fprintf(stderr, "Error al cargar datos binarios.\n");
        free(A); free(T); free(S); free(F); free(y);
        return 1;
    }

    int n_threads = omp_get_max_threads();
    printf("[OpenMP]  K=%d  hilos=%d  n_items=%d\n", K, n_threads, n_items);

    /* ── Búsqueda ─────────────────────────────────────────────────────── */
    float best_w[3], best_auc;
    double t_start = omp_get_wtime();
    random_search_omp(K, A, T, S, F, y, n_items, (unsigned int)seed,
                      best_w, &best_auc);
    double elapsed = omp_get_wtime() - t_start;

    /* ── Consistencia con W* ─────────────────────────────────────────────── */
    float *P_best = (float *)malloc((size_t)n_items * sizeof(float));
    float  scores_best[N_SAMPLES];
    compute_item_scores(best_w, T, S, F, P_best, n_items);
    compute_sample_scores(A, P_best, scores_best, n_items);
    float consistency = compute_consistency(scores_best, y);
    free(P_best);

    printf("  W*           = [%.4f, %.4f, %.4f]\n", best_w[0], best_w[1], best_w[2]);
    printf("  AUC          = %.4f\n", best_auc);
    printf("  Consistencia = %.4f  %s\n", consistency, consistency >= 0.8f ? "✓" : "✗");
    printf("  Tiempo       = %.4f s\n", elapsed);

    /* ── Liberar memoria ──────────────────────────────────────────────── */
    free(A); free(T); free(S); free(F); free(y);
    return 0;
}
