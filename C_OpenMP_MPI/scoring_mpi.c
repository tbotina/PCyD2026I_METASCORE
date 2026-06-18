/*
 * scoring_mpi.c — Nivel 2: Búsqueda aleatoria con MPI (memoria distribuida)
 *
 * Estrategia:
 *   1. Proceso root (rank 0) genera los K candidatos W y los distribuye
 *      equitativamente con MPI_Scatter (K debe ser múltiplo de nprocs;
 *      si no, root añade candidatos de relleno para completar el reparto).
 *   2. Cada proceso evalúa su subconjunto de forma completamente independiente.
 *   3. MPI_Reduce con operación personalizada (MPI_MAXLOC sobre struct {auc, rank})
 *      identifica el proceso con el mejor AUC.
 *   4. MPI_Bcast difunde el W* del proceso ganador a todos los demás.
 *
 * Compilación:
 *   mpicc -O2 -lm scoring_mpi.c -o scoring_mpi
 *
 * Uso:
 *   mpirun -n 4 ./scoring_mpi [K_candidatos] [semilla]
 *
 * Requiere: data/metadata.txt y los archivos .bin generados por generate_data.py
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

/* ═══════════════════════════ Constantes ════════════════════════════════════ */
#define N_SAMPLES     10
#define N_HEALTHY      5
#define N_SICK         5
#define DEFAULT_K  10000
#define DEFAULT_SEED   0
#define DATA_DIR     "../data"

/* ═══════════════════════════ E/S de datos ══════════════════════════════════ */

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

/* ═══════════════════════════ Muestreo del símplex ══════════════════════════ */

static inline float lcg_rand(unsigned long long *state)
{
    *state = (*state * 6364136223846793005ULL) + 1442695040888963407ULL;
    return (float)((*state >> 33) & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

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

/**
 * Genera K candidatos en la matriz buf (K × 3) usando un LCG derivado de seed.
 * Solo el proceso root llama a esta función.
 */
static void generate_candidates(float *buf, int K, unsigned int seed)
{
    unsigned long long state = (unsigned long long)(seed + 1) * 6364136223846793005ULL;
    for (int k = 0; k < K; k++)
        sample_simplex(buf + k * 3, &state);
}

/* ═══════════════════════════ Scoring ═══════════════════════════════════════ */

static void compute_item_scores(const float *w, const float *T, const float *S,
                                 const float *F, float *P, int n_items)
{
    for (int i = 0; i < n_items; i++)
        P[i] = w[0] * T[i] + w[1] * S[i] + w[2] * F[i];
}

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

/* ═══════════════════════════ Búsqueda MPI ══════════════════════════════════ */

/**
 * Evalúa un lote local de candidatos y retorna el mejor (W, AUC) del lote.
 *
 * @param local_cands  Submatriz de candidatos (K_local × 3) del proceso actual.
 * @param K_local      Número de candidatos locales.
 * @param A,T,S,F,y   Datos del problema (todos los procesos los tienen completos).
 * @param n_items      Dimensión.
 * @param best_w       [out] Mejor vector de pesos local (3).
 * @param best_auc     [out] Mejor AUC local.
 */
static void evaluate_local(const float *local_cands, int K_local,
                            const float *A, const float *T, const float *S,
                            const float *F, const int *labels, int n_items,
                            float *best_w, float *best_auc)
{
    *best_auc = -1.0f;
    best_w[0] = best_w[1] = best_w[2] = 1.0f / 3.0f;

    float *P      = (float *)malloc((size_t)n_items * sizeof(float));
    float  scores[N_SAMPLES];

    for (int k = 0; k < K_local; k++) {
        const float *w = local_cands + k * 3;
        compute_item_scores(w, T, S, F, P, n_items);
        compute_sample_scores(A, P, scores, n_items);

        float auc = compute_auc(scores, labels);
        if (auc > *best_auc) {
            *best_auc  = auc;
            best_w[0]  = w[0];
            best_w[1]  = w[1];
            best_w[2]  = w[2];
        }
    }

    free(P);
}

/* ═══════════════════════════ Main ══════════════════════════════════════════ */

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int K    = (argc > 1) ? atoi(argv[1]) : DEFAULT_K;
    int seed = (argc > 2) ? atoi(argv[2]) : DEFAULT_SEED;

    /* Redondear K hacia arriba para que sea múltiplo de nprocs */
    int K_padded = ((K + nprocs - 1) / nprocs) * nprocs;
    int K_local  = K_padded / nprocs;

    /* ── Leer metadatos (todos los procesos) ──────────────────────────── */
    int n_samples, n_items;
    if (read_metadata(&n_samples, &n_items) != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: no se pudo leer metadata.txt.\n");
        MPI_Finalize();
        return 1;
    }

    /* ── Cargar datos (cada proceso lee su propia copia) ───────────────── */
    float *A = read_float_bin("matrix_A.bin", n_samples * n_items);
    float *T = read_float_bin("profile_T.bin", n_items);
    float *S = read_float_bin("profile_S.bin", n_items);
    float *F = read_float_bin("profile_F.bin", n_items);
    int   *y = read_int_bin("labels.bin", n_samples);

    if (!A || !T || !S || !F || !y) {
        fprintf(stderr, "[rank %d] Error al cargar datos.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ── Root genera candidatos y los dispersa ────────────────────────── */
    float *all_cands   = NULL;
    float *local_cands = (float *)malloc((size_t)(K_local * 3) * sizeof(float));

    if (rank == 0) {
        all_cands = (float *)malloc((size_t)(K_padded * 3) * sizeof(float));
        generate_candidates(all_cands, K_padded, (unsigned int)seed);
        printf("[MPI]  K=%d (padded=%d)  procesos=%d  n_items=%d\n",
               K, K_padded, nprocs, n_items);
    }

    double t_start = MPI_Wtime();

    /* Distribuir K_local candidatos (cada uno = 3 floats) a cada proceso */
    MPI_Scatter(all_cands, K_local * 3, MPI_FLOAT,
                local_cands, K_local * 3, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    /* ── Cada proceso evalúa su lote ──────────────────────────────────── */
    float local_best_w[3], local_best_auc;
    evaluate_local(local_cands, K_local,
                   A, T, S, F, y, n_items,
                   local_best_w, &local_best_auc);

    /* ── Reducción: encontrar el proceso con el AUC global máximo ──────── */
    /* MPI_FLOAT_INT permite usar MPI_MAXLOC para obtener {valor, rango} */
    struct { float val; int rank; } local_max = {local_best_auc, rank};
    struct { float val; int rank; } global_max;

    MPI_Reduce(&local_max, &global_max, 1, MPI_FLOAT_INT,
               MPI_MAXLOC, 0, MPI_COMM_WORLD);

    /* Difundir el rank ganador a todos — global_max solo es válido en rank 0 */
    int winner_rank = (rank == 0) ? global_max.rank : 0;
    MPI_Bcast(&winner_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* El proceso ganador difunde su W* a todos */
    float global_best_w[3];
    if (rank == winner_rank)
        memcpy(global_best_w, local_best_w, 3 * sizeof(float));

    MPI_Bcast(global_best_w, 3, MPI_FLOAT, winner_rank, MPI_COMM_WORLD);

    double elapsed = MPI_Wtime() - t_start;

    if (rank == 0) {
        float *P_best     = (float *)malloc((size_t)n_items * sizeof(float));
        float  scores_best[N_SAMPLES];
        compute_item_scores(global_best_w, T, S, F, P_best, n_items);
        compute_sample_scores(A, P_best, scores_best, n_items);
        float consistency = compute_consistency(scores_best, y);
        free(P_best);

        printf("  W*           = [%.4f, %.4f, %.4f]\n",
               global_best_w[0], global_best_w[1], global_best_w[2]);
        printf("  AUC          = %.4f\n", global_max.val);
        printf("  Consistencia = %.4f  %s\n", consistency, consistency >= 0.8f ? "✓" : "✗");
        printf("  Tiempo       = %.4f s\n", elapsed);
    }

    /* ── Liberar memoria ──────────────────────────────────────────────── */
    free(A); free(T); free(S); free(F); free(y);
    free(local_cands);
    if (rank == 0) free(all_cands);

    MPI_Finalize();
    return 0;
}
