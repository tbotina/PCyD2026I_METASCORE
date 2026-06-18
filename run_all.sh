#!/usr/bin/env bash
# run_all.sh — Benchmark completo MetaScore HPC
#
# Ejecuta todas las implementaciones Python y C, mide tiempos y
# genera results/benchmark.csv listo para el informe.
#
# Nota: CUDA se ejecuta en Google Colab — sus filas se añaden
#       manualmente al CSV desde los resultados del notebook.
#
# Uso (desde la raíz del proyecto en WSL):
#   bash run_all.sh [K] [n_items]
#
# Ejemplo:
#   bash run_all.sh 100000 50

K="${1:-100000}"
N_ITEMS="${2:-50}"
SEED=0
CSV="results/benchmark.csv"
C_DIR="C_OpenMP_MPI"
OMP_THREADS=(1 2 4 6 12)
MPI_PROCS=(1 2 4 6)
SEP="════════════════════════════════════════════════════"

# ──────────────────────────── Funciones auxiliares ──────────────────────────

extract_tiempo() {
    echo "$1" | grep -oP '(?<=Tiempo\s{1,10}= )\d+\.\d+' || \
    echo "$1" | grep -oP '(?<=Tiempo       = )\d+\.\d+' || echo "0"
}

extract_auc() {
    echo "$1" | grep -oP '(?<=AUC\s{1,10}= )\d+\.\d+' || \
    echo "$1" | grep -oP '(?<=AUC          = )\d+\.\d+' || echo "0"
}

calc() { awk "BEGIN{printf \"%.4f\", $1}"; }

append_csv() {
    # args: implementacion tiempo_s speedup_abs speedup_rel eficiencia auc nucleos
    echo "$1,${K},${N_ITEMS},$2,$3,$4,$5,$6,$7" >> "${CSV}"
}

# ──────────────────────────── Inicialización ────────────────────────────────
mkdir -p results/plots
echo "implementacion,K,n_items,tiempo_s,speedup_abs,speedup_rel,eficiencia,auc,nucleos" > "${CSV}"

echo "${SEP}"
echo "  Benchmark MetaScore HPC  |  K=${K}  N_items=${N_ITEMS}"
echo "${SEP}"

# ──────────────────────────── Detectar Python ───────────────────────────────
PYTHON=""
for cmd in python python3; do
    if command -v "$cmd" &>/dev/null && "$cmd" -c "import numpy, sklearn" &>/dev/null 2>&1; then
        PYTHON="$cmd"; break
    fi
done

# ══════════════════════════════════════════════════════════════════════════════
# Nivel 1 — Python
# ══════════════════════════════════════════════════════════════════════════════
if [[ -n "$PYTHON" ]]; then
    echo ""
    echo "  [ Nivel 1 — Python secuencial ]"
    out=$("$PYTHON" python/sequential.py --k "${K}" 2>&1)
    echo "$out"
    T_seq=$(extract_tiempo "$out")
    AUC_seq=$(extract_auc "$out")
    append_csv "Python_Secuencial" "${T_seq}" "1.0000" "1.0000" "1.0000" "${AUC_seq}" "1"

    echo ""
    echo "  [ Nivel 1 — Python multicore ]"
    N_CORES=$("$PYTHON" -c "import multiprocessing; print(multiprocessing.cpu_count())")
    T1_mc=""
    for p in 1 2 4 6 "${N_CORES}"; do
        p=$(( p ))  # normalizar
        out=$("$PYTHON" python/multicore.py --k "${K}" --workers "${p}" 2>&1)
        echo "$out"
        tp=$(extract_tiempo "$out")
        auc=$(extract_auc "$out")
        [[ -z "$T1_mc" ]] && T1_mc="$tp"
        s_abs=$(calc "${T_seq}/${tp}")
        s_rel=$(calc "${T1_mc}/${tp}")
        efic=$(calc "${s_rel}/${p}")
        append_csv "Python_Multicore_P${p}" "${tp}" "${s_abs}" "${s_rel}" "${efic}" "${auc}" "${p}"
    done
else
    echo ""
    echo "  [SKIP] Python con numpy/sklearn no disponible en WSL."
    echo "         Ejecuta 'python python/benchmark_python.py' en Windows"
    echo "         y copia las filas al CSV manualmente."
    T_seq="166.9143"   # usar T0 conocido para calcular speedup de C
    echo "         Usando T_seq=${T_seq} s como referencia para speedup C."
fi

# ══════════════════════════════════════════════════════════════════════════════
# Nivel 2 — C + OpenMP
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "${SEP}"
echo "  [ Nivel 2 — C + OpenMP ]"

if [[ ! -f "${C_DIR}/scoring_openmp" ]]; then
    echo "  Compilando..."
    (cd "${C_DIR}" && make scoring_openmp --silent)
fi

T1_omp=""
for t in "${OMP_THREADS[@]}"; do
    out=$(cd "${C_DIR}" && OMP_NUM_THREADS="${t}" ./scoring_openmp "${K}" "${SEED}" 2>/dev/null) || true
    tp=$(extract_tiempo "$out")
    auc=$(extract_auc "$out")
    [[ -z "$T1_omp" ]] && T1_omp="$tp"
    if [[ "$tp" == "0" ]]; then
        echo "  [ERROR] OpenMP P=${t} no produjo salida"
        continue
    fi
    s_abs=$(calc "${T_seq}/${tp}")
    s_rel=$(calc "${T1_omp}/${tp}")
    efic=$(calc "${s_rel}/${t}")
    echo "  P=${t}: T=${tp}s  S_abs=${s_abs}x  S_rel=${s_rel}x  E=${efic}  AUC=${auc}"
    append_csv "C_OpenMP_P${t}" "${tp}" "${s_abs}" "${s_rel}" "${efic}" "${auc}" "${t}"
done

# ══════════════════════════════════════════════════════════════════════════════
# Nivel 2 — C + MPI
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "${SEP}"
echo "  [ Nivel 2 — C + MPI ]"

if [[ ! -f "${C_DIR}/scoring_mpi" ]]; then
    echo "  Compilando..."
    (cd "${C_DIR}" && make scoring_mpi --silent)
fi

T1_mpi=""
for p in "${MPI_PROCS[@]}"; do
    # Promedio de 3 corridas para reducir ruido
    sum=0; count=0
    for i in 1 2 3; do
        out=$(cd "${C_DIR}" && mpirun -n "${p}" ./scoring_mpi "${K}" "${SEED}" 2>/dev/null) || true
        tp=$(extract_tiempo "$out")
        auc=$(extract_auc "$out")
        if [[ "$tp" != "0" && -n "$tp" ]]; then
            sum=$(awk "BEGIN{printf \"%.6f\", ${sum}+${tp}}")
            count=$((count + 1))
        fi
    done
    if [[ "$count" -eq 0 ]]; then
        echo "  [ERROR] MPI P=${p} no produjo salida"
        continue
    fi
    tp=$(awk "BEGIN{printf \"%.4f\", ${sum}/${count}}")
    [[ -z "$T1_mpi" ]] && T1_mpi="$tp"
    s_abs=$(calc "${T_seq}/${tp}")
    s_rel=$(calc "${T1_mpi}/${tp}")
    efic=$(calc "${s_rel}/${p}")
    echo "  P=${p}: T=${tp}s (prom. ${count} runs)  S_abs=${s_abs}x  S_rel=${s_rel}x  E=${efic}  AUC=${auc}"
    append_csv "C_MPI_P${p}" "${tp}" "${s_abs}" "${s_rel}" "${efic}" "${auc}" "${p}"
done

# ──────────────────────────── Resumen ───────────────────────────────────────
echo ""
echo "${SEP}"
echo "  Benchmark completado. Resultados en: ${CSV}"
echo "  Nota: filas CUDA deben añadirse desde los resultados de Google Colab."
echo "${SEP}"
echo ""
column -t -s ',' "${CSV}"
