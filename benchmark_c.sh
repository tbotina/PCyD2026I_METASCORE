#!/usr/bin/env bash
# benchmark_c.sh — Benchmark OpenMP y MPI desde ~/metascore
#
# Uso (desde la raíz del proyecto en WSL):
#   bash benchmark_c.sh [K] [T0_segundos]
#
# Los binarios deben estar en ~/metascore junto con ~/data/ (archivos .bin)
# Resultado guardado en resultados_c.txt en la raíz del proyecto.

K="${1:-100000}"
T0="${2:-16.8158}"
SEED=0
BINARY_DIR="$HOME/metascore"
OUT_FILE="$(pwd)/resultados_c.txt"
SEP="===================================================================="

OMP_THREADS=(1 2 4 6 12)
MPI_PROCS=(1 2 4 6)

# ──────────────────────────── Funciones ─────────────────────────────────────

calc_speedup()  { awk -v t0="$1" -v tp="$2" 'BEGIN{printf "%.4f", t0/tp}'; }
calc_efic()     { awk -v s="$1"  -v p="$2"  'BEGIN{printf "%.4f", s/p}'; }
amdahl_f()      { awk -v s="$1"  -v p="$2"  \
    'BEGIN{f=(p>1&&s>1)?(s-1)/(s*(1-1/p)):0; printf "%.4f",f}'; }
amdahl_smax()   { awk -v f="$1" 'BEGIN{printf "%.2f",(f<1)?1/(1-f):9999}'; }

run_section() {
    local impl="$1"   # "openmp" o "mpi"
    local -n procs=$2

    echo ""
    if [[ "$impl" == "openmp" ]]; then
        echo "  [ OpenMP — memoria compartida ]"
    else
        echo "  [ MPI — memoria distribuida ]"
    fi
    echo ""
    printf "  %-4s  %-10s  %-10s  %-8s  %-10s  %-6s\n" \
        "P" "T(P) [s]" "S_abs" "S_rel" "Eficiencia" "AUC"
    printf "  %-4s  %-10s  %-10s  %-8s  %-10s  %-6s\n" \
        "----" "----------" "----------" "--------" "----------" "------"

    local t1=""
    local best_s_rel=0 best_p=1
    local rows=()

    for p in "${procs[@]}"; do
        if [[ "$impl" == "openmp" ]]; then
            out=$(cd "$BINARY_DIR" && OMP_NUM_THREADS=$p ./scoring_openmp "$K" "$SEED" 2>/dev/null) || true
        else
            out=$(cd "$BINARY_DIR" && mpirun -n "$p" ./scoring_mpi "$K" "$SEED" 2>/dev/null) || true
        fi

        tiempo=$(echo "$out" | grep -oP '(?<=Tiempo       = )\d+\.\d+' || echo "")
        auc=$(echo    "$out" | grep -oP '(?<=AUC          = )\d+\.\d+' || echo "")

        if [[ -z "$tiempo" ]]; then
            printf "  %-4s  %-10s  %-10s  %-8s  %-10s  %-6s\n" \
                "$p" "ERROR" "-" "-" "-" "-"
            continue
        fi

        [[ -z "$t1" ]] && t1="$tiempo"

        s_abs=$(calc_speedup "$T0" "$tiempo")
        s_rel=$(calc_speedup "$t1" "$tiempo")
        efic=$(calc_efic "$s_rel" "$p")

        printf "  %-4s  %-10s  %-10s  %-8s  %-10s  %-6s\n" \
            "$p" "$tiempo" "$s_abs" "$s_rel" "$efic" "$auc"

        if awk -v s="$s_rel" -v b="$best_s_rel" 'BEGIN{exit !(s+0>b+0)}'; then
            best_s_rel="$s_rel"; best_p="$p"
        fi
    done

    local f=$(amdahl_f "$best_s_rel" "$best_p")
    local smax=$(amdahl_smax "$f")
    echo ""
    echo "  Amdahl — mejor punto: P=$best_p, S_rel=${best_s_rel}x"
    echo "    Fraccion paralela  f = $f  ($(awk -v f="$f" 'BEGIN{printf "%.1f",f*100}') %)"
    echo "    Speedup maximo teo.  = ${smax}x"
}

# ──────────────────────────── Verificar binarios ────────────────────────────
for bin in scoring_openmp scoring_mpi; do
    if [[ ! -f "$BINARY_DIR/$bin" ]]; then
        echo "ERROR: $BINARY_DIR/$bin no encontrado."
        echo "Compila primero con:"
        echo "  cd C_OpenMP_MPI && make"
        echo "  cp scoring_openmp scoring_mpi ~/metascore/"
        exit 1
    fi
done

# ──────────────────────────── Ejecutar y guardar ────────────────────────────
{
    echo "$SEP"
    echo "  Benchmark C — MetaScore HPC"
    echo "  K=$K  |  T0 Python secuencial = ${T0} s"
    echo "  Columnas: S_abs = speedup vs Python | S_rel = speedup dentro de C"
    echo "$SEP"

    run_section "openmp" OMP_THREADS

    echo ""
    echo "$SEP"

    run_section "mpi" MPI_PROCS

    echo ""
    echo "$SEP"
} | tee "$OUT_FILE"

echo ""
echo "  Resultados guardados en: $OUT_FILE"
