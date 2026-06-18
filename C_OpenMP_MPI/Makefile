# ────────────────────────────────────────────────────────────────────────────
# Makefile — C_OpenMP_MPI
#
# Targets:
#   all           → compila scoring_openmp y scoring_mpi
#   scoring_openmp → binario OpenMP
#   scoring_mpi    → binario MPI
#   clean          → elimina binarios y archivos objeto
#
# Uso:
#   make                          # compilar todo
#   make OMP_THREADS=8            # cambiar número de hilos en la variable
#   OMP_NUM_THREADS=4 ./scoring_openmp 10000 0
#   mpirun -n 4 ./scoring_mpi 10000 0
# ────────────────────────────────────────────────────────────────────────────

CC      = gcc
MPICC   = mpicc
CFLAGS  = -O2 -Wall -Wextra -std=c11
LDFLAGS = -lm

.PHONY: all clean

all: scoring_openmp scoring_mpi

scoring_openmp: scoring_openmp.c
	$(CC) $(CFLAGS) -fopenmp $< -o $@ $(LDFLAGS)
	@echo "Compilado: $@"

scoring_mpi: scoring_mpi.c
	$(MPICC) $(CFLAGS) $< -o $@ $(LDFLAGS)
	@echo "Compilado: $@"

clean:
	rm -f scoring_openmp scoring_mpi *.o
	@echo "Limpieza completada."
