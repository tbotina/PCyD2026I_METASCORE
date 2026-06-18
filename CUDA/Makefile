# ────────────────────────────────────────────────────────────────────────────
# Makefile — CUDA
#
# Targets:
#   all            → compila scoring_cuda
#   scoring_cuda   → binario CUDA
#   clean          → elimina binarios y archivos objeto
#
# Variables configurables:
#   ARCH   → arquitectura GPU (sm_70 = Volta, sm_80 = Ampere, sm_86 = RTX 3xxx)
#            Consultar: nvidia-smi --query-gpu=compute_cap --format=csv
#
# Uso:
#   make ARCH=sm_86
#   ./scoring_cuda 10000 0
# ────────────────────────────────────────────────────────────────────────────

NVCC   = nvcc
ARCH  ?= sm_70
NVFLAGS = -O2 -arch=$(ARCH) -Xcompiler -Wall

.PHONY: all clean

all: scoring_cuda

scoring_cuda: scoring_kernel.cu
	$(NVCC) $(NVFLAGS) $< -o $@
	@echo "Compilado: $@  (arch=$(ARCH))"

clean:
	rm -f scoring_cuda *.o
	@echo "Limpieza completada."
