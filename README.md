# MetaScore HPC — Optimización Paralela del Sistema de Scoring Metagenómico

**Asignatura:** Programación Concurrente y Distribuida 2026-I  
**Área:** Bioinformática Computacional  
**Stack:** Python · C/OpenMP · C/MPI · CUDA  

---

## Descripción del Problema

Se busca encontrar un vector de pesos **W = (W₁, W₂, W₃)** que maximice el área bajo la curva ROC (**AUC**) en la clasificación binaria de muestras biológicas metagenómicas.

El dataset consiste en **10 muestras** de pacientes:
- 5 muestras **sanas** (`y = 0`)
- 5 muestras **enfermas** (`y = 1`)

Cada muestra se describe mediante **N ítems** (taxones/genomas), cada uno con tres perfiles:

| Símbolo | Perfil | Descripción |
|---------|--------|-------------|
| `Tᵢ` | Taxonómico | Abundancia relativa de microorganismos |
| `Sᵢ` | Ecológico | Variables contextuales no genómicas |
| `Fᵢ` | Funcional | Presencia/ausencia de genes de interés |

### Modelo Matemático

```
Score por ítem:    Pᵢ = W₁·Tᵢ + W₂·Sᵢ + W₃·Fᵢ

Score por muestra: Score = A · P          (A ∈ ℝ^{10×N}, P ∈ ℝ^N)

Función objetivo:  max  AUC(y, Score(W))
                    W

Restricción:       W₁ + W₂ + W₃ = 1,  Wᵢ ≥ 0   (símplex estándar)
```

### Estrategia de Optimización

**Random Search**: Se muestrean K = 100,000 vectores **W** aleatoriamente sobre el símplex y se evalúa el AUC para cada uno. La independencia entre evaluaciones hace que el problema sea *embarazosamente paralelizable*.

---

## Arquitectura del Sistema

| Nivel | Tecnología | Paradigma | Archivo principal |
|-------|-----------|-----------|-------------------|
| 1A | Python secuencial | Baseline | `python/sequential.py` |
| 1B | Python multiprocessing | Multicore | `python/multicore.py` |
| 2A | C + OpenMP | Memoria compartida | `C_OpenMP_MPI/scoring_openmp.c` |
| 2B | C + MPI | Memoria distribuida | `C_OpenMP_MPI/scoring_mpi.c` |
| 3  | CUDA / PyCUDA | GPU masiva | `CUDA/` |

---

## Prerrequisitos

### Nivel 1 — Python

- **Python 3.10** o superior

```bash
pip install numpy scikit-learn matplotlib
```

### Nivel 2 — C (requiere Linux, WSL 2 o macOS)

> En Windows se recomienda usar **WSL 2**.  
> Se usa **MPICH** (no OpenMPI) por compatibilidad con el entorno WSL 2.

| Herramienta | Instalación (Ubuntu/Debian) |
|-------------|----------------------------|
| GCC con OpenMP | `sudo apt install gcc` |
| MPICH | `sudo apt install mpich` |
| Make | `sudo apt install make` |

```bash
# Verificar instalaciones
gcc --version
mpicc --version
mpirun --version
```

### Nivel 3 — CUDA

> La GPU de este equipo es AMD Radeon (integrada). CUDA es exclusivo de NVIDIA.  
> El Nivel 3 se ejecuta en **Google Colab** con GPU NVIDIA T4 gratuita.

- Cuenta de Google con acceso a Google Colab
- Notebook: `CUDA/MetaScore_HPC_Nivel3_Colab.ipynb`

---

## Estructura del Repositorio

```
PCyD_Proyecto_MetaScore_HPC_2026I/
├── data/
│   ├── generate_data.py          # Generador de datos sintéticos
│   ├── matrix_A.npy / .bin       # Matriz de contribución
│   ├── labels.npy / .bin         # Etiquetas binarias
│   ├── profile_T/S/F.npy / .bin  # Perfiles por ítem
│   └── metadata.txt              # Dimensiones: n_samples n_items
├── python/
│   ├── sequential.py             # Nivel 1A — baseline secuencial
│   ├── multicore.py              # Nivel 1B — multiprocessing
│   └── benchmark_python.py       # Automatiza Nivel 1 y genera tabla
├── C_OpenMP_MPI/
│   ├── scoring_openmp.c          # Nivel 2A — OpenMP
│   ├── scoring_mpi.c             # Nivel 2B — MPI
│   └── Makefile
├── CUDA/
│   ├── scoring_kernel.cu         # Nivel 3 — CUDA C
│   ├── scoring_pycuda.py         # Nivel 3 — PyCUDA wrapper
│   └── MetaScore_HPC_Nivel3_Colab.ipynb
├── results/
│   ├── benchmark.csv             # Tabla comparativa de métricas
│   └── plots/                    # Gráficas speedup y eficiencia
├── report/
│   └── informe_tecnico.pdf       # Entregable final
├── run_all.sh                    # Benchmark C completo (WSL)
└── README.md
```

---

## Instalación

### 1. Descargar el repositorio

```bash
cd /ruta/al/proyecto
```

### 2. Crear entorno virtual Python (recomendado)

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate

# Linux / WSL
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias Python

```bash
pip install numpy scikit-learn matplotlib
```

### 4. Generar los datos de prueba

```bash
# N=50 ítems, semilla=42 (valores por defecto)
python data/generate_data.py
```

Crea los archivos `.npy` (Python) y `.bin` + `metadata.txt` (C/CUDA) en `data/`.

### 5. Compilar implementaciones C (en WSL)

```bash
cd C_OpenMP_MPI
make
cd ..
```

Genera los binarios `scoring_openmp` y `scoring_mpi`.

---

## Cómo Ejecutar Cada Nivel

> **Nota:** Generar los datos (`python data/generate_data.py`) antes de ejecutar cualquier implementación.

---

### Nivel 1A — Python Secuencial

```bash
python python/sequential.py --k 100000
```

**Salida esperada:**
```
[Secuencial]  K=100,000  N_items=50
  W*           = [0.9382, 0.0347, 0.0270]
  AUC          = 0.6800
  Consistencia = 0.70  [OK]
  Tiempo       = 166.9143 s
```

---

### Nivel 1B — Python Multicore

```bash
# Usar todos los núcleos disponibles
python python/multicore.py --k 100000

# Especificar número de procesos
python python/multicore.py --k 100000 --workers 4
```

**Benchmark automatizado** (genera tabla de speedup/eficiencia para P=1,2,4,6,12):

```bash
python python/benchmark_python.py
```

---

### Nivel 2A — C + OpenMP

> Los binarios deben ejecutarse **desde dentro de `C_OpenMP_MPI/`** porque el código busca los datos en `../data/`.

```bash
cd C_OpenMP_MPI

# Un único hilo
OMP_NUM_THREADS=1 ./scoring_openmp 100000 0

# Curva de speedup para P = 1 2 4 6 12
for t in 1 2 4 6 12; do
    echo "--- hilos=$t ---"
    OMP_NUM_THREADS=$t ./scoring_openmp 100000 0
done

cd ..
```

**Salida esperada (P=6):**
```
[OpenMP]  K=100000  hilos=6  n_items=50
  W*           = [0.9382, 0.0347, 0.0270]
  AUC          = 0.6800
  Consistencia = 0.70  [OK]
  Tiempo       = 0.0109 s
```

---

### Nivel 2B — C + MPI

> Misma restricción de directorio que OpenMP.

```bash
cd C_OpenMP_MPI

# Con 4 procesos
mpirun -n 4 ./scoring_mpi 100000 0

# Curva de speedup para P = 1 2 4 6
for p in 1 2 4 6; do
    echo "--- procesos=$p ---"
    mpirun -n $p ./scoring_mpi 100000 0
done

cd ..
```

**Salida esperada (P=4):**
```
[MPI]  K=100000 (padded=100000)  procesos=4  n_items=50
  W*           = [0.9382, 0.0347, 0.0270]
  AUC          = 0.6800
  Consistencia = 0.70  [OK]
  Tiempo       = 0.0163 s
```

---

### Nivel 3 — CUDA (Google Colab)

#### Paso 1 — Subir el notebook a Colab

1. Ir a [colab.research.google.com](https://colab.research.google.com)
2. `Archivo → Subir notebook` → seleccionar `CUDA/MetaScore_HPC_Nivel3_Colab.ipynb`
3. `Entorno de ejecución → Cambiar tipo → GPU T4`

#### Paso 2 — Subir archivos de datos

Cuando el notebook lo indique, subir todos los archivos de `data/`:

```
matrix_A.npy    matrix_A.bin
labels.npy      labels.bin
profile_T.npy   profile_T.bin
profile_S.npy   profile_S.bin
profile_F.npy   profile_F.bin
metadata.txt
```

#### Paso 3 — Ajustar T_PYTHON_SEQ

En la celda de parámetros del notebook, asegurarse de que:

```python
T_PYTHON_SEQ = 166.9143   # tiempo medido en Nivel 1A
K = 100_000
```

#### Paso 4 — Ejecutar todas las celdas

```
Entorno de ejecución → Ejecutar todas  (Ctrl + F9)
```

---

### Benchmark Completo C (Linux/WSL)

El script `run_all.sh` ejecuta todos los niveles C, promedia 3 corridas para MPI y genera `results/benchmark.csv`.

```bash
# Desde la raíz del proyecto en WSL
bash run_all.sh 100000 50
```

> El benchmark de Python se ejecuta por separado con `python python/benchmark_python.py` desde Windows,
> ya que requiere Anaconda/numpy. Las filas resultantes se incorporan al CSV automáticamente.

---

## Métricas de Evaluación

| Métrica | Fórmula | Descripción |
|---------|---------|-------------|
| Tiempo | `T = t_fin − t_inicio` | Solo la búsqueda, sin carga de datos |
| Speedup absoluto | `S_abs = T_Python_seq / T_impl` | Aceleración respecto al baseline Python |
| Speedup relativo | `S_rel = T_impl(P=1) / T_impl(P)` | Escalado dentro de la misma tecnología |
| Eficiencia | `E = S_rel / P` | Fracción del speedup ideal (E=1 es óptimo) |
| Ley de Amdahl | `S_max = 1 / (1−f)` | Límite teórico dado f paralelo empírico |
| AUC | Mann-Whitney U | Calidad del clasificador (igual en todas las impl.) |
| Consistencia | `C = (TPR + TNR) / 2` | Exactitud balanceada ∈ [0, 1] |

---

## Resultados Obtenidos (K = 100,000, N\_items = 50)

### Tabla comparativa completa

| Implementación | T (s) | S\_abs | S\_rel | Eficiencia | AUC | Núcleos |
|----------------|---------|--------|--------|------------|-----|---------|
| Python Secuencial | 166.9143 | 1.00× | 1.00× | 1.0000 | 0.6800 | 1 |
| Python Multicore P=1 | 168.4022 | 0.99× | 0.99× | 0.9912 | 0.6800 | 1 |
| Python Multicore P=2 | 91.0876 | 1.83× | 1.83× | 0.9162 | 0.6800 | 2 |
| Python Multicore P=4 | 60.7481 | 2.75× | 2.75× | 0.6869 | 0.6800 | 4 |
| Python Multicore P=6 | 51.4383 | 3.24× | 3.24× | 0.5408 | 0.6800 | 6 |
| Python Multicore P=12 | 45.5424 | 3.67× | 3.67× | 0.3054 | 0.6800 | 12 |
| C + OpenMP P=1 | 0.0561 | 2,975× | 1.00× | 1.0000 | 0.6800 | 1 |
| C + OpenMP P=2 | 0.0286 | 5,836× | 1.96× | 0.9808 | 0.6800 | 2 |
| C + OpenMP P=4 | 0.0149 | 11,202× | 3.77× | 0.9413 | 0.6800 | 4 |
| C + OpenMP P=6 | 0.0109 | 15,313× | 5.15× | 0.8578 | 0.6800 | 6 |
| C + OpenMP P=12 | 0.0095 | 17,570× | 5.91× | 0.4921 | 0.6800 | 12 |
| C + MPI P=1 | 0.0387 | 4,313× | 1.00× | 1.0000 | 0.6800 | 1 |
| C + MPI P=2 | 0.0202 | 8,263× | 1.92× | 0.9579 | 0.6800 | 2 |
| C + MPI P=4 | 0.0163 | 10,239× | 2.37× | 0.5935 | 0.6800 | 4 |
| C + MPI P=6 | 0.0122 | 13,682× | 3.17× | 0.5287 | 0.6800 | 6 |
| CUDA PyCUDA | 0.001706 | 97,848× | N/A | N/A | 0.6800 | GPU T4 |
| CUDA C | 0.000700 | 238,449× | N/A | N/A | 0.6800 | GPU T4 |

### Análisis de Amdahl (fracción paralela empírica)

| Implementación | Mejor punto | S\_rel | f empírico | S\_max teórico |
|----------------|-------------|--------|------------|----------------|
| Python Multicore | P=12 | 3.67× | 79.3 % | 4.84× |
| C + OpenMP | P=12 | 5.91× | 90.6 % | 10.66× |
| C + MPI | P=6 | 3.17× | 75.2 % | 4.03× |

---

## Hardware de Referencia

| Componente | Especificación |
|-----------|---------------|
| Procesador | AMD Ryzen 5 5500U @ 2.10 GHz |
| Núcleos físicos / lógicos | 6 / 12 |
| RAM | 8 GB DDR4 |
| GPU local | AMD Radeon integrada *(sin soporte CUDA)* |
| GPU Nivel 3 | NVIDIA T4 via Google Colab |
| SO | Windows 11 (Python) · WSL 2 Ubuntu 22.04 (C/MPI) |
| MPI | MPICH (no OpenMPI) |

---

## Solución de Problemas

**`FileNotFoundError: matrix_A.npy` / `../data/metadata.txt: No such file`**  
→ Ejecutar primero `python data/generate_data.py`  
→ Para los binarios C, ejecutar **desde dentro de `C_OpenMP_MPI/`**, no desde la raíz del proyecto.

**`make: gcc: command not found`**  
→ `sudo apt install gcc`

**`mpirun: command not found`**  
→ `sudo apt install mpich` (usar MPICH, no OpenMPI)

**`ModuleNotFoundError: No module named 'sklearn'`**  
→ `pip install scikit-learn`

**MPI se cuelga sin producir salida**  
→ Verificar que se compiló con MPICH: `mpicc --version` debe decir MPICH.  
→ Asegurarse de ejecutar desde `C_OpenMP_MPI/` donde están los binarios y `../data/` es accesible.

**El notebook de Colab falla en PyCUDA**  
→ `Entorno de ejecución → Ver recursos` — debe mostrar GPU T4.  
→ Si dice "Sin acelerador", cambiar a GPU T4 y reconectar.
