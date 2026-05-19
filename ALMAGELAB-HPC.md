# Guía de Ejecución en AlmageLab-HPC

Este documento explica paso a paso cómo ejecutar el proyecto MDN-C-CS en el cluster **AlmageLab-HPC** de la UNIMORE.

---

## Índice

1. [Información del Cluster](#información-del-cluster)
2. [Requisitos Previos](#requisitos-previos)
3. [Configuración Inicial](#configuración-inicial)
4. [ Flujo de Ejecución](#flujo-de-ejecución)
5. [Scripts SLURM Disponibles](#scripts-slurm-disponibles)
6. [Comandos Útiles](#comandos-útiles)
7. [Solución de Problemas](#solución-de-problemas)

---

## Información del Cluster

### Datos de Acceso

- **Login Node**: `ailb-login-02.ing.unimore.it`
- **Account**: `cvcs2026`
- **Partition**: `all_serial`
- **GPU**: NVIDIA (1 GPU por job por defecto)

### Estructura de Directorios en el Cluster

```
/homes/<usuario>/cvcs2026/
├── MDN-C-CS/          # Código del proyecto
├── venv/              # Entorno virtual Python
├── checkpoints/       # Modelos descargados (opcional)
└── results/           # Resultados de ejecuciones
```

---

## Requisitos Previos

### 1. Cuenta en AlmageLab-HPC

Tienes que tener:
- Una cuenta activa en el cluster AlmageLab-HPC
- Acceso al account `cvcs2026`
- Autorización para usar GPUs

### 2. Software Requerido en Cluster

El cluster debe tener instalado:
- **Python 3.10 o superior**
- **pip** (gestor de paquetes Python)
- **NVidia driver** (para CUDA/GPU)

Verificar en el cluster:
```bash
python --version
pip --version
nvidia-smi
```

### 3. Requisitos de Recursos

| Recurso | Cantidad | Notas |
|---------|----------|-------|
| CPU | 4-8 cores | Dependiendo del job |
| Memoria | 16-32 GB | Según job |
| GPU | 1 | NVIDIA (CUDAnable) |
| Disk | 40-60 GB | Modelos + resultados |
| Tiempo máximo | 8 horas | En partition `all_serial` |

---

## Configuración Inicial

### Paso 1: Conectarse al Cluster

```bash
ssh <tu-usuario>@ailb-login-02.ing.unimore.it
```

### Paso 2: Copiar el Proyecto

Puedes usar `scp` desde tu máquina local:

```bash
# En tu máquina local (no en el cluster):
scp -r /path/MDN-C-CS <usuario>@ailb-login-02.ing.unimore.it:/homes/<usuario>/cvcs2026/
```

O usar git si el repositorio está en un servidor accesible:

```bash
cd /homes/<usuario>/cvcs2026/
git clone <url-del-repositorio> MDN-C-CS
```

### Paso 3: Ejecutar Setup

El script `slurm/setup_cluster.sh` configura todo automáticamente:

```bash
cd /homes/<usuario>/cvcs2026/MDN-C-CS
bash slurm/setup_cluster.sh
```

Este script:
1. Crea los directorios necesarios
2. Crea/activa el entorno virtual Python
3. Actualiza pip
4. Instala todas las dependencias de `requirements.txt`
5. Verifica que PyTorch y GPU funcionen correctamente

**Nota**: Si el script de setup ya fue ejecutado previamente, puedes omitir este paso. Para actualizar solo las dependencias:

```bash
source /homes/<usuario>/cvcs2026/venv/bin/activate
pip install --upgrade -r requirements.txt
```

---

## Flujo de Ejecución

El workflow típico en el cluster sigue estos pasos:

```mermaid
graph LR
    A[Setup] --> B[Descargar Modelos]
    B --> C[Generar Benchmark]
    C --> D[Ejecutar Evaluación]
    D --> E[Analizar Resultados]
```

### Paso 1: Configurar Entorno (1 sola vez)

```bash
cd /homes/<usuario>/cvcs2026/MDN-C-CS
bash slurm/setup_cluster.sh
```

### Paso 2: Descargar Modelos (aprox. 15-30 min)

Los modelos se descargan desde Hugging Face y se guardan en `data/checkpoints/`.

```bash
sbatch slurm/download_models.sh
```

Para descargar solo un modelo específico:

```bash
# Solo LLaVA-v1.5
sbatch --export=ALL,MODEL=llava15 slurm/download_models.sh

# Solo Qwen3-VL
sbatch --export=ALL,MODEL=qwen3 slurm/download_models.sh
```

**Monitorizar el job**:
```bash
squeue --me
cat /homes/<usuario>/cvcs2026/download_<job_id>.out
```

### Paso 3: Ejecutar Evaluación (aprox. 30-90 min por modelo)

```bash
sbatch slurm/evaluation.sh
```

Este job:
1. Genera el benchmark (sintético, ~100 samples por defecto)
2. Ejecuta la evaluación de los modelos
3. Genera los reportes de análisis

**Monitorizar**:
```bash
squeue --me
cat /homes/<usuario>/cvcs2026/evaluation_<job_id>.out
```

**Ver resultados**:
```bash
cat results/llava15/metrics.json
cat results/qwen3/metrics.json
```

---

## Scripts SLURM Disponibles

### `slurm/download_models.sh`

**Purpose**: Descarga los checkpoints de los modelos desde Hugging Face.

**Resource Request**:
- CPU: 4 cores
- Memory: 32 GB
- Time: 2 horas
- GPU: No requiere (solo disco)

**Uso**:
```bash
# Descargar todos los modelos
sbatch slurm/download_models.sh

# Descargar solo un modelo
sbatch --export=ALL,MODEL=llava15 slurm/download_models.sh
sbatch --export=ALL,MODEL=qwen3 slurm/download_models.sh
sbatch --export=ALL,MODEL=internvl35 slurm/download_models.sh
```

---

### `slurm/evaluation.sh`

**Purpose**: Ejecuta el pipeline completo (benchmark + evaluación + análisis).

**Resource Request**:
- CPU: 8 cores
- Memory: 32 GB
- GPU: 1 (solicita `--gres=gpu:1`)
- Time: 8 horas

**Variables de Entorno Soportadas**:

| Variable | Default | Descripción |
|----------|---------|-------------|
| `NUM_SAMPLES` | 100 | Cantidad de muestras en el benchmark |
| `MAX_SAMPLES` | (todos) | Límite de muestras a evaluar |
| `MODEL` | all | Modelo específico: `llava15`, `qwen3`, `internvl35`, o `all` |
| `OFFLINE` | 1 | Si 1 usa checkpoints locales, si 0 descarga en el job |
| `DTYPE` | auto | Precision: `auto`, `float16`, `bfloat16` |
| `MITIGATION_STUDY` | 0 | Si 1 ejecuta baseline + strict prompt |

**Uso**:

```bash
# Evaluación estándar (todos los modelos por defecto)
sbatch slurm/evaluation.sh

# Solo LLaVA con 200 muestras
sbatch --export=ALL,MODEL=llava15,NUM_SAMPLES=200 slurm/evaluation.sh

# Qwen3-VL con mitigación study (baseline + strict)
sbatch --export=ALL,MODEL=qwen3,MITIGATION_STUDY=1 slurm/evaluation.sh

# Smoke test rápido (20 muestras, solo LLaVA)
sbatch --export=ALL,MODEL=llava15,NUM_SAMPLES=20,MAX_SAMPLES=5 slurm/evaluation.sh

# Modo online (descargar modelos dentro del job)
sbatch --export=ALL,OFFLINE=0 slurm/evaluation.sh
```

---

### `slurm/setup_cluster.sh`

**Purpose**: Configura el entorno en el cluster (1 sola vez).

**Resource Request**: No usa SLURM - se ejecuta en login node.

**Uso**:
```bash
bash slurm/setup_cluster.sh
```

**Lo que hace**:
1. Crea directorios: `/homes/<usuario>/cvcs2026/{MDN-C-CS,venv}`
2. Crea entorno virtual Python si no existe
3. Instala dependencias de `requirements.txt`
4. Verifica PyTorch + GPU

---

### `slurm/jupyter_gpu.sh`

**Purpose**: Inicia JupyterLab con acceso a GPU en un nodo computacional.

**Resource Request**:
- CPU: 4 cores
- Memory: 16 GB
- GPU: 1
- Time: 4 horas

**Uso**:

1. Enviar el job:
```bash
sbatch slurm/jupyter_gpu.sh
```

2. Esperar a que inicie y revisar el log:
```bash
sleep 60
tail -f /homes/<usuario>/cvcs2026/jupyter_<job_id>.out
```

3. Ver el token y URL en los logs (ejemplo):
```
http://0.0.0.0:8888/?token=abc123...
```

4. Hacer tunneling desde tu máquina local:
```bash
# En tu máquina local (no en el cluster):
ssh -L 8888:<nodo-gpu>:8888 <usuario>@ailb-login-02.ing.unimore.it
```

Donde `<nodo-gpu>` es el nombre del nodo que aparece en el log del job.

5. Abrir navegador en: `http://localhost:8888/?token=abc123...`

**Nota**: Solo para desarrollo/interactivo. No usar para evaluaciones largas.

---

## Comandos Útiles

### Gestión de Jobs

```bash
# Ver tus jobs en cola
squeue --me

# Ver detalles de un job específico
scontrol show job <job_id>

# Cancelar un job
scancel <job_id>

# Cancelar todos tus jobs
scancel -u <usuario>
```

### Monitorización

```bash
# Ver uso de GPU
nvidia-smi

# Ver uso de CPU y memoria
htop

# Ver espacio en disco
df -h
du -sh /homes/<usuario>/cvcs2026/

# Ver archivos más grandes
find /homes/<usuario>/cvcs2026 -type f -size +100M -exec du -h {} + | sort -rh | head -20
```

### Ver Logs

```bash
# Logs de evaluation
tail -f /homes/<usuario>/cvcs2026/evaluation_<job_id>.out

# Logs de download
tail -f /homes/<usuario>/cvcs2026/download_<job_id>.out

# Logs de Jupyter
tail -f /homes/<usuario>/cvcs2026/jupyter_<job_id>.out
```

### Ver Resultados

```bash
# Métricas JSON (detalladas)
cat results/llava15/metrics.json

# Métricas JSON (Qwen3)
cat results/qwen3/metrics.json

# Resultados en texto
cat reports/llava15/summary.txt

# Ver gráficos generados
ls -la reports/llava15/plots/
```

---

## Solución de Problemas

### Error: "No GPU available"

**Causa**: El job no está solicitando GPU correctamente o la GPU no está disponible.

**Solución**:
```bash
# Verificar que el script solicite GPU
grep -A2 "gres=gpu" slurm/evaluation.sh

# Verificar disponibilidad de GPU en el nodo
srun --gres=gpu:1 --time=01:00:00 nvidia-smi
```

### Error: "Out of memory" o "CUDA out of memory"

**Causa**: El modelo es demasiado grande para la GPU disponible.

**Solución**:
```bash
# Usar una cantidad menor de muestras
sbatch --export=ALL,NUM_SAMPLES=20,MAX_SAMPLES=5 slurm/evaluation.sh

# Usar descarga offline para evitar cargar modelos en memoria
sbatch --export=ALL,OFFLINE=1 slurm/evaluation.sh

# En evaluation.sh, ajustar pytorch memory allocator
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Error: "ModuleNotFoundError: No module named 'torch'"

**Causa**: El entorno virtual no está activado o las dependencias no están instaladas.

**Solución**:
```bash
# Re-ejecutar setup
bash slurm/setup_cluster.sh

# O manualmente:
source /homes/<usuario>/cvcs2026/venv/bin/activate
pip install -r requirements.txt
```

### Error: "Permission denied" al escribir en directorios

**Causa**: Problemas de permisos en el directorio de trabajo.

**Solución**:
```bash
# Verificar permisos
ls -la /homes/<usuario>/cvcs2026/

# Corregir permisos si es necesario
chmod -R u+rwx /homes/<usuario>/cvcs2026/
```

### Job se queda colgado en "Pending" estado

**Causa**: Cola de jobs larga o recursos no disponibles.

**Solución**:
```bash
# Ver razón del pending
squeue --me --state=PDF
squeue --me --state=PR

# Ver cola de la partition
squeue -p all_serial --state=PDF

# Intentar con tiempo menor para prioridad
sbatch --export=ALL,NUM_SAMPLES=10,MAX_SAMPLES=2 slurm/evaluation.sh
```

### Espacio en disco insuficiente

**Causa**: Los modelos descargados ocupan 20-40 GB.

**Solución**:
```bash
# Ver uso de disco
df -h /homes/<usuario>
du -sh /homes/<usuario>/cvcs2026/

# Limpiar cache de pip (si es necesario)
rm -rf /homes/<usuario>/cvcs2026/venv/cache
pip cache purge

# Limpiar logs viejos
find /homes/<usuario>/cvcs2026 -name "*.out" -mtime +30 -delete
```

---

## Ejemplo Completo de Sesión

```bash
# 1. Conectar al cluster
ssh usuario@ailb-login-02.ing.unimore.it

# 2. Verificar espacio en disco
df -h /homes/usuario

# 3. Ir al directorio del proyecto
cd /homes/usuario/cvcs2026/MDN-C-CS

# 4. Verificar entorno (si ya existe)
source /homes/usuario/cvcs2026/venv/bin/activate
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
deactivate

# 5. Iniciar descarga de modelos (background)
sbatch slurm/download_models.sh
DOWNLOAD_JOB=$(squeue --me | grep download | awk '{print $1}')

# 6. Iniciar evaluación (esperar a que descargue termine)
# O lanzar ambos secuencialmente
sbatch slurm/evaluation.sh
EVAL_JOB=$(squeue --me | grep evaluation | awk '{print $1}')

# 7. Monitorizar
while [ $(squeue --me | grep -c "RUNNING\|PENDING") -gt 0 ]; do
    echo "Jobs running..."
    sleep 60
done

# 8. Ver resultados
cat results/llava15/metrics.json
cat results/qwen3/metrics.json
```

---

## Contacto y Soporte

Para problemas con el cluster AlmageLab-HPC:

- **Documentación**: https://hpc.unimore.it/
- **Support email**: support-hpc@unimore.it
- **Urgente**: Contactar through the HPC portal

---

## Resumen de Comandos Rápidos

| Acción | Comando |
|--------|---------|
| Conectar al cluster | `ssh <user>@ailb-login-02.ing.unimore.it` |
| Setup inicial | `bash slurm/setup_cluster.sh` |
| Descargar modelos | `sbatch slurm/download_models.sh` |
| Ejecutar evaluación | `sbatch slurm/evaluation.sh` |
| Ver jobs | `squeue --me` |
| Cancelar job | `scancel <job_id>` |
| Ver logs | `tail -f /homes/<user>/cvcs2026/*.out` |

---

*Última actualización: 2026-05-19*
