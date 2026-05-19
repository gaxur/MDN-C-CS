# 📁 Estructura de Archivos del Proyecto MDN-C-CS

Documentación completa de todos los archivos y carpetas del proyecto de evaluación de alucinaciones en Modelos de Visión-Lenguaje (VLMs).

---

## 📋 Archivos Raíz

### `README.md`
Archivo principal de documentación del proyecto. Contiene:
- Descripción general del proyecto
- Motivación y objetivos
- Alcance implementado (modelos, métricas, análisis)
- Estructura del proyecto
- Requisitos del sistema
- Instrucciones de instalación

### `QUICKSTART.md`
Guía rápida para comenzar a usar el proyecto. Incluye:
- Pasos para preparar el entorno (venv, instalación de dependencias)
- Descargar checkpoints de modelos
- Generar el benchmark
- Ejecutar evaluaciones
- Analizar resultados

### `REQUIREMENTS.md`
Especificación de dependencias del proyecto. Detalla:
- Librerías principales necesarias (torch, transformers, etc.)
- Versiones recomendadas
- Opciones de instalación

### `requirements.txt`
Archivo de dependencias pip para instalar todos los paquetes necesarios:
- PyTorch y TorchVision
- Transformers (>=4.57.0)
- Herramientas de aceleración y hugging face
- Librerías de análisis (pandas, numpy, matplotlib, seaborn)
- JupyterLab para notebooks interactivas

### `TIMELINE.md`
Cronograma y timeline del proyecto. Contiene:
- Hitos importantes
- Fechas de ejecución
- Fases de desarrollo

### `run.py`
Script interactivo con menú para ejecutar tareas del proyecto. Proporciona:
- Interfaz de línea de comandos para el usuario
- Menú interactivo con opciones principales
- Ejecución de comandos comunes del proyecto

---

## 🐍 Scripts Principal (`src/`)

Contiene el código fuente del proyecto.

### `run_pipeline.py`
Orquestador principal del proyecto. Automatiza el flujo completo:
1. Genera el benchmark de prueba
2. Evalúa los modelos VLM
3. Ejecuta estudios de mitigación (opcional)
4. Genera reportes de análisis
- Verifica disponibilidad de GPU
- Configura rutas de directorios
- Maneja múltiples modelos (llava15, qwen3, internvl35)

### `generate_benchmark.py`
Generador sintético del benchmark de evaluación. Crea:
- Escenas visuales simples (círculos, cuadrados, triángulos)
- Preguntas trampa en 4 categorías:
  - `object_absent`: objetos que no están en la imagen
  - `attribute`: atributos incorrectos (ej: color equivocado)
  - `relation`: relaciones espaciales falsas (izquierda, derecha, arriba, abajo)
  - `count`: conteos incorrectos de objetos
- Pares imagen-pregunta diseñados para provocar alucinaciones
- Datos con formato JSON para evaluación

### `evaluate_vlms.py`
Motor de evaluación de modelos de visión-lenguaje. Realiza:
- Carga de modelos VLM (LLaVA, Qwen, InternVL)
- Procesamiento de imágenes y preguntas
- Inferencia del modelo en el benchmark
- Captura de respuestas y métricas tipo POPE (Probability Of Positive Examples)
- Generación de resultados en JSON
- Soporta diferentes estrategias de prompts (baseline vs strict)

### `download_models.py`
Descargador de checkpoints de modelos. Funciona para:
- Descargar modelos preentrenados desde Hugging Face
- Guardarlos en caché local (`data/checkpoints/`)
- Soporta descarga selectiva de modelos
- Evita cargar modelos completos en memoria durante la descarga
- Modelos soportados: llava15, qwen3, internvl35

### `analyze_results.py`
Analizador de resultados de evaluación. Genera:
- Tablas resumen de métricas de alucinación
- Gráficos comparativos entre modelos
- Análisis por condición (color, relación, conteo)
- Reportes en texto
- Exporta métricas a nivel de condición
- Identifica errores y casos especiales

---

## 📊 Scripts SLURM (`slurm/`)

Contiene scripts para ejecutar trabajos en clusters GPU con SLURM.

### `download_models.sh`
Script SLURM para descargar modelos en el cluster:
- Solicita recursos de GPU
- Ejecuta `download_models.py` en el cluster
- Almacena checkpoints en el sistema de archivos compartido

### `setup_cluster.sh`
Script de configuración inicial del cluster:
- Prepara el entorno en nodos computacionales
- Crea directorios necesarios
- Configura variables de entorno
- Instala dependencias en el cluster

### `evaluation.sh`
Script SLURM para ejecutar evaluaciones de VLMs:
- Solicita recursos GPU específicos
- Ejecuta `evaluate_vlms.py` en paralelo
- Maneja múltiples trabajos de evaluación
- Guarda logs de ejecución

### `jupyter_gpu.sh`
Script para lanzar JupyterLab con acceso a GPU en el cluster:
- Solicita nodo computacional con GPU
- Inicia servidor Jupyter
- Proporciona token de acceso
- Facilita desarrollo interactivo

### `README.md`
Documentación específica para SLURM:
- Instrucciones de uso de scripts
- Ejemplos de configuración
- Parámetros de ejecución
- Tips para optimizar trabajos en cluster

---

## 📓 Notebooks (`notebooks/`)

Contiene notebooks Jupyter interactivas para exploración y análisis.

### `01_load_models.ipynb`
Notebook de prueba e interacción con modelos:
- Carga modelos VLM
- Permite hacer inferencias interactivas
- Permite visualizar imágenes y respuestas
- Facilita debugging y experimentación
- Pruebas rápidas sin ejecutar pipeline completo

---

## 📁 Directorios

### `data/`
Almacenamiento de datos del proyecto:
- **`checkpoints/`**: Modelos preentrenados descargados de Hugging Face
  - Contiene los pesos de LLaVA-v1.5, Qwen3-VL, etc.
  - Generados por `download_models.py`
  - ~20-40 GB según modelos descargados

### `reports/`
Reportes y resultados de análisis:
- **`README.md`**: Documentación sobre reportes
- **`report_template.tex`**: Plantilla LaTeX para generar reportes PDF
- Gráficos generados por `analyze_results.py`
- Tablas resumen de evaluación
- Análisis estadísticos

### `results/`
Resultados de evaluaciones (generado durante ejecución):
- Archivos JSON con respuestas de modelos
- Métricas de alucinación por modelo
- Datos de condiciones específicas
- Logs de ejecución

### `scripts/`
Carpeta adicional para scripts personalizados:
- Actualmente vacía
- Lugar para scripts de utilidad personalizados

---

## 🔄 Flujo de Ejecución General

```
1. Configuración (requirements.txt, QUICKSTART.md)
   ↓
2. Descargar modelos (download_models.py)
   ↓
3. Generar benchmark (generate_benchmark.py)
   ↓
4. Evaluar modelos (evaluate_vlms.py)
   ↓
5. Analizar resultados (analyze_results.py)
   ↓
6. Generar reportes (reports/)
```

---

## 🚀 Inicio Rápido

1. **Setup**: Seguir [QUICKSTART.md](QUICKSTART.md)
2. **Descargar modelos**: `python src/download_models.py --download --model all`
3. **Ejecutar pipeline**: `python src/run_pipeline.py` o usar `python run.py`
4. **Analizar**: Resultados en `reports/` y `results/`

---

## 📝 Notas Importantes

- **GPU necesaria**: Todos los scripts de evaluación requieren GPU con CUDA
- **Espacio disco**: Requiere 20-40 GB para almacenar modelos
- **Modelos por defecto**: LLaVA-v1.5 y Qwen3-VL
- **Cluster**: Los scripts SLURM están optimizados para ejecución en clusters GPU
- **Reproducibilidad**: El pipeline está diseñado para ser reproducible y auditables
