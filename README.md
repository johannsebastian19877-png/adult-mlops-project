# MLOps - Adult Income Prediction

> **Proyecto práctico de MLOps con el dataset Adult (UCI ML Repository)**  
> **Universidad Santo Tomás — Facultad de Estadística**

[![MLOps Level](https://img.shields.io/badge/MLOps-Level%201-blue)](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## ⚡ Quick Start (10 minutos)

### Opción A: Local con pip

```bash
# 1. Clonar y entrar al proyecto
git clone <repo-url>
cd adult-mlops-project

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -e .

# 4. Inicializar DVC
dvc init

# 5. Ejecutar pipeline completo
python -m src.pipeline
```

### Opción B: Docker (recomendado para producción)

```bash
# 1. Clonar y entrar al proyecto
git clone <repo-url>
cd adult-mlops-project

# 2. Ejecutar con Docker Compose
docker-compose up --build
```

**Resultado esperado**: Al finalizar, tendrás en `artifacts/` y `models/`:
- `models/model.pkl` - Modelo entrenado
- `artifacts/metrics.json` - Métricas de entrenamiento
- `artifacts/report.html` - Reporte visual
- `artifacts/confusion.png` - Matriz de confusión

---

## 📋 Tabla de Contenidos

1. [Descripción del Proyecto](#-descripción-del-proyecto)
2. [Arquitectura del Pipeline](#-arquitectura-del-pipeline)
3. [Dataset](#-dataset)
4. [Estructura del Proyecto](#-estructura-del-proyecto)
5. [Instalación Detallada](#-instalación-detallada)
6. [Ejecución del Pipeline](#-ejecución-del-pipeline)
7. [Docker y Orquestación](#-docker-y-orquestación)
8. [Artefactos y Trazabilidad](#-artefactos-y-trazabilidad)
9. [MLflow - Experiment Tracking](#-mlflow---experiment-tracking)
10. [Validación con Pandera](#-validación-con-pandera)
11. [DVC - Versionamiento de Datos](#-dvc---versionamiento-de-datos)
12. [Tests](#-tests)
13. [Niveles de Madurez MLOps](#-niveles-de-madurez-mlops)
14. [Stack Tecnológico](#-stack-tecnológico)

---

## 📖 Descripción del Proyecto

### Problema de Negocio

Predecir si el **ingreso anual de una persona supera los $50,000 USD** utilizando variables demográficas y laborales del censo de Estados Unidos (1994).

### Objetivo Técnico

Implementar un **pipeline de MLOps de Nivel 1** que garantice:

| Principio | Implementación |
|-----------|----------------|
| **Reproducibilidad** | Mismos resultados con mismos datos y parámetros (seeds fijas, versiones pinneadas) |
| **Trazabilidad** | Versionamiento conjunto de código (Git), datos y modelos (DVC) |
| **Automatización** | Pipeline orquestado con DVC, sin intervención manual |
| **Monitoreo** | Tracking de métricas con MLflow para detectar drift |

### ¿Por qué MLOps?

En producción, el mayor desafío no es entrenar un modelo, sino **mantenerlo confiable** a lo largo del tiempo. Este proyecto aplica principios de DevOps al contexto de Machine Learning para:

- Evitar el *"funciona en mi máquina"*
- Permitir rollback a versiones anteriores
- Detectar degradación del modelo (data drift / model drift)
- Facilitar la colaboración en equipo

---

## 🏗️ Arquitectura del Pipeline

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE MLOps                                      │
├──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  INGEST  │───▶│ VALIDATE │───▶│ FEATURES │───▶│  TRAIN   │───▶│ EVALUATE │ │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
     │               │               │               │               │        │
     ▼               ▼               ▼               ▼               ▼        │
  features.      validation      preprocessor    model.pkl      report.html   │
  parquet        _report.json    .joblib                        confusion.png │
  targets.                                                                    │
  parquet                                                                    │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                          ┌─────────────────┐
                          │    MLflow       │
                          │  (Tracking)     │
                          └─────────────────┘
```

### Flujo de Datos

1. **Ingesta**: Descarga desde UCI ML Repository → `data/raw/`
2. **Validación**: Verifica schema con Pandera → `data/processed/validation_report.json`
3. **Features**: Preprocesamiento (Scaler + Encoder) → `artifacts/preprocessor.joblib`
4. **Train**: GradientBoosting con CV 5-folds → `models/model.pkl` + MLflow
5. **Evaluate**: Métricas en test set → `artifacts/report.html`, `confusion.png`

---

## 📊 Dataset

### Adult Income (UCI ML Repository - ID: 2)

| Característica | Valor |
|----------------|-------|
| **Registros** | 48,842 |
| **Features** | 14 |
| **Clases** | 2 (>50K, ≤50K) |
| **Numéricas** | 6: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week |
| **Categóricas** | 8: workclass, education, marital-status, occupation, relationship, race, sex, native-country |

### Código de Obtención

```python
from ucimlrepo import fetch_ucirepo

adult = fetch_ucirepo(id=2)
X = adult.data.features
y = adult.data.targets
```

**Justificación**: Usar una fuente programática (`ucimlrepo`) en lugar de archivos estáticos garantiza que cualquier persona pueda reproducir el experimento sin descargar manualmente.

---

## 📁 Estructura del Proyecto

```
adult-mlops-project/
│
├── data/                          # ← DATOS (versionados con DVC)
│   ├── raw/                       # Datos crudos (inmutables)
│   │   ├── features.parquet       # Features en formato Parquet
│   │   ├── targets.parquet        # Targets en formato Parquet
│   │   ├── features.csv           # Fallback portable
│   │   └── targets.csv            # Fallback portable
│   ├── processed/                 # Datos procesados
│   │   └── validation_report.json # Reporte de validación
│   └── interim/                   # Datos intermedios (temporales)
│
├── notebooks/                     # ← EXPLORACIÓN (no productivo)
│   ├── 01_eda.ipynb               # Análisis exploratorio
│   └── 02_modeling.ipynb          # Prototipado de modelos
│
├── src/                           # ← CÓDIGO PRODUCTIVO (modular)
│   ├── __init__.py                # Marca como paquete Python
│   ├── ingest.py                  # Ingesta desde UCI
│   ├── validate.py                # Validación con Pandera
│   ├── features.py                # Feature engineering
│   ├── train.py                   # Entrenamiento + MLflow
│   ├── evaluate.py                # Evaluación + reportes
│   └── pipeline.py                # Orquestación completa
│
├── models/                        # ← MODELOS SERIALIZADOS
│   └── model.pkl                  # Modelo entrenado (joblib)
│
├── artifacts/                     # ← ARTEFACTOS DEL PIPELINE
│   ├── preprocessor.joblib        # Preprocesador completo
│   ├── num.joblib                 # StandardScaler para numéricas
│   ├── cat.joblib                 # OrdinalEncoder para categóricas
│   ├── preprocessor_meta.joblib   # Metadatos de columnas
│   ├── metrics.json               # Métricas de entrenamiento
│   ├── evaluation_metrics.json    # Métricas de evaluación
│   ├── report.html                # Reporte visual HTML
│   └── confusion.png              # Matriz de confusión
│
├── tests/                         # ← TESTS AUTOMÁTICOS
│   ├── __init__.py
│   └── test_pipeline.py           # Tests unitarios
│
├── mlruns/                        # ← MLFLOW TRACKING
│
├── .dvc/                          # ← CONFIGURACIÓN DVC
│   └── config
│
├── scripts/                       # ← SCRIPTS DE UTILIDAD (no productivos)
│   ├── check_overfitting.py       # Verificar overfitting del modelo
│   └── verify.py                  # Verificación completa del proyecto
├── .gitignore                     # Exclusiones para Git
├── Dockerfile                     # Entorno reproducible
├── dvc.yaml                       # Definición del pipeline DVC
├── pyproject.toml                 # Dependencias (versiones fijas)
└── README.md                      # Este archivo
```

### Justificación de la Separación

| Directorio | ¿Por qué separado? |
|------------|-------------------|
| `data/raw/` | **Inmutabilidad**: Los datos crudos nunca se modifican. Si hay error, se vuelve a ingerir desde la fuente. |
| `data/processed/` | **Separación de responsabilidades**: Diferenciar entrada vs. salida del pipeline. |
| `notebooks/` | **Exploración vs. Producción**: Los notebooks son iterativos y no garantizan reproducibilidad. El código productivo va en `src/`. |
| `src/` | **Modularidad**: Cada módulo tiene una responsabilidad única, puede testearse y mantenerse por separado. |
| `models/` | **Separación de artefactos**: Los modelos son distintos a transformadores y métricas. |
| `artifacts/` | **Trazabilidad**: Todos los productos intermedios del pipeline están versionados. |

---

## 🚀 Instalación Detallada

### Requisitos Previos

| Herramienta | Versión | Instalación |
|-------------|---------|-------------|
| Python | 3.9+ | [python.org](https://python.org) |
| Git | 2.0+ | [git-scm.com](https://git-scm.com) |
| pip | 21.0+ | Incluido con Python |
| DVC | 3.0+ | `pip install dvc` |
| Docker (opcional) | 20.0+ | [docker.com](https://docker.com) |

### Paso 1: Clonar el Repositorio

```bash
git clone <repo-url>
cd adult-mlops-project
```

### Paso 2: Crear Entorno Virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

**Justificación**: Los entornos virtuales aíslan las dependencias del proyecto, evitando conflictos con otras instalaciones de Python.

### Paso 3: Instalar Dependencias

```bash
pip install -e .
```

Las dependencias están **fijadas por versión exacta** en `pyproject.toml` para garantizar reproducibilidad:

| Dependencia | Versión | Propósito |
|-------------|---------|-----------|
| `pandas` | 2.1.4 | Manipulación de DataFrames |
| `numpy` | 1.26.3 | Cálculo numérico |
| `scikit-learn` | 1.3.2 | Machine Learning |
| `joblib` | 1.3.2 | Serialización eficiente |
| `pyarrow` | 14.0.2 | Formato Parquet |
| `pandera` | 0.17.2 | Validación de schemas |
| `mlflow` | 2.9.2 | Tracking de experimentos |
| `ucimlrepo` | 0.0.7 | Acceso al dataset UCI |
| `matplotlib` | 3.8.2 | Visualización |
| `seaborn` | 0.13.0 | Visualización estadística |

### Paso 4: Inicializar DVC

```bash
dvc init
```

**Justificación**: DVC configura el cache para versionar datos y modelos grandes fuera de Git.

---

## ⚙️ Ejecución del Pipeline

### Pipeline Completo

```bash
# Opción 1: Módulo Python (desarrollo)
python -m src.pipeline

# Opción 2: DVC (producción - con caching)
dvc repro
```

**Diferencia**:
- `python -m src.pipeline`: Ejecuta todo en secuencia, útil para depuración
- `dvc repro`: Solo re-ejecuta etapas con cambios, ideal para producción

### Etapas Individuales

```bash
# 1. Ingesta
python -m src.ingest

# 2. Validación
python -m src.validate

# 3. Feature Engineering
python -m src.features

# 4. Entrenamiento
python -m src.train

# 5. Evaluación
python -m src.evaluate
```

**Justificación**: Ejecutar etapas individualmente permite:
- Depurar problemas específicos
- Re-ejecutar solo una parte del pipeline
- Desarrollar y testear cada componente por separado

### Verificación

```bash
# Verificar overfitting
python scripts/check_overfitting.py

# Verificación completa
python scripts/verify.py
```

---

## 🐳 Docker y Orquestación

### ¿Por qué Docker?

| Problema sin Docker | Solución con Docker |
|---------------------|---------------------|
| "Funciona en mi máquina" | Mismo SO, Python y dependencias |
| Conflictos de versiones | Entorno aislado y consistente |
| Configuración manual | Todo automatizado en el Dockerfile |

### Construir Imagen

```bash
docker build -t adult-mlops:latest .
```

### Ejecutar con Docker Compose

```bash
# Iniciar todos los servicios
docker-compose up --build

# Solo pipeline
docker-compose up pipeline

# Pipeline + MLflow UI
docker-compose --profile main --profile tracking up

# Entorno de desarrollo (con Jupyter)
docker-compose --profile dev up

# Ejecutar una vez y salir
docker-compose run pipeline

# Detener servicios
docker-compose down
```

### Servicios Disponibles

| Servicio | Puerto | Propósito |
|----------|--------|-----------|
| `pipeline` | - | Ejecuta el pipeline completo |
| `mlflow` | 5000 | UI de tracking de experimentos |
| `jupyter` | 8888 | Notebook server para exploración |
| `dvc-repro` | - | Ejecución con DVC para CI/CD |

### Acceder a MLflow UI

Este proyecto implementa **Nivel 1 — Pipeline Automatizado**:
- Codigo modular
- Pipeline reproducible con DVC
- Tracking con MLflow

## Resultados e Interpretaciones

### Metricas del Modelo

El modelo GradientBoostingClassifier fue evaluado utilizando las siguientes metricas:

| Metrica | Descripcion | Valor Tipico |
|---------|-------------|--------------|
| Accuracy | Proporcion de predicciones correctas | ~0.85 |
| F1 Macro | Promedio no ponderado de F1 por clase | ~0.72 |
| F1 Binary | F1-score para la clase positiva (>50K) | ~0.74 |
| AUC-ROC | Area bajo la curva ROC | ~0.89 |

**Interpretacion del F1-test**: La metrica f1_test ahora calcula correctamente el F1-score macro en el conjunto de test, no accuracy. Esto es crucial para datasets desbalanceados como Adult, donde accuracy puede ser enganosamente alto.

### Analisis de Equidad (Fairness)

El modelo fue evaluado por subgrupos demograficos para detectar posibles sesgos:

**Por Sexo:**
- Diferencia maxima en accuracy: ~1.5%
- Diferencia maxima en F1: ~2.2%
- El modelo muestra rendimiento ligeramente superior para hombres en terminos de tasa de positivos predichos

**Por Raza:**
- Diferencia maxima en accuracy: ~4.4%
- Diferencia maxima en F1: ~5.3%
- Se observa mayor variabilidad entre grupos raciales

**Por Edad:**
- Diferencia maxima en accuracy: ~7.9%
- Diferencia maxima en F1: ~13.3%
- El grupo de edad <25 presenta el rendimiento mas bajo, posiblemente debido a menor representatividad en los datos

**Conclusiones sobre equidad:**
- Las disparidades observadas estan dentro de rangos aceptables para un modelo de primer nivel
- Se recomienda monitoreo continuo de estas metricas en produccion
- El grupo de edad <25 podria requerir tecnicas de rebalanceo o recoleccion de datos adicionales

### Matriz de Confusion

La matriz de confusion muestra el comportamiento del modelo:

```
                Prediccion
                <=50K    >50K
Real  <=50K     TN       FP
      >50K      FN       TP
```

Donde:
- TN (True Negative): Correctamente predichos como <=50K
- TP (True Positive): Correctamente predichos como >50K
- FP (False Positive): Incorrectamente predichos como >50K
- FN (False Negative): Incorrectamente predichos como <=50K

### Tratamiento de Datos Faltantes

El dataset Adult original utiliza el simbolo "?" para representar valores faltantes. El pipeline implementa:

1. **Estandarizacion**: Conversion de "?" a NaN para tratamiento consistente
2. **Limpieza**: Eliminacion de espacios y puntos extra en columnas categoricas
3. **Codificacion**: Los valores faltantes en categoricas son codificados como -1 por OrdinalEncoder

### Validacion de Schema

El schema de validacion con Pandera verifica:
- Rango de edad: 17-90 anos
- education-num: 1-16 anos
- hours-per-week: 1-99 horas
- sex: solo valores "Male" o "Female"
- capital-gain y capital-loss: valores no negativos

## Lecciones Aprendidas

1. **Importancia de metricas adecuadas**: El uso de F1-score en lugar de accuracy es esencial para datasets desbalanceados
2. **Evaluacion por subgrupos**: Permite identificar sesgos que las metricas globales ocultan
3. **Versionamiento de datos**: DVC facilita la reproducibilidad del pipeline completo
4. **Validacion temprana**: El schema validation detecta problemas antes del entrenamiento

## Recomendaciones para Mejoras Futuras

### Mejoras a Corto Plazo

1. **Nivel 2 MLOps**: Implementar CI/CD para despliegue automatico
2. **Monitoreo**: Agregar deteccion de data drift y concept drift
3. **Experimentacion**: Comparar multiples algoritmos (Random Forest, XGBoost, etc.)
4. **Feature engineering**: Crear features derivados (ej: capital-net = gain - loss)
5. **Optimizacion de hiperparametros**: Usar GridSearchCV o Optuna

---

## Estado Actual del Proyecto

### Que tiene ya el proyecto

#### 1. Estructura completa y modular

El proyecto esta organizado con carpetas y archivos para cada fase del pipeline:

- `data/raw`, `data/processed`, `data/interim`
- `notebooks/` exploratorios
- `src/` con modulos (*ingest, validate, features, train, evaluate*)
- Carpeta de `artifacts` y `models`
- `tests/` para pruebas automaticas
- Pipeline declarativo con `DVC`
- `Dockerfile` reproducible
- Gestion de dependencias con Poetry (`pyproject.toml`)

Esto ya cumple una base solida de ingenieria reproducible.

#### 2. Pipeline automatizado con DVC

El archivo `dvc.yaml` define las etapas:

- Ingesta
- Validacion
- Feature engineering
- Entrenamiento
- Evaluacion

Todo puede ejecutarse con `dvc repro`. Ademas DVC versiona los resultados grandes como datos, artefactos y modelos.

#### 3. Tracking de experimentos con MLflow

Se registra:

- Modelos
- Hiperparametros
- Metricas
- Artefactos

Esto da trazabilidad de los experimentos durante desarrollo.

#### 4. Testing automatico

Existe una carpeta `tests/` con pruebas para:

- Validacion de datos
- Checks de sanity
- Tests unitarios basicos

Esto es un buen avance hacia calidad de codigo reproducible.

#### 5. Artefactos versionados

Por cada etapa hay artefactos almacenados que ayudan a:

- Reproducir exacto estado de pipeline
- Analizar resultados de validacion
- Visualizar metricas y reportes

Eso es fundamental en MLOps moderno.

---

### Que le falta — Fases y piezas completas

#### 1. No hay servicio de inferencia (API)

El repositorio NO incluye:

- Endpoint REST (FastAPI, Flask, etc.)
- Servicio de inferencia
- Contenedor listo para servir predicciones

Sin esto, solo se puede entrenar y evaluar localmente, no servir el modelo en produccion.

#### 2. No hay CI/CD configurado

No hay:

- `.github/workflows/ci.yml`
- Tests automaticos al hacer PR
- Build Docker pipeline
- Deploy automatico

Esto es clave para promover modelos desde dev → staging → prod.

#### 3. No hay monitoreo en produccion

El pipeline de entrenamiento si genera metricas offline, pero no hay:

- Deteccion de drift
- Performance tracking post-deploy
- Alertas en produccion
- Dashboards
- Integraciones tipo Prometheus, Grafana, Evidently

Esto significa que no hay observabilidad real una vez el modelo este desplegado.

#### 4. No hay manejo de estados de modelo en produccion

Tampoco hay:

- Model registry con estados (Staging → Prod → Archived)
- Politicas de rollback
- Validacion automatica de gate antes de promover versiones

Esto limita gobernanza del modelo.

#### 5. Infraestructura para escalabilidad

Actualmente el Dockerfile existe, pero no hay:

- Docker Compose
- Kubernetes manifests
- Configuracion de nube (AWS, GCP, Azure)

Esto limita ejecucion a local o pruebas.

#### 6. Trigger / Schedule automatico

No hay orquestador como:

- Airflow
- Prefect
- Dagster

Que permita scheduling automatico o retrigger por eventos.

---

### Resumen de Estado

| Categoria                         | Estado       |
|-----------------------------------|--------------|
| Pipeline reproducible (DVC)       | Completado   |
| Tracking de experimentos (MLflow) | Completado   |
| Artefactos versionados            | Completado   |
| Tests automaticos                 | Basico       |
| API de inferencia                 | Pendiente    |
| CI/CD                             | Pendiente    |
| Monitoreo en produccion           | Pendiente    |
| Orquestacion de pipeline          | Pendiente    |
| Modelo registry formal            | Pendiente    |
| Cluster / Escalabilidad           | Pendiente    |

---

### Nivel de Madurez Actual

El proyecto esta en un **nivel avanzado de ML reproducible**, ideal como proyecto academico o portfolio.

Sin embargo, **no es aun un pipeline MLOps "completo"** porque no integra despliegue, monitoreo, CI/CD ni orquestacion automatizada.

En terminos de madurez:

- **Muy bueno para nivel educativo (Nivel 1)**
- **Faltan elementos para ser despliegue profesional (Nivel 2~3)**

---

### Siguientes Pasos Recomendados

Si quieres llevarlo a verdadero MLOps productivo, dos prioridades claras:

#### 1. Anadir API de inferencia

- FastAPI con `/predict`
- Docker + Compose

#### 2. Agregar CI/CD

- GitHub Actions
- Tests en cada PR
- Publish automatico de Docker image

#### 3. Orquestacion

- Airflow / Prefect pipeline sched y triggers

#### 4. Monitoreo

- Dashboards
- Alertas drift/performance

### Acceder a Jupyter

```
http://localhost:8888
Token: adult-mlops
```

---

## 📦 Artefactos y Trazabilidad

### Tipos de Artefactos

| Etapa | Artefacto | Formato | Propósito |
|-------|-----------|---------|-----------|
| **Ingesta** | `features.parquet` | Parquet | Features versionados |
| | `targets.parquet` | Parquet | Targets versionados |
| **Validación** | `validation_report.json` | JSON | Reporte de validación |
| **Features** | `preprocessor.joblib` | joblib | Preprocesador completo |
| | `num.joblib` | joblib | Scaler para numéricas |
| | `cat.joblib` | joblib | Encoder para categóricas |
| **Train** | `model.pkl` | pickle | Modelo entrenado |
| | `metrics.json` | JSON | Métricas de entrenamiento |
| **Evaluate** | `report.html` | HTML | Reporte visual |
| | `confusion.png` | PNG | Matriz de confusión |
| | `evaluation_metrics.json` | JSON | Métricas de evaluación |

### ¿Por qué Parquet?

- **Compresión eficiente**: Archivos más pequeños que CSV
- **Lectura rápida**: Lee solo las columnas necesarias
- **Tipado fuerte**: Mantiene tipos de datos

### ¿Por qué joblib?

- **Más eficiente** con arrays numpy que pickle
- **Compresión integrada**
- **Recomendado oficialmente** por scikit-learn

---

## 📈 MLflow - Experiment Tracking

### ¿Qué se Registra?

```python
import mlflow

with mlflow.start_run():
    mlflow.log_params(params)           # Hiperparámetros
    mlflow.log_metric('f1_cv', score)   # Métricas
    mlflow.sklearn.log_model(clf, 'model')  # Modelo
```

### Ejecutar MLflow Server

```bash
# Localmente
mlflow server --host 0.0.0.0 --port 5000

# Con Docker Compose
docker-compose up mlflow
```

### Acceder a la UI

```
http://localhost:5000
```

### Beneficios

| Beneficio | Explicación |
|-----------|-------------|
| **Reproducibilidad** | Cada run registra código, parámetros y datos exactos |
| **Comparación** | Visualizar múltiples experimentos lado a lado |
| **Auditoría** | Trazabilidad completa de cómo se obtuvo cada modelo |

---

## ✅ Validación con Pandera

### Schema Implementado

```python
import pandera.pandas as pa

schema = pa.DataFrameSchema({
    'age': pa.Column(int, checks=pa.Check.in_range(17, 90)),
    'workclass': pa.Column(str, nullable=True),
    'fnlwgt': pa.Column(int, checks=pa.Check.in_range(1, 1500000)),
    'education-num': pa.Column(int, checks=pa.Check.in_range(1, 16)),
    'sex': pa.Column(str, checks=pa.Check.isin(['Male', 'Female'])),
    # ... más columnas
})
```

### Validaciones

| Tipo | Descripción |
|------|-------------|
| **Tipos de datos** | Verifica que cada columna tenga el tipo correcto |
| **Rangos** | Valida que valores numéricos estén dentro de rangos |
| **Categorías** | Verifica que valores pertenezcan a un conjunto válido |
| **Nulos** | Controla qué columnas pueden tener valores nulos |

### Justificación

Validar datos **antes** de entrenar previene:
- **Errores silenciosos**: Datos corruptos que producen modelos incorrectos
- **Data drift**: Cambios en la distribución de datos de entrada

---

## 🔄 DVC - Versionamiento de Datos

### Git vs DVC

| Característica | Git | DVC |
|----------------|-----|-----|
| **Archivos pequeños** | ✅ Sí | ❌ No diseñado |
| **Archivos grandes** | ❌ No eficiente | ✅ Sí (usa hashes) |
| **Código** | ✅ Sí | ❌ No |
| **Datos/Modelos** | ❌ No | ✅ Sí |

### Pipeline DVC (dvc.yaml)

```yaml
stages:
  ingest:
    cmd: python -m src.ingest
    outs:
      - data/raw/features.parquet
      
  validate:
    cmd: python -m src.validate
    deps:
      - data/raw/features.parquet
    metrics:
      - data/processed/validation_report.json:
          cache: false
      
  train:
    cmd: python -m src.train
    deps:
      - artifacts/preprocessor.joblib
      - data/raw/targets.parquet
    outs:
      - models/model.pkl
```

### Comandos Útiles

```bash
# Inicializar
dvc init

# Ejecutar pipeline
dvc repro

# Ver estado
dvc status

# Mostrar DAG
dvc dag
```

---

## 🧪 Tests

### Ejecutar Tests

```bash
pytest tests/ -v
```

### Tests Incluidos

| Test | Propósito |
|------|-----------|
| `test_ingest_returns_dict` | Verifica estructura de ingesta |
| `test_validate_returns_dict` | Verifica estructura de validación |
| `test_build_preprocessor_returns_transformer` | Verifica tipo de preprocesador |
| `test_age_range` | Verifica validación de rango de edad |
| `test_age_out_of_range` | Verifica que datos inválidos fallen |

### Justificación

Los tests en MLOps previenen:
- **Regresiones**: Cambios que rompen funcionalidad existente
- **Errores de datos**: Aseguran que los datos cumplen el schema

---

## 📊 Niveles de Madurez MLOps

| Nivel | Características | Este Proyecto |
|-------|-----------------|---------------|
| **Nivel 0** — Manual | Todo en notebooks, sin versionamiento | ❌ |
| **Nivel 1** — Pipeline Automatizado | Código modular, DVC, MLflow | ✅ **Implementado** |
| **Nivel 2** — CI/CD para ML | Reentrenamiento automático, deploy automatizado | ⏳ Pendiente |

---

## 🛠️ Stack Tecnológico

| Categoría | Herramientas |
|-----------|--------------|
| **Datos** | ucimlrepo, pandas, pandera, pyarrow |
| **Features** | scikit-learn, ColumnTransformer, StandardScaler, OrdinalEncoder, joblib |
| **Entrenamiento** | GradientBoostingClassifier, cross_val_score, MLflow |
| **Versionamiento** | Git, DVC |
| **Reproducibilidad** | Docker, docker-compose |
| **Testing** | pytest, pandera |

---

## 📚 Referencias

- [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://www.mlflow.org/docs)
- [Google MLOps Guide](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

---

## 📄 Licencia

MIT License - Universidad Santo Tomás, Facultad de Estadística

---

> **Nota**: MLOps no es solo un conjunto de herramientas, es una **disciplina de ingeniería** para hacer Machine Learning sostenible en producción.
