# MLOps - Adult Income Prediction

Proyecto práctico con el dataset Adult (UCI ML Repository)  
Universidad Santo Tomás — Facultad de Estadística

## Descripción

Predecir si el ingreso anual de una persona supera los $50K USD, usando variables demográficas y laborales del censo de EE.UU. (1994).

### Dataset

- **48,842 registros**
- **14 features**
- **2 clases** (>50K, ≤50K)
- **6 numéricas**, **8 categóricas**

## Estructura del Proyecto

```
adult-mlops-project/
├── data/
│   ├── raw/           # Datos crudos (versionados con DVC)
│   ├── processed/     # Datos procesados
│   └── interim/       # Datos intermedios
├── notebooks/         # Exploración y prototipado
├── src/               # Módulos Python reutilizables
│   ├── __init__.py
│   ├── ingest.py      # Ingesta de datos
│   ├── validate.py    # Validación de datos
│   ├── features.py    # Feature engineering
│   ├── train.py       # Entrenamiento
│   └── evaluate.py    # Evaluación
├── models/            # Modelos serializados
├── artifacts/         # Artefactos (transformers, métricas, reportes)
├── tests/             # Tests automáticos
├── pyproject.toml     # Configuración del proyecto
├── Dockerfile         # Contenedor reproducible
├── dvc.yaml           # Pipeline DVC
└── README.md
```

## Instalación

```bash
# Clonar repositorio
git clone <repo-url>
cd adult-mlops-project

# Instalar dependencias
pip install -e .

# Configurar DVC
dvc init
dvc pull
```

## Uso

### Ejecutar pipeline completo

```bash
dvc repro
```

### Ejecutar etapas individuales

```bash
# Ingesta
python -m src.ingest

# Validación
python -m src.validate

# Feature engineering
python -m src.features

# Entrenamiento
python -m src.train

# Evaluación
python -m src.evaluate
```

### Construir contenedor Docker

```bash
docker build -t adult-mlops .
docker run adult-mlops
```

## Artefactos

Cada etapa del pipeline genera artefactos versionados:

| Etapa | Artefactos |
|-------|-----------|
| Ingesta | `features.parquet`, `targets.parquet` |
| Validación | `validation_report.json` |
| Features | `preprocessor.joblib`, `num.joblib`, `cat.joblib` |
| Train | `model.pkl`, `metrics.json` |
| Evaluate | `report.html`, `confusion.png` |

## Stack Tecnológico

**Datos**
- ucimlrepo, pandas, pandera, Parquet

**Features**
- scikit-learn, Pipeline, ColumnTransformer, joblib

**Entrenamiento**
- scikit-learn, MLflow

**Versionamiento**
- Git, DVC

**Reproducibilidad**
- Docker, Poetry

**Testing**
- pytest, pandera

## Principios MLOps

### Reproducibilidad
Cada ejecución del pipeline genera los mismos resultados con los mismos datos y parámetros.

### Versionamiento
Datos, código y modelos se versionan juntos para trazabilidad completa.

### Automatización
Pipelines orquestados reducen intervención manual y errores humanos.

### Monitoreo Continuo
Las métricas del modelo se rastrean para detectar degradación (drift).

## Niveles de Madurez MLOps

Este proyecto implementa **Nivel 1 — Pipeline Automatizado**:
- ✅ Código modular
- ✅ Pipeline reproducible con DVC
- ✅ Tracking con MLflow

## Licencia

MIT
