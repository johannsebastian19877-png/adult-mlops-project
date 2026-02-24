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

## Licencia

MIT
