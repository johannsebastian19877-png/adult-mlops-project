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

1. **Nivel 2 MLOps**: Implementar CI/CD para despliegue automatico
2. **Monitoreo**: Agregar deteccion de data drift y concept drift
3. **Experimentacion**: Comparar multiples algoritmos (Random Forest, XGBoost, etc.)
4. **Feature engineering**: Crear features derivados (ej: capital-net = gain - loss)
5. **Optimizacion de hiperparametros**: Usar GridSearchCV o Optuna

## Licencia

MIT
