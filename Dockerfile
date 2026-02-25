# =============================================================================
# Dockerfile - MLOps Pipeline para Adult Income Prediction
# =============================================================================
# Propósito: Crear un entorno reproducible para ejecutar el pipeline completo
# 
# Construcción:
#   docker build -t adult-mlops:latest .
#
# Ejecución:
#   docker run --rm adult-mlops:latest
#
# Con Docker Compose (recomendado):
#   docker-compose up pipeline
# =============================================================================

# Etapa 1: Imagen base con Python 3.11
FROM python:3.11-slim as base

# Variables de entorno para Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# =============================================================================
# Etapa 2: Instalación de dependencias del sistema
# =============================================================================
FROM base as system-deps

# Instalar dependencias necesarias para compilar paquetes
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libopenblas-dev \
    liblapack-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Etapa 3: Instalación de dependencias de Python
# =============================================================================
FROM system-deps as python-deps

# Copiar solo el archivo de dependencias primero (cache de capas Docker)
COPY pyproject.toml .

# Instalar dependencias en una ubicación específica para multi-stage
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .[dev]

# =============================================================================
# Etapa 4: Imagen final con el código fuente
# =============================================================================
FROM python-deps as final

# Copiar código fuente
COPY src/ ./src/

# Copiar archivos de configuración del pipeline
COPY dvc.yaml ./
COPY .dvcignore ./
COPY .gitignore ./

# Copiar scripts de verificación
COPY check_overfitting.py ./
COPY verify.py ./

# Crear directorios necesarios para el pipeline
RUN mkdir -p data/raw data/processed data/interim artifacts models mlruns notebooks tests

# Configurar variables de entorno para MLflow
ENV MLFLOW_TRACKING_URI=file:///app/mlruns \
    MLFLOW_EXPERIMENT_NAME=adult-income

# Health check para verificar que el contenedor está funcionando
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Metadata del contenedor
LABEL maintainer="Universidad Santo Tomás - Facultad de Estadística" \
      description="MLOps pipeline for Adult Income prediction" \
      version="1.0.0"

# Comando por defecto: ejecutar pipeline completo
# Se puede sobrescribir con: docker run adult-mlops python -m src.train
ENTRYPOINT ["python", "-m", "src.pipeline"]
