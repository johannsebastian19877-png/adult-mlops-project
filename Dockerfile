FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar configuración del proyecto
COPY pyproject.toml .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -e .[dev]

# Copiar código fuente
COPY src/ ./src/

# Copiar configuración de DVC
COPY dvc.yaml .dvcignore .gitignore ./

# Crear directorios necesarios
RUN mkdir -p data/raw data/processed data/interim artifacts models notebooks tests

# Configurar MLflow tracking URI (puede sobrescribirse con variable de entorno)
ENV MLFLOW_TRACKING_URI=file:///app/mlruns
ENV MLFLOW_EXPERIMENT_NAME=adult-income

# Comando por defecto: ejecutar pipeline completo
ENTRYPOINT ["python", "-m", "src.pipeline"]
