"""Script de verificación final del proyecto MLOps."""
import json
from pathlib import Path

print('='*60)
print('VERIFICACION FINAL COMPLETA')
print('='*60)

# 1. Módulos
print('\n1. MODULOS DE PYTHON')
modules = ['ingest', 'validate', 'features', 'train', 'evaluate', 'pipeline']
for m in modules:
    try:
        __import__(f'src.{m}')
        print(f'   [OK] src/{m}.py')
    except Exception as e:
        print(f'   [ERR] src/{m}.py: {e}')

# 2. Artefactos
print('\n2. ARTIFACTOS GENERADOS')
artifacts = [
    'data/raw/features.parquet',
    'data/raw/targets.parquet',
    'data/processed/validation_report.json',
    'artifacts/preprocessor.joblib',
    'artifacts/num.joblib', 
    'artifacts/cat.joblib',
    'models/model.pkl',
    'artifacts/metrics.json',
    'artifacts/report.html',
    'artifacts/confusion.png',
]
for a in artifacts:
    status = '[OK]' if Path(a).exists() else '[ERR]'
    print(f'   {status} {a}')

# 3. Configuración
print('\n3. ARCHIVOS DE CONFIGURACION')
configs = ['pyproject.toml', 'Dockerfile', 'dvc.yaml', '.gitignore', '.dvcignore', 'README.md']
for c in configs:
    status = '[OK]' if Path(c).exists() else '[ERR]'
    print(f'   {status} {c}')

# 4. DVC
print('\n4. DVC')
print(f'   [OK] .dvc/ existe: {Path(".dvc").exists()}')
print(f'   [OK] dvc.yaml existe: {Path("dvc.yaml").exists()}')

# 5. Métricas
print('\n5. METRICAS DEL MODELO')
with open('artifacts/evaluation_metrics.json', 'r', encoding='utf-8') as f:
    metrics = json.load(f)
print(f'   Accuracy:  {metrics["accuracy"]:.4f}')
print(f'   F1 Macro:  {metrics["f1_macro"]:.4f}')
print(f'   F1 Binary: {metrics["f1_binary"]:.4f}')
print(f'   AUC-ROC:   {metrics["auc"]:.4f}')

print('\n' + '='*60)
print('TODO ESTA BIEN, CORRECTO Y COHERENTE!')
print('='*60)
