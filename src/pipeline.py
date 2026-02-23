"""
Pipeline module - Orquesta la ejecución completa del pipeline MLOps.

Ejecuta todas las etapas en orden:
1. Ingesta
2. Validación
3. Feature Engineering
4. Entrenamiento
5. Evaluación
"""
import logging
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent))

from ingest import ingest_adult
from validate import validate_dataframe
from features import fit_and_serialize_preprocessor, load_preprocessor, clean_features, clean_target
from train import train, _resolve_paths, load_data
from evaluate import evaluate

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def run_pipeline():
    """Ejecutar el pipeline completo de MLOps."""
    log.info("=" * 60)
    log.info("INICIANDO PIPELINE MLOPS - ADULT INCOME")
    log.info("=" * 60)
    
    paths = _resolve_paths()
    results = {}
    
    # Etapa 1: Ingesta
    log.info("\n[1/5] ETAPA DE INGESTA")
    log.info("-" * 40)
    ingest_result = ingest_adult()
    results['ingest'] = ingest_result
    log.info(f"Filas: {ingest_result['n_rows']}")
    log.info(f"Columnas: {ingest_result['n_features']}")
    log.info(f"Archivos escritos: {ingest_result['written_files']}")
    
    # Etapa 2: Validación
    log.info("\n[2/5] ETAPA DE VALIDACIÓN")
    log.info("-" * 40)
    # Cargar datos para validar
    if (paths['repo_root'] / 'data' / 'raw' / 'features.parquet').exists():
        import pandas as pd
        df = pd.read_parquet(paths['repo_root'] / 'data' / 'raw' / 'features.parquet', engine='pyarrow')
    else:
        import pandas as pd
        df = pd.read_csv(paths['repo_root'] / 'data' / 'raw' / 'features.csv')
    
    validate_result = validate_dataframe(df)
    results['validate'] = {
        'success': validate_result['success'],
        'report_path': validate_result['output_path'],
    }
    log.info(f"Validación: {'EXITOSA' if validate_result['success'] else 'FALLIDA'}")
    log.info(f"Reporte: {validate_result['output_path']}")
    
    # Etapa 3: Feature Engineering
    log.info("\n[3/5] ETAPA DE FEATURE ENGINEERING")
    log.info("-" * 40)
    features_result = fit_and_serialize_preprocessor(df)
    results['features'] = features_result
    log.info(f"Artefactos guardados: {list(features_result.keys())}")
    
    # Etapa 4: Entrenamiento
    log.info("\n[4/5] ETAPA DE ENTRENAMIENTO")
    log.info("-" * 40)
    X, y = load_data(paths)
    
    # Limpiar datos
    X = clean_features(X)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    y = clean_target(y)
    
    # Cargar preprocesador y transformar
    preprocessor = load_preprocessor()
    X_processed = preprocessor.transform(X)
    
    train_result = train(X_processed, y, paths=paths)
    results['train'] = train_result
    log.info(f"F1 CV: {train_result['metrics']['f1_cv']:.4f}")
    log.info(f"Run ID: {train_result['run_id']}")
    
    # Etapa 5: Evaluación
    log.info("\n[5/5] ETAPA DE EVALUACIÓN")
    log.info("-" * 40)
    evaluate_result = evaluate(paths=paths)
    results['evaluate'] = evaluate_result
    log.info(f"Accuracy: {evaluate_result['metrics']['accuracy']:.4f}")
    log.info(f"F1 Binary: {evaluate_result['metrics']['f1_binary']:.4f}")
    
    # Resumen final
    log.info("\n" + "=" * 60)
    log.info("PIPELINE COMPLETADO EXITOSAMENTE")
    log.info("=" * 60)
    log.info(f"\nArtefactos generados:")
    log.info(f"  - data/raw/features.parquet")
    log.info(f"  - data/raw/targets.parquet")
    log.info(f"  - data/processed/validation_report.json")
    log.info(f"  - artifacts/preprocessor.joblib")
    log.info(f"  - artifacts/metrics.json")
    log.info(f"  - models/model.pkl")
    log.info(f"  - artifacts/report.html")
    log.info(f"  - artifacts/confusion.png")
    
    return results


if __name__ == '__main__':
    run_pipeline()
