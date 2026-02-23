"""
Entrenamiento de modelo con MLflow experiment tracking.

Según las diapositivas:
- Usa GradientBoostingClassifier
- Loguea parámetros, métricas y modelo con MLflow
- Usa cross-validation con 5 folds
- Métrica principal: F1 macro
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Hiperparámetros por defecto
DEFAULT_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 5,
    'random_state': 42,
}


def _resolve_paths():
    """Resolver rutas relativas al root del repositorio."""
    repo_root = Path(__file__).resolve().parents[1]
    return {
        'repo_root': repo_root,
        'data_processed': repo_root / 'data' / 'processed',
        'artifacts': repo_root / 'artifacts',
        'models': repo_root / 'models',
    }


def load_data(paths: Dict[str, Path]):
    """Cargar datos procesados (features y targets)."""
    features_file = paths['data_processed'] / 'features_processed.parquet'
    targets_file = paths['data_processed'] / 'targets_processed.parquet'
    
    # Si no existen procesados, intentar con raw
    if not features_file.exists():
        raw_dir = paths['repo_root'] / 'data' / 'raw'
        if (raw_dir / 'features.parquet').exists():
            features_file = raw_dir / 'features.parquet'
        elif (raw_dir / 'features.csv').exists():
            features_file = raw_dir / 'features.csv'
    
    if not targets_file.exists():
        raw_dir = paths['repo_root'] / 'data' / 'raw'
        if (raw_dir / 'targets.parquet').exists():
            targets_file = raw_dir / 'targets.parquet'
        elif (raw_dir / 'targets.csv').exists():
            targets_file = raw_dir / 'targets.csv'
    
    # Leer features
    if str(features_file).endswith('.parquet'):
        X = pd.read_parquet(features_file, engine='pyarrow')
    else:
        X = pd.read_csv(features_file)
    
    # Leer targets
    if str(targets_file).endswith('.parquet'):
        y = pd.read_parquet(targets_file, engine='pyarrow')
    else:
        y = pd.read_csv(targets_file)
    
    return X, y


def train(
    X: pd.DataFrame,
    y: pd.Series,
    params: Optional[Dict[str, Any]] = None,
    experiment_name: str = 'adult-income',
    paths: Optional[Dict[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Entrenar modelo con GradientBoostingClassifier y registrar en MLflow.
    
    Args:
        X: DataFrame con features
        y: Series con targets
        params: Hiperparámetros del modelo
        experiment_name: Nombre del experimento en MLflow
        paths: Diccionario con rutas
    
    Returns:
        Dict con métricas y ruta del modelo guardado
    """
    if params is None:
        params = DEFAULT_PARAMS
    
    if paths is None:
        paths = _resolve_paths()
    
    # Configurar MLflow con tracking URI relativo (evita problemas con espacios en rutas)
    mlflow_dir = paths['repo_root'] / 'mlruns'
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{mlflow_dir.as_posix()}")
    mlflow.set_experiment(experiment_name)
    
    # Dividir datos para validación
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=params.get('random_state', 42), stratify=y
    )
    
    with mlflow.start_run():
        # Crear modelo
        clf = GradientBoostingClassifier(**params)
        
        # Cross-validation (5 folds, F1 macro)
        scores = cross_val_score(
            clf, X_train, y_train,
            cv=5, scoring='f1_macro', n_jobs=-1
        )
        
        # Entrenar modelo final
        clf.fit(X_train, y_train)
        
        # Métricas en test set
        f1_test = clf.score(X_test, y_test)
        
        # Loguear parámetros y métricas
        mlflow.log_params(params)
        mlflow.log_metric('f1_cv', scores.mean())
        mlflow.log_metric('f1_cv_std', scores.std())
        mlflow.log_metric('f1_test', f1_test)
        
        # Loguear modelo
        mlflow.sklearn.log_model(clf, 'model')
        
        # Guardar modelo localmente
        paths['models'].mkdir(parents=True, exist_ok=True)
        model_path = paths['models'] / 'model.pkl'
        joblib.dump(clf, model_path)
        
        # Guardar métricas en JSON
        metrics = {
            'f1_cv': float(scores.mean()),
            'f1_cv_std': float(scores.std()),
            'f1_test': float(f1_test),
            'params': params,
        }
        metrics_path = paths['artifacts'] / 'metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        log.info(f"Modelo guardado en {model_path}")
        log.info(f"Métricas guardadas en {metrics_path}")
        log.info(f"F1 CV: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return {
            'model_path': str(model_path),
            'metrics_path': str(metrics_path),
            'metrics': metrics,
            'run_id': mlflow.active_run().info.run_id,
        }


if __name__ == '__main__':
    import json as json_module
    from src.features import clean_features, clean_target
    
    paths = _resolve_paths()
    
    # Cargar preprocesador
    preprocessor_path = paths['artifacts'] / 'preprocessor.joblib'
    if preprocessor_path.exists():
        preprocessor = joblib.load(preprocessor_path)
        
        # Cargar datos raw
        raw_dir = paths['repo_root'] / 'data' / 'raw'
        if (raw_dir / 'features.parquet').exists():
            X_raw = pd.read_parquet(raw_dir / 'features.parquet', engine='pyarrow')
        else:
            X_raw = pd.read_csv(raw_dir / 'features.csv')
        
        # Limpiar features
        X_raw = clean_features(X_raw)
        
        # Preprocesar
        X = preprocessor.transform(X_raw)
        
        # Cargar targets
        if (raw_dir / 'targets.parquet').exists():
            y = pd.read_parquet(raw_dir / 'targets.parquet', engine='pyarrow')
        else:
            y = pd.read_csv(raw_dir / 'targets.csv')
        
        # Aplanar y limpiar target
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        y = clean_target(y)
        
        # Entrenar
        result = train(X, y, paths=paths)
        print(json_module.dumps(result, indent=2, ensure_ascii=False))
    else:
        log.error("Preprocesador no encontrado. Ejecuta features.py primero.")
