"""
Evaluación de modelo.

Según las diapositivas:
- Genera reporte HTML
- Genera matriz de confusión (confusion.png)
- Calcula métricas: Accuracy, F1, AUC
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo
import matplotlib.pyplot as plt
import seaborn as sns

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _resolve_paths():
    """Resolver rutas relativas al root del repositorio."""
    repo_root = Path(__file__).resolve().parents[1]
    return {
        'repo_root': repo_root,
        'data_raw': repo_root / 'data' / 'raw',
        'data_processed': repo_root / 'data' / 'processed',
        'artifacts': repo_root / 'artifacts',
        'models': repo_root / 'models',
    }


def evaluate(
    model=None,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    preprocessor=None,
    paths: Optional[Dict[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Evaluar modelo y generar reporte.
    
    Args:
        model: Modelo entrenado (o None para cargar desde models/)
        X_test: Features de test (o None para cargar desde data/raw/)
        y_test: Targets de test (o None para cargar desde data/raw/)
        preprocessor: Preprocesador (o None para cargar desde artifacts/)
        paths: Diccionario con rutas
    
    Returns:
        Dict con métricas y paths de artefactos generados
    """
    from src.features import clean_features, clean_target
    
    if paths is None:
        paths = _resolve_paths()
    
    # Asegurar que todas las claves necesarias existan
    if 'data_raw' not in paths:
        paths['data_raw'] = paths['repo_root'] / 'data' / 'raw'
    if 'data_processed' not in paths:
        paths['data_processed'] = paths['repo_root'] / 'data' / 'processed'
    
    # Cargar modelo si no se proporciona
    if model is None:
        model_path = paths['models'] / 'model.pkl'
        if model_path.exists():
            model = joblib.load(model_path)
            log.info(f"Modelo cargado desde {model_path}")
        else:
            raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
    
    # Cargar preprocesador si no se proporciona
    if preprocessor is None:
        preprocessor_path = paths['artifacts'] / 'preprocessor.joblib'
        if preprocessor_path.exists():
            preprocessor = joblib.load(preprocessor_path)
            log.info(f"Preprocesador cargado desde {preprocessor_path}")
        else:
            raise FileNotFoundError(f"Preprocesador no encontrado en {preprocessor_path}")
    
    # Cargar datos si no se proporcionan
    if X_test is None or y_test is None:
        # Leer datos raw
        if (paths['data_raw'] / 'features.parquet').exists():
            X_raw = pd.read_parquet(paths['data_raw'] / 'features.parquet', engine='pyarrow')
            y_raw = pd.read_parquet(paths['data_raw'] / 'targets.parquet', engine='pyarrow')
        else:
            X_raw = pd.read_csv(paths['data_raw'] / 'features.csv')
            y_raw = pd.read_csv(paths['data_raw'] / 'targets.csv')
        
        # Limpiar datos
        X_raw = clean_features(X_raw)
        
        # Aplanar y limpiar target
        if isinstance(y_raw, pd.DataFrame):
            y_raw = y_raw.iloc[:, 0]
        y_raw = clean_target(y_raw)
        
        # Dividir (mismo random_state que en train.py para reproducibilidad)
        from sklearn.model_selection import train_test_split
        X_raw_train, X_raw_test, y_train, y_test = train_test_split(
            X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
        )
        
        # Preprocesar
        X_test = preprocessor.transform(X_raw_test)
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Convertir target a binario (0/1) si es necesario
    # Importante: '>50K' es la clase positiva (1), '<=50K' es la negativa (0)
    if y_test.dtype == object:
        y_test_bin = (y_test == '>50K').astype(int)
    else:
        y_test_bin = y_test
    
    if isinstance(y_pred, pd.Series) or (hasattr(y_pred, 'dtype') and y_pred.dtype == object):
        y_pred_bin = (y_pred == '>50K').astype(int)
    else:
        y_pred_bin = y_pred
    
    # Obtener probabilidades para la clase positiva (>50K)
    if hasattr(model, 'predict_proba'):
        # Sklearn ordena las clases alfabéticamente: ['<=50K', '>50K'] -> [0, 1]
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilidad de clase 1 (>50K)
    else:
        y_pred_proba = None
    
    # Métricas
    accuracy = accuracy_score(y_test_bin, y_pred_bin)
    f1 = f1_score(y_test_bin, y_pred_bin, average='macro')
    f1_binary = f1_score(y_test_bin, y_pred_bin, average='binary')
    auc = roc_auc_score(y_test_bin, y_pred_proba) if y_pred_proba is not None else None
    
    # Matriz de confusión
    cm = confusion_matrix(y_test_bin, y_pred_bin)
    
    # Generar reporte de clasificación
    class_report = classification_report(y_test_bin, y_pred_bin, target_names=['<=50K', '>50K'])
    
    # Guardar métricas
    metrics = {
        'accuracy': float(accuracy),
        'f1_macro': float(f1),
        'f1_binary': float(f1_binary),
        'auc': float(auc) if auc else None,
        'confusion_matrix': cm.tolist(),
    }
    
    # Guardar matriz de confusión como imagen
    paths['artifacts'].mkdir(parents=True, exist_ok=True)
    cm_path = paths['artifacts'] / 'confusion.png'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['<=50K', '>50K'],
                yticklabels=['<=50K', '>50K'])
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    log.info(f"Matriz de confusión guardada en {cm_path}")
    
    # Generar reporte HTML
    auc_str = f"{auc:.4f}" if auc is not None else "N/A"
    html_report = f"""<!DOCTYPE html>
<html>
<head>
    <title>Reporte de Evaluación - Adult Income</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .metric {{ display: inline-block; margin: 10px; padding: 20px; 
                   background: #f0f0f0; border-radius: 8px; min-width: 150px; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #2196F3; }}
        .metric-label {{ color: #666; }}
        pre {{ background: #f5f5f5; padding: 15px; border-radius: 4px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background: #333; color: white; }}
    </style>
</head>
<body>
    <h1>Reporte de Evaluación - Adult Income Prediction</h1>

    <h2>Métricas Principales</h2>
    <div class="metric">
        <div class="metric-value">{accuracy:.4f}</div>
        <div class="metric-label">Accuracy</div>
    </div>
    <div class="metric">
        <div class="metric-value">{f1:.4f}</div>
        <div class="metric-label">F1 Macro</div>
    </div>
    <div class="metric">
        <div class="metric-value">{f1_binary:.4f}</div>
        <div class="metric-label">F1 Binary</div>
    </div>
    <div class="metric">
        <div class="metric-value">{auc_str}</div>
        <div class="metric-label">AUC-ROC</div>
    </div>

    <h2>Matriz de Confusión</h2>
    <table>
        <tr><th></th><th>Pred <=50K</th><th>Pred >50K</th></tr>
        <tr><th>Real <=50K</th><td>{cm[0][0]}</td><td>{cm[0][1]}</td></tr>
        <tr><th>Real >50K</th><td>{cm[1][0]}</td><td>{cm[1][1]}</td></tr>
    </table>

    <h2>Reporte de Clasificación</h2>
    <pre>{class_report}</pre>

    <h2>Imagen</h2>
    <img src="confusion.png" alt="Matriz de Confusión" style="max-width: 500px;">
</body>
</html>
"""
    
    report_path = paths['artifacts'] / 'report.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_report)
    log.info(f"Reporte HTML guardado en {report_path}")
    
    # Guardar métricas en JSON
    metrics_path = paths['artifacts'] / 'evaluation_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    log.info(f"Métricas guardadas en {metrics_path}")
    
    # Imprimir resumen
    print(f"\n{'='*50}")
    print("RESULTADOS DE EVALUACIÓN")
    print(f"{'='*50}")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"F1 Macro:     {f1:.4f}")
    print(f"F1 Binary:    {f1_binary:.4f}")
    if auc:
        print(f"AUC-ROC:      {auc:.4f}")
    print(f"{'='*50}\n")
    
    return {
        'metrics': metrics,
        'report_path': str(report_path),
        'confusion_matrix_path': str(cm_path),
        'metrics_path': str(metrics_path),
    }


if __name__ == '__main__':
    import json as json_module
    
    paths = _resolve_paths()
    result = evaluate(paths=paths)
    print(json_module.dumps(result, indent=2, ensure_ascii=False))
