"""
Smoke tests del pipeline MLOps.

Estos tests verifican que cada componente del pipeline
puede ejecutarse sin errores con datos mínimos.
"""
import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import tempfile
import os

from src.features import (
    build_preprocessor,
    fit_and_serialize_preprocessor,
    preprocess_data,
    NUM_COLS,
    CAT_COLS,
)
from src.train import train, DEFAULT_PARAMS
from src.evaluate import evaluate, evaluate_subgroups


def create_minimal_dataset():
    """Crear un dataset mínimo para tests."""
    n_samples = 100
    
    np.random.seed(42)
    
    df = pd.DataFrame({
        'age': np.random.randint(25, 65, n_samples),
        'fnlwgt': np.random.randint(50000, 400000, n_samples),
        'education-num': np.random.randint(8, 16, n_samples),
        'capital-gain': np.random.randint(0, 5000, n_samples),
        'capital-loss': np.random.randint(0, 500, n_samples),
        'hours-per-week': np.random.randint(30, 60, n_samples),
        'workclass': np.random.choice(['Private', 'Self-emp', 'Local-gov', 'State-gov'], n_samples),
        'education': np.random.choice(['HS-grad', 'Bachelors', 'Masters', 'Some-college'], n_samples),
        'marital-status': np.random.choice(['Married-civ-spouse', 'Never-married', 'Divorced'], n_samples),
        'occupation': np.random.choice(['Prof-specialty', 'Sales', 'Tech-support', 'Adm-clerical'], n_samples),
        'relationship': np.random.choice(['Husband', 'Wife', 'Not-in-family', 'Own-child'], n_samples),
        'race': np.random.choice(['White', 'Black', 'Asian-Pac-Islander'], n_samples),
        'sex': np.random.choice(['Male', 'Female'], n_samples),
        'native-country': np.random.choice(['United-States', 'Mexico', 'Canada', 'India'], n_samples),
    })
    
    # Target binario
    y = pd.Series(np.random.choice(['>50K', '<=50K'], n_samples), name='income')
    
    return df, y


class TestPreprocessingSmoke:
    """Smoke test para preprocesamiento."""

    def test_preprocessor_fit_transform(self):
        """El preprocesador debe poder hacer fit y transform."""
        df, _ = create_minimal_dataset()
        
        preprocessor = build_preprocessor(NUM_COLS, CAT_COLS)
        preprocessor.fit(df)
        result = preprocessor.transform(df)
        
        assert result.shape[0] == 100
        assert result.shape[1] == len(NUM_COLS) + len(CAT_COLS)

    def test_full_preprocessing_pipeline(self):
        """Test completo de preprocesamiento con serialización."""
        df, _ = create_minimal_dataset()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Fit y serialización
            result = fit_and_serialize_preprocessor(df, artifacts_dir=tmp_dir)
            
            # Verificar que se crearon los archivos
            assert Path(result['preprocessor']).exists()
            assert Path(result['meta']).exists()
            
            # Cargar y usar el preprocesador
            preprocessor = load_preprocessor(tmp_dir)
            X_transformed = preprocess_data(df, preprocessor=preprocessor, artifacts_dir=tmp_dir)
            
            assert X_transformed.shape[0] == 100


class TestTrainingSmoke:
    """Smoke test para entrenamiento."""

    def test_train_with_minimal_data(self):
        """El entrenamiento debe funcionar con datos mínimos."""
        df, y = create_minimal_dataset()
        
        # Preprocesar
        preprocessor = build_preprocessor(NUM_COLS, CAT_COLS)
        preprocessor.fit(df)
        X = preprocessor.transform(df)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = {
                'repo_root': Path(tmp_dir),
                'artifacts': Path(tmp_dir) / 'artifacts',
                'models': Path(tmp_dir) / 'models',
            }
            paths['artifacts'].mkdir(parents=True, exist_ok=True)
            paths['models'].mkdir(parents=True, exist_ok=True)
            
            # Entrenar
            result = train(X, y, params=DEFAULT_PARAMS, paths=paths)
            
            # Verificar resultados
            assert 'model_path' in result
            assert 'metrics' in result
            assert Path(result['model_path']).exists()
            assert 'f1_cv' in result['metrics']
            assert 'f1_test' in result['metrics']


class TestEvaluationSmoke:
    """Smoke test para evaluación."""

    def test_evaluate_subgroups(self):
        """La evaluación por subgrupos debe funcionar."""
        df, y = create_minimal_dataset()
        
        # Preparar datos
        y_bin = (y == '>50K').astype(int)
        y_pred = np.random.randint(0, 2, len(y))
        y_proba = np.random.random(len(y))
        
        # Evaluar subgrupos
        subgroup_metrics = evaluate_subgroups(df, y_bin, y_pred, y_proba)
        
        # Verificar que hay métricas por subgrupo
        assert 'sex' in subgroup_metrics or 'race' in subgroup_metrics or 'age' in subgroup_metrics


class TestPipelineIntegration:
    """Tests de integración del pipeline completo."""

    def test_end_to_end_preprocessing_and_training(self):
        """Test end-to-end: preprocesamiento + entrenamiento."""
        df, y = create_minimal_dataset()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Preprocesamiento
            preproc_result = fit_and_serialize_preprocessor(
                df, artifacts_dir=str(tmp_path / 'artifacts')
            )
            
            # Cargar preprocesador
            preprocessor = joblib.load(preproc_result['preprocessor'])
            X = preprocessor.transform(df)
            
            # Entrenamiento
            paths = {
                'repo_root': tmp_path,
                'artifacts': tmp_path / 'artifacts',
                'models': tmp_path / 'models',
            }
            train_result = train(X, y, params=DEFAULT_PARAMS, paths=paths)
            
            # Verificar
            assert Path(train_result['model_path']).exists()
            assert train_result['metrics']['f1_cv'] > 0
            assert train_result['metrics']['f1_cv'] <= 1
