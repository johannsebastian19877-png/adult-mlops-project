"""
Tests para el módulo de features (preprocesador).
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from src.features import (
    clean_target,
    clean_features,
    handle_missing_values,
    build_preprocessor,
    fit_and_serialize_preprocessor,
    load_preprocessor,
    preprocess_data,
    NUM_COLS,
    CAT_COLS,
    MISSING_VALUES,
)


class TestCleanTarget:
    """Tests para la función clean_target."""

    def test_clean_target_removes_dots(self):
        """Debe eliminar puntos al final."""
        y = pd.Series(['>50K.', '<=50K.', '>50K.'])
        result = clean_target(y)
        assert all(result == ['>50K', '<=50K', '>50K'])

    def test_clean_target_removes_whitespace(self):
        """Debe eliminar espacios extra."""
        y = pd.Series([' >50K ', '<=50K', '  >50K'])
        result = clean_target(y)
        assert all(result == ['>50K', '<=50K', '>50K'])

    def test_clean_target_preserves_valid_values(self):
        """Debe preservar valores válidos."""
        y = pd.Series(['>50K', '<=50K'])
        result = clean_target(y)
        assert all(result == ['>50K', '<=50K'])


class TestHandleMissingValues:
    """Tests para la función handle_missing_values."""

    def test_handle_missing_replaces_question_mark(self):
        """Debe reemplazar '?' con NaN."""
        df = pd.DataFrame({
            'workclass': ['Private', '?', 'Self-emp'],
            'occupation': ['Prof', '?', 'Sales'],
        })
        result = handle_missing_values(df)
        assert pd.isna(result.loc[1, 'workclass'])
        assert pd.isna(result.loc[1, 'occupation'])

    def test_handle_missing_replaces_space_question_mark(self):
        """Debe reemplazar ' ?' con NaN."""
        df = pd.DataFrame({
            'workclass': ['Private', ' ?', 'Self-emp'],
        })
        result = handle_missing_values(df)
        assert pd.isna(result.loc[1, 'workclass'])

    def test_handle_missing_preserves_valid_values(self):
        """Debe preservar valores válidos."""
        df = pd.DataFrame({
            'workclass': ['Private', 'Self-emp', 'Local-gov'],
        })
        result = handle_missing_values(df)
        assert all(result['workclass'] == ['Private', 'Self-emp', 'Local-gov'])

    def test_handle_missing_preserves_numeric_columns(self):
        """No debe afectar columnas numéricas."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'workclass': ['Private', '?', 'Self-emp'],
        })
        result = handle_missing_values(df)
        assert all(result['age'] == [25, 30, 35])


class TestCleanFeatures:
    """Tests para la función clean_features."""

    def test_clean_features_strips_whitespace(self):
        """Debe eliminar espacios en blanco."""
        df = pd.DataFrame({
            'workclass': [' Private ', 'Self-emp', ' Local-gov'],
        })
        result = clean_features(df)
        assert all(result['workclass'] == ['Private', 'Self-emp', 'Local-gov'])

    def test_clean_features_removes_trailing_dots(self):
        """Debe eliminar puntos al final."""
        df = pd.DataFrame({
            'workclass': ['Private.', 'Self-emp.', 'Local-gov.'],
        })
        result = clean_features(df)
        assert all(result['workclass'] == ['Private', 'Self-emp', 'Local-gov'])


class TestBuildPreprocessor:
    """Tests para la función build_preprocessor."""

    def test_build_preprocessor_returns_column_transformer(self):
        """Debe retornar un ColumnTransformer."""
        from sklearn.compose import ColumnTransformer
        preprocessor = build_preprocessor(NUM_COLS, CAT_COLS)
        assert isinstance(preprocessor, ColumnTransformer)

    def test_build_preprocessor_has_correct_transformers(self):
        """Debe tener los transformadores correctos."""
        preprocessor = build_preprocessor(NUM_COLS, CAT_COLS)
        names = [name for name, _, _ in preprocessor.transformers]
        assert 'num' in names
        assert 'cat' in names

    def test_build_preprocessor_transforms_data(self):
        """Debe transformar datos correctamente."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'fnlwgt': [100000, 200000, 300000],
            'education-num': [12, 13, 14],
            'capital-gain': [0, 1000, 2000],
            'capital-loss': [0, 100, 200],
            'hours-per-week': [40, 45, 50],
            'workclass': ['Private', 'Self-emp', 'Local-gov'],
            'education': ['HS-grad', 'Bachelors', 'Masters'],
            'marital-status': ['Married', 'Single', 'Divorced'],
            'occupation': ['Prof', 'Sales', 'Tech'],
            'relationship': ['Husband', 'Wife', 'Child'],
            'race': ['White', 'Black', 'Asian'],
            'sex': ['Male', 'Female', 'Male'],
            'native-country': ['USA', 'Mexico', 'Canada'],
        })
        preprocessor = build_preprocessor(NUM_COLS, CAT_COLS)
        preprocessor.fit(df)
        result = preprocessor.transform(df)
        assert result.shape[0] == 3
        assert result.shape[1] == len(NUM_COLS) + len(CAT_COLS)


class TestFitAndSerializePreprocessor:
    """Tests para la función fit_and_serialize_preprocessor."""

    def test_fit_and_serialize_creates_files(self, tmp_path):
        """Debe crear archivos de artefactos."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'fnlwgt': [100000, 200000, 300000],
            'education-num': [12, 13, 14],
            'capital-gain': [0, 1000, 2000],
            'capital-loss': [0, 100, 200],
            'hours-per-week': [40, 45, 50],
            'workclass': ['Private', 'Self-emp', 'Local-gov'],
            'education': ['HS-grad', 'Bachelors', 'Masters'],
            'marital-status': ['Married', 'Single', 'Divorced'],
            'occupation': ['Prof', 'Sales', 'Tech'],
            'relationship': ['Husband', 'Wife', 'Child'],
            'race': ['White', 'Black', 'Asian'],
            'sex': ['Male', 'Female', 'Male'],
            'native-country': ['USA', 'Mexico', 'Canada'],
        })
        
        # Usar tmp_path para evitar escribir en el directorio real
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_artifacts:
            # Mock del repo root
            import src.features as features_module
            original_resolve = Path.__call__
            
            result = fit_and_serialize_preprocessor(df, artifacts_dir=tmp_artifacts)
            
            assert 'preprocessor' in result
            assert 'meta' in result
            assert Path(result['preprocessor']).exists()
            assert Path(result['meta']).exists()


class TestPreprocessData:
    """Tests para la función preprocess_data."""

    def test_preprocess_data_with_missing_values(self):
        """Debe manejar valores faltantes correctamente."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'fnlwgt': [100000, 200000, 300000],
            'education-num': [12, 13, 14],
            'capital-gain': [0, 1000, 2000],
            'capital-loss': [0, 100, 200],
            'hours-per-week': [40, 45, 50],
            'workclass': ['Private', '?', 'Local-gov'],
            'education': ['HS-grad', 'Bachelors', 'Masters'],
            'marital-status': ['Married', 'Single', 'Divorced'],
            'occupation': ['Prof', 'Sales', 'Tech'],
            'relationship': ['Husband', 'Wife', 'Child'],
            'race': ['White', 'Black', 'Asian'],
            'sex': ['Male', 'Female', 'Male'],
            'native-country': ['USA', 'Mexico', 'Canada'],
        })
        
        preprocessor = build_preprocessor(NUM_COLS, CAT_COLS)
        preprocessor.fit(df)
        
        result = preprocess_data(df, preprocessor=preprocessor)
        assert result.shape[0] == 3
