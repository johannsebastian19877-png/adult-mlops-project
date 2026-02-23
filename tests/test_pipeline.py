"""
Tests para el pipeline MLOps.
"""
import pytest
import pandas as pd
from pathlib import Path


class TestIngest:
    """Tests para el módulo de ingesta."""
    
    def test_ingest_returns_dict(self):
        """La ingesta debe retornar un diccionario."""
        from src.ingest import ingest_adult
        result = ingest_adult()
        assert isinstance(result, dict)
    
    def test_ingest_has_required_keys(self):
        """El resultado debe tener las claves requeridas."""
        from src.ingest import ingest_adult
        result = ingest_adult()
        required_keys = ['n_rows', 'n_features', 'written_files']
        for key in required_keys:
            assert key in result


class TestValidate:
    """Tests para el módulo de validación."""
    
    def test_validate_returns_dict(self):
        """La validación debe retornar un diccionario."""
        from src.validate import validate_dataframe
        df = pd.DataFrame({'a': [1, 2, 3]})
        result = validate_dataframe(df)
        assert isinstance(result, dict)
    
    def test_validate_has_success_key(self):
        """El resultado debe tener la clave 'success'."""
        from src.validate import validate_dataframe
        df = pd.DataFrame({'a': [1, 2, 3]})
        result = validate_dataframe(df)
        assert 'success' in result


class TestFeatures:
    """Tests para el módulo de features."""
    
    def test_build_preprocessor_returns_transformer(self):
        """El preprocesador debe ser un ColumnTransformer."""
        from src.features import build_preprocessor
        from sklearn.compose import ColumnTransformer
        
        num_cols = ['age']
        cat_cols = ['workclass']
        preprocessor = build_preprocessor(num_cols, cat_cols)
        
        assert isinstance(preprocessor, ColumnTransformer)


class TestSchema:
    """Tests para el schema de validación."""
    
    def test_age_range(self):
        """La edad debe estar en rango válido."""
        from src.validate import schema
        
        # Datos válidos
        df_valid = pd.DataFrame({
            'age': [25, 50, 70],
            'workclass': ['Private', 'Self-emp', 'Local-gov'],
            'fnlwgt': [100000, 200000, 300000],
            'education': ['HS-grad', 'Bachelors', 'Masters'],
            'education-num': [12, 13, 14],
            'marital-status': ['Married', 'Single', 'Divorced'],
            'occupation': ['Prof', 'Sales', 'Tech'],
            'relationship': ['Husband', 'Wife', 'Child'],
            'race': ['White', 'Black', 'Asian'],
            'sex': ['Male', 'Female', 'Male'],
            'capital-gain': [0, 1000, 5000],
            'capital-loss': [0, 100, 200],
            'hours-per-week': [40, 45, 50],
            'native-country': ['USA', 'Mexico', 'Canada'],
        })
        
        # Debe validar sin errores
        schema.validate(df_valid)
    
    def test_age_out_of_range(self):
        """La edad fuera de rango debe fallar."""
        from src.validate import schema
        from pandera.errors import SchemaError
        
        df_invalid = pd.DataFrame({
            'age': [10],  # Menor que 17
            'workclass': ['Private'],
            'fnlwgt': [100000],
            'education': ['HS-grad'],
            'education-num': [12],
            'marital-status': ['Married'],
            'occupation': ['Prof'],
            'relationship': ['Husband'],
            'race': ['White'],
            'sex': ['Male'],
            'capital-gain': [0],
            'capital-loss': [0],
            'hours-per-week': [40],
            'native-country': ['USA'],
        })
        
        with pytest.raises(SchemaError):
            schema.validate(df_invalid)
