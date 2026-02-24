"""
Tests para validación de schema con Pandera.
"""
import pytest
import pandas as pd
from pandera.errors import SchemaError

from src.validate import schema, validate_dataframe


class TestSchemaValidation:
    """Tests para validación de schema."""

    def test_valid_data_passes_schema(self):
        """Datos válidos deben pasar el schema."""
        df = pd.DataFrame({
            'age': [25, 30, 50],
            'workclass': ['Private', 'Self-emp', 'Local-gov'],
            'fnlwgt': [100000, 200000, 300000],
            'education': ['HS-grad', 'Bachelors', 'Masters'],
            'education-num': [12, 13, 14],
            'marital-status': ['Married-civ-spouse', 'Never-married', 'Divorced'],
            'occupation': ['Prof-specialty', 'Sales', 'Tech-support'],
            'relationship': ['Husband', 'Not-in-family', 'Wife'],
            'race': ['White', 'Black', 'Asian-Pac-Islander'],
            'sex': ['Male', 'Female', 'Male'],
            'capital-gain': [0, 1000, 5000],
            'capital-loss': [0, 100, 200],
            'hours-per-week': [40, 45, 50],
            'native-country': ['United-States', 'Mexico', 'Canada'],
        })
        
        # No debe lanzar excepción
        schema.validate(df)

    def test_age_below_range_fails(self):
        """Edad menor a 17 debe fallar."""
        df = pd.DataFrame({
            'age': [15],
            'workclass': ['Private'],
            'fnlwgt': [100000],
            'education': ['HS-grad'],
            'education-num': [12],
            'marital-status': ['Never-married'],
            'occupation': ['Prof-specialty'],
            'relationship': ['Not-in-family'],
            'race': ['White'],
            'sex': ['Male'],
            'capital-gain': [0],
            'capital-loss': [0],
            'hours-per-week': [40],
            'native-country': ['United-States'],
        })
        
        with pytest.raises(SchemaError):
            schema.validate(df, lazy=True)

    def test_age_above_range_fails(self):
        """Edad mayor a 90 debe fallar."""
        df = pd.DataFrame({
            'age': [95],
            'workclass': ['Private'],
            'fnlwgt': [100000],
            'education': ['HS-grad'],
            'education-num': [12],
            'marital-status': ['Never-married'],
            'occupation': ['Prof-specialty'],
            'relationship': ['Not-in-family'],
            'race': ['White'],
            'sex': ['Male'],
            'capital-gain': [0],
            'capital-loss': [0],
            'hours-per-week': [40],
            'native-country': ['United-States'],
        })
        
        with pytest.raises(SchemaError):
            schema.validate(df, lazy=True)

    def test_invalid_sex_fails(self):
        """Sexo inválido debe fallar."""
        df = pd.DataFrame({
            'age': [25],
            'workclass': ['Private'],
            'fnlwgt': [100000],
            'education': ['HS-grad'],
            'education-num': [12],
            'marital-status': ['Never-married'],
            'occupation': ['Prof-specialty'],
            'relationship': ['Not-in-family'],
            'race': ['White'],
            'sex': ['Invalid'],
            'capital-gain': [0],
            'capital-loss': [0],
            'hours-per-week': [40],
            'native-country': ['United-States'],
        })
        
        with pytest.raises(SchemaError):
            schema.validate(df, lazy=True)

    def test_education_num_out_of_range_fails(self):
        """education-num fuera de rango debe fallar."""
        df = pd.DataFrame({
            'age': [25],
            'workclass': ['Private'],
            'fnlwgt': [100000],
            'education': ['HS-grad'],
            'education-num': [20],  # Fuera de rango (1-16)
            'marital-status': ['Never-married'],
            'occupation': ['Prof-specialty'],
            'relationship': ['Not-in-family'],
            'race': ['White'],
            'sex': ['Male'],
            'capital-gain': [0],
            'capital-loss': [0],
            'hours-per-week': [40],
            'native-country': ['United-States'],
        })
        
        with pytest.raises(SchemaError):
            schema.validate(df, lazy=True)

    def test_hours_per_week_out_of_range_fails(self):
        """hours-per-week fuera de rango debe fallar."""
        df = pd.DataFrame({
            'age': [25],
            'workclass': ['Private'],
            'fnlwgt': [100000],
            'education': ['HS-grad'],
            'education-num': [12],
            'marital-status': ['Never-married'],
            'occupation': ['Prof-specialty'],
            'relationship': ['Not-in-family'],
            'race': ['White'],
            'sex': ['Male'],
            'capital-gain': [0],
            'capital-loss': [0],
            'hours-per-week': [120],  # Fuera de rango (1-99)
            'native-country': ['United-States'],
        })
        
        with pytest.raises(SchemaError):
            schema.validate(df, lazy=True)

    def test_capital_gain_negative_fails(self):
        """capital-gain negativo debe fallar."""
        df = pd.DataFrame({
            'age': [25],
            'workclass': ['Private'],
            'fnlwgt': [100000],
            'education': ['HS-grad'],
            'education-num': [12],
            'marital-status': ['Never-married'],
            'occupation': ['Prof-specialty'],
            'relationship': ['Not-in-family'],
            'race': ['White'],
            'sex': ['Male'],
            'capital-gain': [-100],  # Negativo no permitido
            'capital-loss': [0],
            'hours-per-week': [40],
            'native-country': ['United-States'],
        })
        
        with pytest.raises(SchemaError):
            schema.validate(df, lazy=True)


class TestValidateDataframe:
    """Tests para la función validate_dataframe."""

    def test_validate_returns_dict(self):
        """Debe retornar un diccionario."""
        df = pd.DataFrame({
            'age': [25, 30],
            'workclass': ['Private', 'Self-emp'],
            'fnlwgt': [100000, 200000],
            'education': ['HS-grad', 'Bachelors'],
            'education-num': [12, 13],
            'marital-status': ['Married', 'Single'],
            'occupation': ['Prof', 'Sales'],
            'relationship': ['Husband', 'Wife'],
            'race': ['White', 'Black'],
            'sex': ['Male', 'Female'],
            'capital-gain': [0, 1000],
            'capital-loss': [0, 100],
            'hours-per-week': [40, 45],
            'native-country': ['USA', 'Mexico'],
        })
        
        result = validate_dataframe(df)
        assert isinstance(result, dict)

    def test_validate_has_required_keys(self):
        """Debe tener las claves requeridas."""
        df = pd.DataFrame({
            'age': [25, 30],
            'workclass': ['Private', 'Self-emp'],
            'fnlwgt': [100000, 200000],
            'education': ['HS-grad', 'Bachelors'],
            'education-num': [12, 13],
            'marital-status': ['Married', 'Single'],
            'occupation': ['Prof', 'Sales'],
            'relationship': ['Husband', 'Wife'],
            'race': ['White', 'Black'],
            'sex': ['Male', 'Female'],
            'capital-gain': [0, 1000],
            'capital-loss': [0, 100],
            'hours-per-week': [40, 45],
            'native-country': ['USA', 'Mexico'],
        })
        
        result = validate_dataframe(df)
        required_keys = ['success', 'schema_issues', 'null_counts', 'dtypes', 'duplicates']
        for key in required_keys:
            assert key in result

    def test_validate_success_is_true_for_valid_data(self):
        """success debe ser True para datos válidos."""
        df = pd.DataFrame({
            'age': [25, 30],
            'workclass': ['Private', 'Self-emp'],
            'fnlwgt': [100000, 200000],
            'education': ['HS-grad', 'Bachelors'],
            'education-num': [12, 13],
            'marital-status': ['Married', 'Single'],
            'occupation': ['Prof', 'Sales'],
            'relationship': ['Husband', 'Wife'],
            'race': ['White', 'Black'],
            'sex': ['Male', 'Female'],
            'capital-gain': [0, 1000],
            'capital-loss': [0, 100],
            'hours-per-week': [40, 45],
            'native-country': ['USA', 'Mexico'],
        })
        
        result = validate_dataframe(df)
        assert result['success'] is True

    def test_validate_reports_null_counts(self):
        """Debe reportar conteo de nulos."""
        df = pd.DataFrame({
            'age': [25, None, 30],
            'workclass': ['Private', 'Self-emp', None],
            'fnlwgt': [100000, 200000, 300000],
            'education': ['HS-grad', 'Bachelors', 'Masters'],
            'education-num': [12, 13, 14],
            'marital-status': ['Married', 'Single', 'Divorced'],
            'occupation': ['Prof', 'Sales', 'Tech'],
            'relationship': ['Husband', 'Wife', 'Child'],
            'race': ['White', 'Black', 'Asian'],
            'sex': ['Male', 'Female', 'Male'],
            'capital-gain': [0, 1000, 2000],
            'capital-loss': [0, 100, 200],
            'hours-per-week': [40, 45, 50],
            'native-country': ['USA', 'Mexico', 'Canada'],
        })
        
        result = validate_dataframe(df)
        assert 'null_counts' in result
        assert result['null_counts']['age'] == 1
        assert result['null_counts']['workclass'] == 1

    def test_validate_detects_duplicates(self):
        """Debe detectar duplicados."""
        df = pd.DataFrame({
            'age': [25, 25, 30],
            'workclass': ['Private', 'Private', 'Self-emp'],
            'fnlwgt': [100000, 100000, 200000],
            'education': ['HS-grad', 'HS-grad', 'Bachelors'],
            'education-num': [12, 12, 13],
            'marital-status': ['Married', 'Married', 'Single'],
            'occupation': ['Prof', 'Prof', 'Sales'],
            'relationship': ['Husband', 'Husband', 'Wife'],
            'race': ['White', 'White', 'Black'],
            'sex': ['Male', 'Male', 'Female'],
            'capital-gain': [0, 0, 1000],
            'capital-loss': [0, 0, 100],
            'hours-per-week': [40, 40, 45],
            'native-country': ['USA', 'USA', 'Mexico'],
        })
        
        result = validate_dataframe(df)
        assert result['duplicates']['count'] == 1
        assert result['duplicates']['has_duplicates'] is True
