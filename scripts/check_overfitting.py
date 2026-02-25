"""Verificar overfitting del modelo."""
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from src.features import clean_features, clean_target, load_preprocessor

# Cargar datos
X = pd.read_parquet('data/raw/features.parquet')
y = pd.read_parquet('data/raw/targets.parquet').iloc[:, 0]

# Limpiar
X = clean_features(X)
y = clean_target(y)

# Cargar preprocesador
preprocessor = load_preprocessor()
X_processed = preprocessor.transform(X)

# Dividir
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

# Cargar modelo
model = joblib.load('models/model.pkl')

# Métricas en train
y_train_pred = model.predict(X_train)
y_train_bin = (y_train == '>50K').astype(int)
y_train_pred_bin = (y_train_pred == '>50K').astype(int)

train_acc = accuracy_score(y_train_bin, y_train_pred_bin)
train_f1 = f1_score(y_train_bin, y_train_pred_bin, average='binary')

# Métricas en test
y_test_pred = model.predict(X_test)
y_test_bin = (y_test == '>50K').astype(int)
y_test_pred_bin = (y_test_pred == '>50K').astype(int)

test_acc = accuracy_score(y_test_bin, y_test_pred_bin)
test_f1 = f1_score(y_test_bin, y_test_pred_bin, average='binary')

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train_bin, cv=5, scoring='f1_macro')

print('='*60)
print('VERIFICACION DE OVERFITTING')
print('='*60)
print()
print('Metricas en TRAIN:')
print(f'  Accuracy: {train_acc:.4f}')
print(f'  F1 Binary: {train_f1:.4f}')
print()
print('Metricas en TEST:')
print(f'  Accuracy: {test_acc:.4f}')
print(f'  F1 Binary: {test_f1:.4f}')
print()
print('Diferencias (Train - Test):')
print(f'  Accuracy diff: {train_acc - test_acc:.4f}')
print(f'  F1 Binary diff: {train_f1 - test_f1:.4f}')
print()
print('Cross-Validation (5 folds):')
print(f'  F1 CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')
print()

# Criterios de overfitting
acc_diff = train_acc - test_acc
f1_diff = train_f1 - test_f1

print('='*60)
print('DIAGNOSTICO')
print('='*60)

if acc_diff > 0.1 or f1_diff > 0.1:
    print('ALERTA: Posible sobreentrenamiento detectado')
    print(f'  - Diferencia accuracy: {acc_diff:.4f} (> 0.1)')
    print(f'  - Diferencia F1: {f1_diff:.4f} (> 0.1)')
elif acc_diff > 0.05 or f1_diff > 0.05:
    print('PRECAUCION: Ligero sobreentrenamiento')
    print(f'  - Diferencia accuracy: {acc_diff:.4f}')
    print(f'  - Diferencia F1: {f1_diff:.4f}')
else:
    print('OK: No hay sobreentrenamiento significativo')
    print(f'  - Diferencia accuracy: {acc_diff:.4f}')
    print(f'  - Diferencia F1: {f1_diff:.4f}')

print()
print('CV vs Test:')
cv_f1 = cv_scores.mean()
if abs(cv_f1 - test_f1) > 0.05:
    print(f'  Diferencia: {abs(cv_f1 - test_f1):.4f} - Verificar consistencia')
else:
    print(f'  Diferencia: {abs(cv_f1 - test_f1):.4f} - Consistente')
