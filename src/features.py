from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import joblib


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
  """Return an unfitted ColumnTransformer for numerical and categorical columns."""
  num = Pipeline([('scaler', StandardScaler())])
  cat = Pipeline([('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
  ct = ColumnTransformer([('num', num, num_cols), ('cat', cat, cat_cols)], remainder='drop')
  return ct


def fit_and_serialize_preprocessor(
  df: pd.DataFrame,
  num_cols: Optional[List[str]] = None,
  cat_cols: Optional[List[str]] = None,
  artifacts_dir: str = 'artifacts',
) -> Dict[str, str]:
  """Fit preprocessor on `df`, save each transformer as a .joblib artifact.

  Returns a dict with paths to saved artifacts.
  """
  repo_root = Path(__file__).resolve().parents[1]
  artifacts_path = (repo_root / artifacts_dir).resolve()
  artifacts_path.mkdir(parents=True, exist_ok=True)

  # Infer columns if not provided
  if num_cols is None or cat_cols is None:
    inferred_num = df.select_dtypes(include=['number']).columns.tolist()
    inferred_cat = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = num_cols or inferred_num
    cat_cols = cat_cols or inferred_cat

  preprocessor = build_preprocessor(num_cols, cat_cols)
  # Fit the column transformer
  preprocessor.fit(df)

  saved: Dict[str, str] = {}

  # Save full preprocessor
  preproc_path = artifacts_path / 'preprocessor.joblib'
  joblib.dump(preprocessor, preproc_path)
  saved['preprocessor'] = str(preproc_path)

  # Extract and save individual transformers if present
  # ColumnTransformer stores transformers_ after fit as tuples (name, transformer, columns)
  for name, transformer, cols in preprocessor.transformers_:
    # Skip passthrough or drop
    if transformer == 'drop' or transformer == 'passthrough':
      continue
    safe_name = name.replace(' ', '_')
    path = artifacts_path / f"{safe_name}.joblib"
    joblib.dump(transformer, path)
    saved[safe_name] = str(path)

  # Also save metadata about which columns were used
  meta = {'num_cols': num_cols, 'cat_cols': cat_cols}
  meta_path = artifacts_path / 'preprocessor_meta.joblib'
  joblib.dump(meta, meta_path)
  saved['meta'] = str(meta_path)

  return saved


def load_preprocessor(artifacts_dir: str = 'artifacts') -> ColumnTransformer:
  """Load and return the full preprocessor from artifacts."""
  repo_root = Path(__file__).resolve().parents[1]
  preproc_path = (repo_root / artifacts_dir / 'preprocessor.joblib').resolve()
  return joblib.load(preproc_path)


if __name__ == '__main__':
  # Simple CLI: fit on data/raw/features.csv or parquet and serialize artifacts
  import json
  repo_root = Path(__file__).resolve().parents[1]
  raw = repo_root / 'data' / 'raw'
  df = None
  if (raw / 'features.csv').exists():
    df = pd.read_csv(raw / 'features.csv')
  elif (raw / 'features.parquet').exists():
    df = pd.read_parquet(raw / 'features.parquet', engine='pyarrow')

  if df is None:
    print('No raw features file found. Place features.csv or features.parquet in data/raw')
  else:
    saved = fit_and_serialize_preprocessor(df)
    print(json.dumps(saved, indent=2, ensure_ascii=False))
