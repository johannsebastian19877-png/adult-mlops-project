import os
import warnings
import json
import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError

# Suppress pandera top-level import FutureWarning if present
os.environ.setdefault('DISABLE_PANDERA_IMPORT_WARNING', 'True')
warnings.filterwarnings('ignore', category=FutureWarning)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Simple schema: extend as needed for your dataset
schema = pa.DataFrameSchema({
  'age': pa.Column(int, checks=pa.Check.in_range(17, 90)),
  'workclass': pa.Column(str, nullable=True),
  'education-num': pa.Column(int, checks=pa.Check.in_range(1, 16)),
  # add other columns from the Adult dataset here if you want strict validation
})


def validate_dataframe(df: pd.DataFrame, output_dir: str = 'data/processed') -> Dict[str, Any]:
  """Validate `df` against the schema and produce a JSON report.

  The report includes:
  - schema_issues (from pandera)
  - null_counts per column
  - detected dtypes
  - duplicate row count
  - numeric summaries (mean, std, min, max, quantiles)
  - categorical top-value distributions (top 10)

  Returns a dict with results and writes `validation_report.json` to `output_dir`.
  """
  repo_root = Path(__file__).resolve().parents[1]
  out_path = (repo_root / output_dir).resolve()
  out_path.mkdir(parents=True, exist_ok=True)

  report: Dict[str, Any] = {
    'success': True,
    'schema_issues': None,
    'null_counts': None,
    'dtypes': None,
    'duplicates': None,
    'numeric_summary': None,
    'categorical_distribution': None,
  }

  # 1) Schema validation (lazy to collect all failures)
  try:
    schema.validate(df, lazy=True)
    log.info('Schema validation passed')
  except SchemaError as err:
    log.warning('Schema validation failed')
    report['success'] = False
    try:
      report['schema_issues'] = err.failure_cases.to_dict(orient='records')
    except Exception:
      report['schema_issues'] = str(err)
  except Exception as err:
    log.exception('Unexpected schema validation error')
    report['success'] = False
    report['schema_issues'] = str(err)

  # 2) Null counts
  try:
    nulls = df.isnull().sum()
    report['null_counts'] = {str(c): int(v) for c, v in nulls.items()}
  except Exception as err:
    report['null_counts'] = str(err)

  # 3) Dtypes
  try:
    report['dtypes'] = {str(c): str(t) for c, t in df.dtypes.items()}
  except Exception as err:
    report['dtypes'] = str(err)

  # 4) Duplicates
  try:
    dup = int(df.duplicated().sum())
    report['duplicates'] = {'count': dup, 'has_duplicates': dup > 0}
  except Exception as err:
    report['duplicates'] = str(err)

  # 5) Numeric summary
  try:
    num = df.select_dtypes(include=['number'])
    numeric_summary: Dict[str, Any] = {}
    for col in num.columns:
      s = num[col].dropna()
      numeric_summary[col] = {
        'count': int(s.count()),
        'mean': float(s.mean()) if s.size else None,
        'std': float(s.std()) if s.size else None,
        'min': float(s.min()) if s.size else None,
        '25%': float(s.quantile(0.25)) if s.size else None,
        '50%': float(s.median()) if s.size else None,
        '75%': float(s.quantile(0.75)) if s.size else None,
        'max': float(s.max()) if s.size else None,
      }
    report['numeric_summary'] = numeric_summary
  except Exception as err:
    report['numeric_summary'] = str(err)

  # 6) Categorical distribution (top 10)
  try:
    cat = df.select_dtypes(include=['object', 'category'])
    cat_dist: Dict[str, Any] = {}
    for col in cat.columns:
      vc = cat[col].value_counts(dropna=False).head(10)
      cat_dist[col] = {str(k): int(v) for k, v in vc.items()}
    report['categorical_distribution'] = cat_dist
  except Exception as err:
    report['categorical_distribution'] = str(err)

  # Write report to disk
  report_path = out_path / 'validation_report.json'
  with report_path.open('w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

  report['output_path'] = str(report_path)
  return report


if __name__ == '__main__':
  # Run a quick example: attempt to load `data/raw/features.csv` or parquet and validate
  repo_root = Path(__file__).resolve().parents[1]
  raw_dir = repo_root / 'data' / 'raw'
  sample = None
  if (raw_dir / 'features.csv').exists():
    sample = pd.read_csv(raw_dir / 'features.csv')
  elif (raw_dir / 'features.parquet').exists():
    sample = pd.read_parquet(raw_dir / 'features.parquet', engine='pyarrow')

  if sample is None:
    log.info('No sample file found at %s. Import `validate_dataframe` and call it with your DataFrame.', raw_dir)
  else:
    res = validate_dataframe(sample)
    print(json.dumps(res, indent=2, ensure_ascii=False))
