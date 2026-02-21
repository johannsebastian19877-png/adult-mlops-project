# src/ingest.py
import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from ucimlrepo import fetch_ucirepo

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _resolve_output_path(output_dir: str) -> Path:
    # Resolve relative to the repository root (one level above src/)
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / output_dir).resolve()


def ingest_adult(output_dir: str = 'data/raw') -> Dict[str, Any]:
    """Fetch the UCI adult dataset and save features/targets to output_dir.

    Tries to write Parquet using `pyarrow`; if that fails, falls back to CSV.
    Returns a small summary dict.
    """
    adult = fetch_ucirepo(id=2)

    # Validate fetched object
    features = None
    targets = None
    try:
        features = adult.data.features
        targets = adult.data.targets
    except Exception:
        # best-effort attribute access
        if hasattr(adult, 'data'):
            data = getattr(adult, 'data')
            features = getattr(data, 'features', None)
            targets = getattr(data, 'targets', None)
    if features is None or targets is None:
        raise RuntimeError('Fetched dataset does not contain expected features/targets')

    out_path = _resolve_output_path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Try parquet first, but also write CSV for portability
    written = []
    parquet_ok = False
    try:
        features.to_parquet(out_path / 'features.parquet', engine='pyarrow')
        targets.to_parquet(out_path / 'targets.parquet', engine='pyarrow')
        written.extend(['features.parquet', 'targets.parquet'])
        parquet_ok = True
        log.info('Wrote parquet files to %s', out_path)
    except Exception as e:
        log.warning('Parquet write failed (%s). Will attempt CSV only.', e)

    # Always attempt CSV as well (either fallback or alongside parquet)
    try:
        features.to_csv(out_path / 'features.csv', index=False)
        targets.to_csv(out_path / 'targets.csv', index=False)
        written.extend(['features.csv', 'targets.csv'])
        log.info('Wrote CSV files to %s', out_path)
    except Exception as e2:
        log.error('Failed to write CSV: %s', e2)
        if not parquet_ok:
            # Only raise if neither format could be written
            raise

    return {
        'n_rows': int(len(features)),
        'n_features': int(features.shape[1]) if hasattr(features, 'shape') else None,
        'target_dist': {str(k): v for k, v in (targets.value_counts().to_dict().items() if hasattr(targets, 'value_counts') else [])},
        'written_files': written,
        'output_dir': str(out_path)
    }


if __name__ == '__main__':
    import json
    try:
        res = ingest_adult()
        print(json.dumps(res, indent=2, ensure_ascii=False))
    except Exception as e:
        log.exception('Ingest failed: %s', e)
