import os

import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, 'output', 'models')
DATA_PATH = os.path.join(MODEL_DIR, 'modeling_dataset.csv')
OUTPUT_PATH = os.path.join(MODEL_DIR, 'inspection_priority_index.csv')


def percentile_rank(series):
    return series.rank(pct=True, method='average')


def main():
    df = pd.read_csv(DATA_PATH)

    # Transparent operational score, not a predictive ML model.
    # We keep the direct administrative outcome as the largest component.
    df['RATE_SCORE'] = percentile_rank(df['VIOLATION_RATE']) * 100
    df['CPI_PRESSURE_SCORE'] = percentile_rank(df['CPI_TOTAL']) * 100
    df['CLIMATE_LOAD_SCORE'] = percentile_rank(df['THI']) * 100

    df['INSPECTION_PRIORITY_SCORE'] = (
        0.70 * df['RATE_SCORE']
        + 0.20 * df['CPI_PRESSURE_SCORE']
        + 0.10 * df['CLIMATE_LOAD_SCORE']
    )

    out_cols = [
        'YEAR_MONTH',
        'SIDO',
        'TARGET_COUNT',
        'RESTAURANT_DENSITY',
        'VIOLATION_RATE',
        'CPI_TOTAL',
        'THI',
        'RATE_SCORE',
        'CPI_PRESSURE_SCORE',
        'CLIMATE_LOAD_SCORE',
        'INSPECTION_PRIORITY_SCORE',
        'TARGET',
    ]

    result = df[out_cols].sort_values(
        ['INSPECTION_PRIORITY_SCORE', 'VIOLATION_RATE'],
        ascending=False,
    )
    result.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(result.head(15).round(3).to_string(index=False))
    print(OUTPUT_PATH)


if __name__ == '__main__':
    main()
