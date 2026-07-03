import os

import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, 'output', 'models')
DATA_PATH = os.path.join(MODEL_DIR, 'modeling_dataset.csv')
OUTPUT_PATH = os.path.join(MODEL_DIR, 'inspection_priority_index.csv')
SENSITIVITY_OUTPUT_PATH = os.path.join(MODEL_DIR, 'priority_weight_sensitivity.csv')


def percentile_rank(series):
    return series.rank(pct=True, method='average')


def add_priority_components(df):
    df = df.copy()
    df['DATE'] = pd.to_datetime(df['YEAR_MONTH'])
    df = df.sort_values(['SIDO', 'DATE'])

    previous_3m_avg = (
        df.groupby('SIDO')['VIOLATION_RATE']
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    df['RATE_TREND'] = (
        (df['VIOLATION_RATE'] - previous_3m_avg) / (previous_3m_avg + 0.01)
    ).replace([float('inf'), -float('inf')], 0).fillna(0)
    df['RATE_TREND'] = df['RATE_TREND'].clip(
        lower=df['RATE_TREND'].quantile(0.01),
        upper=df['RATE_TREND'].quantile(0.99),
    )

    df['HOTSPOT_REPEAT_COUNT_3M'] = (
        df.groupby('SIDO')['TARGET']
        .transform(lambda s: s.rolling(3, min_periods=1).sum())
    )
    df['HOTSPOT_REPEAT_SCORE'] = df['HOTSPOT_REPEAT_COUNT_3M'] / 3 * 100

    df['RATE_SCORE'] = percentile_rank(df['VIOLATION_RATE']) * 100
    df['TREND_SCORE'] = percentile_rank(df['RATE_TREND']) * 100
    df['CPI_PRESSURE_SCORE'] = percentile_rank(df['CPI_TOTAL']) * 100

    climate_thresholds = {
        'THI': df['THI'].quantile(0.8),
        'TEMP_MAX': df['TEMP_MAX'].quantile(0.8),
        'HUMIDITY': df['HUMIDITY'].quantile(0.8),
        'RAIN': df['RAIN'].quantile(0.8),
    }
    hot_humid = (df['TEMP_MAX'] >= climate_thresholds['TEMP_MAX']) & (
        df['HUMIDITY'] >= climate_thresholds['HUMIDITY']
    )
    df['CLIMATE_CAUTION_FLAG'] = (
        (df['THI'] >= climate_thresholds['THI'])
        | (df['TEMP_MAX'] >= climate_thresholds['TEMP_MAX'])
        | (df['HUMIDITY'] >= climate_thresholds['HUMIDITY'])
        | (df['RAIN'] >= climate_thresholds['RAIN'])
        | hot_humid
    ).astype(int)

    df['FSIPI'] = (
        0.55 * df['RATE_SCORE']
        + 0.20 * df['TREND_SCORE']
        + 0.15 * df['CPI_PRESSURE_SCORE']
        + 0.10 * df['HOTSPOT_REPEAT_SCORE']
    )
    df['FSIPI_GRADE'] = pd.cut(
        df['FSIPI'],
        bins=[-0.01, 40, 60, 80, 100],
        labels=['4등급·일반', '3등급·관찰', '2등급·높음', '1등급·매우 높음'],
    )
    return df


def build_weight_sensitivity(df):
    weight_sets = {
        '기본안_55_20_15_10': (0.55, 0.20, 0.15, 0.10),
        '발생률중심안_70_15_10_5': (0.70, 0.15, 0.10, 0.05),
        '균형안_45_25_20_10': (0.45, 0.25, 0.20, 0.10),
    }
    scores = {}
    top_sets = {}
    for name, weights in weight_sets.items():
        scores[name] = (
            weights[0] * df['RATE_SCORE']
            + weights[1] * df['TREND_SCORE']
            + weights[2] * df['CPI_PRESSURE_SCORE']
            + weights[3] * df['HOTSPOT_REPEAT_SCORE']
        )
        top_sets[name] = set(df.assign(_score=scores[name]).nlargest(10, '_score').index)

    base = '기본안_55_20_15_10'
    rows = []
    for name in weight_sets:
        overlap = len(top_sets[base] & top_sets[name])
        rows.append({
            'Weight_Set': name,
            'Rate_Weight': weight_sets[name][0],
            'Trend_Weight': weight_sets[name][1],
            'Economy_Weight': weight_sets[name][2],
            'Repeat_Weight': weight_sets[name][3],
            'Top10_Overlap_With_Base': overlap,
            'Top10_Overlap_Ratio': overlap / 10,
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(DATA_PATH)
    df = add_priority_components(df)

    out_cols = [
        'YEAR_MONTH',
        'SIDO',
        'TARGET_COUNT',
        'RESTAURANT_DENSITY',
        'VIOLATION_RATE',
        'CPI_TOTAL',
        'THI',
        'RATE_SCORE',
        'RATE_TREND',
        'TREND_SCORE',
        'CPI_PRESSURE_SCORE',
        'HOTSPOT_REPEAT_COUNT_3M',
        'HOTSPOT_REPEAT_SCORE',
        'CLIMATE_CAUTION_FLAG',
        'FSIPI',
        'FSIPI_GRADE',
        'TARGET',
    ]

    result = df[out_cols].sort_values(
        ['FSIPI', 'VIOLATION_RATE'],
        ascending=False,
    )
    result.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    sensitivity = build_weight_sensitivity(df)
    sensitivity.to_csv(SENSITIVITY_OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(result.head(15).round(3).to_string(index=False))
    print(sensitivity.to_string(index=False))
    print(OUTPUT_PATH)
    print(SENSITIVITY_OUTPUT_PATH)


if __name__ == '__main__':
    main()
