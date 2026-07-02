import os

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, 'output', 'models')
DATA_PATH = os.path.join(MODEL_DIR, 'modeling_dataset.csv')


def fit_logit_with_pvalues(df, features, target='TARGET'):
    X = df[features].astype(float)
    y = df[target].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(C=np.inf, max_iter=1000)
    model.fit(X_scaled, y)

    probs = model.predict_proba(X_scaled)
    design = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])
    weights = np.diagflat(np.prod(probs, axis=1))

    try:
        cov = np.linalg.inv(design.T @ weights @ design)
        stderr = np.sqrt(np.diagonal(cov))
        z_values = np.append(model.intercept_, model.coef_[0]) / stderr
        p_values = [2 * (1 - stats.norm.cdf(abs(z))) for z in z_values][1:]
    except np.linalg.LinAlgError:
        p_values = [np.nan] * len(features)

    return pd.DataFrame({
        'Feature': features,
        'OddsRatio': np.exp(model.coef_[0]),
        'P_Value': p_values,
    })


def model_spec_sensitivity(df):
    specs = {
        'FULL_CURRENT': ['THI', 'CPI_AGRI', 'CPI_TOTAL', 'RESTAURANT_DENSITY'],
        'CPI_TOTAL_ONLY_ECON': ['THI', 'CPI_TOTAL', 'RESTAURANT_DENSITY'],
        'CPI_AGRI_ONLY_ECON': ['THI', 'CPI_AGRI', 'RESTAURANT_DENSITY'],
        'CPI_TOTAL_SIMPLE': ['CPI_TOTAL'],
        'CPI_AGRI_SIMPLE': ['CPI_AGRI'],
    }

    rows = []
    for spec_name, features in specs.items():
        result = fit_logit_with_pvalues(df, features)
        for _, row in result.iterrows():
            rows.append({
                'Check_Type': 'MODEL_SPEC',
                'Spec': spec_name,
                'Threshold_Quantile': 0.80,
                'Feature': row['Feature'],
                'OddsRatio': row['OddsRatio'],
                'P_Value': row['P_Value'],
                'Target_1_Count': int(df['TARGET'].sum()),
                'Rows_Used': len(df),
            })
    return pd.DataFrame(rows)


def hotspot_threshold_sensitivity(df):
    rows = []
    features = ['THI', 'CPI_TOTAL', 'RESTAURANT_DENSITY']

    for quantile in [0.75, 0.80, 0.90]:
        tmp = df.copy()
        threshold = tmp['VIOLATION_RATE'].quantile(quantile)
        tmp['TARGET_SENS'] = (tmp['VIOLATION_RATE'] >= threshold).astype(int)
        result = fit_logit_with_pvalues(tmp, features, target='TARGET_SENS')

        for _, row in result.iterrows():
            rows.append({
                'Check_Type': 'HOTSPOT_THRESHOLD',
                'Spec': f'TOP_{round((1 - quantile) * 100)}pct',
                'Threshold_Quantile': quantile,
                'Feature': row['Feature'],
                'OddsRatio': row['OddsRatio'],
                'P_Value': row['P_Value'],
                'Target_1_Count': int(tmp['TARGET_SENS'].sum()),
                'Rows_Used': len(tmp),
            })
    return pd.DataFrame(rows)


def ml_baseline_check(df):
    features = ['THI', 'CPI_AGRI', 'CPI_TOTAL', 'RESTAURANT_DENSITY']
    X = df[features]
    y = df['TARGET'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    majority_class = int(y_train.value_counts().idxmax())
    y_majority = np.repeat(majority_class, len(y_test))

    rows = [
        {
            'Model': 'LightGBM',
            'Accuracy': accuracy_score(y_test, y_pred),
            'Balanced_Accuracy': balanced_accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'ROC_AUC': roc_auc_score(y_test, y_prob),
        },
        {
            'Model': 'Majority_Baseline',
            'Accuracy': accuracy_score(y_test, y_majority),
            'Balanced_Accuracy': balanced_accuracy_score(y_test, y_majority),
            'Precision': precision_score(y_test, y_majority, zero_division=0),
            'Recall': recall_score(y_test, y_majority, zero_division=0),
            'F1': f1_score(y_test, y_majority, zero_division=0),
            'ROC_AUC': 0.5,
        },
    ]
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(DATA_PATH)

    sensitivity = pd.concat([
        model_spec_sensitivity(df),
        hotspot_threshold_sensitivity(df),
    ], ignore_index=True)
    sensitivity.to_csv(
        os.path.join(MODEL_DIR, 'sensitivity_analysis.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    ml_metrics = ml_baseline_check(df)
    ml_metrics.to_csv(
        os.path.join(MODEL_DIR, 'ml_baseline_metrics.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    print(sensitivity.round(4).to_string(index=False))
    print()
    print(ml_metrics.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
