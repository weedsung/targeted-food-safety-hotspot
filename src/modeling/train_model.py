import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import os
import joblib

# 경로 설정 (현재 파일 위치가 src/modeling/ 내부이므로 3단계 상위가 루트 디렉토리임)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'integrated_data_mart.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'output', 'models')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
os.makedirs(MODEL_DIR, exist_ok=True)

def train_targeting_model():
    print("[1/3] 데이터셋 구성 및 음성 샘플(Non-Violation) 생성...")
    df = pd.read_csv(DATA_PATH)
    
    # 1. 모든 시도-연월 조합 생성 (Negative Sample 생성용)
    all_sidos = df['SIDO'].unique()
    all_months = df['YEAR_MONTH'].unique()
    
    base_grid = []
    for m in all_months:
        for s in all_sidos:
            base_grid.append({'YEAR_MONTH': m, 'SIDO': s})
    df_base = pd.DataFrame(base_grid)
    
    # 2. 실제 위반 건수 집계
    violation_counts = df.groupby(['YEAR_MONTH', 'SIDO']).size().reset_index(name='TARGET_COUNT')
    
    # 3. 데이터 병합
    df_model = pd.merge(df_base, violation_counts, on=['YEAR_MONTH', 'SIDO'], how='left').fillna(0)
    
    # 4. 특징(Feature) 데이터 로드 (전체 기간 데이터 확보를 위해 개별 파일 사용)
    df_x1 = pd.read_csv(os.path.join(PROCESSED_DIR, 'x1_weather_monthly.csv'))
    df_x2 = pd.read_csv(os.path.join(PROCESSED_DIR, 'x2_economy_clean.csv'))
    df_x3 = pd.read_csv(os.path.join(PROCESSED_DIR, 'x3_mdis_2023_baseline.csv'))
    
    # X1(기후) 전국 평균으로 단순화
    x1_features = df_x1.groupby('YEAR_MONTH').agg({
        'TEMP_AVG': 'mean', 'TEMP_MAX': 'max', 'HUMIDITY': 'mean', 'RAIN': 'mean', 'THI': 'mean'
    }).reset_index()
    
    # X2(물가) 피벗
    x2_features = df_x2.drop_duplicates(['YEAR_MONTH', 'ITEM']).pivot(index='YEAR_MONTH', columns='ITEM', values='CPI').reset_index()
    
    # 병합
    df_model = pd.merge(df_model, x1_features, on='YEAR_MONTH', how='left')
    df_model = pd.merge(df_model, x2_features, on='YEAR_MONTH', how='left')
    df_model = pd.merge(df_model, df_x3, on='SIDO', how='left')
    
    # 결측치 처리 (밀집도가 없는 데이터 등 제거)
    before_drop = len(df_model)
    df_model = df_model.dropna(subset=['RESTAURANT_DENSITY', 'THI', 'CPI_TOTAL'])
    print(f"모델링 데이터 결측 제거: {before_drop}행 -> {len(df_model)}행")
    
    # 5. 핵심: 위험도 재정의 (식당 1,000개당 위반 발생률 계산)
    # 단순 건수가 아닌 '비율'을 사용하여 대도시 편향(Bias) 제거
    df_model['VIOLATION_RATE'] = (df_model['TARGET_COUNT'] / df_model['RESTAURANT_DENSITY']) * 1000
    
    # 상위 20% 비율을 기록한 시공간을 '고위험 핫스팟(Target=1)'으로 정의
    threshold = df_model['VIOLATION_RATE'].quantile(0.8)
    df_model['TARGET'] = (df_model['VIOLATION_RATE'] >= threshold).astype(int)
    
    print(f"\n[핫스팟 정의] 상위 20% 위반율 임계점: {threshold:.2f}건 (식당 1,000개당)")
    print(f"데이터 클래스 분포 (1=핫스팟, 0=일반): \n{df_model['TARGET'].value_counts()}")
    
    import scipy.stats as stat
    
    # 6. 모델 학습 (Logistic Regression for Odds Ratio & P-value)
    print("[2/3] 로지스틱 회귀 학습 및 종합 통계 지표 도출...")
    X_cols = ['THI', 'CPI_AGRI', 'CPI_TOTAL', 'RESTAURANT_DENSITY']
    X = df_model[X_cols]
    y = df_model['TARGET']
    
    # 단순 상관계수 계산 (위반 건수 vs 독립변수)
    correlations = [df_model['TARGET_COUNT'].corr(df_model[col]) for col in X_cols]
    
    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 정밀한 통계 검정을 위해 penalty=None (규제 없음) 사용
    log_reg = LogisticRegression(C=np.inf, max_iter=1000)
    log_reg.fit(X_scaled, y)
    
    # 오즈비 계산
    odds_ratios = np.exp(log_reg.coef_[0])
    
    # P-value 계산 (Wald Test)
    predProbs = log_reg.predict_proba(X_scaled)
    X_design = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])
    V = np.diagflat(np.prod(predProbs, axis=1))
    
    try:
        covLogit = np.linalg.inv(np.dot(np.dot(X_design.T, V), X_design))
        std_err = np.sqrt(np.diagonal(covLogit))
        wald_z = np.append(log_reg.intercept_, log_reg.coef_[0]) / std_err
        p_values = [2 * (1 - stat.norm.cdf(np.abs(z))) for z in wald_z][1:] # intercept 제외
    except Exception as e:
        print(f"P-value 계산 중 오류 발생 (단순 오즈비만 출력합니다): {e}")
        p_values = [np.nan] * len(X_cols)

    or_df = pd.DataFrame({
        'Feature': X_cols,
        'Correlation (vs Count)': correlations,
        'OddsRatio': odds_ratios,
        'P_Value': p_values
    })
    
    print("\n" + "="*60)
    print(" [핵심 통계 결과: 오즈비 및 유의성 검정 종합표] ")
    print("="*60)
    print(or_df.to_string(index=False))
    print("="*60 + "\n")
    
    # 7. 모델 학습 (LightGBM for Accuracy)
    print("[3/3] LightGBM 고성능 예측 모델 학습...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    lgb_model.fit(X_train, y_train)

    y_pred = lgb_model.predict(X_test)
    y_proba = lgb_model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"LightGBM 검증 성능: accuracy={accuracy:.3f}, auc={auc:.3f}")
    
    # 모델 저장
    joblib.dump(log_reg, os.path.join(MODEL_DIR, 'logistic_reg_model.pkl'))
    joblib.dump(lgb_model, os.path.join(MODEL_DIR, 'lightgbm_model.pkl'))
    
    # 통합 통계 데이터 저장 (추가된 종합본)
    or_df.to_csv(os.path.join(MODEL_DIR, 'odds_ratios.csv'), index=False)

    feature_importance = pd.Series(lgb_model.feature_importances_, index=X_cols)
    cpi_corr = X.corr()['CPI_TOTAL'].reindex(X_cols).fillna(0)

    comprehensive_df = or_df.copy()
    comprehensive_df['Correlation_with_CPI_TOTAL'] = comprehensive_df['Feature'].map(cpi_corr.to_dict())
    comprehensive_df['LightGBM_Split_Count'] = comprehensive_df['Feature'].map(feature_importance.to_dict())
    comprehensive_df['Category'] = comprehensive_df['Feature'].map({
        'CPI_TOTAL': 'Economic Pressure',
        'CPI_AGRI': 'Economic Pressure',
        'THI': 'Climate Pressure',
        'RESTAURANT_DENSITY': 'Spatial Baseline',
    })
    comprehensive_df['Conclusion_Summary'] = comprehensive_df['Feature'].map({
        'CPI_TOTAL': 'Positive hotspot-risk signal after density control',
        'CPI_AGRI': 'Significant but interpreted cautiously due to CPI multicollinearity',
        'THI': 'Negative association under restaurant-normalized hotspot target',
        'RESTAURANT_DENSITY': 'High count correlation but no direct hotspot-rate effect',
    })
    comprehensive_df = comprehensive_df[[
        'Feature', 'Category', 'Correlation_with_CPI_TOTAL',
        'Correlation (vs Count)', 'OddsRatio', 'P_Value',
        'LightGBM_Split_Count', 'Conclusion_Summary'
    ]].rename(columns={'Correlation (vs Count)': 'Correlation_with_Violations'})
    comprehensive_df.to_csv(os.path.join(MODEL_DIR, 'comprehensive_statistics.csv'), index=False)

    df_model.to_csv(os.path.join(MODEL_DIR, 'modeling_dataset.csv'), index=False, encoding='utf-8-sig')

    summary = pd.DataFrame([{
        'Rows_Used': len(df_model),
        'Months_Used': df_model['YEAR_MONTH'].nunique(),
        'Sidos_Used': df_model['SIDO'].nunique(),
        'Hotspot_Threshold_Per_1000_Restaurants': threshold,
        'Target_0_Count': int((df_model['TARGET'] == 0).sum()),
        'Target_1_Count': int((df_model['TARGET'] == 1).sum()),
        'LightGBM_Accuracy': accuracy,
        'LightGBM_AUC': auc,
    }])
    summary.to_csv(os.path.join(MODEL_DIR, 'model_training_summary.csv'), index=False)
    
    print(f"\n모델 학습 및 결과 저장 완료! (위치: {MODEL_DIR})")
    return or_df

if __name__ == "__main__":
    train_targeting_model()
