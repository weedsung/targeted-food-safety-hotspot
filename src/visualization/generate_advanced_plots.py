import pandas as pd
import numpy as np
import os
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.getcwd(), 'scratch', 'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 및 스타일 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('dark_background') # 포스터 다크 테마에 맞춤

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'integrated_data_mart.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'eda_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_advanced_plots():
    print("고품질 데이터 시각화 생성 시작...")
    df = pd.read_csv(DATA_PATH)
    
    # 데이터셋 로드 (train_model.py와 동일한 로직 사용)
    df_x1 = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'x1_weather_monthly.csv'))
    df_x2 = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'x2_economy_clean.csv'))
    df_x3 = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'x3_mdis_2023_baseline.csv'))
    
    # 1. 시계열 듀얼 액시스 차트 (물가 상승 vs 위반율)
    print("1. 시계열 듀얼 액시스 차트 생성 중...")
    violation_counts = df.groupby('YEAR_MONTH').size().reset_index(name='COUNT')
    cpi_total = df_x2[df_x2['ITEM'] == 'CPI_TOTAL'][['YEAR_MONTH', 'CPI']].groupby('YEAR_MONTH').mean().reset_index()
    time_df = pd.merge(violation_counts, cpi_total, on='YEAR_MONTH', how='inner')
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.set_xlabel('기간 (Year-Month)', fontsize=12, color='white')
    ax1.set_ylabel('위반 건수 (건)', color='#FF4B4B', fontsize=14, fontweight='bold')
    sns.lineplot(data=time_df, x='YEAR_MONTH', y='COUNT', ax=ax1, color='#FF4B4B', marker='o', linewidth=3, label='적발 건수')
    ax1.tick_params(axis='y', labelcolor='#FF4B4B')
    ax1.tick_params(axis='x', rotation=45, colors='white')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('소비자물가지수 (CPI)', color='#00D4FF', fontsize=14, fontweight='bold')
    sns.lineplot(data=time_df, x='YEAR_MONTH', y='CPI', ax=ax2, color='#00D4FF', marker='s', linewidth=3, linestyle='--', label='물가지수')
    ax2.tick_params(axis='y', labelcolor='#00D4FF')
    
    plt.title('경제가 흔들리면 위생도 무너진다: 물가 상승과 식품 위반의 동기화 현상', fontsize=18, fontweight='bold', color='white', pad=20)
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'advanced_01_dual_axis_trend.png'), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    # 2. 저장된 실제 모델에서 Feature Importance 생성
    print("2. 저장된 LightGBM 모델 중요도 차트 생성 중...")
    import joblib
    MODEL_PATH = os.path.join(BASE_DIR, 'output', 'models', 'lightgbm_model.pkl')

    if not os.path.exists(MODEL_PATH):
        print("[WARNING] lightgbm_model.pkl 없음. train_model.py 먼저 실행 필요.")
        return

    lgb_model = joblib.load(MODEL_PATH)
    # 저장된 모델의 실제 피처 (train_model.py와 동일한 4개)
    actual_features = ['THI', 'CPI_AGRI', 'CPI_TOTAL', 'RESTAURANT_DENSITY']
    feature_name_map = {
        'CPI_TOTAL':          '총 소비자물가지수 (경제)',
        'CPI_AGRI':           '농축수산물 물가 (식재료)',
        'THI':                '기후 불쾌지수 (날씨)',
        'RESTAURANT_DENSITY': '식당 밀집도 (환경)',
    }

    feature_imp = pd.DataFrame({
        'Feature': actual_features,
        'Value':   lgb_model.feature_importances_,
        'Label':   [feature_name_map[f] for f in actual_features]
    }).sort_values('Value', ascending=True).reset_index(drop=True)

    with plt.style.context('default'):
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        fig, ax = plt.subplots(figsize=(10, 5))

        colors = ['#FADBD8', '#EC7063', '#C0392B', '#7B241C']
        bars = ax.barh(feature_imp['Label'], feature_imp['Value'],
                       color=colors[:len(feature_imp)])

        for bar, val in zip(bars, feature_imp['Value']):
            ax.text(bar.get_width() + max(feature_imp['Value']) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{int(val)}', va='center', ha='left',
                    fontsize=12, fontweight='bold', color='#2C3E50')

        ax.set_title('AI 모델 변수 중요도 (LightGBM Feature Importance)\n'
                     '(스플릿 횟수 기준 — 저장된 lightgbm_model.pkl 기준)',
                     fontsize=14, fontweight='bold', pad=20, color='#2C3E50')
        ax.set_xlabel('중요도 점수 (Split Count)', fontsize=11, color='#2C3E50')
        ax.tick_params(colors='#2C3E50')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'advanced_02_feature_importance.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    # 히트맵용 X 데이터 준비 (상관관계 차트에 사용)
    all_sidos = df['SIDO'].unique()
    all_months = df_x1['YEAR_MONTH'].unique()
    df_base = pd.DataFrame([{'YEAR_MONTH': m, 'SIDO': s} for m in all_months for s in all_sidos])
    viol_cnt = df.groupby(['YEAR_MONTH', 'SIDO']).size().reset_index(name='TARGET_COUNT')
    df_model = pd.merge(df_base, viol_cnt, on=['YEAR_MONTH', 'SIDO'], how='left').fillna(0)
    x1_f = df_x1.groupby('YEAR_MONTH').agg({'THI': 'mean', 'TEMP_AVG': 'mean', 'HUMIDITY': 'mean'}).reset_index()
    x2_f = df_x2.drop_duplicates(['YEAR_MONTH', 'ITEM']).pivot(index='YEAR_MONTH', columns='ITEM', values='CPI').reset_index()
    df_model = pd.merge(df_model, x1_f, on='YEAR_MONTH', how='left')
    df_model = pd.merge(df_model, x2_f, on='YEAR_MONTH', how='left')
    df_model = pd.merge(df_model, df_x3, on='SIDO', how='left')
    df_model = df_model.dropna(subset=['RESTAURANT_DENSITY', 'THI', 'CPI_TOTAL'])
    X = df_model[['THI', 'CPI_AGRI', 'CPI_TOTAL', 'RESTAURANT_DENSITY']]
    
    # 3. 상관관계 히트맵 (4개 변수 전체)
    print("3. 다변량 상관관계 히트맵 생성 중...")
    X_all = df_model[['THI', 'CPI_AGRI', 'CPI_TOTAL', 'RESTAURANT_DENSITY']].copy()
    X_all.columns = ['기후 불쾌지수\n(THI)', '농축수산물 물가\n(CPI_AGRI)', '총 소비자물가지수\n(CPI_TOTAL)', '식당 밀집도\n(RESTAURANT_DENSITY)']

    with plt.style.context('default'):
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = X_all.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5},
                    annot_kws={"size": 13, "weight": "bold"}, ax=ax)
        ax.set_title('독립 변수 간 상관관계 매트릭스 (Multicollinearity Check)',
                     fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
        ax.tick_params(colors='#2C3E50')
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'advanced_03_correlation_heatmap.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    print(f"완료! 고품질 포스터용 시각화 자료가 {OUTPUT_DIR}에 저장되었습니다.")

if __name__ == "__main__":
    generate_advanced_plots()
