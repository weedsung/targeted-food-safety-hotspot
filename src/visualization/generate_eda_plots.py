import pandas as pd
import os
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.getcwd(), 'scratch', 'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'integrated_data_mart.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'eda_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_plots():
    df = pd.read_csv(DATA_PATH)
    
    # 1. 월별 위반 건수 추이
    monthly_counts = df.groupby('YEAR_MONTH').size().reset_index(name='COUNT')
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly_counts, x='YEAR_MONTH', y='COUNT', marker='o', color='red', linewidth=2)
    plt.title('월별 식품 위생 위반 행정처분 건수 추이 (2024-2026)', fontsize=15)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_monthly_violations.png'))
    plt.close()

    # 2. 물가(소비자물가지수) vs 위반 건수
    economy_impact = df.groupby('YEAR_MONTH').agg({
        'CPI_TOTAL': 'mean', 
        'PRSDNT_NM': 'count'
    }).rename(columns={'PRSDNT_NM': 'VIOLATION_COUNT'}).reset_index()
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=economy_impact, x='YEAR_MONTH', y='VIOLATION_COUNT', ax=ax1, color='red', marker='o', label='위반 건수')
    ax1.set_ylabel('위반 건수', color='red', fontsize=12)
    
    ax2 = ax1.twinx()
    sns.lineplot(data=economy_impact, x='YEAR_MONTH', y='CPI_TOTAL', ax=ax2, color='blue', marker='s', label='총 소비자물가지수(CPI)')
    ax2.set_ylabel('물가지수', color='blue', fontsize=12)
    
    plt.title('경제적 압박(물가 상승)과 위생 위반의 상관관계', fontsize=15)
    ax1.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_economy_vs_violation.png'))
    plt.close()

    # 3. 불쾌지수(THI) vs 위반 건수
    weather_impact = df.groupby('YEAR_MONTH').agg({
        'THI': 'mean', 
        'PRSDNT_NM': 'count'
    }).rename(columns={'PRSDNT_NM': 'VIOLATION_COUNT'}).reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.regplot(data=weather_impact, x='THI', y='VIOLATION_COUNT', color='orange', scatter_kws={'s':100})
    plt.title('기후적 압박(불쾌지수)과 위반 건수의 상관성', fontsize=15)
    plt.xlabel('평균 불쾌지수 (THI)')
    plt.ylabel('월별 위반 건수')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_weather_correlation.png'))
    plt.close()

    print(f"EDA 시각화 완료! 결과물 저장 위치: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_plots()
