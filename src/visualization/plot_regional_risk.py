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

def plot_regional_risk():
    print("1. 지역별 위생 취약 지수 분석 중...")
    df = pd.read_csv(DATA_PATH)
    
    # 지역별 위반 건수 및 평균 밀집도(식당 수) 집계
    regional_stats = df.groupby('SIDO').agg({
        'PRSDNT_NM': 'count',
        'RESTAURANT_DENSITY': 'mean'
    }).rename(columns={'PRSDNT_NM': 'VIOLATION_COUNT'}).reset_index()
    
    # 1,000개당 위반율 계산
    regional_stats['VIOLATION_RATE'] = (regional_stats['VIOLATION_COUNT'] / regional_stats['RESTAURANT_DENSITY']) * 1000
    regional_stats = regional_stats.sort_values('VIOLATION_RATE', ascending=False)

    # 시각화: 상단은 위반율(막대), 하단은 위반건수 vs 밀집도(버블)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # 차트 1: 위반율 랭킹
    sns.barplot(data=regional_stats, x='SIDO', y='VIOLATION_RATE', palette='Reds_r', ax=ax1)
    ax1.set_title('지역별 식당 1,000개당 위생 위반율 (취약 지역 순위)', fontsize=15, pad=20)
    ax1.set_ylabel('위반율 (건/1,000개 식당)')
    ax1.tick_params(axis='x', rotation=45)

    # 차트 2: 건수 vs 밀집도 관계
    sns.scatterplot(data=regional_stats, x='RESTAURANT_DENSITY', y='VIOLATION_COUNT', 
                    size='VIOLATION_RATE', hue='VIOLATION_RATE', sizes=(100, 1000),
                    palette='coolwarm', ax=ax2, legend=False)
    
    # 지역명 라벨링
    for i in range(regional_stats.shape[0]):
        ax2.text(regional_stats.RESTAURANT_DENSITY[i], regional_stats.VIOLATION_COUNT[i], 
                 regional_stats.SIDO[i], fontsize=10, ha='center', va='center', fontweight='bold')

    ax2.set_title('식당 밀집도 대비 실제 위반 건수 (버블 크기=위반율)', fontsize=15, pad=20)
    ax2.set_xlabel('식당 밀집도 (지역 내 총 식당 수 추정)')
    ax2.set_ylabel('총 위반 건수')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'advanced_01_regional_risk.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"완료: {save_path}")

if __name__ == "__main__":
    plot_regional_risk()
