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

def plot_industry_sensitivity():
    print("2. 업종별 경제 압박 민감도 분석 중...")
    df = pd.read_csv(DATA_PATH)
    
    # 업종별, 월별 위반 건수 집계
    industry_monthly = df.groupby(['YEAR_MONTH', 'INDUTY_CD_NM']).size().reset_index(name='COUNT')
    
    # CPI 데이터 결합 (대표적으로 CPI_TOTAL 사용)
    cpi_data = df[['YEAR_MONTH', 'CPI_TOTAL']].drop_duplicates()
    merged = pd.merge(industry_monthly, cpi_data, on='YEAR_MONTH')
    
    # 각 업종별로 CPI와 위반 건수의 상관계수 계산
    industries = merged['INDUTY_CD_NM'].unique()
    sensitivity = []
    
    for ind in industries:
        subset = merged[merged['INDUTY_CD_NM'] == ind]
        if len(subset) > 5: # 데이터가 충분한 경우만
            corr = subset['COUNT'].corr(subset['CPI_TOTAL'])
            sensitivity.append({'Industry': ind, 'Correlation': corr})
            
    sens_df = pd.DataFrame(sensitivity).sort_values('Correlation', ascending=False)
    
    # 시각화: 롤리팝 차트
    plt.figure(figsize=(10, 8))
    plt.hlines(y=sens_df['Industry'], xmin=0, xmax=sens_df['Correlation'], color='skyblue', alpha=0.7, linewidth=3)
    plt.plot(sens_df['Correlation'], sens_df['Industry'], "o", markersize=10, color='blue', alpha=0.6)
    
    plt.axvline(0, color='grey', linestyle='--', alpha=0.5)
    plt.title('업종별 물가 상승(CPI)에 따른 위반 건수 민감도\n(상관계수가 높을수록 경제 압박에 취약)', fontsize=15, pad=20)
    plt.xlabel('상관계수 (CPI vs 위반 건수)')
    plt.ylabel('업종명')
    plt.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'advanced_02_industry_sensitivity.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"완료: {save_path}")

if __name__ == "__main__":
    plot_industry_sensitivity()
