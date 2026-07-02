import pandas as pd
import os
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.getcwd(), 'scratch', 'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'integrated_data_mart.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'eda_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_violation_text():
    print("3. 위반 사유(텍스트) 핵심 키워드 분석 중...")
    df = pd.read_csv(DATA_PATH)
    
    # 위반 내용(VILTCN) 추출 및 정제
    texts = df['VILTCN'].dropna().astype(str).tolist()
    
    # 간단한 단어 추출 (조사 제외 등은 한계가 있으므로 명사 위주 패턴 매칭)
    # 실제 공공데이터 특성상 '유통기한', '위생', '시설', '건강진단' 등이 주요 단어임
    words = []
    for text in texts:
        # 한글 단어(2글자 이상)만 추출
        found = re.findall(r'[가-힣]{2,}', text)
        words.extend(found)
    
    # 불용어 처리 (공공데이터 공통어구)
    stopwords = ['내용', '위반', '준수', '사항', '의거', '따른', '대한', '있음', '실시', '미실시', '하여', '관리']
    filtered_words = [w for w in words if w not in stopwords]
    
    word_counts = Counter(filtered_words).most_common(20)
    word_df = pd.DataFrame(word_counts, columns=['Keyword', 'Frequency'])
    
    # 시각화
    plt.figure(figsize=(10, 8))
    sns.barplot(data=word_df, x='Frequency', y='Keyword', palette='viridis')
    
    plt.title('주요 위생 위반 키워드 분석 (Top 20)\n(경제 압박 시 식당들이 포기하는 관리 항목)', fontsize=15, pad=20)
    plt.xlabel('출현 빈도')
    plt.ylabel('핵심 키워드')
    plt.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'advanced_03_violation_keywords.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"완료: {save_path}")

if __name__ == "__main__":
    plot_violation_text()
