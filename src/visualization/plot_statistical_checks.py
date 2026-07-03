import os

import matplotlib
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.getcwd(), 'scratch', 'matplotlib'))
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, 'output', 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'eda_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def save_correlation_summary(df):
    predictors = ['THI', 'CPI_AGRI', 'CPI_TOTAL', 'RESTAURANT_DENSITY']
    outcomes = ['TARGET_COUNT', 'VIOLATION_RATE', 'TARGET']
    label_map = {
        'THI': '불쾌지수(THI)',
        'CPI_AGRI': '농축수산 CPI',
        'CPI_TOTAL': '총 CPI',
        'RESTAURANT_DENSITY': '음식점업 사업체 수',
        'TARGET_COUNT': '절대 행정처분 건수',
        'VIOLATION_RATE': '식당 1,000개당 발생률',
        'TARGET': 'Top 20% 핫스팟',
    }

    corr = df[predictors + outcomes].corr().loc[predictors, outcomes]
    corr.index = [label_map[i] for i in corr.index]
    corr.columns = [label_map[c] for c in corr.columns]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    sns.heatmap(
        corr,
        annot=True,
        fmt='.3f',
        cmap='RdBu_r',
        center=0,
        linewidths=0.5,
        linecolor='#E2E8F0',
        cbar_kws={'label': 'Pearson r'},
        ax=ax,
    )
    ax.set_title('상관분석 요약: 절대 건수와 보정 발생률은 다르게 움직임', fontsize=14, fontweight='bold', pad=16)
    ax.set_xlabel('분석 결과 변수')
    ax.set_ylabel('설명 변수')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'stat_01_correlation_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_model_spec_sensitivity(sens):
    specs = [
        ('FULL_CURRENT', '전체 모형'),
        ('CPI_TOTAL_ONLY_ECON', '총 CPI 모형'),
        ('CPI_AGRI_ONLY_ECON', '농축수산 CPI 모형'),
        ('CPI_TOTAL_SIMPLE', '총 CPI 단순'),
        ('CPI_AGRI_SIMPLE', '농축수산 CPI 단순'),
    ]
    rows = []
    for spec, label in specs:
        sub = sens[(sens['Check_Type'] == 'MODEL_SPEC') & (sens['Spec'] == spec)]
        if spec in ['FULL_CURRENT', 'CPI_TOTAL_ONLY_ECON', 'CPI_TOTAL_SIMPLE']:
            feature = 'CPI_TOTAL'
        else:
            feature = 'CPI_AGRI'
        row = sub[sub['Feature'] == feature].iloc[0]
        rows.append({
            'Spec_Label': label,
            'Feature': feature,
            'OddsRatio': row['OddsRatio'],
            'P_Value': row['P_Value'],
        })
    plot_df = pd.DataFrame(rows)

    colors = ['#DC2626' if v > 1 else '#2563EB' for v in plot_df['OddsRatio']]
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    bars = ax.barh(plot_df['Spec_Label'], plot_df['OddsRatio'], color=colors, alpha=0.88)
    ax.axvline(1.0, color='#111827', linestyle='--', linewidth=1.3)
    ax.set_xlabel('오즈비(OR)')
    ax.set_title('경제 변수 민감도: CPI 신호는 전체 모형에서만 강하게 나타남', fontsize=14, fontweight='bold', pad=14)
    ax.set_xlim(0.5, 1.55)
    ax.grid(axis='x', alpha=0.25)

    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        ax.text(
            row['OddsRatio'] + 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"OR {row['OddsRatio']:.2f}, p={row['P_Value']:.3f}",
            va='center',
            fontsize=9,
            color='#111827',
        )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'stat_02_model_spec_sensitivity.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_hotspot_threshold_sensitivity(sens):
    sub = sens[sens['Check_Type'] == 'HOTSPOT_THRESHOLD'].copy()
    sub = sub[sub['Feature'].isin(['THI', 'CPI_TOTAL', 'RESTAURANT_DENSITY'])]
    label_map = {
        'THI': 'THI',
        'CPI_TOTAL': '총 CPI',
        'RESTAURANT_DENSITY': '음식점업 사업체 수',
        'TOP_25pct': '상위 25%',
        'TOP_20pct': '상위 20%',
        'TOP_10pct': '상위 10%',
    }
    sub['Spec_Label'] = sub['Spec'].map(label_map)
    sub['Feature_Label'] = sub['Feature'].map(label_map)
    order = ['상위 25%', '상위 20%', '상위 10%']

    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    sns.lineplot(
        data=sub,
        x='Spec_Label',
        y='OddsRatio',
        hue='Feature_Label',
        marker='o',
        linewidth=2.4,
        sort=False,
        ax=ax,
    )
    ax.axhline(1.0, color='#111827', linestyle='--', linewidth=1.1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel('오즈비(OR)')
    ax.set_xlabel('핫스팟 운영 기준')
    ax.set_title('핫스팟 기준 민감도: 상위 20%는 집중도와 표본 수의 절충 기준', fontsize=14, fontweight='bold', pad=14)
    ax.grid(axis='y', alpha=0.25)
    ax.legend(title='변수')

    counts = sub.drop_duplicates('Spec_Label').set_index('Spec_Label')['Target_1_Count'].to_dict()
    for i, label in enumerate(order):
        if label in counts:
            ax.text(i, ax.get_ylim()[0] + 0.02, f"n={int(counts[label])}", ha='center', va='bottom', fontsize=9, color='#475569')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'stat_03_hotspot_threshold_sensitivity.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_priority_index_components(priority):
    top = priority.head(10).copy()
    top['Label'] = top['YEAR_MONTH'] + '\n' + top['SIDO']

    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.6])

    ax0 = fig.add_subplot(gs[0, 0])
    weights = pd.DataFrame({
        'Component': ['발생률', '증가추세', '경제압박', '반복핫스팟'],
        'Weight': [55, 20, 15, 10],
    })
    ax0.bar(weights['Component'], weights['Weight'], color=['#DC2626', '#F97316', '#F59E0B', '#2563EB'])
    ax0.set_ylim(0, 65)
    ax0.set_ylabel('가중치(%)')
    ax0.set_title('FSIPI 산출 공식', fontsize=13, fontweight='bold')
    ax0.tick_params(axis='x', rotation=20)
    for i, row in weights.iterrows():
        ax0.text(i, row['Weight'] + 2, f"{row['Weight']}%", ha='center', fontweight='bold')

    ax1 = fig.add_subplot(gs[0, 1])
    colors = np.where(top['CLIMATE_CAUTION_FLAG'] == 1, '#DC2626', '#2563EB')
    ax1.barh(top['Label'], top['FSIPI'], color=colors, alpha=0.85)
    ax1.invert_yaxis()
    ax1.set_xlabel('FSIPI 점수')
    ax1.set_title('상위 점검 후보 시도-월', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.25)
    for y, (_, row) in enumerate(top.iterrows()):
        flag = ' 기후주의' if row['CLIMATE_CAUTION_FLAG'] == 1 else ''
        ax1.text(row['FSIPI'] + 0.6, y, f"{row['FSIPI']:.1f}{flag}", va='center', fontsize=8.5)

    fig.suptitle('식품위생 점검 우선순위 지수(FSIPI): 예측모형이 아닌 운영형 정렬 지표', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'stat_04_priority_index_components.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    df = pd.read_csv(os.path.join(MODEL_DIR, 'modeling_dataset.csv'))
    sens = pd.read_csv(os.path.join(MODEL_DIR, 'sensitivity_analysis.csv'))
    priority = pd.read_csv(os.path.join(MODEL_DIR, 'inspection_priority_index.csv'))

    save_correlation_summary(df)
    save_model_spec_sensitivity(sens)
    save_hotspot_threshold_sensitivity(sens)
    save_priority_index_components(priority)
    print('statistical check plots saved')


if __name__ == '__main__':
    main()
