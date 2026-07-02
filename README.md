# Targeted Food Safety Hotspot Analysis

국가데이터 활용대회 포스터 제출을 위한 식품접객업 위생 핫스팟 분석 프로젝트입니다.

핵심 목표는 절대 위반 건수 중심의 단속 관행을 보완하는 것입니다. 식당 수가 많은 지역은 위반 건수도 자연스럽게 많아지므로, 이 프로젝트는 `식당 1,000개당 위반율`을 기준으로 고위험 시도-월을 재정의하고, 경제/기후/공간 지표를 결합해 점검 우선순위를 제안합니다.

## Key Outputs

- Final poster PDF: `output/pdf/food_safety_statistical_poster.pdf`
- Defensible poster wording: `docs/poster_defensible_storyline.md`
- Data reliability audit: `docs/data_reliability_audit.md`
- Before/after model comparison: `output/models/model_comparison_before_after.csv`
- Model summary: `output/models/model_training_summary.csv`

## Current Statistical Summary

After fixing path handling and region normalization:

| Feature | Odds Ratio | p-value | Interpretation |
| --- | ---: | ---: | --- |
| `CPI_TOTAL` | 1.39 | 0.090 | Policy signal: hotspot odds increase direction |
| `THI` | 0.76 | 0.062 | Negative association under restaurant-normalized hotspot target |
| `RESTAURANT_DENSITY` | 1.00 | 0.980 | No direct effect on hotspot rate after normalization |
| `CPI_AGRI` | 0.62 | 0.013 | Significant but interpreted cautiously due to CPI multicollinearity |

LightGBM validation AUC is 0.537. Therefore, the model is presented as a targeting support prototype, not as a high-performance automated decision model.

## Reproduction

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run preprocessing:

```powershell
python src\data_prep\data_preprocessing.py
```

Train models and generate summary tables:

```powershell
python src\modeling\train_model.py
```

Regenerate charts:

```powershell
python src\visualization\generate_eda_plots.py
python src\visualization\generate_advanced_plots.py
python src\visualization\plot_regional_risk.py
python src\visualization\plot_industry_sensitivity.py
python src\visualization\plot_violation_text.py
python src\visualization\plot_model_forest.py
python src\visualization\plot_whatif_scenario.py
```

Build the poster PDF:

```powershell
python src\visualization\build_poster_pdf.py
```

## Public Data Policy

Raw data and row-level processed CSV files are excluded from the public repository. They may include large public-use microdata, administrative row details, or fields inappropriate for direct public GitHub upload. See `data/processed/README.md` and `docs/publication_notes.md`.

## Project Structure

```text
data/
  processed/README.md
docs/
  data_reliability_audit.md
  poster_defensible_storyline.md
  publication_notes.md
output/
  dashboard/
  eda_plots/
  models/
  pdf/
src/
  data_prep/
  modeling/
  visualization/
```
