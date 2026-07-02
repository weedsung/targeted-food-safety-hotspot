import pandas as pd
import os
import glob

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

os.makedirs(PROCESSED_DIR, exist_ok=True)

VALID_SIDOS = [
    '서울특별시', '부산광역시', '대구광역시', '인천광역시', '광주광역시',
    '대전광역시', '울산광역시', '세종특별자치시', '경기도', '강원특별자치도',
    '충청북도', '충청남도', '전북특별자치도', '전라남도', '경상북도',
    '경상남도', '제주특별자치도',
]

SIDO_ALIASES = {
    '강원도': '강원특별자치도',
    '전라북도': '전북특별자치도',
    '전북': '전북특별자치도',
    '제주도': '제주특별자치도',
}

def extract_sido(address):
    """주소 문자열에서 유효한 시도명을 안정적으로 추출한다."""
    if not isinstance(address, str):
        return pd.NA

    normalized = address.strip()
    for sido in VALID_SIDOS:
        if normalized.startswith(sido) or f' {sido} ' in f' {normalized} ':
            return sido

    for alias, canonical in SIDO_ALIASES.items():
        if normalized.startswith(alias) or f' {alias} ' in f' {normalized} ':
            return canonical

    return pd.NA

def process_food_safety():
    """식약처 데이터(Y) 전처리: 식품접객업 필터링 및 날짜 처리"""
    print("[1/4] 식약처 데이터(Y) 전처리 시작...")
    path = os.path.join(RAW_DIR, 'food_safety', 'food_safety_violations.csv')
    df = pd.read_csv(path)
    
    # 1. 식품접객업(식당, 카페 등) 필터링
    keywords = '음식|제과|다방|유흥|단란'
    df_filtered = df[df['INDUTY_CD_NM'].str.contains(keywords, na=False)].copy()
    
    # 2. 처분일자(YYYYMMDD)를 datetime으로 변환 후 연-월(YYYY-MM) 추출
    # DSPS_DCSNDT(처분확정일자)가 없는 경우 DSPS_BGNDT(처분시작일) 사용
    df_filtered['DATE_RAW'] = df_filtered['DSPS_DCSNDT'].fillna(df_filtered['DSPS_BGNDT']).astype(str)
    # 잘못된 날짜 형식 처리 (예: 길이가 8이 아닌 경우 등)
    df_filtered['DATE_RAW'] = df_filtered['DATE_RAW'].str.replace(r'[^0-9]', '', regex=True)
    df_filtered = df_filtered[df_filtered['DATE_RAW'].str.len() == 8]
    
    df_filtered['DATE'] = pd.to_datetime(df_filtered['DATE_RAW'], format='%Y%m%d', errors='coerce')
    df_filtered = df_filtered.dropna(subset=['DATE'])
    df_filtered['YEAR_MONTH'] = df_filtered['DATE'].dt.strftime('%Y-%m')
    
    # 지역 추출 (시도 단위): 단순 첫 토큰 추출 대신 유효 시도명 기준 매칭
    df_filtered['SIDO'] = df_filtered['ADDR'].map(extract_sido)

    invalid_sido = df_filtered[df_filtered['SIDO'].isna()].copy()
    invalid_path = os.path.join(PROCESSED_DIR, 'invalid_sido_rows.csv')
    invalid_sido.to_csv(invalid_path, index=False, encoding='utf-8-sig')
    df_filtered = df_filtered[df_filtered['SIDO'].notna()].copy()
    
    print(f"  -> 식품접객업 필터링 완료: {len(df_filtered)}건 (최초 {len(df)}건)")
    print(f"  -> 시도명 추출 실패 검토 파일: {invalid_path} ({len(invalid_sido)}건)")
    return df_filtered

def process_weather():
    """기상청 데이터(X1) 전처리: 월별 평균 및 불쾌지수 계산"""
    print("[2/4] 기상청 데이터(X1) 전처리 시작...")
    path = glob.glob(os.path.join(RAW_DIR, 'kma_weather', '*.csv'))[0]
    df = pd.read_csv(path, encoding='cp949')
    
    # 컬럼명 영어로 통일 (기온, 강수량, 습도 등)
    df.columns = ['STN_ID', 'STN_NM', 'DATE', 'TEMP_AVG', 'TEMP_MIN', 'TEMP_MAX', 'RAIN', 'HUMIDITY']
    
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['YEAR_MONTH'] = df['DATE'].dt.strftime('%Y-%m')
    
    # 불쾌지수(THI) 파생변수 생성 (여름철 식품 부패/위생 불량 위험도 대용)
    # THI = 1.8 * T - 0.55 * (1 - H/100) * (1.8 * T - 26) + 32 (T: 섭씨온도, H: 상대습도%)
    df['THI'] = 1.8 * df['TEMP_AVG'] - 0.55 * (1 - df['HUMIDITY']/100.0) * (1.8 * df['TEMP_AVG'] - 26) + 32
    
    # 월별, 지점별 집계
    monthly_weather = df.groupby(['YEAR_MONTH', 'STN_NM']).agg({
        'TEMP_AVG': 'mean',
        'TEMP_MAX': 'max',
        'HUMIDITY': 'mean',
        'RAIN': 'sum',
        'THI': 'mean'
    }).reset_index()
    
    print(f"  -> 기후 데이터 월별 집계 완료: {len(monthly_weather)}건")
    return monthly_weather

def process_economy():
    """KOSIS 물가 데이터(X2) 전처리: Wide -> Long 형식 변환"""
    print("[3/5] KOSIS 물가 데이터(X2) 전처리 시작...")
    path = os.path.join(RAW_DIR, 'kosis_economy', 'kosis_consumer_price_index.csv')
    
    try:
        df = pd.read_csv(path, encoding='cp949')
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='utf-8')
    
    # 컬럼명이 깨졌을 경우를 대비해 위치 기반으로 지정 (0: 시도, 1: 품목, 2~: 날짜)
    id_vars = [df.columns[0], df.columns[1]]
    date_vars = df.columns[2:]
    
    # Melt: 가로 데이터를 세로 데이터로 변환
    df_long = df.melt(id_vars=id_vars, value_vars=date_vars, var_name='RAW_DATE', value_name='CPI')
    
    # 한글 컬럼명 정리
    df_long.columns = ['SIDO', 'ITEM', 'YEAR_MONTH', 'CPI']
    
    # 시점(YEAR_MONTH) 형식 정리 (2024.01 -> 2024-01)
    df_long['YEAR_MONTH'] = df_long['YEAR_MONTH'].str.replace('.', '-', regex=False).str.strip()
    
    # 주요 항목 명칭 영어로 변경 (유연한 매칭)
    df_long['ITEM'] = df_long['ITEM'].str.strip()
    
    # '총지수'가 포함된 항목 -> CPI_TOTAL
    df_long.loc[df_long['ITEM'].str.contains('총지수'), 'ITEM_ENG'] = 'CPI_TOTAL'
    # '농축수산물'이 포함된 항목 -> CPI_AGRI
    df_long.loc[df_long['ITEM'].str.contains('농축수산물'), 'ITEM_ENG'] = 'CPI_AGRI'
    
    # 분석에 필요한 항목만 필터링
    df_long = df_long[df_long['ITEM_ENG'].notna()].copy()
    df_long['ITEM'] = df_long['ITEM_ENG']
    
    print(f"  -> 물가 데이터 변환 및 매핑 완료: {len(df_long)}건")
    return df_long[['SIDO', 'ITEM', 'YEAR_MONTH', 'CPI']]

def process_mdis():
    """MDIS 전국사업체조사(X3) 전처리: 공간 베이스라인(2023) 추출"""
    print("[4/5] MDIS 전국사업체조사(X3) 전처리 시작...")
    path = glob.glob(os.path.join(RAW_DIR, 'mdis_business', '2023_*.csv'))[0]
    
    # 헤더가 없는 마이크로데이터이므로 임의의 컬럼명 할당
    cols = ['YEAR', 'SIDO_CD', 'SIGUNGU_CD', 'EMD_CD', 'FRANCHISE', 'IND_SECTION', 'EMP_CNT', 'SALES', 'ETC']
    df = pd.read_csv(path, encoding='cp949', header=None, names=cols, dtype=str)
    
    # 'I' (숙박 및 음식점업) 데이터만 필터링
    df_food = df[df['IND_SECTION'] == 'I'].copy()
    
    # 시도 코드(SIDO_CD)를 실제 지역명과 매핑 (주요 코드 기준)
    sido_map = {'11': '서울특별시', '21': '부산광역시', '22': '대구광역시', '23': '인천광역시', 
                '24': '광주광역시', '25': '대전광역시', '26': '울산광역시', '29': '세종특별자치시',
                '31': '경기도', '32': '강원특별자치도', '33': '충청북도', '34': '충청남도',
                '35': '전북특별자치도', '36': '전라남도', '37': '경상북도', '38': '경상남도', '39': '제주특별자치도'}
    df_food['SIDO'] = df_food['SIDO_CD'].map(sido_map)
    
    # 시도별 식당 수(밀집도) 집계 (공간 베이스라인)
    df_agg = df_food.groupby('SIDO').size().reset_index(name='RESTAURANT_DENSITY')
    print(f"  -> 2023년 기준 지역별 식당 밀집도 집계 완료: {len(df_agg)}개 지역")
    return df_agg

def merge_data():
    """모든 전처리된 데이터를 연월(YEAR_MONTH) 기준으로 병합하여 통합 마트 생성"""
    print("\n[5/5] 데이터 병합(Merge) 시작...")
    
    df_y = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_food_safety_filtered.csv'))
    df_x1 = pd.read_csv(os.path.join(PROCESSED_DIR, 'x1_weather_monthly.csv'))
    df_x2 = pd.read_csv(os.path.join(PROCESSED_DIR, 'x2_economy_clean.csv'))
    df_x3 = pd.read_csv(os.path.join(PROCESSED_DIR, 'x3_mdis_2023_baseline.csv'))
    
    # KOSIS(X2) 피벗 처리: 항목(ITEM)을 컬럼으로
    # 중복 제거 및 피벗 (이미 process_economy에서 YEAR_MONTH로 정리됨)
    df_x2_pivot = df_x2.drop_duplicates(['YEAR_MONTH', 'ITEM']).pivot(index='YEAR_MONTH', columns='ITEM', values='CPI').reset_index()
    
    # 기상청(X1)은 일단 전국 월별 평균으로 집계하여 병합 (추후 지점별 매핑 고도화 가능)
    x1_national = df_x1.groupby('YEAR_MONTH').agg({
        'TEMP_AVG': 'mean', 'TEMP_MAX': 'max', 'HUMIDITY': 'mean', 'RAIN': 'mean', 'THI': 'mean'
    }).reset_index()
    
    # Y + X1 병합
    df_merged = pd.merge(df_y, x1_national, on='YEAR_MONTH', how='left')
    # Y + X1 + X2 병합
    df_merged = pd.merge(df_merged, df_x2_pivot, on='YEAR_MONTH', how='left')
    # Y + X1 + X2 + X3(공간 베이스라인) 병합
    df_merged = pd.merge(df_merged, df_x3, on='SIDO', how='left')
    
    output_path = os.path.join(PROCESSED_DIR, 'integrated_data_mart.csv')
    df_merged.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  -> 최종 통합 데이터 마트 생성 완료! 총 {len(df_merged)}건")
    print(f"  -> 경로: {output_path}")

if __name__ == "__main__":
    df_y = process_food_safety()
    df_x1 = process_weather()
    df_x2 = process_economy()
    df_x3 = process_mdis()
    
    df_y.to_csv(os.path.join(PROCESSED_DIR, 'y_food_safety_filtered.csv'), index=False, encoding='utf-8-sig')
    df_x1.to_csv(os.path.join(PROCESSED_DIR, 'x1_weather_monthly.csv'), index=False, encoding='utf-8-sig')
    df_x2.to_csv(os.path.join(PROCESSED_DIR, 'x2_economy_clean.csv'), index=False, encoding='utf-8-sig')
    df_x3.to_csv(os.path.join(PROCESSED_DIR, 'x3_mdis_2023_baseline.csv'), index=False, encoding='utf-8-sig')
    
    merge_data()
    print("\n완료되었습니다! 이제 EDA 및 모델링 단계로 넘어갈 수 있습니다.")
