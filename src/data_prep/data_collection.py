import requests
import pandas as pd
import os

API_KEY = "f3a20cc0476147108a54"
SERVICE_ID = "I0470"  # 행정처분결과 (전체)
FILE_PATH = "data/raw/food_safety_violations.csv"

def fetch_data():
    os.makedirs(os.path.dirname(FILE_PATH), exist_ok=True)
    all_data = []
    start_idx = 1
    chunk_size = 500
    
    print("API 연결 테스트 중...")
    # 첫 번째 요청으로 전체 데이터 건수 확인
    url = f"http://openapi.foodsafetykorea.go.kr/api/{API_KEY}/{SERVICE_ID}/json/1/1"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"API 요청 실패: HTTP {response.status_code}")
        print(f"응답 내용: {response.text}")
        return
        
    try:
        data = response.json()
    except Exception as e:
        print(f"JSON 파싱 실패! 응답 내용:\n{response.text}")
        return
        
    if SERVICE_ID not in data:
        print("API 응답 오류 또는 키/권한 문제:", data)
        return
        
    print("성공 응답 구조:", list(data[SERVICE_ID].keys()))
    
    try:
        total_count = int(data[SERVICE_ID].get('total_count', data[SERVICE_ID].get('list_total_count', 0)))
    except Exception as e:
        print("카운트 추출 실패:", data[SERVICE_ID])
        return
        
    print(f"총 {total_count}건의 적발 데이터가 확인되었습니다!")
    
    fetch_count = total_count
    print(f"전체 {fetch_count}건의 데이터를 모두 수집합니다. (API 호출이 많아 시간이 다소 소요될 수 있습니다)")
        
    for i in range(1, fetch_count + 1, chunk_size):
        end_idx = min(i + chunk_size - 1, fetch_count)
        print(f"데이터 수집 중... ({i} ~ {end_idx})")
        
        req_url = f"http://openapi.foodsafetykorea.go.kr/api/{API_KEY}/{SERVICE_ID}/json/{i}/{end_idx}"
        res = requests.get(req_url).json()
        
        rows = res.get(SERVICE_ID, {}).get('row', [])
        all_data.extend(rows)
        
    df = pd.DataFrame(all_data)
    df.to_csv(FILE_PATH, index=False, encoding='utf-8-sig')
    print(f"✅ 성공적으로 {len(df)}건의 데이터를 '{FILE_PATH}'에 저장했습니다.")

if __name__ == "__main__":
    fetch_data()
