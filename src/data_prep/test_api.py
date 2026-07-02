import requests

API_KEY = "f3a20cc0476147108a54"
SERVICE_ID = "I0470"

# CHNG_DT 로 테스트 (이후 자료 출력)
url1 = f"http://openapi.foodsafetykorea.go.kr/api/{API_KEY}/{SERVICE_ID}/json/1/1/CHNG_DT=20190101"
res1 = requests.get(url1).json()
print(f"CHNG_DT=20190101 결과 건수: {res1.get(SERVICE_ID, {}).get('total_count', '0')}")

# DSPS_DCSNDT (확정일자) 로 테스트
url2 = f"http://openapi.foodsafetykorea.go.kr/api/{API_KEY}/{SERVICE_ID}/json/1/1/DSPS_DCSNDT=20190101"
res2 = requests.get(url2).json()
print(f"DSPS_DCSNDT=20190101 결과 건수: {res2.get(SERVICE_ID, {}).get('total_count', '0')}")
