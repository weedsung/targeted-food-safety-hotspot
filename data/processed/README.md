# Processed Data Policy

행 단위 처리 데이터 CSV는 공개 GitHub 저장소에 포함하지 않습니다.

이 프로젝트의 원자료 및 처리 데이터에는 대용량 공공용 마이크로데이터, 행정처분 세부 행, 전화번호 컬럼 등 공개 저장소에 그대로 올리기 부적절한 필드가 포함될 수 있습니다.

재현 시에는 `src/data_prep/data_preprocessing.py`를 사용해 원자료에서 다시 생성합니다. 공개 저장소에는 집계 모델 결과(`output/models/*.csv`), 시각화, 최종 포스터 PDF만 포함합니다.
