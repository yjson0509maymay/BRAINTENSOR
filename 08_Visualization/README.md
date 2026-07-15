# 08_Visualization

전처리 결과 QC용 시각 자료(컨택트시트, 단계별 비교 이미지)를 모아두는 폴더입니다.

## 현재 상태

이전 컨택트시트(`qc_0713_final/`)는 재생성 예정이라 비워둔 상태입니다. 303명 전체를 슬라이스 레벨(뇌실 z=0.5, 중뇌 z=0.3)로 육안 검사한 컨택트시트를 다시 만들면 이 폴더에 채워 넣으세요.

## 생성 방법

`01_Preprocessing/스크립트/qc_check.py`의 `slice_contact_sheets()` / `stage_montage()` 함수로 생성합니다.

```bash
python ../01_Preprocessing/스크립트/qc_check.py \
  --data-csv ../01_Preprocessing/data_0713.csv \
  --stage-root ../01_Preprocessing/전처리_0713 \
  --qc-out .
```

QC 방법론(수치 검사는 놓치는 회전·형태 왜곡을 컨택트시트 육안 검사로 보완하는 이유 등)은 [`../01_Preprocessing/README.md`](../01_Preprocessing/README.md)와 `qc_check.py` 상단 주석 참고.
