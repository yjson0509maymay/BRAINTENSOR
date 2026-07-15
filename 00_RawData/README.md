# 00_RawData

주논문(Priyadharshini et al., 2024, *Scientific Reports* 14:23394) 재현에 쓰인 PPMI 원본 데이터 안내 폴더입니다.

## 이 폴더에 실제로 들어있는 것

- `README_원본데이터안내.txt` — 예전 버전 안내문 (이 README로 대체됨)

## 실제 원본 데이터는 여기서 제외됨

원본 PPMI DICOM(303명, 약 4.5GB)과 IDA 메타데이터 XML은 용량 문제로 이 저장소(GitHub)에는 포함하지 않았습니다. 로컬에서는 다음 경로에 있습니다.

```text
00_RawData/303_1713/303_0713_dataset/
├── data/PPMI/<Subject>/<Description>/<AcqDate>/<ImageDataID>/*.dcm       # 원본 DICOM, 303명
└── metadata/PPMI/<Subject>/<Description>/<AcqDate>/<ImageDataID>/*.xml  # IDA 메타데이터, 303명
```

- 303명 전원 Baseline(BL) 방문, T2 계열(Axial/FLAIR) 시퀀스
- Control 110 / Prodromal 58 / PD 135 (주논문 Table과 동일 구성)
- 코호트 전체 목록과 컬럼 설명은 [`01_Preprocessing/data_0713.csv`](../01_Preprocessing/data_0713.csv) 참고

## 원본 데이터를 직접 받아야 하는 경우

1. [PPMI/IDA](https://ida.loni.usc.edu/)에서 `data_0713.csv`의 `Subject` + `Image Data ID` 조합으로 동일 이미지를 다운로드
2. `data/PPMI/<Subject>/...` 구조로 배치
3. `data_0713.csv`의 `raw_dicom_dir` 컬럼이 이 경로를 가리키므로, 다른 위치에 두었다면 해당 컬럼을 새 경로로 갱신해야 `01_Preprocessing/스크립트/run_staged_preprocessing.py`가 정상 동작합니다
