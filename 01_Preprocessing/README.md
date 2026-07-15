# 01_Preprocessing

PPMI 원본 DICOM을 주논문이 서술한 6단계 전처리(뇌 추출, N4 bias correction, MNI152 정합, 강도 정규화, 56³ 리사이즈)로 변환하는 코드와 결과물입니다.

## 구성

| 경로 | 내용 |
|---|---|
| `data_0713.csv` | 최종 303명 코호트 마스터 목록. `sample_id, Subject, Image Data ID, Group, label, Sex, Age, Visit, Description, Acq Date, raw_dicom_dir, dicom_file_count` |
| `스크립트/run_staged_preprocessing.py` | `data_0713.csv`를 읽어 6단계 전처리를 배치 실행하는 메인 스크립트 |
| `스크립트/preparing.py`, `brainprep_pipeline.py` | BET/N4/FLIRT/정규화/리사이즈 등 단계별 함수 구현 |
| `스크립트/qc_check.py` | 전처리 결과 QC 도구 (수치 검사 + 컨택트시트 시각 검사 + 문제 단계 자동 추적) |
| `전처리_0713/` | 6단계 전처리 산출물 (`01_raw_nifti` ~ `06_resized`, 각 303개 `.nii.gz`). **용량이 커서(6.7GB) 이 저장소에는 포함하지 않음** — `.gitignore` 참고 |
| `로그_QC_0713/` | QC 로그 및 이슈 트래커 (`303_이슈트래커_및_최종명단.xlsx` 등) |
| `전처리_0713_개별파라미터조정.md` | 3건의 QC 문제 케이스(sub-4126, sub-40893, sub-71093)에 적용한 파라미터 조정 내역과 근거 |
| `303_rerun_wsl_README.md`, `PPMI_작업리스트.txt` | 재실행 과정 메모 |

## 코호트 구성

Control 110 / Prodromal 58 / PD 135 = 303명, 전원 Baseline 방문, T2 계열 시퀀스. 주논문이 보고한 그룹 구성과 정확히 일치합니다.

## 원본 데이터 위치

`data_0713.csv`의 `raw_dicom_dir` 컬럼은 `../00_RawData/303_1713/303_0713_dataset/data/PPMI/...`를 가리킵니다. 원본 DICOM 자체는 용량 문제로 이 저장소에 포함되지 않았으므로, 처음부터 재실행하려면 `00_RawData/README.md`를 참고해 원본을 먼저 준비해야 합니다.

## 재실행 방법

```bash
python 스크립트/run_staged_preprocessing.py \
  --data-csv data_0713.csv \
  --output-root 전처리_0713 \
  --mni-template <MNI152 템플릿 경로> \
  --preparing-path 스크립트/preparing.py
```

이미 존재하는 단계 출력은 건너뛰므로(`--overwrite` 미지정 시) 중단된 지점부터 이어서 실행됩니다.

## 알려진 배포 관행

주논문 자체 인용(ref.31=Smith 2002)에 따라 뇌 추출은 FSL BET을 사용했습니다. 논문에 미기재된 세부값(stride/padding 등)은 최상위 `PREPROCESSING_DEVIATIONS.md`에 근거와 함께 기록되어 있습니다.
