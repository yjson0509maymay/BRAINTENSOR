# PPMI T2 MRI 논문 재현 데이터 패키지

## 개요

이 패키지는 주논문의 T2-weighted MRI 전처리 및 모델링 과정을 재현하기 위한 최종 데이터와 관련 문서입니다.

- 전체 피험자: 303명
- Control: 110명
- Prodromal: 58명
- Parkinson's disease (PD): 135명
- 최종 영상: 3D NIfTI (.nii.gz), 56 x 56 x 56 voxel
- 정합 공간: MNI152
- 모델 입력 폴더: `06_resized/`

원본 DICOM과 1~5단계 중간 결과는 용량을 줄이기 위해 포함하지 않았습니다. `06_resized/`는 모델 학습에 바로 사용할 수 있는 최종 전처리 결과입니다.

## 업로드 파일과 이름 대응표

현재 이름을 그대로 사용해도 됩니다. 더 명확한 이름을 원할 때만 권장 이름을 사용하십시오.

| 현재 이름 | 선택적 권장 이름 | 내용 |
|---|---|---|
| `06_resized/` | `t2_mni152_56x56x56/` | 뇌 추출, N4 보정, MNI152 정합, 강도 정규화 및 56³ 리사이즈가 완료된 NIfTI 303개 |
| `data.csv` | `cohort_metadata.csv` | 영상과 피험자, PPMI Image ID, 진단군 및 숫자 라벨을 연결하는 메타데이터 |
| `preparing.py` | `preprocessing_core.py` | DICOM 변환, BET, N4, MNI152 정합, 정규화, 리사이즈 및 증강 함수 |
| `preprocessing_log.csv` | `preprocessing_manifest.csv` | 303개 표본의 전처리 상태와 영상 형태를 기록한 QC 로그 |
| `requirements.txt` | 변경하지 않음 | 전처리에 사용한 Python 패키지와 고정 버전 |
| `run_staged_preprocessing.py` | `run_preprocessing_pipeline.py` | 전처리를 단계별로 실행하고 결과와 오류를 기록하는 코드 |
| `주논문_nature.pdf` | `reference_paper_nature.pdf` | 전처리와 모델 재현 기준이 된 주논문 |
| `README.md` | 변경하지 않음 | 패키지 구성과 사용 주의사항 |

기존 코드와의 호환성을 위해 현재 이름을 유지하고 이 README를 함께 보관하는 방법을 권장합니다.

## 권장 구조

```text
ppmi_t2_project/
├── README.md
├── 06_resized/
│   ├── sub-3008_I366281.nii.gz
│   ├── ...
│   └── 총 303개 NIfTI
├── data.csv
├── preprocessing_log.csv
├── preparing.py
├── run_staged_preprocessing.py
├── requirements.txt
└── 주논문_nature.pdf
```

## 영상 파일명과 CSV 연결

파일명 규칙은 `sub-{Subject}_{Image Data ID}.nii.gz`입니다. 예: `sub-3008_I366281.nii.gz`

확장자를 제외한 이름이 두 CSV의 `sample_id`와 일치합니다.

**중요:** `06_resized/` 내부의 개별 NIfTI 파일명은 바꾸지 마십시오. 변경하면 `data.csv`와 `preprocessing_log.csv`의 `sample_id`도 함께 바꿔야 합니다. 상위 폴더명만 바꾸는 것은 가능합니다.

## data.csv

303개 행과 다음 열을 포함합니다.

| 열 | 의미 |
|---|---|
| `sample_id` | NIfTI 파일명과 연결되는 표본 ID |
| `Subject` | PPMI 피험자 ID |
| `Image Data ID` | PPMI 영상 시리즈 ID |
| `Group` | `Control`, `Prodromal`, `PD` |
| `label` | Control = 0, Prodromal = 1, PD = 2 |
| `Sex`, `Age` | 성별과 촬영 당시 나이 |
| `Visit` | PPMI 방문 시점; 이 코호트는 baseline으로 구성 |
| `Description` | 원본 MRI 시리즈 설명 |
| `Acq Date` | 촬영일 |
| `raw_dicom_dir` | 전처리에 사용했던 원래 컴퓨터의 DICOM 경로 |
| `dicom_file_count` | 선택된 시리즈의 DICOM 파일 수 |

`raw_dicom_dir`의 `E:\ppmi_dti\...` 경로는 다른 컴퓨터에 존재하지 않으며 최종 영상 학습에는 사용하지 않습니다. 다른 컴퓨터에서는 `sample_id + ".nii.gz"`로 `06_resized/`에서 영상을 찾으십시오.

## preprocessing_log.csv

303개 표본의 ID, 진단군, 처리 상태, 오류 메시지, echo 선택, 원본/최종 형태와 단계별 시간을 기록합니다. 일부 시간 칸이 비어 있어도 실패를 뜻하지 않습니다. `status`와 `message`로 판단하십시오.

전달 데이터는 303개 모두 존재하며 전부 56 x 56 x 56으로 검증되었습니다.

## 전처리 순서

1. DICOM을 3D NIfTI로 변환
2. 논문이 인용한 FSL BET 방식으로 뇌 추출
3. ANTs N4 bias-field correction
4. FSL FLIRT 12-DOF affine 방식으로 MNI152 표준 공간에 정합
5. 0이 아닌 뇌 voxel에 z-score 정규화 후 [-5, 5]로 제한
6. 56 x 56 x 56으로 리사이즈

`06_resized/`는 6단계까지 완료된 모델 입력입니다.

## 데이터 분할과 증강

전달된 NIfTI에는 데이터 증강이 미리 적용되지 않았습니다. 먼저 `Subject` 단위로 train, validation, test를 분리한 다음 **train 세트에만** 증강을 적용하십시오. 분할 전에 증강하면 데이터 누출이 발생할 수 있습니다. 가능하면 세 진단군 비율도 계층화하여 유지하십시오.

## 환경 설치

기록된 환경은 Python 3.12.3입니다.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

원본 DICOM부터 다시 전처리할 때 사용한 외부 도구는 FSL 6.0, ANTs 2.6.5, dcm2niix v1.0.20260416입니다. 이 도구들은 최종 NIfTI로 모델을 학습할 때는 필요하지 않습니다.

PyTorch, TensorFlow 또는 MONAI 같은 학습 패키지는 모델 구현을 선택한 뒤 `requirements.txt`에 추가해야 합니다.

## 다른 컴퓨터에서 확인할 사항

- `06_resized/`에 NIfTI가 정확히 303개 있는가?
- 모든 `data.csv.sample_id`에 대응하는 영상이 있는가?
- 모든 영상이 56 x 56 x 56인가?
- 진단군이 Control 110, Prodromal 58, PD 135인가?
- 증강 전에 피험자 단위 분할을 완료했는가?

## 전처리를 처음부터 재실행할 경우

- `preparing.py`: 개별 전처리 함수
- `run_staged_preprocessing.py`: 단계별 배치 실행과 로그 기록

전체 전처리를 다시 실행하려면 업로드 파일 외에 원본 PPMI DICOM과 MNI152 템플릿이 필요합니다. 이 대용량 원본 자료는 현재 클라우드 패키지에 포함되지 않았습니다.
