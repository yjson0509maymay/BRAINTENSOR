# `전처리_ref21order_v1` 상세 기록 (v1, 최종 채택본)

작성일: 2026-07-20
스크립트: `스크립트/preparing_ref21order_v1.py`
출력 폴더: `01_Preprocessing/전처리_ref21order_v1/`
관련 문서: [`../03_Model_Training/DEVIATIONS.md`](../03_Model_Training/DEVIATIONS.md),
[`../PREPROCESSING_DEVIATIONS.md`](../PREPROCESSING_DEVIATIONS.md),
[`전처리_변형_종합비교.md`](전처리_변형_종합비교.md)(원본·v1·v2·z-score 4종 비교),
[`../03_Model_Training/세션_기록_전처리_재구현_및_CNN_재학습.md`](../03_Model_Training/세션_기록_전처리_재구현_및_CNN_재학습.md),
[`../03_Model_Training/용어설명_시드와_클래스붕괴.md`](../03_Model_Training/용어설명_시드와_클래스붕괴.md)(시드·붕괴 개념 설명)

---

## 1. 이게 뭔가 (한 줄 요약)

기존 `preparing.py`와 별개로 만든 전처리 파이프라인. 뇌추출은 기존과 같이 BET을
쓰되, **정합·정규화를 참고문헌21 방식(ANTsPy + PD 특화 아틀라스, min-max)으로
바꾸고, 전체 단계 순서는 주논문 Methods 원문의 4단계 목록을 그대로 따름.**
여러 실험용 변형(v2: N4 추가, zscore: 정규화 방식 변경) 중 **가장 성능이 좋고
안정적이었던 버전**으로, 현재 시점 기준 채택된 버전.

## 2. 파이프라인 — 단계별 상세

```
[raw NIfTI (01_raw_nifti/*.nii.gz)]
     │
     ▼
① BET (FSL bet, -R -f 0.5 -g 0)                    뇌추출
     │
     ▼
② ANTsPy 정합 (ants.registration,                   정합 — PD 특화 아틀라스
   type_of_transform="SyN", MNIPD25-T1MPRAGE-1mm)     로 affine+nonlinear 정합
     │
     ▼
③ Min-max 정규화 ((I-Min)/(Max-Min),                정규화 — 뇌 영역(0 아닌
   뇌 영역 기준, 배경은 0 유지)                         복셀) 기준 [0,1] 범위
     │
     ▼
④ [N4 bias correction — 기본 비활성]                 주논문 Methods 목록에
                                                      없어서 기본적으로 건너뜀
     │
     ▼
⑤ 리사이즈 56×56×56 (scipy.ndimage.zoom,             주논문 명시값
   trilinear, order=1)
     │
     ▼
[최종 출력: 01_Preprocessing/전처리_ref21order_v1/{sample_id}.nii.gz]
```

| 함수 | 역할 | 위치 |
|---|---|---|
| `run_skull_stripping_bet` | ① BET 호출(네이티브 없으면 WSL 경유) | 스크립트 178~197줄 |
| `run_registration_antspy` | ② ANTsPy 정합(비-ASCII 경로 자동 스테이징) | 221~266줄 |
| `normalize_minmax` / `normalize_zscore` | ③ 정규화(토글 가능, 기본 minmax) | 292~338줄 |
| `run_n4_field_correction` | ④ N4(기본 미호출, `--enable-bias-correction` 시에만) | 269~281줄 |
| `resize_nifti` | ⑤ 리사이즈 | 341~352줄 |
| `preprocess_one_from_nifti` | ①~⑤ 전체 순서 조립 | 448~491줄 |
| `run_batch_from_nifti` | 전체 배치 실행 + 로그/매니페스트 기록 | 606~691줄 |

## 3. 각 선택의 근거

| 항목 | 채택 값 | 근거 |
|---|---|---|
| 뇌추출 도구 | **BET** (FSL, frac=0.5, robust) | 주논문 Results(463~465줄, 각주31=Smith 2002=BET 원 논문) 직접 인용 + 참고문헌22와 일치. 참고문헌21은 ROBEX을 쓰지만 이번 변경 범위 아님(기존 유지) |
| 정합 도구/아틀라스 | **ANTsPy**(`type_of_transform="SyN"`) + **MNIPD25-T1MPRAGE-1mm**(PD 특화 아틀라스) | 참고문헌21 Table3 Step3 원문: "Use Antspyx to align the image with the atlas" 그대로. 주논문 본문엔 정합 도구/아틀라스명이 전혀 없어 참고문헌21을 근거로 채택 |
| 정규화 공식 | **Min-max**: `(I-Min)/(Max-Min)`, 뇌 영역 기준 | 참고문헌21 Eq.1-3 그대로. z-score(참고문헌22 방식)도 코드에 구현돼있으나 min-max로 채택(→ 4번 항목) 기본값으로 채택 |
| Bias correction(N4) | **기본 비활성** | 주논문 Methods(217~218줄) "brain extraction, registration, normalization and data augmentation" 4단계 목록에 bias correction이 없음. N4를 뇌추출 직후 넣어본 버전(v2)이 오히려 성능이 낮고 불안정해서(→ `전처리_변형_종합비교.md`) 이 판단이 실험적으로도 뒷받침됨 |
| 리사이즈 크기 | 56×56×56 | 주논문 명시값(변경 없음) |

## 4. 실험적으로 검증된 것 (다른 버전과 비교)

이 폴더(min-max, N4 없음)가 채택된 건 그냥 "논문 근거가 더 그럴듯해서"가 아니라
**실제로 CNN 학습 결과가 가장 좋았기 때문**임:

| 변형 | 정규화 | N4 위치 | 평균 Accuracy(2시드) | 평균 F1(2시드) |
|---|---|---|---|---|
| **이 폴더 (전처리_ref21order_v1)** | min-max | 없음 | **56.52%** | **52.59%** |
| 전처리_ref21order_v2 | min-max | BET 직후(항상) | 46.74% | 41.57% |
| 전처리_ref21order_zscore | z-score | 없음 | 47.27% | 33.73% |

시드별 상세(zscore): seed42 Acc 45.65%/F1 20.90%(**Control+Prodromal 동시 붕괴**,
PD만 recall 1.00로 살아남음) / seed43 Acc 48.89%/F1 46.56%(붕괴 없음). 세 변형
중 min-max(이 폴더)가 평균 Accuracy·F1 모두 가장 높고 붕괴 빈도도 가장 낮음 —
참고문헌21의 min-max 정규화 채택이 실험적으로도 뒷받침됨.

(원본 전처리(`전처리_0713_v2`, z-score+FLIRT+MNI152)와의 비교는
`../03_Model_Training/세션_기록_전처리_재구현_및_CNN_재학습.md` 참조)

## 5. 배치 실행 결과 (2026-07-19)

- 입력: `전처리_0713/01_raw_nifti/` (304개 파일, DICOM→NIfTI 변환은 이전 세션에서 완료된 것 재사용)
- **결과: 304개 중 303개 성공(292 신규 처리 + 11 이전 부분실행에서 이미 완료), 1개 실패**
  - 실패: `sub-40067_I1396169` — ITK "orthonormal direction cosines" 에러(이 피험자
    NIfTI 헤더의 방향 코사인 행렬이 완전 직교가 아니라 ITK가 거부 — 데이터 자체 결함,
    파이프라인 버그 아님. v2/zscore 버전에서도 같은 샘플이 재현됨)
- **총 소요시간: 7074.5초(약 118분)**, 샘플당 평균 20~30초(BET 5~6초 + ANTsPy 정합 15~25초)
- 로그: `preprocessing_log_ref21order_20260719_130725.csv`(샘플별 상태·소요시간),
  `run_manifest_20260719_130725.json`(배치 설정 요약, 아래 원문)

```json
{
  "timestamp": "20260719_130725",
  "normalization": "minmax",
  "enable_bias_correction": false,
  "target_shape": [56, 56, 56],
  "bet_frac": 0.5,
  "total": 304, "success": 292, "failed": 1, "skipped": 11,
  "elapsed_sec": 7074.5
}
```

## 6. 실행 중 발견/해결한 환경 이슈

1. **FSL(`bet`)/ANTs CLI가 네이티브 Windows에 없음** — WSL(Ubuntu) 안에만 설치돼
   있어서, 스크립트가 자동으로 `wsl bash -c ...`를 경유하도록 구현(`run_via_wsl_clean_staging`).
2. **FSL 바이너리가 경로에 공백·한글이 있으면 자체적으로 실패**함(셸 따옴표와 무관 —
   `fslinfo`로 직접 확인). WSL 내부 `/tmp/ppmi_stage_<uuid>`(공백·한글 없는 경로)로
   파일을 복사해 처리 후 원래 위치로 되돌리는 방식으로 우회. 프로젝트 폴더명(`전처리_ref21order_v1`
   등 한글)은 전혀 바꾸지 않음.
3. **ANTsPy(ITK)도 별도로 비-ASCII(한글) 경로에서 "Could not create ImageIO object"
   에러**로 실패 — 처음엔 몰랐다가 실제 배치에서 전 샘플이 실패하면서 발견함. FSL과
   원인이 다르므로 별도 우회 로직(`_stage_ascii_in`) 추가.
4. **FSL은 `FSLOUTPUTTYPE` 환경변수가 없으면 무조건 실패** — WSL 호출 시 항상
   `FSLOUTPUTTYPE=NIFTI_GZ`를 명시적으로 지정하도록 수정.
5. **stdout 완전 버퍼링**으로 배치 실행 중 로그가 실시간으로 안 보이던 문제 —
   `sys.stdout.reconfigure(line_buffering=True)`로 해결.

## 7. 사용 방법 (재현 커맨드)

```bash
python preparing_ref21order_v1.py \
  --from-nifti-dir "01_Preprocessing/전처리_0713/01_raw_nifti" \
  --output-dir "01_Preprocessing/전처리_ref21order_v1" \
  --atlas-path "00_RawData/atlas/PD25-T1MPRAGE-template-1mm.nii.gz"
  # --normalization minmax (기본값, 생략 가능)
  # --enable-bias-correction 는 주지 않음(기본 비활성 유지)
```

`--overwrite` 없이 실행하면 이미 존재하는 출력 파일은 자동 스킵(이어서 재실행 가능).

## 8. 이 데이터로 CNN 학습한 결과 (참고)

`train_ablation.py --variant base`, classifier 초기화 스케일 보정 적용 기준:

| seed | Accuracy | F1(macro) | 클래스 붕괴 |
|---|---|---|---|
| 42 | 60.87% | 61.15% | 없음 |
| 43 | 52.17% | 44.03% | 없음(Prodromal 약함) |

상세는 `../03_Model_Training/클래스_붕괴_분석_및_대응.md`,
`../03_Model_Training/세션_기록_전처리_재구현_및_CNN_재학습.md` 참조.
