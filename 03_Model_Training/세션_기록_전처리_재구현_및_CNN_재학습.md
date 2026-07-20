# 전처리 재구현(ref21order) + Base CNN 재학습 — 세션 종합 기록

작성일: 2026-07-19
관련 문서: [`DEVIATIONS.md`](DEVIATIONS.md), [`클래스_붕괴_분석_및_대응.md`](클래스_붕괴_분석_및_대응.md),
[`용어설명_시드와_클래스붕괴.md`](용어설명_시드와_클래스붕괴.md), [`../PREPROCESSING_DEVIATIONS.md`](../PREPROCESSING_DEVIATIONS.md),
[`../01_Preprocessing/전처리_변형_종합비교.md`](../01_Preprocessing/전처리_변형_종합비교.md)
(원본·v1·v2·z-score 4종 전처리 변형 종합 비교 - v1이 가장 우수)

---

## 1. 전처리에 무엇이 포함됐나

새 스크립트 `01_Preprocessing/스크립트/preparing_ref21order_v1.py`로 기존 `preparing.py`와
별개의 전처리 파이프라인을 만듦. 참고문헌21(같은 저자 그룹의 T2 PD 논문)의 순서를
우선 채택하고, 주논문 Methods 원문("brain extraction, registration, normalization
and data augmentation")의 4단계 순서를 그대로 따름.

**파이프라인 순서**: DICOM/raw NIfTI → **BET**(뇌추출, FSL) → **ANTsPy 정합**
(affine+nonlinear, MNIPD25-T1MPRAGE-1 아틀라스 — 참고문헌21 Table3 그대로) →
**정규화**(min-max 활성, z-score는 코드에 남겨두되 비활성 토글) → **[N4 bias
correction: 기본 비활성]**(주논문 Methods 4단계 목록에 없어서) → **리사이즈 56³**
(주논문 명시값)

- 아틀라스: `00_RawData/atlas/PD25-T1MPRAGE-template-1mm.nii.gz`
  (McGill NIST 공식 배포, https://nist.mni.mcgill.ca/multi-contrast-pd25-atlas/ 에서 다운로드)
- FSL(`bet`)/ANTs(`N4BiasFieldCorrection`)이 네이티브 Windows에 없어서 **WSL(Ubuntu)을
  자동 경유**하도록 구현. FSL 바이너리가 경로에 공백·한글이 있으면 자체적으로 실패하는
  문제를 발견해, WSL 내부의 공백·한글 없는 임시 경로(`/tmp/ppmi_stage_*`)로 파일을
  복사해 처리 후 원래 위치로 되돌리는 방식으로 우회(프로젝트 폴더명 자체는 변경 안 함).
- ANTsPy(ITK)도 별도로 비-ASCII(한글) 경로에서 "Could not create ImageIO object" 에러로
  실패하는 걸 확인 — 비-ASCII 경로면 임시 ASCII 경로로 스테이징하는 방식으로 별도 우회.

## 2. 전처리 배치 실행 — 완료 상태와 소요시간

- 입력: `01_Preprocessing/전처리_0713/01_raw_nifti/` (기존 세션에서 이미 DICOM→NIfTI
  변환까지 끝나 있던 304개 파일 재사용 — 이 환경엔 원본 DICOM 소스가 없어서)
- 출력: `01_Preprocessing/전처리_ref21order_v1/`
- **결과: 304개 중 303개 성공, 1개 실패**
  - 실패 샘플: `sub-40067_I1396169` — "ITK only supports orthonormal direction
    cosines" 에러(이 피험자 NIfTI 헤더의 방향 코사인 행렬이 완전 직교가 아니라
    ITK가 거부함, 실제 임상 데이터에서 가끔 나오는 데이터 품질 이슈)
  - 나머지 303개는 정상 완료
- **총 소요시간: 7074.5초 (약 118분, 2시간 미만)**, 샘플당 평균 20~30초
  (BET 5~6초 + ANTsPy 정합 15~30초)
- 로그: `preprocessing_log_ref21order_20260719_130725.csv`(샘플별 상태·소요시간),
  `run_manifest_20260719_130725.json`(배치 설정 요약)
- 진행 중 발견한 이슈와 수정: WSL stdout 완전 버퍼링으로 실시간 로그가 안 보이던
  문제 → `sys.stdout.reconfigure(line_buffering=True)`로 해결. 샘플별 소요시간이
  콘솔에만 찍히고 CSV엔 누락돼있던 것도 발견해서 `elapsed_sec` 컬럼 추가.

## 3. CNN 학습 과정과 진행 상황

`train_ablation.py --variant base`(`CNN3D_Base`, 8-layer)로 반복 실행. 하이퍼파라미터는
**논문 Table 3 Study 2, Base 열 값을 계속 고정**: `epoch=30, batch_size=32, lr=0.01,
optimizer=Adam` — 이번 세션에서 이 값들을 바꾼 적은 없음.

| 실행 시각 | 전처리 | seed | 보정 | Accuracy | F1(macro) | 비고 |
|---|---|---|---|---|---|---|
| 07-17 20:36 | 원본(0713_v2) | 42 | - | 34.78% | 34.44% | 마지막 epoch 그대로 사용(초기 버전) |
| 07-17 21:31 | 원본(0713_v2) | 42 | - | 47.83% | 43.68% | +gradient clipping, +best-checkpoint |
| 07-18 03:44 | 원본(0713_v2) | 42 | - | 47.83% | 43.68% | 결과 저장 형식만 변경(로직 동일, 재현 확인용) |
| 07-19 13:13 | **새(ref21order)** | 42 | X | 56.52% | 46.76% | Control 붕괴 |
| 07-19 13:18 | **새(ref21order)** | 43 | X | 50.00% | 34.52% | Prodromal 붕괴 |
| 07-19 13:33 | **새(ref21order)** | 42 | **O** | **60.87%** | **61.15%** | 붕괴 없음, 지금까지 최고 |
| 07-19 13:35 | **새(ref21order)** | 43 | **O** | 52.17% | 44.03% | 붕괴 없음(Prodromal 약함) |
| 07-19 13:44 | 원본(0713_v2) | 42 | **O** | 52.17% | 52.80% | 붕괴 없음 |
| 07-19 13:48 | 원본(0713_v2) | 43 | **O** | 39.13% | 28.30% | Prodromal 붕괴 재발 |

(참고: 논문 Study2 목표는 Accuracy 84.96%, Precision 87.34%, Recall 81.54%,
F1 85.32% — 4번 항목에서 왜 이 숫자와 직접 비교가 어려운지 설명)

학습 1회당 소요시간은 약 60~70초. 매 실행 결과는 `results/ablation_base_{시각}_acc{정확도}.json`
파일로 개별 저장(덮어쓰기 없음), `training_log.txt`/`.csv`에도 누적 기록됨.

## 4. `train_ablation.py`에 적용한 수정 내역

**주의**: 아래는 모두 논문이 명시한 하이퍼파라미터(epoch/batch/lr/optimizer)나
모델 구조는 건드리지 않은, "논문 미기재 → 자체 결정" 항목입니다. epoch/batch
자체를 바꾼 적은 없습니다.

1. **Gradient clipping** 추가(`max_norm=1.0`, 기본 활성) — 로짓 폭주 완화 목적
2. **Best-validation-checkpoint 선택** — 기존엔 마지막 epoch(30) 가중치로 test
   평가했는데, 논문 Table5의 "Selected the optimal hyperparameters based on best
   validation performance" 서술에 맞춰 val acc가 가장 높았던 epoch의 가중치로
   test하도록 변경
3. **Classifier 출력 스케일 보정**(`calibrate_classifier_scale`, 기본 활성,
   `--no_calibrate_init`로 끌 수 있음) — 클래스 붕괴 완화 목적. 상세 원인·효과는
   [`클래스_붕괴_분석_및_대응.md`](클래스_붕괴_분석_및_대응.md) 참조
4. **결과 저장 방식**: `ablation_{variant}_result.json` 고정 파일명(재실행 시
   덮어써짐) → `results/ablation_{variant}_{시각}_acc{정확도}.json`으로 변경,
   summary 블록을 파일 맨 앞에 배치해 핵심 지표를 바로 볼 수 있게 함
5. **콘솔 표/txt로그/csv 로그 형식**: "논문 | 우리 모델 | 비고(차이%p)" 3열 →
   "실험값 | 논문값" 2열로 단순화, 차이(%p) 컬럼 제거(사용자 요청 반영)
6. **`--csv_path`/`--image_dir` 기본값**이 stale했던 문제(존재하지 않는
   `data_final_303.csv`, `전처리06_리사이즈_최종`을 가리키고 있었음) 발견 →
   세션 초반엔 CLI 인자로 `data_0713_wsl_v2.csv` + `전처리_0713_v2/06_resized`를
   매번 직접 지정해 우회했으나, **스크립트 기본값 자체는 안 바뀐 채로 남아있었음**.
   2026-07-20, 여러 전처리 변형을 비교해 `전처리_ref21order_v1`(min-max, N4 없음)를
   표준으로 채택하기로 하면서 기본값을 `data_0713_wsl_v2.csv` +
   `전처리_ref21order_v1`로 최종 변경(이제 `--image_dir` 안 줘도 이 경로 사용).

## 5. 주논문과 안 맞는(또는 애매한) 부분 총정리

| 항목 | 주논문 서술 | 실제 구현/발견 사항 |
|---|---|---|
| 전처리 순서 | Methods(p.4): 뇌추출→정합→정규화→증강(bias correction 언급 없음). Results(p.9, ref.31=Smith2002=BET): skull stripping+field correction을 "초기 단계"로 묶어 서술 | 논문 **내부에서 두 서술이 서로 다름**. 새 스크립트는 Methods 순서를 채택(N4 기본 비활성) |
| 뇌추출 알고리즘 | Results는 BET(ref.31) 인용, 그런데 "brain extraction... as described in reference21,22"라고도 함 | **참고문헌21은 ROBEX을 씀** — 논문이 가리키는 두 문헌끼리도 서로 다름. 우리는 BET 채택(Results 직접 인용 + 참고문헌22와 일치) |
| 정합 아틀라스/방식 | 논문 본문엔 구체 아틀라스명 전혀 없음. 참고문헌21은 MNIPD25(PD특화)+ANTsPy affine+nonlinear | 원본 파이프라인은 범용 MNI152+FSL FLIRT affine-only. **새 스크립트에서 처음으로 참고문헌21 방식(MNIPD25+ANTsPy) 구현** |
| 정규화 공식 | 참고문헌21=min-max, 참고문헌22=z-score, 논문 본문엔 명시 없음 | 원본은 z-score(참고문헌22), **새 스크립트는 min-max(참고문헌21)로 전환**해서 비교 중 |
| Study1 vs Study2 비교 기준 | Table3 Study2(84.96%)는 "Grid search + best validation performance 선택" 결과(Table5 명시) | 즉 **64가지 하이퍼파라미터 조합 중 최고값**이라, 우리의 단일 실행(그것도 튜닝 없이)과 액면 그대로 비교하는 건 원래도 무리가 있음(이전 세션에서 논의됨) |
| 데이터 증강 기법 | "data augmentation" 단어만 언급, 구체 기법 전혀 없음 | 회전(±8°)/flip(50%)/스케일(±5%)/노이즈로 자체 결정(기존 `dataset.py`, 이번 세션 변경 없음) |
| 학습 시드 | **논문에 전혀 언급 없음** | seed=42를 프로젝트 자체 결정으로 고정(자체 결정 사항, 이번 세션에 42/43 두 개로 테스트) |
| Classifier 초기화/gradient clipping/스케일 보정 | 전혀 언급 없음 | 전부 자체 결정 — 이번 세션에 추가한 학습 안정화 조치(4번 항목 참조) |

## 6. 종합 결론

- 새 전처리(ref21order) + classifier 스케일 보정을 같이 적용한 조합(seed42)이
  지금까지 실행 중 **가장 높은 성능(Acc 60.87%, F1 61.15%)**을 기록했지만,
  시드 하나만 바꿔도(seed43) F1이 44.03%로 떨어지고, 원본 전처리+보정 조합에서는
  seed43에서 붕괴가 재발함 — **"새 전처리가 확실히 더 낫다"고 결론 내리기엔
  아직 표본(시드 2개)이 부족**함.
- 논문 목표치(84.96%)와는 여전히 큰 격차가 있는데, 이건 (a) Study2 숫자 자체가
  그리드서치 최고값이라는 비교 기준 문제, (b) classifier 구조상 근본적인
  과대적합 소지, (c) 작은 데이터셋(212 train) 등 여러 요인이 겹쳐 있어 단일
  원인으로 설명되지 않음.
- 다음 단계로 논의된 것: 여러 시드로 반복 실행해 평균/분산을 내서 "새 전처리
  효과"와 "학습 불안정성"을 통계적으로 분리해보는 것.
