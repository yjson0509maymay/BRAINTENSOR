# 전처리 ref21order v1 vs v2 — N4 위치 비교

작성일: 2026-07-19
관련 스크립트: `스크립트/preparing_ref21order.py`(v1), `스크립트/preparing_ref21order_v2.py`(v2)
관련 문서: [`../PREPROCESSING_DEVIATIONS.md`](../PREPROCESSING_DEVIATIONS.md),
[`../03_Model_Training/세션_기록_전처리_재구현_및_CNN_재학습.md`](../03_Model_Training/세션_기록_전처리_재구현_및_CNN_재학습.md)

---

## 1. 왜 두 버전이 생겼나 (근거)

주논문이 전처리를 설명하는 문장이 **본문 내 두 군데서 서로 다름** (상세 설명은
`03_Model_Training/세션_기록_...md` 5번 항목 참조):

| 위치 | 원문 | 포함된 단계 |
|---|---|---|
| Methods, p.4 (217~218줄) | "brain extraction, registration, normalization and data augmentation as described in the reference21,22" | 뇌추출 → 정합 → 정규화 → 증강. **bias correction 언급 없음** |
| Table3 앞, p.9 (463~465줄, 각주31=Smith 2002=BET) | "The NifTi file is further subjected to pre-processing, which includes **skull stripping and field correction**... Finally, the input size is converted to 56x56x56" | 뇌추출 + **field correction**을 묶어 초기 단계로 서술 |

**v1**은 Methods 목록을 우선시(N4 기본 비활성) / **v2**는 Table3 앞 서술을 우선시
(N4를 뇌추출 직후 항상 적용) — 어느 근거가 실제 성능에 더 부합하는지 실험적으로
비교하기 위해 별도 스크립트로 나눠 실행함.

## 2. 파이프라인 순서 비교

| 단계 | v1 (`preparing_ref21order.py`) | v2 (`preparing_ref21order_v2.py`) |
|---|---|---|
| 1 | BET(뇌추출) | BET(뇌추출) |
| 2 | ANTsPy 정합(MNIPD25, affine+nonlinear) | **N4 bias correction** |
| 3 | 정규화(min-max) | ANTsPy 정합(MNIPD25, affine+nonlinear) |
| 4 | N4 — **기본 비활성**(옵션으로만 켤 수 있음, 켜면 정규화 이후) | 정규화(min-max) |
| 5 | 리사이즈 56³ | 리사이즈 56³ |

**바뀐 건 N4의 유무·위치뿐**입니다. 뇌추출 도구(BET), 정합 도구/아틀라스(ANTsPy+MNIPD25),
정규화 공식(min-max)은 v1·v2 동일 — 이 부분은 이번 비교의 변수가 아님.

## 3. 현재(v2) 전처리 과정 — 단계별 상세

```
[raw NIfTI]
     │
     ▼
① BET (FSL, frac=0.5, robust)               ─ 뇌추출. 주논문 Results(ref.31=Smith 2002)
     │                                          + 참고문헌22와 일치하는 선택
     ▼
② N4BiasFieldCorrection (ANTs)               ─ 밝기 불균일 보정. 주논문 Table3 앞
     │                                          서술(뇌추출+field correction을 초기
     │                                          단계로 묶음) 근거로 이 위치에 배치
     ▼
③ ANTsPy 정합 (SyN: affine+nonlinear,        ─ 참고문헌21 Table3 Step3 그대로.
   MNIPD25-T1MPRAGE-1 아틀라스)                 PD 특화 아틀라스
     │
     ▼
④ Min-max 정규화 ((I-Min)/(Max-Min),         ─ 참고문헌21 Eq.1-3 그대로
   뇌 영역 기준)
     │
     ▼
⑤ 리사이즈 56×56×56 (trilinear)              ─ 주논문 명시값
     │
     ▼
[최종 학습용 NIfTI]  →  01_Preprocessing/전처리_ref21order_v2/
```

(FSL/ANTs 네이티브 실행파일이 없는 환경이라 ①②는 내부적으로 WSL을 자동 경유하고,
경로에 한글/공백이 있으면 임시 클린 경로로 스테이징 후 결과만 되돌리는 방식으로
동작 — 프로젝트 폴더 구조·이름은 그대로 유지됨. 상세는 스크립트 상단 주석 참조.)

## 4. 결과 비교

| | v1 (N4 비활성, Methods 순서) | v2 (N4, BET 직후, Table3 앞 서술 순서) |
|---|---|---|
| 전처리 성공/실패 | 303/304 성공(1건 ITK 헤더 오류, `sub-40067_I1396169`) | 303/304 성공(**같은** 샘플이 이번엔 N4 단계에서 실패 - 동일 원인) |
| 전처리 소요시간 | 7074.5초(약 118분) | 8266.4초(약 138분, N4 단계 추가로 더 걸림) |
| Base CNN Accuracy (seed42) | **60.87%** | 43.48% |
| Base CNN F1 macro (seed42) | **61.15%** | 37.71% |
| 클래스 붕괴 (seed42) | 없음 | **PD 붕괴**(precision/recall/f1 전부 0.00) |
| Base CNN Accuracy (seed43) | **52.17%** | 50.00% |
| Base CNN F1 macro (seed43) | **44.03%** | 45.43% |
| 클래스 붕괴 (seed43) | 없음(Prodromal 약함: 0.33/0.12/0.18) | 없음(Prodromal 약함: 0.40/0.25/0.31) |
| **평균 Accuracy(2시드)** | **56.52%** | 46.74% |
| **평균 F1 macro(2시드)** | **52.59%** | 41.57% |

(둘 다 classifier 초기화 스케일 보정 적용 상태에서 비교. 논문 Study2 목표: Acc 84.96%, F1 85.32%)

### 결론

**N4 bias correction을 뇌추출 직후(정합 이전)에 추가한 v2가, 안 넣은 v1보다
오히려 성능이 낮고 불안정함.** 2개 시드 평균으로 Accuracy -9.8%p, F1 -11.0%p
차이가 났고, v2 seed42에서는 v1에선 안 보이던 클래스 완전 붕괴(PD)까지
재발함. 표본이 시드 2개뿐이라 확정적 결론은 아니지만, 적어도 이번 데이터·
설정에서는 **"주논문 Methods 목록(bias correction 없음)을 따르는 v1 쪽이
더 나은 선택"**으로 잠정 판단. Table3 앞 서술(뇌추출+field correction을
초기 단계로 묶은 문장)을 굳이 우선시할 근거가 실험적으로는 약해진 셈.

같은 샘플(`sub-40067_I1396169`)이 v1·v2 양쪽에서 실패했다는 점도 이 판단과는
무관한 별개 이슈 — 데이터 자체의 헤더 결함이라 어느 파이프라인을 쓰든
재현됨(우연이 아님).
