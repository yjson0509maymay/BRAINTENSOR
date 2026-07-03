# 모델링 코드 재작성 - 논문과의 차이점 및 결정 사항 기록

작성일: 2026-07-03
대상: 03_Model/models.py, dataset.py, classifiers.py, train.py, smoke_test.py,
      02_FeatureEngineering/fusion.py, feature_optimization.py

프로젝트 지침 6항("새로운 전처리나 임의의 변경은 적용하지 말고, 필요한 경우 반드시
이유를 명시한 후 진행")에 따라, 논문에 명시되지 않아 자체적으로 결정한 모든 사항을
이 문서에 기록합니다. 논문에 명시된 값은 모두 05_Document/모델_아키텍처_분석.md를
그대로 따랐습니다.

## 1. 3D-CNN (models.py: CNN3D)

| 항목 | 논문 명시 여부 | 채택 값 | 근거/이유 |
|---|---|---|---|
| 필터 진행(32-64-128-256-512-1024-512-256) | 명시(Figure 2) | 그대로 사용 | - |
| 커널 크기 (3,3,3) | 명시(Figure 2) | 그대로 사용 | - |
| MaxPool (2,2,2) x3 | 명시(Figure 2) | 그대로 사용 | - |
| Stride, Padding | **미명시** | stride=1, padding=1(SAME) | Figure 2의 단계별 출력 크기가 pool 전까지 불변으로 표기되어 있어, 이를 만족하는 유일한 표준 조합으로 채택 |
| Flatten 차원(12544) | 명시(Figure 2)하나 산술 불일치(7×7×7×256=87808) | AdaptiveAvgPool3d로 depth축만 1로 축소 → 7×7×256=12544 | 논문 명시 수치(12544)를 재현 목표로 우선시. flatten_mode="full"(87808) 옵션도 병행 제공 |
| FC-1, FC-2 = 1000 features | 명시(본문 p.5) | 그대로 사용 | - |
| FV-3(FC-1,FC-2 결합) 차원 | Eq.2는 elementwise max(→1000)이나 Fig.1 라벨은 ×2000 | concatenation 채택(2000차원) | 두 서술이 불일치하며, 최종 차원이 2000이 되는 concatenation을 채택(모델_아키텍처_분석.md 불일치사항 3 참조) |

## 2. 3D-ResNet (models.py: ResNet3D)

| 항목 | 논문 명시 여부 | 채택 값 | 근거/이유 |
|---|---|---|---|
| 15 layers = 5 Residual Block × 3 unit | 명시(본문) | 그대로 사용 | - |
| 채널 수(필터 폭) | **전혀 미명시**(Figure 3에 수치 없음) | 64→128→256→512→512 (표준 ResNet 더블링) | 논문에 근거가 없어 통상적인 ResNet 채널 확장 관행을 자체 적용 |
| 커널 크기 | 미명시 | (3,3,3), 3D-CNN과 동일 통일 | 일관성을 위한 자체 결정 |
| Residual Unit 내부 구조(main path 2conv + skip path 1conv) | 명시(Figure 3 확대도) | 그대로 구현 | - |
| FC-4 차원 | 본문(1000) vs Fig.1(×2000) 불일치 | 1000 채택 | 본문 서술("1000 features are extracted from the 14th layer")이 더 구체적 근거이므로 채택 |
| Block 간 MaxPool | 명시(구조상 존재) | 채택, 마지막은 AdaptiveAvgPool3d(1,1,1) | 최종 크기를 고정하기 위한 자체 결정(입력 크기 변화에 견고) |

## 3. CCA 특징 융합 (fusion.py)

| 항목 | 논문 명시 여부 | 채택 값 | 근거/이유 |
|---|---|---|---|
| 공분산행렬 기반 CCA 절차 | 명시(Eq.3-7) | scikit-learn CCA(NIPALS 알고리즘) 사용 | 수학적으로 동일한 절차, 검증된 구현체 사용 |
| CCA 성분 개수(n_components) | **미명시** | min(n_samples-1, 100) | 학습 샘플 수 및 두 특징 차원(2000, 1000)에 안전하게 맞춘 자체 결정 |
| Z1, Z2 결합 방식 | 미명시 | concatenate | 논문이 결합 방식을 서술하지 않아 가장 단순한 방식 채택 |

## 4. WOA 특징 최적화 (feature_optimization.py)

| 항목 | 논문 명시 여부 | 채택 값 | 근거/이유 |
|---|---|---|---|
| b, threshold, Population, Iteration, bounds | 명시(Table 7) | 그대로 사용(100, 200, [0,1], thres=0.5) | - |
| Fitness 가중치 alpha | **미명시** | 0.99 | WOA 특징선택 문헌의 통상값(정확도에 절대적 가중치) |
| Fitness 내부 KNN의 K값 | **미명시** | 5 | scikit-learn 근사 기본값 |
| 독립 실행 횟수(20회) | 명시(Table 7) | train.py에서 반복 실행 시 적용 예정(smoke test는 1회) | 전체 실행은 GPU 환경에서 수행 |

## 5. 분류기 (classifiers.py)

| 항목 | 논문 명시 여부 | 채택 값 | 근거/이유 |
|---|---|---|---|
| SVM/KNN/GB/RF 하이퍼파라미터 | **전혀 미명시** | scikit-learn 기본값 | 논문에 근거 없음 |
| 평가지표 평균 방식 | 미명시 | weighted average | 클래스 불균형(110/58/135) 고려, 기존 프로젝트 세션 관행 유지 |

## 6. 학습 설정 (train.py)

| 항목 | 논문 명시 여부 | 채택 값 | 근거/이유 |
|---|---|---|---|
| CNN LR=0.001, ResNet LR=0.0001, Epoch=30, L2=0.0001 | 명시(Table 5) | 그대로 사용 | - |
| Optimizer=Grid search 자체의 재실행 | 명시(Table 5)이나 재탐색은 미수행 | 논문이 보고한 최종 채택 설정(Adam, 상기 LR)을 직접 사용 | 그리드서치 전체 재실행은 계산비용이 지나치게 크며, 논문이 이미 "최적 설정"으로 보고한 값을 직접 재현하는 것이 프로젝트 목표(재현)에 부합 |
| Batch size | 미명시(Study2 그리드 32/64 후보) | 32 | 두 후보 중 메모리 안전한 값 채택 |
| k-fold(CNN/ResNet 학습 자체) | 명시(k=5)이나 본 구현은 단일 70/15/15 hold-out만 실행 | Table 5의 k=5는 그리드서치 시 내부 검증용으로 해석, 별도 재구현 미수행 | 그리드서치를 재실행하지 않기로 한 5번 항목 결정과 일관 |
| Model 3/4 평가 k-fold(5/10/15) | 명시(Table 6) | 그대로 구현(dataset.py get_kfold_splits) | - |

## 7. 스모크 테스트 관련 (CPU 환경 제약)

- 본 프로젝트의 코드 검증 환경은 GPU가 없는 2-core CPU 환경으로, 실제 30-epoch 전체
  학습은 수행하지 않았습니다(사용자 요청에 따름).
- 실제 마운트된 06_resized 데이터 폴더("전처리06_리사이즈_최종_v2")에서 파일 콘텐츠
  직접 읽기가 간헐적으로 실패하는 환경 이슈(한글 경로명 관련, 메타데이터 조회는
  성공하나 파일 열기가 실패)가 있어, 스모크 테스트는 실제 CSV의 sample_id/label
  메타데이터는 그대로 사용하되, 실제 MRI 볼륨 픽셀 값은 동일 shape(56,56,56)의
  synthetic(무작위) 데이터로 대체하여 코드 동작만 검증했습니다. 이는 코드의 기계적
  정합성(shape 흐름, backward, optimizer step)만을 검증하는 것이 목적이며, 실제 학습
  정확도는 검증 대상이 아닙니다.
- CNN 배치=2에서 Optimizer 1 step에 약 21초가 소요되었고, ResNet은 배치=2에서 메모리
  부족(OOM)으로 실패하여 배치=1로 축소 후 검증했습니다. 이는 CPU 전용/비최적화 빌드
  (Ubuntu 22.04 apt 패키지의 PyTorch 1.8.1, MKL 미포함)의 한계이며, RTX 4060 GPU
  환경에서는 문제가 되지 않을 것으로 예상됩니다.
