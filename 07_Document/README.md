# 07_Document

주논문, 참조문헌 원문, 그리고 이 프로젝트가 만든 논문-vs-논문·논문-vs-실제 비교 자료 모음입니다.

## 원문 PDF

| 파일 | 내용 |
|---|---|
| `주논문_nature.pdf` (+ `_한글.pdf`) | Priyadharshini et al., 2024, *Scientific Reports* 14:23394 — 재현 대상 주논문 |
| `ref21.pdf` (+ `_한글.pdf`) | Priyadharshini et al., 2024, *Alexandria Engineering Journal* 107:568-582 — 주논문이 전처리 방법론 출처로 인용(ref.21), 같은 저자의 선행 연구 |
| `ref22.pdf` (+ `_한글.pdf`) | Ullah et al., 2023, *IEEE Access* — 주논문이 ref.21과 함께 전처리 방법론 출처로 인용(ref.22). 원래는 뇌종양 등급분류 논문 |
| `ref23.pdf` | Marino et al., 2012 — T2 강조영상 선택 근거로 인용 |
| `ref31.pdf` | Smith, 2002, "BET: Brain Extraction Tool" — 뇌 추출 도구(BET) 원조 논문 |

## 비교 분석 자료 (이 프로젝트가 작성)

| 파일 | 내용 |
|---|---|
| `논문_비교_ref21_ref22_vs_주논문.xlsx` | 주논문 vs ref21 vs ref22 — 데이터 수집 스펙, 전처리 방법(3자), 모델링 방법 비교. 순수 논문 간 비교, 실제 구현은 포함하지 않음 |
| `논문_레퍼런스_인용목적_정리.xlsx` | 주논문이 인용한 참고문헌 37개 전체의 인용 목적/맥락 정리 |
| `전처리비교_논문_vs_실제.xlsx`, `파이프라인비교_논문_vs_실제.xlsx` | 주논문 서술 vs 이 프로젝트의 실제 전처리 구현 비교 |
| `모델링비교_논문_vs_실제.xlsx` | 주논문 서술 vs 이 프로젝트의 실제 모델링 코드(`02_Model_Definition`~`05_Model_Evaluation`) 비교 |
| `모델_아키텍처_분석.md`, `_상세.docx`, `_발표용.pptx` | 3D-CNN/3D-ResNet 아키텍처 논문 Figure 전수 검토 결과 |
| `전처리_모델링_재현성_검증_보고서.docx` | 재현성 검증 종합 보고서 |

## 관련 문서

원본 데이터/전처리 코드 자체는 [`../01_Preprocessing`](../01_Preprocessing), 모델 코드는 [`../02_Model_Definition`](../02_Model_Definition)~[`../05_Model_Evaluation`](../05_Model_Evaluation) 참고.
