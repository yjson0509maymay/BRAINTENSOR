# 05_Model_Evaluation

Model-3/Model-4(CCA·WOA 융합 특징)를 최종 ML 분류기로 평가하는 단계입니다.

## 구성

| 파일 | 내용 |
|---|---|
| `ml_classifiers_kfold_eval.py` | SVM/KNN/GB(Gradient Boosting)/RF 4개 분류기 정의(`get_classifiers()`)와 k-fold 평가 하니스(`kfold_evaluate()`). `train.py`는 이 중 GB를 고정으로 사용해 Model-3/4를 5/10/15-fold로 평가함 |

## 논문에 없어서 자체 결정한 값

- 각 분류기 하이퍼파라미터: 논문 전혀 미기재 → scikit-learn 기본값 사용
- Precision/Recall/F1: weighted-average (클래스 불균형 Control 110/Prodromal 58/PD 135 고려)
- k-fold 평가 시 분류기를 GB로 고정 (논문 본문 서술상 GB가 대체로 최고 성능이라 판단)

## 빠른 확인

```bash
python ml_classifiers_kfold_eval.py
# 5-fold GB avg: {'accuracy': ..., 'precision': ..., 'recall': ..., 'f1': ...}
```
