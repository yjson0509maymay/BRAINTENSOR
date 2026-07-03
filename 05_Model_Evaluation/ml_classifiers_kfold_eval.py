# -*- coding: utf-8 -*-
"""
classifiers.py - SVM/KNN/GB/RF 분류기 + k-fold 평가 하니스

근거: 모델_아키텍처_분석.md Table 4(Analysis 1-6, SVM/KNN/GB/RF 비교),
Table 6(Model 3/4 k-fold=5,10,15 Accuracy/Recall/Precision/F1-score)

[논문에 기재되지 않아 프로젝트에서 자체 결정한 값 - DEVIATIONS.md 참조]
- 각 분류기의 세부 하이퍼파라미터(SVM 커널/C/gamma, KNN의 K, GB/RF 트리 수 등):
  논문 전혀 미기재. scikit-learn 기본값을 사용하고 그대로 명시.
- 평가지표 계산: accuracy_score, precision/recall/f1은 클래스 불균형(Control 110,
  Prodromal 58, PD 135)을 고려해 weighted-average를 기본으로 사용
  (기존 프로젝트 세션에서도 weighted avg를 사용해온 관행 유지).
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def get_classifiers():
    """Table 4에 언급된 4개 분류기. 하이퍼파라미터는 논문 미기재 -> sklearn 기본값 사용."""
    return {
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
        "GB": GradientBoostingClassifier(),
        "RF": RandomForestClassifier(),
    }


def evaluate_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def kfold_evaluate(X, y, kfold_splits, clf_name="GB"):
    """
    kfold_splits: dataset.get_kfold_splits() 스타일의 (train_idx, test_idx) 또는
                  (X_train,y_train,X_test,y_test) 튜플 리스트를 받아 k-fold 평균 성능 산출.
    clf_name: get_classifiers() 키 중 하나. 논문 본문 서술상 GB가 대체로 최고 성능(Table 4).
    """
    results = []
    for fold_i, (train_idx, test_idx) in enumerate(kfold_splits):
        clf = get_classifiers()[clf_name]
        metrics = evaluate_classifier(clf, X[train_idx], y[train_idx], X[test_idx], y[test_idx])
        metrics["fold"] = fold_i
        results.append(metrics)

    agg = {
        k: float(np.mean([r[k] for r in results]))
        for k in ("accuracy", "precision", "recall", "f1")
    }
    return agg, results


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 20))
    y = rng.integers(0, 3, size=60)
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(X, y))
    agg, per_fold = kfold_evaluate(X, y, splits, clf_name="GB")
    print("5-fold GB avg:", agg)
