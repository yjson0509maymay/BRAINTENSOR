# -*- coding: utf-8 -*-
"""
feature_optimization.py - WOA(Whale Optimization Algorithm) 기반 이진 특징 선택

근거: 모델_아키텍처_분석.md 2-3절(WOA, Figure 4, Eq.8-17) 및 Table 7(파라미터 값)
- Constant(b)=1, Threshold(thres)=0.5, Population Size=100, Number of iterations=200,
  Search bounds=[0,1], No. of independent runs=20 (Table 7 원문 그대로)
- Fitness function Eq.17: f(theta) = alpha*gamma_R(D) + (1-alpha)*|R|/|N|
  gamma_R(D) = KNN 오류율(error rate), |R| = 선택된 특징 수, |N| = 전체 특징 수

[논문에 기재되지 않아 프로젝트에서 자체 결정한 값 - DEVIATIONS.md 참조]
- alpha (fitness 가중치): Table 7에 값이 없음. 특징선택 WOA 문헌의 통상값인
  alpha=0.99(정확도에 절대적 가중치, 특징 수 축소는 부차적 목표)를 채택.
- Fitness 계산용 KNN의 K값: 논문 미기재. K=5(scikit-learn 기본 근사값)를 사용.
- 연속값->이진값 변환: Threshold(thres)=0.5 초과 시 1(선택), 이하 시 0(비선택)
  - Table 7의 threshold 값을 그대로 이진화 기준으로 사용.
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def _fitness(mask, X, y, alpha=0.99, knn_k=5, cv=3):
    n_total = X.shape[1]
    n_selected = int(mask.sum())
    if n_selected == 0:
        return 1.0  # 특징이 하나도 선택되지 않으면 최악의 fitness
    X_sel = X[:, mask.astype(bool)]
    knn = KNeighborsClassifier(n_neighbors=min(knn_k, len(y) - 1))
    try:
        scores = cross_val_score(knn, X_sel, y, cv=min(cv, len(np.unique(y))))
        acc = scores.mean()
    except Exception:
        acc = 0.0
    error_rate = 1.0 - acc
    return alpha * error_rate + (1 - alpha) * (n_selected / n_total)


def binary_woa_feature_selection(X, y, population_size=100, iterations=200,
                                  b=1.0, threshold=0.5, alpha=0.99, seed=42,
                                  verbose=False):
    """
    이진 WOA로 X(N, D)에서 최적 특징 서브셋을 선택.
    반환: best_mask(D,) bool 배열, best_fitness(float), history(list)
    """
    rng = np.random.default_rng(seed)
    n_features = X.shape[1]

    # Population 초기화: 연속값 위치 [0,1] 구간 (Table 7 bounds)
    positions = rng.uniform(0, 1, size=(population_size, n_features))

    def to_binary(pos):
        return (pos > threshold).astype(np.float64)

    fitness = np.array([_fitness(to_binary(positions[i]), X, y, alpha=alpha)
                         for i in range(population_size)])
    best_idx = np.argmin(fitness)
    best_pos = positions[best_idx].copy()
    best_fit = fitness[best_idx]
    history = [best_fit]

    for t in range(iterations):
        a = 2 - t * (2 / iterations)  # a: 2 -> 0 선형 감소
        for i in range(population_size):
            r1, r2 = rng.random(), rng.random()
            A = 2 * a * r1 - a
            C = 2 * r2
            p = rng.random()
            l = rng.uniform(-1, 1)

            if p < 0.5:
                if abs(A) < 1:
                    # Encircling prey (Eq.8-10)
                    D = np.abs(C * best_pos - positions[i])
                    positions[i] = best_pos - A * D
                else:
                    # Exploration: 무작위 개체 기준 탐색 (Eq.14-16)
                    rand_idx = rng.integers(0, population_size)
                    D = np.abs(C * positions[rand_idx] - positions[i])
                    positions[i] = positions[rand_idx] - A * D
            else:
                # Bubble-net attack: spiral update (Eq.11-13)
                D_prime = np.abs(best_pos - positions[i])
                positions[i] = D_prime * np.exp(b * l) * np.cos(2 * np.pi * l) + best_pos

            positions[i] = np.clip(positions[i], 0, 1)

        fitness = np.array([_fitness(to_binary(positions[i]), X, y, alpha=alpha)
                             for i in range(population_size)])
        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_fit:
            best_fit = fitness[gen_best_idx]
            best_pos = positions[gen_best_idx].copy()
        history.append(best_fit)
        if verbose and (t % max(1, iterations // 10) == 0):
            print(f"  [WOA] iter {t}/{iterations}  best_fitness={best_fit:.4f}")

    best_mask = to_binary(best_pos).astype(bool)
    return best_mask, best_fit, history


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 40))
    y = rng.integers(0, 3, size=60)
    mask, fit, hist = binary_woa_feature_selection(
        X, y, population_size=5, iterations=3, verbose=True
    )
    print("selected features:", mask.sum(), "/", len(mask), "best_fitness:", fit)
