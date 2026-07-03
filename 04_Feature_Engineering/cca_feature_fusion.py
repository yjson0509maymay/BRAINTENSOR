# -*- coding: utf-8 -*-
"""
fusion.py - CCA(Canonical Correlation Analysis) 특징 융합

근거: 모델_아키텍처_분석.md - "CCA (Canonical Correlation Analysis) / 3D-CNN과
3D-ResNet의 특징 융합(Feature fusion) / 공분산 행렬 S_AA, S_BB, S_AB 계산 →
변환행렬 W_A, W_B → 고유값 분해 → 정준상관 변수(CCDF) Z1, Z2" (논문 Eq.3-7, p.6)

[논문에 기재되지 않아 프로젝트에서 자체 결정한 값 - DEVIATIONS.md 참조]
- CCA 성분 개수(n_components): 논문에 수치 없음. 학습 샘플 수 및 두 특징벡터
  차원(FV-3=2000, FC-4=1000)에 안전하게 맞도록 min(n_samples-1, 100)을 기본값으로 사용.
- 구현체: scikit-learn의 CCA(sklearn.cross_decomposition.CCA)를 사용. 이는 논문이
  서술한 공분산행렬 기반 고전적 CCA와 수학적으로 동일한 절차(NIPALS 알고리즘 기반)임.
- 융합 방식: 두 정준상관변수 Z1, Z2를 concatenate하여 최종 융합 벡터 생성
  (논문은 "정준상관 변수 Z1, Z2"라고만 서술, 결합 방식은 명시하지 않음).
"""
import numpy as np
from sklearn.cross_decomposition import CCA


def cca_fuse(fv_a, fv_b, n_components=None, fitted_cca=None):
    """
    fv_a, fv_b: (N, D_a), (N, D_b) 특징 행렬 (예: FV-3(3D-CNN), FC-4(3D-ResNet))
    fitted_cca: 이미 학습된 CCA 객체가 있으면 재사용(테스트 세트 변환 시 사용,
                train 세트로 fit한 CCA를 그대로 재사용해야 데이터 누출을 방지함)
    반환: fused (N, 2*n_components), cca 객체
    """
    n_samples = fv_a.shape[0]
    if n_components is None:
        n_components = max(1, min(n_samples - 1, fv_a.shape[1], fv_b.shape[1], 100))

    if fitted_cca is not None:
        cca = fitted_cca
        z1, z2 = cca.transform(fv_a, fv_b)
    else:
        cca = CCA(n_components=n_components)
        z1, z2 = cca.fit_transform(fv_a, fv_b)

    fused = np.concatenate([z1, z2], axis=1)
    return fused, cca


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    fv3 = rng.normal(size=(30, 2000))   # FV-3 (3D-CNN)
    fc4 = rng.normal(size=(30, 1000))   # FC-4 (3D-ResNet)
    fused, cca = cca_fuse(fv3, fc4)
    print("fused shape:", fused.shape, "n_components:", cca.n_components)
