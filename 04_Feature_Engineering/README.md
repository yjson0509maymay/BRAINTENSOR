# 04_Feature_Engineering

3D-CNN(FV-3)과 3D-ResNet(FC-4) 특징을 융합·최적화하는 Model-3/Model-4 단계입니다.

## 구성

| 파일 | 내용 | 근거 |
|---|---|---|
| `cca_feature_fusion.py` | CCA(정준상관분석)로 FV-3(2000차원)와 FC-4(1000차원)를 융합 → Model-3. `sklearn.cross_decomposition.CCA` 사용 | 논문 Eq.3-7 |
| `woa_feature_selection.py` | WOA(고래 최적화 알고리즘)로 융합 특징 중 최적 서브셋 선택 → Model-4. Population=100, Iteration=200, threshold=0.5 | 논문 Table 7, Eq.8-17 |

## 논문에 없어서 자체 결정한 값

- CCA 성분 개수: `min(n_samples-1, 100)`
- WOA fitness 가중치 alpha=0.99, fitness 계산용 KNN K=5

자세한 근거는 각 파일 상단 docstring 참고. 논문-vs-실제 비교 요약은 [`../07_Document/모델링비교_논문_vs_실제.xlsx`](../07_Document/모델링비교_논문_vs_실제.xlsx)에 있습니다.

## 빠른 확인

```bash
python cca_feature_fusion.py     # fused shape: (30, 200) 형태 출력
python woa_feature_selection.py  # selected features: N/40 형태 출력
```
