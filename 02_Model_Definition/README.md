# 02_Model_Definition

주논문 Figure 2/3에 명시된 3D-CNN(Model-1), 3D-ResNet(Model-2) 아키텍처 재구현입니다.

## 구성

| 파일 | 내용 |
|---|---|
| `models.py` | `CNN3D`(24계층, 필터 32→64→128→256→512→1024→512→256, FC-1/FC-2 각 1000차원 → concat하여 FV-3 2000차원) / `ResNet3D`(15계층 = Residual Block 5개×3유닛, FC-4 1000차원) |
| `ablation_models.py` | Table 3 ablation 후보 중 `CNN3D_Base`(8계층, Stage 1)만 구현. Variant1(9계층)·Variant2(17계층)는 미구현 |

## 논문에 없어서 자체 결정한 값 (근거는 파일 상단 docstring 참고)

- Conv3D stride=1, padding=1
- 3D-CNN Flatten 차원을 논문 명시값(12544)에 맞추기 위한 depth-collapse 방식
- FV-3 결합을 element-wise max 대신 concatenation으로 채택 (논문 Eq.2와 Figure 1 라벨이 서로 모순되어, 최종 차원이 맞는 쪽을 선택)
- 3D-ResNet 채널 폭(64→128→256→512→512, 논문 Figure 3에 수치 없음)

자세한 논문-vs-실제 비교는 [`../07_Document/모델링비교_논문_vs_실제.xlsx`](../07_Document/모델링비교_논문_vs_실제.xlsx) 참고.

## 빠른 확인

```bash
python models.py
# CNN3D  logits: torch.Size([2, 3]) | fv3: torch.Size([2, 2000]) | flatten_dim: 12544
# ResNet3D logits: torch.Size([2, 3]) | fc4: torch.Size([2, 1000])
```
