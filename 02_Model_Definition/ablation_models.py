# -*- coding: utf-8 -*-
"""
ablation_models.py - 논문 Ablation Study(Table 3, Study 1) 재현용 3D-CNN 변형 모델

근거: 주논문_nature.pdf 본문 "Ablation study" 절(502~516행, pdftotext 추출 기준),
07_Document/모델_아키텍처_분석.md Table 3 인용부.

논문은 최종 채택 모델(Variant 3, 24-layer, models.py의 CNN3D)에 도달하기까지
4단계 ablation 실험을 수행했습니다. 본 파일은 그 중 Base/Variant1/Variant2를
단계적으로 구현합니다(사용자 요청에 따라 Base부터 순서대로 작성).

[논문 원문 그대로 인용 - Base 모델, Stage 1]
"In stage 1, the base model was designed with 8 layers, starting with the input
layer, followed by two 3D-convolutional (3D-conv) layers, an activation layer,
a pooling layer, a Batch Normalization (BN) layer, a flattened layer, and a
fully connected SoftMax layer, which resulted in a model accuracy of 82.02%."

8-layer 구성 (원문 그대로):
  1. Input
  2-3. Conv3D x2
  4. Activation (ReLU)
  5. Pooling (MaxPool3D)
  6. BatchNorm3D
  7. Flatten
  8. Fully Connected + SoftMax(classifier)

[논문에 기재되지 않아 프로젝트에서 자체 결정한 값 - DEVIATIONS.md 반영 예정]
- Conv 채널 폭(filter 수): 논문은 Base 모델의 채널 수를 명시하지 않음. 최종 채택
  모델(Variant 3, models.py CNN3D)의 첫 conv 진행이 32->64->128이므로, 그 앞부분
  2개 conv와 동일한 채널(32->64)을 사용해 변형 간 계열 일관성을 유지함.
- Conv stride/padding: models.py와 동일하게 stride=1, padding=1(3x3x3 커널, SAME)
  사용 - 공간 크기를 conv 전후로 보존하기 위함(Figure 2 해석과 동일 근거).
- Pooling: kernel=2, stride=2 (56 -> 28). 논문 "pooling layer"는 종류(Max/Average)를
  Table 3 Study 2에서 그리드 서치 대상으로 명시했을 뿐 Base 모델 자체의 선택은
  기재하지 않음. 최종 채택 모델과 동일하게 MaxPool3d를 기본값으로 사용.
- Flatten 이후 별도의 은닉 Dense(1000차원 FC-1/FC-2) 없이 곧바로 분류기로 연결.
  논문이 "flattened layer, and a fully connected SoftMax layer"라고 명시해 은닉
  Dense를 언급하지 않았으므로, 원문 그대로 Flatten -> Linear(num_classes) 구조로
  구현함 (Variant 3의 FC-1/FC-2/2000차원 FV-3 구조는 이 Base 모델에는 해당 없음).
"""
import torch
import torch.nn as nn


class CNN3D_Base(nn.Module):
    """Ablation Study Stage 1 - Base 모델 (8-layer, 논문 보고 정확도 82.02%)

    구조: Conv3D(1->32) -> Conv3D(32->64) -> ReLU -> MaxPool3D(2) -> BatchNorm3D
          -> Flatten -> Linear(-> num_classes)

    입력: (N, 1, 56, 56, 56)  (전처리06_리사이즈_최종 규격)
    """

    def __init__(self, num_classes=3, in_channels=1, input_size=56):
        super().__init__()
        self.input_size = input_size

        # 레이어 2-3: Conv3D x2 (채널 32 -> 64), 공간 크기 보존(stride=1, padding=1)
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)

        # 레이어 4: Activation (ReLU) - 두 conv 다음에 한 번만 적용
        # (논문이 "two 3D-conv layers, an activation layer"로 단수 표기한 것을
        #  그대로 따름 - conv1/conv2를 거친 뒤 activation을 1회 적용)
        self.relu = nn.ReLU(inplace=True)

        # 레이어 5: Pooling
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)  # 56 -> 28

        # 레이어 6: BatchNorm3D
        self.bn = nn.BatchNorm3d(64)

        # 레이어 7: Flatten
        pooled_size = input_size // 2  # 56 -> 28
        self.flatten_dim = 64 * pooled_size * pooled_size * pooled_size  # 64*28*28*28 = 1,404,928
        self.flatten = nn.Flatten()

        # 레이어 8: Fully Connected + SoftMax 분류기
        # (SoftMax 자체는 nn.CrossEntropyLoss에 내장되어 있으므로 forward는 logits만 반환)
        self.classifier = nn.Linear(self.flatten_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        """He 초기화 (ReLU 계열 활성화에 적합) - models.py와 동일한 프로젝트 관행"""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.bn(x)
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits


def _expected_shapes(input_size=56, batch_size=2, num_classes=3):
    """torch 없이도 shape 계산식만 검증할 수 있도록 분리한 순수 함수 (셀프 체크용)."""
    pooled = input_size // 2
    flatten_dim = 64 * pooled ** 3
    return {
        "input": (batch_size, 1, input_size, input_size, input_size),
        "after_conv1_conv2": (batch_size, 64, input_size, input_size, input_size),
        "after_pool": (batch_size, 64, pooled, pooled, pooled),
        "flatten_dim": flatten_dim,
        "output": (batch_size, num_classes),
    }


if __name__ == "__main__":
    # torch가 설치된 환경(사용자 로컬 GPU 머신)에서 실행 시 실제 forward pass shape 검증
    shapes = _expected_shapes()
    print("[계산값] 예상 shape:", shapes)
    try:
        model = CNN3D_Base(num_classes=3)
        dummy = torch.randn(2, 1, 56, 56, 56)
        out = model(dummy)
        print(f"[실행 검증] input={tuple(dummy.shape)} -> output={tuple(out.shape)}")
        assert out.shape == (2, 3), "출력 shape이 예상과 다름"
        print("[통과] CNN3D_Base forward pass 정상 동작.")
    except NameError:
        print("[안내] torch가 설치되지 않은 환경입니다. 계산값만 확인했습니다.")
