# -*- coding: utf-8 -*-
"""
models.py - 논문 사양 기반 3D-CNN / 3D-ResNet 아키텍처 재구현

근거 문서: 05_Document/모델_아키텍처_분석.md (Priyadharshini et al., Sci Rep 2024, 14:23394,
Figure 1-4, Table 5-8 전수 검토 결과)

이 파일은 이전 세션의 train_variant3_pd25_affine.py에 있던 Global Max Pooling 기반
축소 구조(Flatten=256차원)를 완전히 폐기하고, 논문 Figure 2/3에 명시된 채널 폭과
Flatten 차원을 그대로 따르도록 처음부터 다시 작성되었습니다.

[논문에 명시된 값 - 그대로 사용]
- 3D-CNN 필터 진행: 32-64-128-[pool]-[bn]-256-512-1024-[pool]-[bn]-512-256-[pool]-[bn]
- 3D-CNN 커널 크기: (3,3,3)
- 3D-CNN MaxPool size: (2,2,2), 총 3회
- 3D-CNN Flatten 차원: 12544 (Figure 2 원문 표기)
- 3D-CNN FC-1/FC-2: 각 1000개 특징 (본문 p.5)
- 3D-ResNet: 15 layers, Residual Block 5개(각 3 Residual Unit = 15 unit), Conv3D Block 2개(선행)
- 3D-ResNet FC-4: 1000개 특징 (본문 p.5 "1000 features are extracted from the 14th layer")
- Activation: ReLU, Eq.(1)
- Optimizer/LR/Epoch/k-fold/weight decay: Table 5 (train.py에서 사용)

[논문에 기재되지 않아 프로젝트에서 자체 결정한 값 - DEVIATIONS.md 참조]
- Stride, Padding (Conv3D 전 레이어): stride=1, padding=1 (SAME) 사용.
  근거: Figure 2의 단계별 출력 크기(56->56->56->28(pool)->...)가 pool 전에는
  공간크기가 변하지 않는 것으로 표기되어 있어, conv 자체는 크기를 보존해야 함.
  이를 만족하는 유일한 표준 조합이 stride=1, padding=1(3x3x3 커널 기준)이므로 채택.
- Flatten=12544를 만들기 위한 방법(depth_collapse): 최종 conv 출력(7,7,7,256)에서
  depth(D) 축만 AdaptiveAvgPool3d로 1로 축소 -> (1,7,7,256) -> flatten 12544.
  논문 자체의 산술 불일치(7*7*7*256=87808 != 12544)를 그대로 두는 대신,
  Figure 2에 명시된 최종 수치(12544)를 재현 목표로 우선시한 해석적 결정.
- FV-3(3D-CNN 내부 FC-1+FC-2 융합) 방식: concatenation 채택(2000차원).
  본문 Eq.2는 elementwise max로 서술되어 있으나 그 경우 결과 차원이 1000이 되어
  Figure 1의 "FV-3 x2000" 라벨과 모순되므로, 최종 차원이 2000이 되는 concatenation을
  채택하고 이유를 명시함(모델_아키텍처_분석.md 불일치사항 3 참조).
- 3D-ResNet 채널 폭/필터 수: 논문 Figure 3에 전혀 수치가 없어(전 항목 "논문에
  기재되지 않음"), 표준 ResNet 채널 더블링 관행(64->128->256->512->512)을
  자체 적용함. 커널 크기는 3D-CNN과 동일하게 (3,3,3)로 통일.
"""
import torch
import torch.nn as nn


class CNN3D(nn.Module):
    """Model-1: 3D-CNN (Figure 2 사양)"""

    def __init__(self, num_classes=3, in_channels=1, flatten_mode="depth_collapse"):
        super().__init__()
        self.conv0 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1)   # Input Layer, Filter=32
        self.conv1 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)            # Conv3D-1, Filter=64
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)           # Conv3D-2, Filter=128
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)                            # MaxPool: 56->28
        self.bn1 = nn.BatchNorm3d(128)

        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)          # Conv3D-3, Filter=256
        self.conv4 = nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1)          # Conv3D-4, Filter=512
        self.conv5 = nn.Conv3d(512, 1024, kernel_size=3, stride=1, padding=1)         # Conv3D-5, Filter=1024
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)                            # MaxPool: 28->14
        self.bn2 = nn.BatchNorm3d(1024)

        self.conv6 = nn.Conv3d(1024, 512, kernel_size=3, stride=1, padding=1)         # Conv3D-6, Filter=512
        self.conv7 = nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1)          # Conv3D-7, Filter=256
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)                            # MaxPool: 14->7
        self.bn3 = nn.BatchNorm3d(256)

        self.relu = nn.ReLU(inplace=True)

        self.flatten_mode = flatten_mode
        if flatten_mode == "depth_collapse":
            self.spatial_collapse = nn.AdaptiveAvgPool3d((1, 7, 7))
            flatten_dim = 1 * 7 * 7 * 256   # = 12544, 논문 Figure 2 명시값과 일치
        elif flatten_mode == "full":
            self.spatial_collapse = nn.Identity()
            flatten_dim = 7 * 7 * 7 * 256   # = 87808, 산술적으로 정확하나 논문 표기와 불일치
        else:
            raise ValueError(f"Unknown flatten_mode: {flatten_mode}")

        self.flatten_dim = flatten_dim
        self.fc1 = nn.Linear(flatten_dim, 1000)   # FC-1: 1000 features (본문 p.5)
        self.fc2 = nn.Linear(flatten_dim, 1000)   # FC-2: 1000 features (본문 p.5)
        self.classifier = nn.Linear(2000, num_classes)  # FV-3(2000) -> Softmax(3)

    def forward(self, x, return_features=False):
        x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.pool3(x)
        x = self.bn3(x)

        x = self.spatial_collapse(x)
        x = torch.flatten(x, start_dim=1)

        fc1_feat = self.fc1(x)
        fc2_feat = self.fc2(x)
        fv3 = torch.cat([fc1_feat, fc2_feat], dim=1)

        logits = self.classifier(fv3)
        if return_features:
            return logits, {"fc1": fc1_feat, "fc2": fc2_feat, "fv3": fv3}
        return logits


class ResidualUnit3D(nn.Module):
    """Figure 3 Residual Unit: main path(conv-bn-relu-conv-bn) + skip path(conv-bn) -> add -> relu"""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class ResidualBlock3D(nn.Module):
    """Block = 3 Residual Unit (Figure 3: Block당 3 unit, 총 5 Block = 15 unit)"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.unit0 = ResidualUnit3D(in_ch, out_ch, stride=1)
        self.unit1 = ResidualUnit3D(out_ch, out_ch, stride=1)
        self.unit2 = ResidualUnit3D(out_ch, out_ch, stride=1)

    def forward(self, x):
        x = self.unit0(x)
        x = self.unit1(x)
        x = self.unit2(x)
        return x


class ResNet3D(nn.Module):
    """Model-2: Improved 3D-ResNet (Figure 3 사양, 15 layers = 5 Residual Block x 3 unit)"""

    def __init__(self, num_classes=3, in_channels=1):
        super().__init__()
        # Conv3D Block x2 (선행 stem, Figure 3 Stage 2-3)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        # 5 Residual Block, 각 뒤에 MaxPool3D (Figure 3 순서 그대로)
        self.block0 = ResidualBlock3D(64, 64)
        self.pool0 = nn.MaxPool3d(kernel_size=2, stride=2)      # 56->28
        self.block1 = ResidualBlock3D(64, 128)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)      # 28->14
        self.block2 = ResidualBlock3D(128, 256)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)      # 14->7
        self.block3 = ResidualBlock3D(256, 512)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)      # 7->3
        self.block4 = ResidualBlock3D(512, 512)
        self.pool4 = nn.AdaptiveAvgPool3d((1, 1, 1))            # 3->1 (전역 평균 풀링, 잔여 크기 무관하게 고정 출력)

        self.fc4 = nn.Linear(512, 1000)          # FC-4: 1000 features (본문 p.5)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x, return_features=False):
        x = self.stem(x)
        x = self.pool0(self.block0(x))
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        x = torch.flatten(x, start_dim=1)

        fc4_feat = self.fc4(x)
        logits = self.classifier(fc4_feat)

        if return_features:
            return logits, {"fc4": fc4_feat}
        return logits


if __name__ == "__main__":
    cnn = CNN3D()
    resnet = ResNet3D()
    x = torch.randn(2, 1, 56, 56, 56)
    logits_c, feats_c = cnn(x, return_features=True)
    logits_r, feats_r = resnet(x, return_features=True)
    print("CNN3D  logits:", logits_c.shape, "| fv3:", feats_c["fv3"].shape, "| flatten_dim:", cnn.flatten_dim)
    print("ResNet3D logits:", logits_r.shape, "| fc4:", feats_r["fc4"].shape)
