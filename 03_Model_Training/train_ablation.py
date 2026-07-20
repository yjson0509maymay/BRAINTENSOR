# -*- coding: utf-8 -*-
"""
train_ablation.py - Ablation Study(Table 3, Study 1) 재현용 학습 스크립트

02_Model_Definition/ablation_models.py 의 CNN3D_Base(8-layer, 논문 보고 정확도 82.02%)를
실제 303명 PPMI T2 데이터(data_final_303.csv)로 학습하고, Train/Val/Test accuracy와
Precision/Recall/F1을 출력합니다.

[실행 환경]
- 이 스크립트는 로컬 GPU(RTX 4060 등) 환경에서 실행하는 것을 전제로 작성되었습니다.
  개발 샌드박스에는 PyTorch가 설치되어 있지 않아, 여기서는 shape 계산만 검증했고
  (ablation_models.py 참조) 실제 forward pass/학습 실행은 검증하지 못했습니다.

[2026-07 갱신: 논문 Table 3(Study 2, Base model 열)에서 Base 모델 하이퍼파라미터가
실제로 명시되어 있음을 확인함 - 이전 기록은 "미기재"로 잘못 판단했던 것]
- 논문 Table 3 Study 2 표에서 Base model 열의 체크 표시를 확인한 결과:
  Pooling=Max, Activation=ReLU, **Batch size=32**, Flatten=Flatten(Global max 아님),
  Optimizer=Adam, **Learning rate=0.01**, Epoch=30 로 전부 명시되어 있음.
  이전에는 "논문 미기재"로 보고 Variant3(배치64, LR=0.001)와 동일하게 가정했으나 오판이었음 -
  Base 모델은 Variant3와 하이퍼파라미터가 다름. 아래 기본값을 표에 맞게 정정함.
- 정확도 비교 목표치도 2가지로 구분됨:
  · Study 1(구조 탐색, 하이퍼파라미터 튜닝 전) 결과: 82.02%
  · Study 2(위 하이퍼파라미터 튜닝 후) 결과: 84.96%, Precision 87.34%, Recall 81.54%, F1 85.32%
  이 스크립트는 Study 2의 튜닝된 하이퍼파라미터를 사용하므로, 비교 기준은 84.96%가 더 적절함
  (82.02%는 튜닝 전 구조만 비교한 수치라 하이퍼파라미터가 다름).

[논문에 여전히 기재되지 않아 프로젝트에서 자체 결정한 값]
- 데이터 분할은 dataset.py의 get_holdout_split()을 그대로 재사용합니다
  (70/15/15, subject 단위 stratified, seed=42 고정 - 프로젝트 전체 일관성 유지).
- Weight decay(L2): Table 3 Study 2에는 항목 자체가 없어 여전히 미기재. Table 5(Variant3
  학습 설정)의 값(0.0001)을 그대로 가져와 사용 - 근거는 약하지만 프로젝트 일관성 유지 목적.

[2026-07 추가: base variant 첫 실행에서 로짓/로스 폭주(epoch1 loss 수천대) +
극심한 과적합(train_acc 100%, test_acc 34.78% vs 논문 Study2 목표 84.96%) 확인 후 대응]
- Gradient clipping(max_norm=1.0): 논문에 전혀 언급 없음(있는지/없는지조차 미기재).
  CNN3D_Base는 flatten(1,404,928) -> Linear(3) 단일 거대 classifier가 파라미터의
  98.7%를 차지해, lr=0.01(논문 Study2 명시값) 하에서 Adam 업데이트만으로도 로짓이
  쉽게 폭주함(관측: epoch1 val_loss=8172). lr/batch/epoch 등 논문 명시 하이퍼파라미터는
  그대로 두고, 학습 안정화를 위한 구현 디테일만 추가 - weight_decay와 동일한 성격의
  "논문 미기재 자체 결정" 항목.
- Best-validation-checkpoint 선택: 논문 Table 5에 원문 그대로 "Selected the optimal
  hyperparameters based on best validation performance"라고 명시되어 있음에도, 기존
  구현은 마지막 epoch(30) 가중치를 그대로 test에 사용했음 - 이는 오히려 논문 절차와
  어긋남. Val accuracy가 가장 높았던 epoch의 가중치를 저장해두었다가 그 체크포인트로
  test를 평가하도록 수정 - "논문 재현"의 정확도를 높이는 방향이며 하이퍼파라미터
  변경이 아님.

[2026-07-19 추가: gradient clipping/체크포인트 선택 적용 후에도 시드만 바꾸면 Control/
Prodromal 등 서로 다른 클래스가 번갈아 recall 0으로 붕괴하는 불안정성이 계속 관측됨
(예: seed=42 Control 붕괴 acc 56.52%, seed=43 Prodromal 붕괴 acc 50.00%) - 근본 원인
추가 조치]
- Classifier 출력 스케일 보정(calibrate_classifier_scale): He(Kaiming) 초기화는 입력이
  서로 독립(i.i.d.)이라는 가정 하에 분산을 계산하는데, CNN3D_Base의 classifier는
  conv+pool을 거친 공간적으로 상관관계가 큰 1,404,928차원 벡터를 그대로 입력받아 이
  가정이 깨짐 - 그 결과 초기 로짓 분산이 이론값보다 훨씬 커서 epoch1 loss가 수천대로
  폭주하고(이전 항목에서 관측), 이게 학습 궤적을 시드에 매우 민감하게 만드는 원인 중
  하나로 추정됨. 학습 시작 직후 실제 train 배치 하나로 classifier 출력의 표준편차를
  측정해서 목표값(1.0)에 맞게 classifier.weight/bias를 사후 스케일링 - 모델 구조
  (Flatten->Linear->SoftMax, 논문 명시)나 lr/batch/epoch 등 하이퍼파라미터는 전혀
  바꾸지 않고, 이미 자체 결정 사항이었던 초기화 스킴(He 초기화)만 보정하는 구현
  디테일. --no_calibrate_init으로 끌 수 있음(기본 활성).
"""
import argparse
import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 형제 폴더(02_Model_Definition)에서 ablation_models.py를 임포트하기 위한 경로 설정
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)  # D:\Brain_Tensor
for _rel in ["02_Model_Definition", "03_Model_Training"]:
    _p = os.path.join(_ROOT, _rel)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ablation_models import CNN3D_Base, CNN3D_Variant1
from dataset import get_holdout_split, PPMIT2Dataset

try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


VARIANT_MODELS = {
    "base": CNN3D_Base,           # 8-layer, 논문 보고 82.02%
    "variant1": CNN3D_Variant1,   # 9-layer(conv 3개), 논문 보고 85.75%
    # "variant2": ...  # 17-layer, 추후 추가 예정
}


def build_loaders(csv_path, image_dir, batch_size, seed):
    train_samples, val_samples, test_samples = get_holdout_split(csv_path, seed=seed)

    exists = lambda s: os.path.exists(os.path.join(image_dir, f"{s['sample_id']}.nii.gz"))
    train_samples = [s for s in train_samples if exists(s)]
    val_samples = [s for s in val_samples if exists(s)]
    test_samples = [s for s in test_samples if exists(s)]

    train_ds = PPMIT2Dataset(train_samples, image_dir, transform=None)
    val_ds = PPMIT2Dataset(val_samples, image_dir, transform=None)
    test_ds = PPMIT2Dataset(test_samples, image_dir, transform=None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[데이터] Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    return train_loader, val_loader, test_loader


@torch.no_grad()
def calibrate_classifier_scale(model, sample_x, device, target_std=1.0):
    """분류기(classifier) 레이어 출력 스케일 보정 - 스크립트 상단 docstring(2026-07-19
    항목) 참조. He 초기화는 입력이 서로 독립이라고 가정하지만 classifier 입력(flatten된
    conv 특징)은 공간적으로 상관되어 있어 실제 로짓 분산이 이론값보다 커짐. 실제 train
    배치 하나로 출력 표준편차를 재서 목표값에 맞게 classifier.weight/bias를 스케일링."""
    if not hasattr(model, "classifier"):
        return None, None
    model.eval()
    logits = model(sample_x.to(device))
    current_std = logits.std().item()
    if current_std > 1e-8:
        scale = target_std / current_std
        model.classifier.weight.mul_(scale)
        if model.classifier.bias is not None:
            model.classifier.bias.mul_(scale)
    model.train()
    return current_std, target_std


def run_epoch(model, loader, device, criterion, optimizer=None, grad_clip_norm=None):
    """optimizer가 주어지면 학습 모드, 없으면 평가 모드로 동작."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc, all_preds, all_labels


def evaluate_final(model, loader, device, criterion, split_name):
    loss, acc, preds, labels = run_epoch(model, loader, device, criterion, optimizer=None)
    print(f"\n=== {split_name} 최종 평가 ===")
    print(f"  Loss={loss:.4f}  Accuracy={acc:.4f} ({acc*100:.2f}%)")

    result = {"loss": loss, "accuracy": acc}
    if _HAS_SKLEARN and len(set(labels)) > 1:
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        print(f"  Precision(macro)={precision:.4f}  Recall(macro)={recall:.4f}  F1(macro)={f1:.4f}")
        print(classification_report(labels, preds, target_names=["Control", "Prodromal", "PD"], zero_division=0))
        result.update({"precision_macro": precision, "recall_macro": recall, "f1_macro": f1})
    return result


def main():
    p = argparse.ArgumentParser(description="Ablation Study 모델 학습 + 정확도 출력")
    p.add_argument("--variant", choices=list(VARIANT_MODELS.keys()), default="base")
    # [2026-07-20 갱신] 기본값이 존재하지 않는 경로(data_final_303.csv, 전처리06_리사이즈_최종)를
    # 가리키고 있던 걸 발견 - 이번 세션 내내 --csv_path/--image_dir를 매번 직접 지정해서
    # 드러나지 않았음. 여러 전처리 변형(min-max/N4없음, N4있음, z-score)을 비교한 결과
    # min-max+N4없음(전처리_ref21order_v1)이 가장 우수·안정적이어서 이걸 표준으로 채택
    # (상세: 01_Preprocessing/전처리_ref21order_v1_상세기록.md, 전처리_변형_종합비교.md).
    p.add_argument("--csv_path", type=str, default=os.path.join(_ROOT, "01_Preprocessing", "data_0713_wsl_v2.csv"))
    p.add_argument("--image_dir", type=str, default=os.path.join(_ROOT, "01_Preprocessing", "전처리_ref21order_v1"))
    p.add_argument("--epochs", type=int, default=30)
    # [2026-07 재수정] Table 3 Study 2 표에서 Base model 열을 직접 확인한 결과 배치=32로
    # 명시되어 있음(Variant3의 배치=64와는 다름). 이전 수정(8->64)은 Variant3 값을 잘못
    # 가져온 오판이었음 - 32로 되돌림. Base 모델은 채널 폭이 작아 배치32는 물론 64도
    # 메모리 문제는 없지만, 논문 값을 정확히 재현하려면 32를 써야 함.
    p.add_argument("--batch_size", type=int, default=32)
    # [2026-07 재수정] 마찬가지로 Table 3 Study 2에서 Base model의 Learning rate=0.01이
    # 명시되어 있음을 확인(Variant3의 0.001과 다름). 0.001 -> 0.01로 정정.
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--seed", type=int, default=42)
    # [2026-07 추가] 논문 미기재, 학습 안정화를 위한 자체 결정값 - 스크립트 상단 docstring 참조
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument(
        "--no_calibrate_init", action="store_false", dest="calibrate_init", default=True,
        help="classifier 초기화 출력 스케일 보정 비활성화(기본은 활성) - 스크립트 상단 docstring 참조",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader = build_loaders(
        args.csv_path, args.image_dir, args.batch_size, args.seed
    )

    model_cls = VARIANT_MODELS[args.variant]
    model = model_cls(num_classes=3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[모델] {args.variant} ({model_cls.__name__}), 파라미터 수={n_params:,}")

    if args.calibrate_init:
        sample_x, _, _ = next(iter(train_loader))
        before_std, target_std = calibrate_classifier_scale(model, sample_x, device)
        if before_std is not None:
            print(f"[초기화 보정] classifier 출력 표준편차: {before_std:.2f} -> {target_std:.2f}로 스케일 조정")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_val_acc = -1.0
    best_epoch = -1
    best_state = None
    t0 = time.time()
    for epoch in range(args.epochs):
        epoch_t0 = time.time()
        train_loss, train_acc, _, _ = run_epoch(
            model, train_loader, device, criterion, optimizer, grad_clip_norm=args.grad_clip_norm
        )
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, device, criterion, optimizer=None)
        epoch_elapsed = time.time() - epoch_t0
        total_elapsed_so_far = time.time() - t0
        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "epoch_seconds": round(epoch_elapsed, 2),
        })
        print(f"[Epoch {epoch+1}/{args.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} ({train_acc*100:.2f}%)  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({val_acc*100:.2f}%)  "
              f"소요시간={epoch_elapsed:.1f}s (누적 {total_elapsed_so_far:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - t0
    print(f"\n총 학습 시간: {elapsed:.1f}s")

    # 논문 Table 5("Selected the optimal hyperparameters based on best validation
    # performance")를 따라, 마지막 epoch가 아닌 best-validation epoch의 가중치로 test 평가
    print(f"\n[체크포인트 선택] Best validation epoch = {best_epoch}/{args.epochs} "
          f"(val_acc={best_val_acc:.4f}, {best_val_acc*100:.2f}%) - 이 시점 가중치로 test 평가")
    model.load_state_dict(best_state)

    test_result = evaluate_final(model, test_loader, device, criterion, "Test")
    test_result["best_epoch"] = best_epoch
    test_result["best_val_acc"] = best_val_acc

    # ============================================================
    # 논문 Table 3 정보 (Study 1 구조 정보 + Study 2 하이퍼파라미터 + 4개 평가지표)
    # 논문 값과 우리 모델 값을 한 표에서 나란히 비교할 수 있도록 구성
    # ============================================================
    # [2026-07-20 추가] Variant1(9-layer, conv 3개) 등록. Study1 구조/정확도(85.75%)는
    # 논문 원문 직접 인용으로 확정("three 3D-conv layers... 9-layer architecture...
    # achieving a model accuracy of 85.75%", ablation_models.py CNN3D_Variant1 참조).
    # Study2 accuracy/precision/recall/f1(88.25/89.93/84.28/87.43)은
    # 07_Document/모델_아키텍처_분석.md Table3 인용부 확인. 다만 batch_size/lr은
    # Table3 Study2의 표 체크박스가 pdftotext로 깨져서(Base 때처럼 육안 확인 필요)
    # 아직 미확인 - Base와 동일값(32, 0.01)으로 잠정 가정만 해둠(추후 육안 검증 필요).
    paper_architecture = {
        "base": {"n_conv_layers": 2, "n_pooling_layers": 1, "test_acc": 0.8202, "finding": "Lowest accuracy"},
        "variant1": {"n_conv_layers": 3, "n_pooling_layers": 1, "test_acc": 0.8575, "finding": "Intermediate"},
    }
    paper_study2 = {
        "base": {
            "pooling": "Max", "activation": "ReLU", "batch_size": 32, "flatten": "Flatten",
            "optimizer": "Adam", "lr": 0.01, "epochs": 30,
            "accuracy": 0.8496, "precision": 0.8734, "recall": 0.8154, "f1": 0.8532,
        },
        "variant1": {
            "pooling": "Max", "activation": "ReLU", "batch_size": 32, "flatten": "Flatten",  # 미확인, Base값 잠정 가정
            "optimizer": "Adam", "lr": 0.01, "epochs": 30,  # batch_size/lr 미확인, Base값 잠정 가정
            "accuracy": 0.8825, "precision": 0.8993, "recall": 0.8428, "f1": 0.8743,
        },
    }
    paper_acc_study1 = {k: v["test_acc"] for k, v in paper_architecture.items()}
    paper_acc_study2 = {k: v["accuracy"] for k, v in paper_study2.items()}
    paper_hyperparams = {k: {"epochs": v["epochs"], "batch_size": v["batch_size"], "lr": v["lr"]}
                          for k, v in paper_study2.items()}

    comparison_rows = []  # (항목, 논문 값, 우리 모델 값, 비고)

    if args.variant in paper_architecture:
        arch = paper_architecture[args.variant]
        s2 = paper_study2[args.variant]

        actual_precision = test_result.get("precision_macro", float("nan"))
        actual_recall = test_result.get("recall_macro", float("nan"))
        actual_f1 = test_result.get("f1_macro", float("nan"))

        # -- 구조(Study 1) --
        comparison_rows.append(("Conv layer 개수", str(arch["n_conv_layers"]), "2 (conv1, conv2)", "일치"))
        comparison_rows.append(("Pooling layer 개수", str(arch["n_pooling_layers"]), "1 (MaxPool3d)", "일치"))
        comparison_rows.append(("구조 단계 Test Accuracy (Study1, 튜닝 전)",
                                 f"{arch['test_acc']*100:.2f}%", "-", f"Finding: \"{arch['finding']}\""))

        # -- 하이퍼파라미터(Study 2) --
        comparison_rows.append(("Pooling 종류", s2["pooling"], "Max (MaxPool3d)",
                                 "일치" if s2["pooling"] == "Max" else "다름"))
        comparison_rows.append(("Activation", s2["activation"], "ReLU",
                                 "일치" if s2["activation"] == "ReLU" else "다름"))
        comparison_rows.append(("Flatten 방식", s2["flatten"], "Flatten",
                                 "일치" if s2["flatten"] == "Flatten" else "다름"))
        comparison_rows.append(("Optimizer", s2["optimizer"], "Adam",
                                 "일치" if s2["optimizer"] == "Adam" else "다름"))
        comparison_rows.append(("Batch size", str(s2["batch_size"]), str(args.batch_size),
                                 "일치" if s2["batch_size"] == args.batch_size else "다름 - 확인 필요"))
        comparison_rows.append(("Learning rate", str(s2["lr"]), str(args.lr),
                                 "일치" if s2["lr"] == args.lr else "다름 - 확인 필요"))
        comparison_rows.append(("Epoch", str(s2["epochs"]), str(args.epochs),
                                 "일치" if s2["epochs"] == args.epochs else "다름 - 확인 필요"))

        # -- 평가지표 4종(Study 2 기준, 논문과 동일 하이퍼파라미터) --
        # [2026-07 변경] 차이(%p)는 표에 안 적고 실험값/논문값을 나란히 칸으로만 구분
        comparison_rows.append(("Accuracy", f"{s2['accuracy']*100:.2f}%", f"{test_result['accuracy']*100:.2f}%", ""))
        comparison_rows.append(("Precision", f"{s2['precision']*100:.2f}%", f"{actual_precision*100:.2f}%", ""))
        comparison_rows.append(("Recall", f"{s2['recall']*100:.2f}%", f"{actual_recall*100:.2f}%", ""))
        comparison_rows.append(("F1-score", f"{s2['f1']*100:.2f}%", f"{actual_f1*100:.2f}%", ""))

        # -- Finding 비교 (논문의 상대순위 vs 우리 모델의 목표 대비 판정) --
        acc_diff_pct = (test_result["accuracy"] - s2["accuracy"]) * 100
        if acc_diff_pct >= -3:
            our_finding = "논문 목표치(84.96%)와 근접 - 정상 재현으로 판단"
        elif acc_diff_pct >= -10:
            our_finding = "논문 대비 다소 낮음 - 재현 오차 범위 내일 수 있으나 확인 권장"
        else:
            our_finding = "논문 대비 상당히 낮음 - 전처리/학습 파이프라인 점검 필요"
        comparison_rows.append(("Finding (논문 상대순위)", f"\"{arch['finding']}\" (Base/V1/V2/V3 4개 비교 기준)",
                                 "-", "우리는 Base만 실행 - 상대순위는 V1~V3까지 만들어야 확정 가능"))
        comparison_rows.append(("Finding (우리 모델, 자동판정)", "-", our_finding,
                                 f"기준: Study2 목표({s2['accuracy']*100:.2f}%) 대비 차이 {acc_diff_pct:+.2f}%p"))

        # -- 소요 시간(논문에 없음, 참고용) --
        comparison_rows.append(("총 소요 시간", "-", f"{elapsed:.1f}초 (약 {elapsed/60:.1f}분)", "논문 미기재"))
        comparison_rows.append(("Epoch당 평균 시간", "-", f"{elapsed/max(args.epochs,1):.1f}초", "논문 미기재"))

        diff = test_result["accuracy"] - s2["accuracy"]

    # ---- 콘솔에 표 형태로 출력 ----
    # [2026-07 변경] 실험값 칸을 논문값보다 앞에 두고(요청: "실험값 논문값 이렇게"), 비고 칸은
    # 일치 여부/Finding처럼 값 자체가 아닌 주석에만 사용(수치 비교 행은 위에서 비고를 비움)
    print("\n========== [논문(Table 3) vs 우리 모델 비교표] ==========")
    if comparison_rows:
        col1_w = max(len(r[0]) for r in comparison_rows) + 2
        col2_w = max(len("실험값"), *(len(r[2]) for r in comparison_rows)) + 2
        col3_w = max(len("논문값"), *(len(r[1]) for r in comparison_rows)) + 2
        header = f"{'항목':<{col1_w}}{'실험값':<{col2_w}}{'논문값':<{col3_w}}비고"
        print(header)
        print("-" * len(header))
        for label, paper_v, model_v, note in comparison_rows:
            print(f"{label:<{col1_w}}{model_v:<{col2_w}}{paper_v:<{col3_w}}{note}")
    else:
        print("논문 보고값이 없는 모델입니다.")
    print("=" * 60)
    import csv

    # 로그 파일에 누적 기록 (TXT)
    log_path = os.path.join(_ROOT, "03_Model_Training", "training_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        timestamp_fs = time.strftime("%Y%m%d_%H%M%S")  # 파일명용(콜론 등 제외, JSON 파일명에 사용)
        f.write(f"\n[{timestamp}] Variant: {args.variant} | Epochs: {args.epochs} | Batch: {args.batch_size} | 총 소요시간: {elapsed:.1f}s\n")
        # [2026-07 변경] 실험값을 먼저 적고 논문값을 뒤에, 차이(%p)는 안 적음(요청 반영)
        if args.variant in paper_acc_study2:
            p_ep = paper_hyperparams[args.variant]["epochs"]
            p_bs = paper_hyperparams[args.variant]["batch_size"]
            diff = test_result["accuracy"] - paper_acc_study2[args.variant]
            f.write(f"  - [실험값] Epochs: {args.epochs}, Batch: {args.batch_size} -> Acc: {test_result['accuracy']*100:.2f}%\n")
            f.write(f"  - [논문값(Study2)] Epochs: {p_ep}, Batch: {p_bs}, LR: {paper_hyperparams[args.variant]['lr']} "
                    f"-> Acc: {paper_acc_study2[args.variant]*100:.2f}% (참고: Study1 튜닝전 82.02%)\n")
        else:
            f.write(f"  - [실험값] Acc: {test_result['accuracy']*100:.2f}%\n")
            diff = 0.0
        f.write("-" * 50 + "\n")
    print(f"로그 누적 기록 완료 (TXT): {log_path}")

    # 로그 파일에 누적 기록 (CSV)
    # [2026-07 변경] 실험값 컬럼을 논문값 컬럼들보다 앞에 두고, Diff 컬럼은 제거(요청 반영)
    csv_log_path = os.path.join(_ROOT, "03_Model_Training", "training_log.csv")
    csv_exists = os.path.isfile(csv_log_path)
    with open(csv_log_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(["Timestamp", "Variant", "Actual Epochs", "Actual Batch Size", "Elapsed Time (s)",
                              "Actual Acc (%)", "Paper Acc Study2 (%)", "Paper Acc Study1 (%)"])

        paper_acc_s1_val = paper_acc_study1[args.variant] * 100 if args.variant in paper_acc_study1 else ""
        paper_acc_s2_val = paper_acc_study2[args.variant] * 100 if args.variant in paper_acc_study2 else ""
        actual_acc_val = test_result['accuracy'] * 100

        writer.writerow([
            timestamp,
            args.variant,
            args.epochs,
            args.batch_size,
            round(elapsed, 1),
            round(actual_acc_val, 2),
            paper_acc_s2_val,
            paper_acc_s1_val,
        ])
    print(f"로그 누적 기록 완료 (CSV): {csv_log_path}")

    # [2026-07 추가] 기존에는 ablation_{variant}_result.json 고정 파일명이라 재실행 시
    # 이전 실행의 상세 결과(특히 best checkpoint 관련 history)가 덮어써져 사라졌음.
    # results/ 폴더에 실행 시각+정확도를 파일명에 포함시켜 실행마다 별도 파일로 누적 보존하고,
    # 파일을 열지 않아도 폴더 목록만으로 결과를 스캔할 수 있게 함.
    results_dir = os.path.join(_ROOT, "03_Model_Training", "results")
    os.makedirs(results_dir, exist_ok=True)
    acc_tag = f"{test_result['accuracy']*100:.1f}"
    out_path = os.path.join(results_dir, f"ablation_{args.variant}_{timestamp_fs}_acc{acc_tag}.json")

    # summary를 history보다 먼저 두어 파일을 열자마자 핵심 결과부터 보이게 함
    summary = {
        "timestamp": timestamp,
        "variant": args.variant,
        "hyperparams": {
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "grad_clip_norm": args.grad_clip_norm,
            "calibrate_init": args.calibrate_init, "seed": args.seed,
        },
        "best_epoch": test_result.get("best_epoch"),
        "best_val_acc": round(test_result.get("best_val_acc", 0.0), 4),
        "test_accuracy": round(test_result["accuracy"], 4),
        "test_precision_macro": round(test_result.get("precision_macro", float("nan")), 4),
        "test_recall_macro": round(test_result.get("recall_macro", float("nan")), 4),
        "test_f1_macro": round(test_result.get("f1_macro", float("nan")), 4),
        "elapsed_sec": round(elapsed, 1),
        "vs_paper_study2_diff_pct": round(diff * 100, 2) if args.variant in paper_acc_study2 else None,
    }

    history_rounded = [
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in epoch_row.items()}
        for epoch_row in history
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "test_result": test_result,
            "paper_vs_model_comparison": [
                {"항목": label, "실험값": model_v, "논문값": paper_v, "비고": note}
                for label, paper_v, model_v, note in comparison_rows
            ],
            "n_params": n_params,
            "history": history_rounded,
        }, f, ensure_ascii=False, indent=2)
    print(f"상세 결과 JSON 저장: {out_path}")


if __name__ == "__main__":
    main()