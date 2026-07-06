# -*- coding: utf-8 -*-
"""
train_ablation.py - Ablation Study(Table 3, Study 1) 재현용 학습 스크립트

02_Model_Definition/ablation_models.py 의 CNN3D_Base(8-layer, 논문 보고 정확도 82.02%)를
실제 303명 PPMI T2 데이터(data_final_303.csv)로 학습하고, Train/Val/Test accuracy와
Precision/Recall/F1을 출력합니다. epoch별 시간/loss/accuracy는 CSV 로그로도 기록됩니다.

[실행 환경]
- 이 스크립트는 로컬 GPU(RTX 4060 등) 환경에서 실행하는 것을 전제로 작성되었습니다.

[2026-07-06 갱신: Data Augmentation 재도입]
- 최초 실행 결과 Train acc 99% vs Val/Test acc 37~53%의 극심한 과적합이 관측됨
  (train_loss도 epoch마다 불규칙하게 요동 - epoch 30에서 15.48까지 치솟음).
  원인: (1) 증강 없이 (2) Train 212개(Prodromal는 41개뿐)의 작은 데이터로 (3) Dropout도
  없는 단순 구조(1.4M차원 flatten -> 바로 3클래스 Linear)를 학습했기 때문.
  논문 본문도 "Data augmentation was applied"(ref.21/22 인용)라고 명시하므로,
  dataset.py에 복원한 augment_volume_3d()를 Train 세트에만 다시 적용함(기본값 on).
  Validation/Test에는 절대 적용하지 않음(데이터 누출 방지, 논문 방법론과 동일 원칙).

[논문에 기재되지 않아 프로젝트에서 자체 결정한 값 - DEVIATIONS.md 반영 예정]
- Ablation Study 1(Base/Variant1/Variant2 구조 비교) 자체의 학습 하이퍼파라미터는
  논문에 명시되어 있지 않습니다(Study 2에서 그리드서치 대상 하이퍼파라미터만 기재).
  최종 채택 모델(Variant 3, Table 5)과 동일한 Adam/lr=0.001/CrossEntropyLoss를
  기본값으로 사용하고, epoch/batch_size는 CLI 인자로 조절 가능하게 둡니다.
- 데이터 분할은 dataset.py의 get_holdout_split()을 그대로 재사용합니다
  (70/15/15, subject 단위 stratified, seed=42 고정 - 프로젝트 전체 일관성 유지).
- 증강 파라미터(회전 ±8도, 50% flip, scale 0.95~1.05, shift ±0.05, noise std 0.01)는
  논문에 구체 수치가 없어 프로젝트 자체 결정.
"""
import argparse
import csv
import os
import sys
import json
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
for _rel in ["02_Model_Definition", "03_Model_Training"]:
    _p = os.path.join(_ROOT, _rel)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ablation_models import CNN3D_Base
from dataset import get_holdout_split, PPMIT2Dataset, augment_volume_3d

try:
    from sklearn.metrics import precision_recall_fscore_support, classification_report
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


VARIANT_MODELS = {
    "base": CNN3D_Base,
}


def build_loaders(csv_path, image_dir, batch_size, seed, use_augmentation=True):
    train_samples, val_samples, test_samples = get_holdout_split(csv_path, seed=seed)
    exists = lambda s: os.path.exists(os.path.join(image_dir, f"{s['sample_id']}.nii.gz"))
    train_samples = [s for s in train_samples if exists(s)]
    val_samples = [s for s in val_samples if exists(s)]
    test_samples = [s for s in test_samples if exists(s)]

    # Train 세트에만 온라인 증강 적용(매 epoch 새 랜덤 변형). Val/Test는 절대 증강하지 않음.
    train_transform = (lambda x: augment_volume_3d(x, np.random.default_rng())) if use_augmentation else None
    train_ds = PPMIT2Dataset(train_samples, image_dir, transform=train_transform)
    val_ds = PPMIT2Dataset(val_samples, image_dir, transform=None)
    test_ds = PPMIT2Dataset(test_samples, image_dir, transform=None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[데이터] Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)} "
          f"(Train 증강={'적용' if use_augmentation else '미적용'})")
    return train_loader, val_loader, test_loader


def run_epoch(model, loader, device, criterion, optimizer=None):
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


def write_csv_log(csv_path, history):
    fields = ["epoch", "timestamp", "train_loss", "train_acc", "val_loss", "val_acc",
              "epoch_seconds", "cumulative_seconds"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


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
    p.add_argument("--csv_path", type=str, default=r"D:\Brain_Tensor\01_Preprocessing\data_final_303.csv")
    p.add_argument("--image_dir", type=str, default=r"D:\Brain_Tensor\01_Preprocessing\전처리06_리사이즈_최종")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_augmentation", action="store_true",
                   help="Train 세트 증강을 끄고 싶을 때 사용 (기본값: 증강 사용)")
    p.add_argument("--log_csv", type=str, default=None,
                   help="epoch별 시간/loss/accuracy 기록 CSV 경로 (기본: ablation_<variant>_training_log.csv)")
    args = p.parse_args()
    if args.log_csv is None:
        _tag = "noaug" if args.no_augmentation else "aug"
        args.log_csv = f"ablation_{args.variant}_{_tag}_training_log.csv"

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader = build_loaders(
        args.csv_path, args.image_dir, args.batch_size, args.seed,
        use_augmentation=not args.no_augmentation,
    )

    model_cls = VARIANT_MODELS[args.variant]
    model = model_cls(num_classes=3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[모델] {args.variant} ({model_cls.__name__}), 파라미터 수={n_params:,}")
    print(f"[로그] epoch별 기록 CSV: {args.log_csv} (매 epoch마다 갱신됨)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        epoch_t0 = time.time()
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, device, criterion, optimizer=None)
        epoch_elapsed = time.time() - epoch_t0
        total_elapsed_so_far = time.time() - t0

        history.append({
            "epoch": epoch + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "epoch_seconds": round(epoch_elapsed, 2),
            "cumulative_seconds": round(total_elapsed_so_far, 2),
        })
        print(f"[Epoch {epoch+1}/{args.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} ({train_acc*100:.2f}%)  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({val_acc*100:.2f}%)  "
              f"소요시간={epoch_elapsed:.1f}s (누적 {total_elapsed_so_far:.1f}s)")

        write_csv_log(args.log_csv, history)

    elapsed = time.time() - t0
    print(f"\n총 학습 시간: {elapsed:.1f}s")

    test_result = evaluate_final(model, test_loader, device, criterion, "Test")

    paper_acc = {"base": 0.8202}
    if args.variant in paper_acc:
        diff = test_result["accuracy"] - paper_acc[args.variant]
        print(f"\n[논문 대비] 논문 보고 정확도={paper_acc[args.variant]*100:.2f}%  "
              f"실제 학습 정확도={test_result['accuracy']*100:.2f}%  "
              f"차이={diff*100:+.2f}%p")

    augmentation_used = not args.no_augmentation
    tag = "aug" if augmentation_used else "noaug"
    out_path = f"ablation_{args.variant}_{tag}_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "history": history,
            "test_result": test_result,
            "elapsed_sec": elapsed,
            "n_params": n_params,
            "variant": args.variant,
            "augmentation_used": augmentation_used,
            "run_finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hyperparams": {
                "epochs": args.epochs, "batch_size": args.batch_size,
                "lr": args.lr, "weight_decay": args.weight_decay, "seed": args.seed,
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"결과 저장(JSON): {out_path}  (augmentation_used={augmentation_used})")
    print(f"결과 저장(CSV, epoch별 시간/loss/accuracy): {args.log_csv}")


if __name__ == "__main__":
    main()
