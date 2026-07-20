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

[논문에 기재되지 않아 프로젝트에서 자체 결정한 값 - DEVIATIONS.md 반영 예정]
- Ablation Study 1(Base/Variant1/Variant2 구조 비교) 자체의 학습 하이퍼파라미터는
  논문에 명시되어 있지 않습니다(Study 2에서 그리드서치 대상 하이퍼파라미터만 기재).
  최종 채택 모델(Variant 3, Table 5)과 동일한 Adam/lr=0.001/CrossEntropyLoss를
  기본값으로 사용하고, epoch/batch_size는 CLI 인자로 조절 가능하게 둡니다.
- 데이터 분할은 dataset.py의 get_holdout_split()을 그대로 재사용합니다
  (70/15/15, subject 단위 stratified, seed=42 고정 - 프로젝트 전체 일관성 유지).
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

from ablation_models import CNN3D_Base
from dataset import get_holdout_split, PPMIT2Dataset

try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


VARIANT_MODELS = {
    "base": CNN3D_Base,   # 8-layer, 논문 보고 82.02%
    # "variant1": ...  # 9-layer, 추후 추가 예정
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


def run_epoch(model, loader, device, criterion, optimizer=None):
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
    p.add_argument("--csv_path", type=str, default=os.path.join(_ROOT, "01_Preprocessing", "data_final_303.csv"))
    p.add_argument("--image_dir", type=str, default=os.path.join(_ROOT, "01_Preprocessing", "전처리06_리사이즈_최종"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--seed", type=int, default=42)
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
            "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "epoch_seconds": round(epoch_elapsed, 2),
        })
        print(f"[Epoch {epoch+1}/{args.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} ({train_acc*100:.2f}%)  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({val_acc*100:.2f}%)  "
              f"소요시간={epoch_elapsed:.1f}s (누적 {total_elapsed_so_far:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n총 학습 시간: {elapsed:.1f}s")

    test_result = evaluate_final(model, test_loader, device, criterion, "Test")

    # 논문 보고값과 비교 (Base 모델 기준 82.02%, Table 3 Study 1)
    paper_acc = {"base": 0.8202}
    paper_hyperparams = {"base": {"epochs": 30, "batch_size": 64}}
    
    print("\n========== [논문 vs 실제 실행 결과 비교] ==========")
    if args.variant in paper_acc:
        p_ep = paper_hyperparams[args.variant]["epochs"]
        p_bs = paper_hyperparams[args.variant]["batch_size"]
        
        print(f"1. 하이퍼파라미터 비교:")
        print(f"   - [논문] Epochs: {p_ep}, Batch Size: {p_bs}")
        print(f"   - [실제] Epochs: {args.epochs}, Batch Size: {args.batch_size}")
        
        diff = test_result["accuracy"] - paper_acc[args.variant]
        print(f"2. 성능(Accuracy) 비교:")
        print(f"   - [논문] {paper_acc[args.variant]*100:.2f}%")
        print(f"   - [실제] {test_result['accuracy']*100:.2f}%")
        print(f"   - [차이] {diff*100:+.2f}%p")
    else:
        print("논문 보고값이 없는 모델입니다.")
    print("====================================================")
    import csv
    
    # 로그 파일에 누적 기록 (TXT)
    log_path = os.path.join(_ROOT, "03_Model_Training", "training_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n[{timestamp}] Variant: {args.variant} | Epochs: {args.epochs} | Batch: {args.batch_size} | 총 소요시간: {elapsed:.1f}s\n")
        if args.variant in paper_acc:
            p_ep = paper_hyperparams[args.variant]["epochs"]
            p_bs = paper_hyperparams[args.variant]["batch_size"]
            diff = test_result["accuracy"] - paper_acc[args.variant]
            f.write(f"  - [논문] Epochs: {p_ep}, Batch: {p_bs} -> Acc: {paper_acc[args.variant]*100:.2f}%\n")
            f.write(f"  - [실제] Epochs: {args.epochs}, Batch: {args.batch_size} -> Acc: {test_result['accuracy']*100:.2f}%\n")
            f.write(f"  - [차이] {diff*100:+.2f}%p\n")
        else:
            f.write(f"  - [실제] Acc: {test_result['accuracy']*100:.2f}%\n")
            diff = 0.0
        f.write("-" * 50 + "\n")
    print(f"로그 누적 기록 완료 (TXT): {log_path}")

    # 로그 파일에 누적 기록 (CSV)
    csv_log_path = os.path.join(_ROOT, "03_Model_Training", "training_log.csv")
    csv_exists = os.path.isfile(csv_log_path)
    with open(csv_log_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(["Timestamp", "Variant", "Actual Epochs", "Actual Batch Size", "Elapsed Time (s)", "Paper Acc (%)", "Actual Acc (%)", "Diff (%p)"])
        
        paper_acc_val = paper_acc[args.variant]*100 if args.variant in paper_acc else ""
        actual_acc_val = test_result['accuracy']*100
        diff_val = diff*100 if args.variant in paper_acc else ""
        
        writer.writerow([
            timestamp, 
            args.variant, 
            args.epochs, 
            args.batch_size, 
            round(elapsed, 1), 
            paper_acc_val, 
            round(actual_acc_val, 2), 
            round(diff_val, 2) if diff_val != "" else ""
        ])
    print(f"로그 누적 기록 완료 (CSV): {csv_log_path}")

    out_path = f"ablation_{args.variant}_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"history": history, "test_result": test_result, "elapsed_sec": elapsed,
                    "n_params": n_params, "variant": args.variant}, f, ensure_ascii=False, indent=2)
    print(f"상세 결과 JSON 저장: {out_path}")


if __name__ == "__main__":
    main()
