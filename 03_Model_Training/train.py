# -*- coding: utf-8 -*-
"""
train.py - PPMI T2 MRI 논문 재현 전체 파이프라인 오케스트레이션

실행 순서 (Figure 1 전체 데이터 흐름 그대로):
  1. 3D-CNN(Model-1) 학습 - Table 5 사양 (LR=0.001, Adam, Epoch=30, L2=0.0001)
  2. 3D-ResNet(Model-2) 학습 - Table 5 사양 (LR=0.0001, Adam, Epoch=30, L2=0.0001)
  3. 두 백본에서 특징 추출 (FV-3 = 3D-CNN의 FC-1+FC-2 concat, FC-4 = 3D-ResNet)
  4. CCA 특징 융합 -> Model-3 (without optimization), Table 6 방식으로 5/10/15-fold 평가
  5. WOA 특징 최적화 -> Model-4 (with optimization), Table 6 방식으로 5/10/15-fold 평가
  6. 결과 보고서(Markdown) 생성

[실행 환경]
- 실제 30-epoch 전체 학습은 RTX 4060 GPU 환경(사용자 로컬 PC)에서 실행하는 것을 전제로
  작성되었습니다. GPU가 없는 환경에서는 --smoke_test 플래그로 5~10개 샘플, 1 epoch,
  WOA population=3/iteration=2 등 초소형 설정으로 코드 정상 동작만 검증합니다.
- --smoke_test 는 아키텍처 자체를 축소하지 않습니다(models.py 그대로 사용). 오직
  데이터 개수/epoch/WOA 반복 횟수만 축소합니다.

[논문에 기재되지 않아 프로젝트에서 자체 결정한 값]
- CNN/ResNet 학습 시 배치 크기: Table 5에 고정값 없음(Study2 그리드 서치 대상).
  기본값 batch_size=32 사용(그리드 후보값 32/64 중 메모리 안전한 쪽 채택).
- Grid Search(Table 5 "Optimization method: Grid search") 자체의 재실행은 계산비용이
  매우 크므로 수행하지 않고, 논문이 보고한 최종 채택 설정(Variant 3, Adam, LR 0.001/0.0001,
  Epoch=30)을 직접 사용합니다. DEVIATIONS.md에 기록.
"""
import argparse
import os
import sys
import time
import json

import numpy as np
import torch
import torch.nn as nn

# [v2 폴더 재구성] 이 파일은 03_Model_Training에 위치하며, 임포트 대상 모듈은
# 02_Model_Definition(models.py), 04_Feature_Engineering(cca_feature_fusion.py,
# woa_feature_selection.py), 05_Model_Evaluation(ml_classifiers_kfold_eval.py)에
# 나뉘어 있습니다. 실행 단계별로 폴더를 분리한 결과이므로, 아래처럼 각 형제 폴더를
# sys.path에 추가하여 임포트합니다(패키지화하지 않고 폴더 분리만 한 상태이기 때문).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)  # D:\Brain_Tensor
for _rel in ["02_Model_Definition", "03_Model_Training",
             "04_Feature_Engineering", "05_Model_Evaluation"]:
    _p = os.path.join(_ROOT, _rel)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import CNN3D, ResNet3D
from dataset import get_holdout_split, get_kfold_splits, PPMIT2Dataset, augment_volume_3d
from torch.utils.data import DataLoader
from cca_feature_fusion import cca_fuse
from woa_feature_selection import binary_woa_feature_selection
from ml_classifiers_kfold_eval import get_classifiers, evaluate_classifier, kfold_evaluate


def train_backbone(model, train_loader, val_loader, device, epochs=30, lr=0.001,
                    weight_decay=0.0001, log_prefix="CNN"):
    """Table 5 사양: Adam, CrossEntropyLoss, L2 weight_decay, epoch 고정, scheduler 없음"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += x.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += x.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        history.append({"epoch": epoch + 1, "train_loss": train_loss / max(train_total, 1),
                         "train_acc": train_acc, "val_acc": val_acc})
        print(f"  [{log_prefix}] epoch {epoch+1}/{epochs}  "
              f"train_loss={history[-1]['train_loss']:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

    return model, history


@torch.no_grad()
def extract_features(model, loader, device, feature_key):
    model.eval()
    feats, labels, ids = [], [], []
    for x, y, sid in loader:
        x = x.to(device)
        _, feat_dict = model(x, return_features=True)
        feats.append(feat_dict[feature_key].cpu().numpy())
        labels.append(y.numpy())
        ids.extend(sid)
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0), ids


def run_pipeline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---------- 1. 데이터 분할 (Table 5: 70% Train, 15%+15% Test/Val) ----------
    train_samples, val_samples, test_samples = get_holdout_split(args.csv_path, seed=args.seed)

    exists = lambda s: os.path.exists(os.path.join(args.image_dir, f"{s['sample_id']}.nii.gz"))
    train_samples = [s for s in train_samples if exists(s)]
    val_samples = [s for s in val_samples if exists(s)]
    test_samples = [s for s in test_samples if exists(s)]

    if args.smoke_test:
        n = args.smoke_n
        train_samples = train_samples[:max(n - 2, 2)]
        val_samples = val_samples[:1] if val_samples else val_samples
        test_samples = test_samples[:1] if test_samples else test_samples
        print(f"[SMOKE TEST] 샘플 수 축소: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    train_ds = PPMIT2Dataset(train_samples, args.image_dir,
                              transform=lambda x: augment_volume_3d(x, np.random.default_rng()))
    val_ds = PPMIT2Dataset(val_samples, args.image_dir, transform=None)
    test_ds = PPMIT2Dataset(test_samples, args.image_dir, transform=None)

    batch_size = args.smoke_batch if args.smoke_test else args.batch_size
    train_loader = DataLoader(train_ds, batch_size=min(batch_size, max(len(train_ds), 1)), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, max(len(val_ds), 1)), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=min(batch_size, max(len(test_ds), 1)), shuffle=False)

    epochs = 1 if args.smoke_test else args.epochs

    # ---------- 2. 3D-CNN(Model-1) 학습 ----------
    print("\n=== [1/5] 3D-CNN (Model-1) 학습 ===")
    cnn = CNN3D(num_classes=3)
    cnn, cnn_history = train_backbone(cnn, train_loader, val_loader, device,
                                       epochs=epochs, lr=0.001, weight_decay=0.0001,
                                       log_prefix="3D-CNN")

    # ---------- 3. 3D-ResNet(Model-2) 학습 ----------
    print("\n=== [2/5] 3D-ResNet (Model-2) 학습 ===")
    resnet = ResNet3D(num_classes=3)
    resnet, resnet_history = train_backbone(resnet, train_loader, val_loader, device,
                                             epochs=epochs, lr=0.0001, weight_decay=0.0001,
                                             log_prefix="3D-ResNet")

    # ---------- 4. 특징 추출 ----------
    print("\n=== [3/5] 특징 추출 (FV-3, FC-4) ===")
    full_loader = DataLoader(
        PPMIT2Dataset(train_samples + val_samples + test_samples, args.image_dir, transform=None),
        batch_size=min(batch_size, max(len(train_samples) + len(val_samples) + len(test_samples), 1)),
        shuffle=False,
    )
    fv3, labels, ids = extract_features(cnn, full_loader, device, "fv3")
    fc4, _, _ = extract_features(resnet, full_loader, device, "fc4")
    print(f"  FV-3 shape={fv3.shape}, FC-4 shape={fc4.shape}")

    # ---------- 5. CCA 융합 -> Model-3 ----------
    print("\n=== [4/5] CCA 특징 융합 (Model-3, without optimization) ===")
    fused, cca = cca_fuse(fv3, fc4)
    print(f"  fused shape={fused.shape}")

    # ---------- 6. WOA 특징 최적화 -> Model-4 ----------
    print("\n=== [5/5] WOA 특징 최적화 (Model-4, with optimization) ===")
    woa_pop = args.smoke_woa_pop if args.smoke_test else 100     # Table 7: Population Size=100
    woa_iter = args.smoke_woa_iter if args.smoke_test else 200   # Table 7: Number of iterations=200
    mask, best_fit, woa_history = binary_woa_feature_selection(
        fused, labels, population_size=woa_pop, iterations=woa_iter, b=1.0, threshold=0.5,
        seed=args.seed, verbose=True,
    )
    optimized = fused[:, mask]
    print(f"  선택된 특징: {mask.sum()}/{len(mask)}  best_fitness={best_fit:.4f}")

    # ---------- 7. k-fold 평가 (Table 6: k=5,10,15) ----------
    results = {}
    if not args.smoke_test:
        from sklearn.model_selection import StratifiedKFold
        for k in (5, 10, 15):
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=args.seed)
            splits3 = list(skf.split(fused, labels))
            splits4 = list(skf.split(optimized, labels))
            agg3, _ = kfold_evaluate(fused, labels, splits3, clf_name="GB")
            agg4, _ = kfold_evaluate(optimized, labels, splits4, clf_name="GB")
            results[f"model3_{k}fold"] = agg3
            results[f"model4_{k}fold"] = agg4
            print(f"  [k={k}] Model-3(GB): {agg3}")
            print(f"  [k={k}] Model-4(GB): {agg4}")

    return {
        "cnn_history": cnn_history,
        "resnet_history": resnet_history,
        "fv3_shape": list(fv3.shape),
        "fc4_shape": list(fc4.shape),
        "fused_shape": list(fused.shape),
        "n_selected_features": int(mask.sum()),
        "woa_best_fitness": float(best_fit),
        "kfold_results": results,
    }


def build_arg_parser():
    p = argparse.ArgumentParser(description="PPMI T2 MRI 논문 재현 전체 파이프라인")
    p.add_argument("--csv_path", type=str, default=r"d:\Brain_Tensor\01_Preprocessing\data_final_303.csv")
    p.add_argument("--image_dir", type=str, default=r"d:\Brain_Tensor\01_Preprocessing\전처리06_리사이즈_최종")
    p.add_argument("--epochs", type=int, default=30)         # Table 5
    p.add_argument("--batch_size", type=int, default=32)     # Study2 그리드 후보(32/64) 중 채택
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke_test", action="store_true", help="5~10개 샘플/1epoch로 코드 동작만 검증")
    p.add_argument("--smoke_n", type=int, default=8)
    p.add_argument("--smoke_batch", type=int, default=2)
    p.add_argument("--smoke_woa_pop", type=int, default=3)
    p.add_argument("--smoke_woa_iter", type=int, default=2)
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    t0 = time.time()
    result = run_pipeline(args)
    elapsed = time.time() - t0
    result["elapsed_sec"] = elapsed
    print(f"\n총 소요 시간: {elapsed:.2f}s")

    out_path = "smoke_test_result.json" if args.smoke_test else "full_run_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {out_path}")
