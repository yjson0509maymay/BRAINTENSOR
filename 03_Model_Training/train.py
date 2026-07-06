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

[2026-07-03 갱신: 논문 Table 3(Study 2, 3D-CNN ablation)에서 배치 크기가 실제로
명시되어 있음을 확인함]
- 3D-CNN(Model-1) 학습 배치: Table 3 Study 2의 최고 성능 구성인 Variant 3(Accuracy
  93.41%)가 Batch size=64를 사용함이 확인되어, 유효 배치(effective batch)를 64로
  맞춥니다. 다만 이 아키텍처는 56^3 입력에서 채널을 최대 1024까지 사용해 배치 64를
  물리적으로 그대로 GPU에 올리면 활성화(activation) 메모리만 수십 GB가 필요할 수
  있어(DEVIATIONS.md 참조), 물리 배치(--batch_size, 기본 8)로 실제 GPU에 로드하고
  gradient accumulation(기본 8회 누적)으로 유효 배치 64를 재현합니다. 이는 논문과
  수치적으로 동일한 그래디언트 업데이트를 만들어내는 표준 기법입니다.
- 3D-ResNet(Model-2) 배치: Table 3은 3D-CNN 전용 ablation이라 3D-ResNet의 배치 크기는
  여전히 논문에 명시되어 있지 않습니다. 기존 프로젝트 관행(두 백본에 동일 배치 파라미터
  공유)을 유지해 3D-CNN과 동일한 유효 배치(64)를 적용합니다 - DEVIATIONS.md 참조.
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
from dataset import get_holdout_split, get_kfold_splits, PPMIT2Dataset
from torch.utils.data import DataLoader
from cca_feature_fusion import cca_fuse
from woa_feature_selection import binary_woa_feature_selection
from ml_classifiers_kfold_eval import get_classifiers, evaluate_classifier, kfold_evaluate


def train_backbone(model, train_loader, val_loader, device, epochs=30, lr=0.001,
                    weight_decay=0.0001, log_prefix="CNN", accum_steps=1):
    """Table 5 사양: Adam, CrossEntropyLoss, L2 weight_decay, epoch 고정, scheduler 없음.

    accum_steps: gradient accumulation 횟수. Table 3(Study 2, Variant 3)에서 3D-CNN의
    유효 배치가 64로 확인되었으나, 이 아키텍처는 56^3 입력에서 채널을 최대 1024까지
    쓰기 때문에 배치 64를 물리적으로 그대로 GPU에 올리면 활성화 메모리가 지나치게
    커진다(DEVIATIONS.md 참조). 따라서 DataLoader의 물리 배치(--batch_size)는 작게
    유지하고, accum_steps번 그래디언트를 누적한 뒤 한 번만 optimizer.step()을 호출해
    "물리 배치 x accum_steps = 유효 배치 64"를 재현한다. 이는 논문의 배치 64와
    수치적으로 동일한 그래디언트 업데이트를 만들어내는 표준 기법이다.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        optimizer.zero_grad()
        n_batches = len(train_loader)
        for step, (x, y, _) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y) / accum_steps
            loss.backward()

            is_last_batch = (step + 1) == n_batches
            if (step + 1) % accum_steps == 0 or is_last_batch:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accum_steps * x.size(0)
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

    train_ds = PPMIT2Dataset(train_samples, args.image_dir, transform=None)
    val_ds = PPMIT2Dataset(val_samples, args.image_dir, transform=None)
    test_ds = PPMIT2Dataset(test_samples, args.image_dir, transform=None)

    batch_size = args.smoke_batch if args.smoke_test else args.batch_size
    train_loader = DataLoader(train_ds, batch_size=min(batch_size, max(len(train_ds), 1)), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, max(len(val_ds), 1)), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=min(batch_size, max(len(test_ds), 1)), shuffle=False)

    epochs = 1 if args.smoke_test else args.epochs

    # 유효 배치(effective batch) 재현: Table 3 Study2 Variant3 확인값(64)을
    # 물리 배치(args.batch_size, GPU에 실제로 올라가는 크기) x accum_steps로 구성
    accum_steps = 1 if args.smoke_test else max(1, args.effective_batch_size // args.batch_size)
    if not args.smoke_test:
        print(f"[배치 설정] 물리 배치={args.batch_size}, 누적 횟수={accum_steps}, "
              f"유효 배치={args.batch_size * accum_steps} (논문 Table 3 Variant3 목표=64)")

    # ---------- 2. 3D-CNN(Model-1) 학습 ----------
    print("\n=== [1/5] 3D-CNN (Model-1) 학습 ===")
    cnn = CNN3D(num_classes=3)
    cnn, cnn_history = train_backbone(cnn, train_loader, val_loader, device,
                                       epochs=epochs, lr=0.001, weight_decay=0.0001,
                                       log_prefix="3D-CNN", accum_steps=accum_steps)

    # ---------- 3. 3D-ResNet(Model-2) 학습 ----------
    print("\n=== [2/5] 3D-ResNet (Model-2) 학습 ===")
    resnet = ResNet3D(num_classes=3)
    resnet, resnet_history = train_backbone(resnet, train_loader, val_loader, device,
                                             epochs=epochs, lr=0.0001, weight_decay=0.0001,
                                             log_prefix="3D-ResNet", accum_steps=accum_steps)

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
    p.add_argument("--batch_size", type=int, default=8,
                   help="물리 배치 크기: GPU에 실제로 한 번에 올라가는 샘플 수. "
                        "VRAM 제약에 맞춰 조정(예: 12GB급 GPU는 8 권장)")
    p.add_argument("--effective_batch_size", type=int, default=64,
                   help="유효 배치 크기: Table 3 Study2 Variant3(3D-CNN 최고 성능 구성, "
                        "Accuracy 93.41%%)에서 확인된 논문 채택값. batch_size보다 크면 "
                        "gradient accumulation으로 재현(accum_steps = effective_batch_size // batch_size)")
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
