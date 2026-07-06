# -*- coding: utf-8 -*-
"""
smoke_test.py - CPU 환경에서 코드 정상 동작만 검증하는 경량 스모크 테스트 (단계별 분리 실행판)

CPU 2코어 환경에서는 3D-CNN 1회 forward+backward만도 수 초가 소요되므로,
모델별로 나누어 실행할 수 있도록 --model 인자를 지원합니다.
"""
import sys, os, time, json, argparse

# [v2 폴더 재구성] models.py는 02_Model_Definition, dataset.py는 03_Model_Training(본 파일과 동일 폴더)에 위치
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
for _rel in ["02_Model_Definition", "03_Model_Training"]:
    _p = os.path.join(_ROOT, _rel)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn
torch.set_num_threads(2)

from dataset import _load_samples, PPMIT2Dataset
from torch.utils.data import DataLoader
from models import CNN3D, ResNet3D


def log(msg):
    print(msg, flush=True)


def run_model_smoke(model_name, model, lr, batch_x, batch_y, n_opt_steps):
    report = {}
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    t0 = time.time()
    logits, feats = model(batch_x, return_features=True)
    report["forward_time"] = time.time() - t0
    report["logits_shape"] = list(logits.shape)
    log(f"[OK] {model_name} Forward Pass: input={tuple(batch_x.shape)} -> logits={tuple(logits.shape)} "
        f"({report['forward_time']:.2f}s)")

    t0 = time.time()
    loss = criterion(logits, batch_y)
    report["initial_loss"] = loss.item()
    log(f"[OK] {model_name} Loss 계산: loss={loss.item():.4f} ({time.time()-t0:.2f}s)")

    losses = []
    t0 = time.time()
    for step in range(n_opt_steps):
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        log(f"    step {step+1}/{n_opt_steps}: loss={loss.item():.4f}")
    report["opt_step_losses"] = losses
    report["opt_step_time"] = time.time() - t0
    log(f"[OK] {model_name} Optimizer {n_opt_steps} step 완료 ({report['opt_step_time']:.2f}s)")

    feat_key = "fv3" if model_name == "3D-CNN" else "fc4"
    with torch.no_grad():
        _, feats = model(batch_x, return_features=True)
    np.save(f"/tmp/smoke_data/feat_{model_name}.npy", feats[feat_key].numpy())
    np.save(f"/tmp/smoke_data/labels.npy", batch_y.numpy())
    report["feat_shape"] = list(feats[feat_key].shape)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", default="/tmp/smoke_data/data.csv")
    p.add_argument("--image_dir", default="/tmp/smoke_data/images")
    p.add_argument("--model", choices=["cnn", "resnet"], required=True)
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--n_opt_steps", type=int, default=1)
    args = p.parse_args()

    t_start = time.time()
    samples = _load_samples(args.csv_path)[:args.n_samples]
    log(f"[OK] 1_데이터로딩: {len(samples)}개 샘플: {[s['sample_id'] for s in samples]}")

    ds = PPMIT2Dataset(samples, args.image_dir, transform=None)
    x0, y0, sid0 = ds[0]
    log(f"[OK] 2_전처리(증강 없음): shape={tuple(x0.shape)}, dtype={x0.dtype}, label={y0.item()}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    batch_x, batch_y, batch_ids = next(iter(loader))
    log(f"[OK] 3_DataLoader배치: X={tuple(batch_x.shape)}, y={tuple(batch_y.shape)}, ids={list(batch_ids)}")

    if args.model == "cnn":
        model = CNN3D(num_classes=3)
        rep = run_model_smoke("3D-CNN", model, 0.001, batch_x, batch_y, args.n_opt_steps)
    else:
        model = ResNet3D(num_classes=3)
        rep = run_model_smoke("3D-ResNet", model, 0.0001, batch_x, batch_y, args.n_opt_steps)

    rep["total_elapsed_sec"] = time.time() - t_start
    rep["batch_x_shape"] = list(batch_x.shape)
    rep["batch_y_shape"] = list(batch_y.shape)
    out_path = f"/tmp/smoke_data/report_{args.model}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    log(f"\n총 소요시간: {rep['total_elapsed_sec']:.2f}s -> {out_path} 저장")


if __name__ == "__main__":
    main()
