# -*- coding: utf-8 -*-
"""
dataset.py - PPMI T2 MRI 데이터셋 로더 및 논문 사양 평가 분할

근거: 모델_아키텍처_분석.md Table 5(3D-CNN/3D-ResNet 학습: 70% Train, 15%+15% Test/Val,
k-fold=5), Table 6(Model 3/4: 5-fold, 10-fold, 15-fold cross validation).

이 버전은 두 가지 분할 전략을 모두 제공합니다:

  1. get_holdout_split()  - 3D-CNN/3D-ResNet 자체 학습용 (Table 5 사양: 70% Train,
     나머지 30%를 Test/Validation으로 사용. 두 세트의 정확한 구분 비율은 논문에
     명시되어 있지 않아 15%/15%로 균등 분할하는 기존 프로젝트 관행을 유지함 - DEVIATIONS.md 참조)
  2. get_kfold_splits()   - Model 3/4(CCA 융합 + WOA 최적화 후 ML 분류기) 평가용
     (Table 6 사양: k=5, 10, 15 모두 지원)

모든 분할은 피험자(Subject) 단위로 수행되어 동일 피험자의 여러 스캔이 train/val/test
사이에 걸쳐 누출(leakage)되지 않도록 합니다 (기존 프로젝트 원칙 유지).

[Data Augmentation 관련 이력]
- 2026-07-06(1차): augment_volume_3d()를 프로젝트 결정으로 제거함.
- 2026-07-06(2차, 본 버전): Ablation Base(8-layer) 모델을 실제 303명 데이터로 학습한
  결과 Train acc 99% vs Val/Test acc 37~53%의 극심한 과적합이 관측되어(train_loss도
  epoch마다 불규칙하게 요동), 증강을 다시 도입함. 근거: 원 논문 본문도 "Data
  augmentation was applied"(ref.21/22 인용, 구체 기법 미기재)라고 명시하고 있어
  증강 자체는 논문 방법론과 일치하며, 특히 Prodromal 클래스(train 41개)처럼 표본이
  적은 상황에서 증강 없이 3D-CNN을 학습하는 것은 비현실적이라고 판단함.
  적용 파라미터(회전 ±8도, 50% flip, scale 0.95~1.05, shift ±0.05, gaussian noise
  std 0.01)는 논문에 구체 수치가 없어 프로젝트 자체 결정 - DEVIATIONS.md 참조.
"""
import os
import csv
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import rotate


def _load_samples(csv_path):
    samples = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)
    return samples


def _group_by_subject(samples):
    subject_to_samples = {}
    for s in samples:
        subject_to_samples.setdefault(s["Subject"], []).append(s)
    return subject_to_samples


def _subject_labels(subject_to_samples):
    subjects = list(subject_to_samples.keys())
    labels = [int(subject_to_samples[s][0]["label"]) for s in subjects]
    return subjects, labels


class PPMIT2Dataset(Dataset):
    """56x56x56 T2 MRI 볼륨 데이터셋 (전처리06_리사이즈_최종 / 06_resized 기준)"""

    def __init__(self, samples, image_dir, transform=None, cache=False):
        self.samples = samples
        self.image_dir = image_dir
        self.transform = transform
        self.cache = cache
        if cache:
            self.cached_data = {}
            for s in samples:
                sid = s["sample_id"]
                path = os.path.join(image_dir, f"{sid}.nii.gz")
                self.cached_data[sid] = nib.load(path).get_fdata(dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        sid = s["sample_id"]
        label = int(s["label"])

        if self.cache:
            data = self.cached_data[sid].copy()
        else:
            path = os.path.join(self.image_dir, f"{sid}.nii.gz")
            data = nib.load(path).get_fdata(dtype=np.float32)

        if self.transform:
            data = self.transform(data)

        data = np.expand_dims(data, axis=0)  # (1,56,56,56)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long), sid


def augment_volume_3d(data, rng=None, use_rotation=True):
    """학습(Train) 세트에만 on-the-fly 적용. 논문은 증강 사용을 명시(ref.21/22)하나
    구체 기법은 미기재 - 파라미터는 프로젝트 자체 결정(DEVIATIONS.md 참조)."""
    if rng is None:
        rng = np.random.default_rng()
    aug = data.copy()
    if use_rotation:
        angle = float(rng.uniform(-8.0, 8.0))
        axes = [(0, 1), (0, 2), (1, 2)][int(rng.integers(0, 3))]
        aug = rotate(aug, angle=angle, axes=axes, reshape=False, order=1, mode="nearest")
    if rng.random() < 0.5:
        axis = int(rng.integers(0, 3))
        aug = np.flip(aug, axis=axis)
    scale = float(rng.uniform(0.95, 1.05))
    shift = float(rng.uniform(-0.05, 0.05))
    noise = rng.normal(0.0, 0.01, size=aug.shape)
    aug = aug * scale + shift + noise
    return aug.astype(np.float32)


def get_holdout_split(csv_path, seed=42, train_frac=0.70, val_frac=0.15):
    """
    3D-CNN / 3D-ResNet 자체 학습용 분할 (Table 5: 70% Train, 15%+15% Test/Val)
    피험자 단위 stratified 분할.
    """
    rng = np.random.default_rng(seed)
    samples = _load_samples(csv_path)
    subject_to_samples = _group_by_subject(samples)
    subjects, labels = _subject_labels(subject_to_samples)

    label_to_subs = {}
    for sub, lab in zip(subjects, labels):
        label_to_subs.setdefault(lab, []).append(sub)

    train_subs, val_subs, test_subs = [], [], []
    for lab, subs in label_to_subs.items():
        subs = np.array(subs)
        rng.shuffle(subs)
        n = len(subs)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        train_subs.extend(subs[:n_train])
        val_subs.extend(subs[n_train:n_train + n_val])
        test_subs.extend(subs[n_train + n_val:])

    def expand(subs):
        out = []
        for sub in subs:
            out.extend(subject_to_samples[sub])
        return out

    return expand(train_subs), expand(val_subs), expand(test_subs)


def get_kfold_splits(csv_path, k=5, seed=42):
    """
    Model 3/4 (CCA 융합 + WOA 최적화) 평가용 k-fold 분할 (Table 6: k=5, 10, 15 지원)
    피험자 단위 stratified k-fold. 각 fold: (train_samples, test_samples) 튜플을 yield.
    """
    if k not in (5, 10, 15):
        raise ValueError(f"논문 Table 6은 k=5,10,15만 사용합니다 (요청값: {k})")

    samples = _load_samples(csv_path)
    subject_to_samples = _group_by_subject(samples)
    subjects, labels = _subject_labels(subject_to_samples)

    label_to_subs = {}
    for sub, lab in zip(subjects, labels):
        label_to_subs.setdefault(lab, []).append(sub)

    rng = np.random.default_rng(seed)
    # 라벨별로 subject를 섞은 뒤 k개 fold로 균등 분배 (stratified k-fold, subject-level)
    label_folds = {}
    for lab, subs in label_to_subs.items():
        subs = np.array(subs)
        rng.shuffle(subs)
        label_folds[lab] = np.array_split(subs, k)

    for fold_idx in range(k):
        test_subs = []
        train_subs = []
        for lab in label_folds:
            for i, chunk in enumerate(label_folds[lab]):
                if i == fold_idx:
                    test_subs.extend(chunk)
                else:
                    train_subs.extend(chunk)

        def expand(subs):
            out = []
            for sub in subs:
                out.extend(subject_to_samples[sub])
            return out

        yield expand(train_subs), expand(test_subs)


def get_dataloaders(csv_path, image_dir, batch_size=8, seed=42, use_augmentation=True):
    train_samples, val_samples, test_samples = get_holdout_split(csv_path, seed=seed)

    exists = lambda s: os.path.exists(os.path.join(image_dir, f"{s['sample_id']}.nii.gz"))
    train_samples = [s for s in train_samples if exists(s)]
    val_samples = [s for s in val_samples if exists(s)]
    test_samples = [s for s in test_samples if exists(s)]

    train_transform = (lambda x: augment_volume_3d(x, np.random.default_rng())) if use_augmentation else None
    train_ds = PPMIT2Dataset(train_samples, image_dir, transform=train_transform)
    val_ds = PPMIT2Dataset(val_samples, image_dir, transform=None)
    test_ds = PPMIT2Dataset(test_samples, image_dir, transform=None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Holdout Split: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)} "
          f"(augmentation={'on' if use_augmentation else 'off'})")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    csv_path = r"d:\Brain_Tensor\01_Preprocessing\data_final_303.csv"
    for k in (5, 10, 15):
        folds = list(get_kfold_splits(csv_path, k=k))
        sizes = [(len(tr), len(te)) for tr, te in folds]
        print(f"k={k}: fold sizes (train,test) = {sizes}")
