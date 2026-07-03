# pip install dicom2nifti nibabel scipy numpy
#
# External neuroimaging tools expected for full reproduction:
#   FSL: bet, flirt
#   ANTs: N4BiasFieldCorrection
#
# Paper-aligned preprocessing summary:
#   DICOM -> 3D NIfTI -> skull stripping(BET) -> field correction(N4)
#   -> registration(MNI152) -> intensity normalization -> optional augmentation
#   -> resize to 56x56x56

import argparse
import csv
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import dicom2nifti
import dicom2nifti.settings as settings
import nibabel as nib
import numpy as np
from scipy.ndimage import rotate, zoom


# PPMI DICOM series sometimes fail validation because of tiny slice-spacing
# rounding differences. The original script used the same setting.
settings.disable_validate_slice_increment()


DEFAULT_TARGET_SHAPE = (56, 56, 56)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def require_executable(command):
    executable = shutil.which(command)
    if executable is None:
        raise RuntimeError(
            f"Required command not found: {command}. "
            "Install FSL/ANTs or pass a valid executable on PATH."
        )
    return executable


def run_command(command, step_name):
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{step_name} failed: {message}")
    return result


def convert_dicom_to_nifti(dicom_dir, output_nifti_path):
    temp_nifti_dir = tempfile.mkdtemp(prefix="dcm2nii_")
    try:
        dicom2nifti.convert_directory(
            dicom_dir, temp_nifti_dir, compression=True, reorient=True
        )
        nifti_files = [
            f for f in os.listdir(temp_nifti_dir)
            if f.endswith(".nii") or f.endswith(".nii.gz")
        ]
        if not nifti_files:
            raise RuntimeError("NIfTI conversion produced no file; check DICOM series.")

        converted_path = os.path.join(temp_nifti_dir, sorted(nifti_files)[0])
        shutil.copy2(converted_path, output_nifti_path)
        return output_nifti_path
    finally:
        shutil.rmtree(temp_nifti_dir, ignore_errors=True)


def run_skull_stripping_bet(input_path, output_path, frac=0.5):
    # Reference used by the paper for skull stripping:
    # Smith, S. Fast robust automated brain extraction. Hum Brain Mapp. 2002.
    bet = require_executable("bet")
    command = [
        bet,
        input_path,
        output_path,
        "-R",
        "-f",
        str(frac),
        "-g",
        "0",
    ]
    run_command(command, "FSL BET skull stripping")
    return output_path


def run_n4_field_correction(input_path, output_path):
    n4 = require_executable("N4BiasFieldCorrection")
    command = [
        n4,
        "-d",
        "3",
        "-i",
        input_path,
        "-o",
        output_path,
    ]
    run_command(command, "N4 field/bias correction")
    return output_path


def run_mni_registration_flirt(input_path, output_path, reference_path):
    if not reference_path:
        raise RuntimeError("MNI152 template path is required for registration.")
    if not os.path.exists(reference_path):
        raise RuntimeError(f"MNI152 template not found: {reference_path}")

    flirt = require_executable("flirt")
    matrix_path = output_path.replace(".nii.gz", "_mni.mat").replace(".nii", "_mni.mat")
    command = [
        flirt,
        "-in",
        input_path,
        "-ref",
        reference_path,
        "-out",
        output_path,
        "-omat",
        matrix_path,
        "-dof",
        "12",
        "-interp",
        "trilinear",
    ]
    run_command(command, "FSL FLIRT MNI152 registration")
    return output_path


def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise RuntimeError(f"Expected 3D NIfTI, got shape {data.shape}: {path}")
    return img, data


def normalize_intensity(input_path, output_path):
    img, data = load_nifti(input_path)
    finite = np.isfinite(data)
    brain_mask = finite & (data != 0)
    if not np.any(brain_mask):
        raise RuntimeError("Cannot normalize: no non-zero brain voxels found.")

    values = data[brain_mask]
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-6:
        raise RuntimeError("Cannot normalize: brain voxel standard deviation is too small.")

    normalized = np.zeros_like(data, dtype=np.float32)
    normalized[brain_mask] = (data[brain_mask] - mean) / std
    normalized = np.clip(normalized, -5.0, 5.0)
    nib.save(nib.Nifti1Image(normalized, img.affine, img.header), output_path)
    return output_path


def resize_nifti(input_path, output_path, target_shape=DEFAULT_TARGET_SHAPE, order=1):
    img, data = load_nifti(input_path)
    original_shape = data.shape
    factors = np.array(target_shape, dtype=float) / np.array(original_shape, dtype=float)
    resized_data = zoom(data, factors, order=order)

    new_affine = img.affine.copy()
    new_affine[:3, :3] = img.affine[:3, :3] / factors

    resized_img = nib.Nifti1Image(resized_data.astype(np.float32), new_affine, img.header)
    nib.save(resized_img, output_path)
    return original_shape


def augment_volume(data, rng):
    # The main paper only states that augmentation was used via refs. 21,22
    # without publishing exact parameters. These conservative 3D transforms are
    # therefore kept configurable and should be applied to training data only.
    augmented = data.copy()
    angle = float(rng.uniform(-8.0, 8.0))
    axes = [(0, 1), (0, 2), (1, 2)][int(rng.integers(0, 3))]
    augmented = rotate(augmented, angle=angle, axes=axes, reshape=False, order=1, mode="nearest")

    if rng.random() < 0.5:
        axis = int(rng.integers(0, 3))
        augmented = np.flip(augmented, axis=axis)

    scale = float(rng.uniform(0.95, 1.05))
    shift = float(rng.uniform(-0.05, 0.05))
    noise = rng.normal(0.0, 0.01, size=augmented.shape)
    augmented = augmented * scale + shift + noise
    return augmented.astype(np.float32)


def save_augmented_resized(
    input_path,
    output_dir,
    base_name,
    target_shape=DEFAULT_TARGET_SHAPE,
    augment_count=0,
    seed=42,
):
    if augment_count <= 0:
        return []

    img, data = load_nifti(input_path)
    rng = np.random.default_rng(seed)
    outputs = []

    for idx in range(augment_count):
        augmented = augment_volume(data, rng)
        temp_img = nib.Nifti1Image(augmented, img.affine, img.header)
        temp_path = os.path.join(output_dir, f"{base_name}_aug{idx + 1:02d}_pre_resize.nii.gz")
        out_path = os.path.join(output_dir, f"{base_name}_aug{idx + 1:02d}.nii.gz")
        nib.save(temp_img, temp_path)
        resize_nifti(temp_path, out_path, target_shape=target_shape)
        os.remove(temp_path)
        outputs.append(out_path)

    return outputs


def preprocess_one(
    dicom_dir,
    output_dir,
    name,
    mni_template,
    target_shape=DEFAULT_TARGET_SHAPE,
    bet_frac=0.5,
    augment_count=0,
    seed=42,
    keep_intermediate=True,
):
    ensure_dir(output_dir)
    work_dir = os.path.join(output_dir, "_work", name)
    ensure_dir(work_dir)

    raw_nifti = os.path.join(work_dir, f"{name}_00_raw.nii.gz")
    brain_nifti = os.path.join(work_dir, f"{name}_01_bet.nii.gz")
    corrected_nifti = os.path.join(work_dir, f"{name}_02_n4.nii.gz")
    registered_nifti = os.path.join(work_dir, f"{name}_03_mni152.nii.gz")
    normalized_nifti = os.path.join(work_dir, f"{name}_04_norm.nii.gz")
    final_nifti = os.path.join(output_dir, f"{name}.nii.gz")

    convert_dicom_to_nifti(dicom_dir, raw_nifti)
    original_shape = load_nifti(raw_nifti)[1].shape

    run_skull_stripping_bet(raw_nifti, brain_nifti, frac=bet_frac)
    run_n4_field_correction(brain_nifti, corrected_nifti)
    run_mni_registration_flirt(corrected_nifti, registered_nifti, mni_template)
    normalize_intensity(registered_nifti, normalized_nifti)
    resize_nifti(normalized_nifti, final_nifti, target_shape=target_shape)
    augmented_outputs = save_augmented_resized(
        normalized_nifti,
        output_dir,
        name,
        target_shape=target_shape,
        augment_count=augment_count,
        seed=seed,
    )

    if not keep_intermediate:
        shutil.rmtree(work_dir, ignore_errors=True)

    return {
        "original_shape": "x".join(map(str, original_shape)),
        "final_shape": "x".join(map(str, target_shape)),
        "augmented_count": len(augmented_outputs),
    }


def find_dicom_series_dirs(root):
    dirs = []
    for dirpath, _, filenames in os.walk(root):
        if any(f.lower().endswith(".dcm") for f in filenames):
            dirs.append(dirpath)
    return sorted(dirs)


def derive_output_name(dicom_dir, ppmi_root):
    try:
        rel_parts = os.path.relpath(dicom_dir, ppmi_root).split(os.sep)
        subject_id = rel_parts[0]
    except ValueError:
        subject_id = "unknown"
    image_id = os.path.basename(dicom_dir.rstrip(os.sep))
    return f"sub-{subject_id}_{image_id}"


def run_batch(
    ppmi_root,
    output_dir,
    mni_template,
    target_shape=DEFAULT_TARGET_SHAPE,
    skip_existing=True,
    bet_frac=0.5,
    augment_count=0,
    seed=42,
    keep_intermediate=True,
):
    ensure_dir(output_dir)
    series_dirs = find_dicom_series_dirs(ppmi_root)
    total = len(series_dirs)
    print(f"Found DICOM series folders: {total}\n")

    results = []
    n_ok = n_fail = n_skip = 0

    for i, dicom_dir in enumerate(series_dirs, start=1):
        name = derive_output_name(dicom_dir, ppmi_root)
        out_path = os.path.join(output_dir, name + ".nii.gz")
        prefix = f"[{i}/{total}] {name}"

        if skip_existing and os.path.exists(out_path):
            print(f"{prefix} - exists, skipped")
            n_skip += 1
            results.append({
                "name": name,
                "status": "skipped",
                "original_shape": "",
                "final_shape": "",
                "augmented_count": "",
                "message": "",
                "source": dicom_dir,
            })
            continue

        try:
            info = preprocess_one(
                dicom_dir=dicom_dir,
                output_dir=output_dir,
                name=name,
                mni_template=mni_template,
                target_shape=target_shape,
                bet_frac=bet_frac,
                augment_count=augment_count,
                seed=seed + i,
                keep_intermediate=keep_intermediate,
            )
            print(
                f"{prefix} - done "
                f"(original {info['original_shape']} -> {info['final_shape']}, "
                f"aug {info['augmented_count']})"
            )
            n_ok += 1
            results.append({
                "name": name,
                "status": "ok",
                "original_shape": info["original_shape"],
                "final_shape": info["final_shape"],
                "augmented_count": info["augmented_count"],
                "message": "",
                "source": dicom_dir,
            })
        except Exception as exc:
            print(f"{prefix} - failed: {exc}")
            n_fail += 1
            results.append({
                "name": name,
                "status": "failed",
                "original_shape": "",
                "final_shape": "",
                "augmented_count": "",
                "message": str(exc),
                "source": dicom_dir,
            })

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"preprocessing_log_{stamp}.csv")
    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "status",
                "original_shape",
                "final_shape",
                "augmented_count",
                "message",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n===== Batch complete =====")
    print(f"success {n_ok} / failed {n_fail} / skipped {n_skip} (total {total})")
    print(f"log: {log_path}")


def parse_shape(value):
    parts = value.lower().replace(",", "x").split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be like 56x56x56")
    return tuple(int(part) for part in parts)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="PPMI T2 DICOM preprocessing for paper-style 3D-CNN input."
    )
    parser.add_argument("--ppmi-root", default=r"E:\ppmi_dti\raw\data\PPMI")
    parser.add_argument("--output-dir", default=r"E:\ppmi_dti\preparing\nifti")
    parser.add_argument(
        "--mni-template",
        default=os.environ.get("MNI152_TEMPLATE", ""),
        help="Path to MNI152_T1_1mm_brain.nii.gz or another MNI152 reference.",
    )
    parser.add_argument("--target-shape", type=parse_shape, default=DEFAULT_TARGET_SHAPE)
    parser.add_argument("--bet-frac", type=float, default=0.5)
    parser.add_argument(
        "--augment-count",
        type=int,
        default=0,
        help=(
            "Offline augmentations per subject. Use after train/test split only "
            "to avoid leakage; default keeps only original samples."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-keep-intermediate", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_batch(
        ppmi_root=args.ppmi_root,
        output_dir=args.output_dir,
        mni_template=args.mni_template,
        target_shape=args.target_shape,
        skip_existing=not args.overwrite,
        bet_frac=args.bet_frac,
        augment_count=args.augment_count,
        seed=args.seed,
        keep_intermediate=not args.no_keep_intermediate,
    )
