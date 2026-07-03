import argparse
import csv
import importlib.util
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import dicom2nifti
import dicom2nifti.settings as dcm_settings
import nibabel as nib
import numpy as np
import pydicom


dcm_settings.disable_validate_slice_increment()

STAGES = [
    ("01_raw_nifti", "DICOM converted to a reoriented 3D NIfTI volume."),
    ("02_bet", "FSL BET brain extraction output."),
    ("03_n4", "ANTs N4 bias-field corrected brain volume."),
    ("04_mni152", "FSL FLIRT 12-DOF registration to the MNI152 template."),
    ("05_normalized", "Non-zero brain voxels z-score normalized and clipped to [-5, 5]."),
    ("06_resized", "Final model input resized to 56 x 56 x 56 voxels."),
]


def load_preparing(path):
    spec = importlib.util.spec_from_file_location("paper_preparing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_echo_time(path):
    try:
        ds = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            specific_tags=["EchoTime"],
            force=True,
        )
        value = getattr(ds, "EchoTime", None)
        return float(value) if value not in (None, "") else None
    except Exception:
        return None


def select_t2_echo_files(dicom_dir):
    files = [
        str(Path(dicom_dir) / name)
        for name in os.listdir(dicom_dir)
        if name.lower().endswith(".dcm")
    ]
    echo_groups = {}
    unknown = []
    for path in files:
        echo_time = read_echo_time(path)
        if echo_time is None:
            unknown.append(path)
        else:
            echo_groups.setdefault(round(echo_time, 4), []).append(path)
    if len(echo_groups) <= 1:
        return files, next(iter(echo_groups), None), False
    selected_te = max(echo_groups)
    return echo_groups[selected_te], selected_te, True


def convert_t2_dicom(dicom_dir, output_path):
    selected_files, echo_time, echo_filtered = select_t2_echo_files(dicom_dir)
    input_dir = dicom_dir
    filtered_dir = None
    converted_dir = tempfile.mkdtemp(prefix="t2_nifti_")
    try:
        if echo_filtered:
            filtered_dir = tempfile.mkdtemp(prefix="t2_echo_")
            for index, source in enumerate(selected_files):
                target = os.path.join(filtered_dir, f"slice_{index:06d}.dcm")
                shutil.copy2(source, target)
            input_dir = filtered_dir

        dicom2nifti.convert_directory(input_dir, converted_dir, compression=True, reorient=True)
        candidates = []
        for name in os.listdir(converted_dir):
            if not (name.endswith(".nii") or name.endswith(".nii.gz")):
                continue
            path = os.path.join(converted_dir, name)
            image = nib.load(path)
            data = image.get_fdata(dtype=np.float32)
            if data.ndim == 4:
                data = data[..., -1]
                image = nib.Nifti1Image(data.astype(np.float32), image.affine, image.header)
                collapsed = os.path.join(converted_dir, "collapsed_" + name)
                nib.save(image, collapsed)
                path = collapsed
            if data.ndim == 3:
                candidates.append((int(np.prod(data.shape)), path, data.shape))
        if not candidates:
            raise RuntimeError("DICOM conversion produced no 3D NIfTI volume")
        _, selected_path, shape = max(candidates, key=lambda item: item[0])
        shutil.copy2(selected_path, output_path)
        return echo_time, echo_filtered, "x".join(map(str, shape))
    finally:
        shutil.rmtree(converted_dir, ignore_errors=True)
        if filtered_dir:
            shutil.rmtree(filtered_dir, ignore_errors=True)


def ensure_readmes(output_root):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    root_text = """# Paper-aligned T2 preprocessing\n\nInput: `E:/ppmi_dti/raw/data.csv` and raw DICOM folders.\n\nPipeline: DICOM to NIfTI, BET, N4, MNI152 registration, intensity normalization, and 56x56x56 resize. Each stage folder is the input to the next stage. FLAIR data is excluded from this cohort.\n"""
    (root / "README.md").write_text(root_text, encoding="utf-8")
    for index, (name, description) in enumerate(STAGES):
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        previous_name = "raw DICOM from data.csv" if index == 0 else STAGES[index - 1][0]
        next_name = "model training input" if index == len(STAGES) - 1 else STAGES[index + 1][0]
        text = (
            f"# {name}\n\nPurpose: {description}\n\n"
            f"Input: `{previous_name}`\n\nOutput: one `.nii.gz` per sample.\n\n"
            f"Next: `{next_name}`\n"
        )
        (folder / "README.md").write_text(text, encoding="utf-8")
    for name, purpose in [
        ("logs", "Batch and per-sample preprocessing logs."),
        ("visualization", "Before/after figures and PPT-ready pipeline images."),
    ]:
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "README.md").write_text(
            f"# {name}\n\nPurpose: {purpose}\n", encoding="utf-8"
        )


def stage_path(root, stage, sample_id):
    return str(Path(root) / stage / f"{sample_id}.nii.gz")


def platform_path(path):
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", path)
    if os.name != "nt" and match:
        drive = match.group(1).lower()
        tail = match.group(2).replace("\\", "/")
        return f"/mnt/{drive}/{tail}"
    return path


def process_sample(row, config):
    start_total = time.time()
    sample_id = row["sample_id"]
    preparing = load_preparing(config["preparing_path"])
    timings = {}
    result = {
        "sample_id": sample_id,
        "Subject": row["Subject"],
        "Image Data ID": row["Image Data ID"],
        "Group": row["Group"],
        "status": "ok",
        "message": "",
        "selected_echo_time": "",
        "echo_filtered": "no",
        "original_shape": "",
        "final_shape": "56x56x56",
    }
    paths = {stage: stage_path(config["output_root"], stage, sample_id) for stage, _ in STAGES}
    try:
        if config["overwrite"] or not os.path.exists(paths["01_raw_nifti"]):
            started = time.time()
            echo_time, echo_filtered, shape = convert_t2_dicom(
                platform_path(row["raw_dicom_dir"]), paths["01_raw_nifti"]
            )
            timings["01_seconds"] = time.time() - started
            result["selected_echo_time"] = "" if echo_time is None else echo_time
            result["echo_filtered"] = "yes" if echo_filtered else "no"
            result["original_shape"] = shape
        else:
            image = nib.load(paths["01_raw_nifti"])
            result["original_shape"] = "x".join(map(str, image.shape))

        operations = [
            ("02_bet", preparing.run_skull_stripping_bet, (paths["01_raw_nifti"], paths["02_bet"])),
            ("03_n4", preparing.run_n4_field_correction, (paths["02_bet"], paths["03_n4"])),
            ("04_mni152", preparing.run_mni_registration_flirt, (paths["03_n4"], paths["04_mni152"], config["mni_template"])),
            ("05_normalized", preparing.normalize_intensity, (paths["04_mni152"], paths["05_normalized"])),
            ("06_resized", preparing.resize_nifti, (paths["05_normalized"], paths["06_resized"])),
        ]
        for stage, function, arguments in operations:
            if config["overwrite"] or not os.path.exists(paths[stage]):
                started = time.time()
                function(*arguments)
                timings[stage[:2] + "_seconds"] = time.time() - started
    except Exception as exc:
        result["status"] = "failed"
        result["message"] = str(exc)
    result.update({key: round(value, 3) for key, value in timings.items()})
    result["total_seconds"] = round(time.time() - start_total, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description="Staged paper-aligned T2 preprocessing.")
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mni-template", required=True)
    parser.add_argument("--preparing-path", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = read_rows(args.data_csv)
    if args.limit > 0:
        rows = rows[: args.limit]
    ensure_readmes(args.output_root)
    config = {
        "output_root": args.output_root,
        "mni_template": args.mni_template,
        "preparing_path": args.preparing_path,
        "overwrite": args.overwrite,
    }
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_sample, row, config): row for row in rows}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[{index}/{len(rows)}] {result['sample_id']} - {result['status']} "
                f"({result['total_seconds']}s)",
                flush=True,
            )
            fields = sorted({key for item in results for key in item})
            write_rows(Path(args.output_root) / "logs" / "preprocessing_log.csv", results, fields)

    ok = sum(item["status"] == "ok" for item in results)
    failed = len(results) - ok
    print(f"Complete: success={ok}, failed={failed}, total={len(results)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
