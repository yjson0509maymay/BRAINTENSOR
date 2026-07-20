# pip install dicom2nifti nibabel scipy numpy antspyx
#
# External neuroimaging tools expected for full reproduction:
#   FSL: bet
#   ANTs: N4BiasFieldCorrection (CLI)
#   ANTsPy (antspyx, python package "ants"): registration
#
# [2026-07-19 신규 작성] preparing_ref21order.py(v1)의 후속 버전 - N4 위치 재검토
#
# v1과의 유일한 차이점: Bias Field Correction(N4) 위치.
#   - v1: 주논문 Methods(p.4) "brain extraction, registration, normalization and data
#     augmentation" 4단계 목록을 우선시해, 이 목록에 없는 N4는 기본 비활성화(옵션으로만 켤 수
#     있었고, 켜면 정규화 이후에 적용).
#   - v2(이 파일): 주논문 Table3 앞부분(본문 463~465줄, ref.31=Smith 2002=BET 인용부)의
#     "The NifTi file is further subjected to pre-processing, which includes skull
#     stripping and field correction..." 서술을 근거로 채택. 이 문장은 skull stripping과
#     field correction을 "초기 단계"로 묶어서 서술하므로, N4를 **뇌추출(BET) 직후, 정합
#     이전**에 항상 적용하도록 변경(v1처럼 옵션이 아니라 기본 파이프라인에 포함).
#     -> 이건 기존 preparing.py(최초 버전)가 원래 채택했던 순서와 같음. 다만 정합
#     도구/아틀라스(ANTsPy+MNIPD25)와 정규화(min-max)는 v1 그대로 유지 - 이 부분은
#     참고문헌21을 따르기로 한 별도 결정이라 이번 변경 범위가 아님.
#
# 즉 파이프라인 순서: BET -> **N4(항상 적용)** -> ANTsPy 정합(MNIPD25) -> 정규화(min-max) -> 리사이즈
# (v1 순서:            BET -> ANTsPy 정합(MNIPD25) -> 정규화(min-max) -> [N4 기본 비활성] -> 리사이즈)
#
# 주논문이 전처리를 설명하는 두 서술이 서로 다르다는 점 자체는 v1과 동일하게 남아있음
# (Methods 목록엔 N4가 없고, Table3 앞 서술엔 있음 - 03_Model_Training/
# 세션_기록_전처리_재구현_및_CNN_재학습.md 5번 항목 및 03_Model_Training/
# DEVIATIONS.md 참조). v1은 Methods를, v2는 Table3 앞 서술을 근거로 삼음 - 두 버전을
# 나란히 실행/비교해서 어느 쪽이 실제 성능에 유리한지 실험적으로 확인하는 것이 목적.
#
# 실행 전 필요 사항(미충족 시 즉시 에러로 안내):
#   - pip install antspyx
#   - MNIPD25-T1MPRAGE-1 아틀라스 파일(예: 00_RawData/atlas/PD25-T1MPRAGE-template-1mm.nii.gz)

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

import dicom2nifti
import dicom2nifti.settings as settings
import nibabel as nib
import numpy as np
from scipy.ndimage import rotate, zoom

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

settings.disable_validate_slice_increment()


DEFAULT_TARGET_SHAPE = (56, 56, 56)  # 주논문 명시값(56x56x56), 변경 없음


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


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


# FSL(bet)/ANTs CLI(N4BiasFieldCorrection)가 네이티브 Windows PATH에 없는 경우(이 프로젝트
# 환경처럼 WSL 안에만 설치된 경우) 자동으로 WSL을 경유해서 실행. FSL/ANTs의 C 기반 바이너리는
# 경로에 공백이나 비-ASCII(한글 등) 문자가 있으면 자체적으로 실패하는 것을 v1에서 확인함 -
# WSL 내부의 공백/한글 없는 임시 경로(/tmp/ppmi_stage_<uuid>)로 복사한 뒤 처리하고, 결과만
# 원래 위치로 복사해 돌려줌(v1과 동일 우회 로직).
FSL_BIN = "/usr/local/fsl/bin"
ANTS_BIN = "/usr/lib/ants"
FSL_ENV_PREFIX = (
    f"PATH={FSL_BIN}:{ANTS_BIN}:/usr/bin:/bin FSLDIR=/usr/local/fsl FSLOUTPUTTYPE=NIFTI_GZ"
)


def _wsl_available():
    return shutil.which("wsl") is not None


def _to_wsl_path(win_path):
    win_path = os.path.abspath(win_path)
    drive, rest = os.path.splitdrive(win_path)
    drive_letter = drive.rstrip(":").lower()
    rest = rest.replace("\\", "/")
    return f"/mnt/{drive_letter}{rest}"


def _run_wsl_bash(bash_command, step_name):
    result = subprocess.run(
        ["wsl", "bash", "-c", bash_command],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{step_name} failed(WSL): {message}")
    return result


def run_via_wsl_clean_staging(input_path, output_path, build_command, step_name):
    """input_path -> (WSL 내부 공백/한글 없는 임시 경로로 복사) -> build_command 실행
    -> 결과를 output_path(원래 경로, 공백/한글 포함 가능)로 복사."""
    if not _wsl_available():
        raise RuntimeError(
            f"{step_name}: 네이티브 실행 파일도 없고 WSL도 사용할 수 없습니다. "
            "FSL/ANTs를 Windows PATH에 설치하거나 WSL을 사용 가능하게 하세요."
        )

    stage_id = uuid.uuid4().hex[:12]
    stage_dir = f"/tmp/ppmi_stage_{stage_id}"
    stage_in = f"{stage_dir}/in.nii.gz"
    stage_out = f"{stage_dir}/out.nii.gz"
    input_wsl = _to_wsl_path(input_path)

    _run_wsl_bash(f"mkdir -p {stage_dir} && cp '{input_wsl}' {stage_in}", f"{step_name}(입력 스테이징)")
    try:
        command = build_command(stage_in, stage_out)
        _run_wsl_bash(f"{FSL_ENV_PREFIX} {command}", step_name)

        ensure_dir(os.path.dirname(os.path.abspath(output_path)))
        output_wsl = _to_wsl_path(output_path)
        _run_wsl_bash(f"cp {stage_out} '{output_wsl}'", f"{step_name}(결과 복사)")
    finally:
        _run_wsl_bash(f"rm -rf {stage_dir}", f"{step_name}(정리)")

    return output_path


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
    # 주논문 Results(ref.31=Smith 2002=BET 인용)와 참고문헌22 모두와 일치하는 선택(v1과 동일).
    bet = shutil.which("bet")
    if bet is not None:
        env = os.environ.copy()
        env.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")
        result = subprocess.run(
            [bet, input_path, output_path, "-R", "-f", str(frac), "-g", "0"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(f"FSL BET skull stripping failed: {result.stderr or result.stdout}")
        return output_path

    def build_bet_command(stage_in, stage_out):
        return f"{FSL_BIN}/bet {stage_in} {stage_out} -R -f {frac} -g 0"

    return run_via_wsl_clean_staging(input_path, output_path, build_bet_command, "FSL BET skull stripping")


def _has_non_ascii(path):
    try:
        path.encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def _stage_ascii_in(path):
    """ants(ITK)가 비-ASCII(한글 등) 경로에서 'Could not create ImageIO object' 에러로
    실패하는 것을 v1에서 확인함. 경로에 비-ASCII 문자가 있으면 임시 ASCII 경로로 복사."""
    if not _has_non_ascii(os.path.abspath(path)):
        return path, None
    tmp_dir = tempfile.mkdtemp(prefix="ants_stage_")
    tmp_path = os.path.join(tmp_dir, "img.nii.gz")
    shutil.copy2(path, tmp_path)
    return tmp_path, tmp_dir


def run_registration_antspy(input_path, output_path, atlas_path):
    # 참고문헌21 Table 3, Step 3 그대로: ANTsPy로 MNIPD25-T1MPRAGE-1 아틀라스에
    # affine + non-linear 정합(v1과 동일 - 이번 v2 변경 범위 아님).
    try:
        import ants
    except ImportError as exc:
        raise RuntimeError(
            "ANTsPy(antspyx)가 설치되어 있지 않습니다. 'pip install antspyx' 실행 후 "
            "다시 시도하세요."
        ) from exc

    if not atlas_path:
        raise RuntimeError(
            "MNIPD25-T1MPRAGE-1 아틀라스 경로가 필요합니다(--atlas-path). "
            "로컬에 없다면 https://www.nitrc.org/projects/mni_pd25 등에서 받아 지정하세요."
        )
    if not os.path.exists(atlas_path):
        raise RuntimeError(f"아틀라스 파일을 찾을 수 없습니다: {atlas_path}")

    safe_input, input_tmp = _stage_ascii_in(input_path)
    safe_atlas, atlas_tmp = _stage_ascii_in(atlas_path)
    output_is_non_ascii = _has_non_ascii(os.path.abspath(output_path))
    out_tmp = None
    try:
        if output_is_non_ascii:
            out_tmp = tempfile.mkdtemp(prefix="ants_stage_out_")
            safe_output = os.path.join(out_tmp, "out.nii.gz")
        else:
            safe_output = output_path

        fixed = ants.image_read(safe_atlas)
        moving = ants.image_read(safe_input)
        result = ants.registration(fixed=fixed, moving=moving, type_of_transform="SyN")
        ants.image_write(result["warpedmovout"], safe_output)

        if output_is_non_ascii:
            ensure_dir(os.path.dirname(os.path.abspath(output_path)))
            shutil.copy2(safe_output, output_path)
    finally:
        for d in (input_tmp, atlas_tmp, out_tmp):
            if d:
                shutil.rmtree(d, ignore_errors=True)

    return output_path


def run_n4_field_correction(input_path, output_path):
    # [v2] 이제 항상 호출됨(BET 직후) - 주논문 Table3 앞 서술(ref.31=BET 인용부에서
    # skull stripping과 field correction을 묶어 서술) 근거. 네이티브 우선, 없으면
    # WSL 경유 + 클린 경로 스테이징(v1과 동일 로직).
    n4 = shutil.which("N4BiasFieldCorrection")
    if n4 is not None:
        command = [n4, "-d", "3", "-i", input_path, "-o", output_path]
        run_command(command, "N4 field/bias correction")
        return output_path

    def build_n4_command(stage_in, stage_out):
        return f"{ANTS_BIN}/N4BiasFieldCorrection -d 3 -i {stage_in} -o {stage_out}"

    return run_via_wsl_clean_staging(input_path, output_path, build_n4_command, "N4 field/bias correction")


def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise RuntimeError(f"Expected 3D NIfTI, got shape {data.shape}: {path}")
    return img, data


def normalize_minmax(input_path, output_path):
    # 참고문헌21 Eq.1-3 그대로(v1과 동일 - 이번 v2 변경 범위 아님).
    img, data = load_nifti(input_path)
    finite = np.isfinite(data)
    brain_mask = finite & (data != 0)
    if not np.any(brain_mask):
        raise RuntimeError("Cannot normalize: no non-zero brain voxels found.")

    values = data[brain_mask]
    min_value = float(values.min())
    max_value = float(values.max())
    value_range = max_value - min_value
    if value_range < 1e-6:
        raise RuntimeError("Cannot normalize: brain voxel intensity range is too small.")

    normalized = np.zeros_like(data, dtype=np.float32)
    normalized[brain_mask] = (data[brain_mask] - min_value) / value_range
    nib.save(nib.Nifti1Image(normalized, img.affine, img.header), output_path)
    return output_path


def normalize_zscore(input_path, output_path):
    # 참고문헌22 방식(v1과 동일). 현재 비활성 - 필요 시 --normalization zscore로 전환.
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


NORMALIZATION_FUNCS = {
    "minmax": normalize_minmax,   # 기본값(활성) - 참고문헌21
    "zscore": normalize_zscore,   # 비활성(토글로만 사용) - 참고문헌22
}


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
    # 논문/참고문헌 모두 증강 기법 미기재 - 기존과 동일(프로젝트 자체 결정,
    # DEVIATIONS.md 6번 항목 참조). 이 파일에서 바뀐 부분 아님.
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
    input_path, output_dir, base_name, target_shape=DEFAULT_TARGET_SHAPE,
    augment_count=0, seed=42,
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


def preprocess_one_from_nifti(
    raw_nifti_path, output_dir, name, atlas_path, normalization="minmax",
    target_shape=DEFAULT_TARGET_SHAPE, bet_frac=0.5,
    augment_count=0, seed=42, keep_intermediate=False,
):
    """[v2] 순서: 뇌추출(BET) -> N4(항상 적용) -> 정합(ANTsPy) -> 정규화 -> 리사이즈.
    DICOM 대신 이미 변환된 raw NIfTI(예: 01_raw_nifti)에서 시작."""
    ensure_dir(output_dir)
    work_dir = os.path.join(output_dir, "_work", name)
    ensure_dir(work_dir)

    brain_nifti = os.path.join(work_dir, f"{name}_01_bet.nii.gz")
    corrected_nifti = os.path.join(work_dir, f"{name}_02_n4.nii.gz")
    registered_nifti = os.path.join(work_dir, f"{name}_03_reg.nii.gz")
    normalized_nifti = os.path.join(work_dir, f"{name}_04_norm.nii.gz")
    final_nifti = os.path.join(output_dir, f"{name}.nii.gz")

    normalize_fn = NORMALIZATION_FUNCS[normalization]
    original_shape = load_nifti(raw_nifti_path)[1].shape

    run_skull_stripping_bet(raw_nifti_path, brain_nifti, frac=bet_frac)
    run_n4_field_correction(brain_nifti, corrected_nifti)
    run_registration_antspy(corrected_nifti, registered_nifti, atlas_path)
    normalize_fn(registered_nifti, normalized_nifti)

    resize_nifti(normalized_nifti, final_nifti, target_shape=target_shape)
    augmented_outputs = save_augmented_resized(
        normalized_nifti, output_dir, name, target_shape=target_shape,
        augment_count=augment_count, seed=seed,
    )

    if not keep_intermediate:
        shutil.rmtree(work_dir, ignore_errors=True)

    return {
        "original_shape": "x".join(map(str, original_shape)),
        "final_shape": "x".join(map(str, target_shape)),
        "augmented_count": len(augmented_outputs),
    }


def write_run_manifest(output_dir, stamp, manifest):
    manifest_path = os.path.join(output_dir, f"run_manifest_{stamp}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path


def run_batch_from_nifti(
    raw_nifti_dir, output_dir, atlas_path, normalization="minmax",
    target_shape=DEFAULT_TARGET_SHAPE, skip_existing=True,
    bet_frac=0.5, augment_count=0, seed=42, keep_intermediate=False,
):
    ensure_dir(output_dir)
    raw_files = sorted(f for f in os.listdir(raw_nifti_dir) if f.endswith(".nii.gz"))
    total = len(raw_files)
    t_start = time.time()
    print(f"Found raw NIfTI files: {total}\n")
    print(f"[v2] Normalization: {normalization} (active) | pipeline order: "
          f"BET -> N4(항상 적용) -> ANTsPy registration -> normalize -> resize {target_shape}\n")

    results = []
    n_ok = n_fail = n_skip = 0

    for i, fname in enumerate(raw_files, start=1):
        name = fname[: -len(".nii.gz")]
        raw_path = os.path.join(raw_nifti_dir, fname)
        out_path = os.path.join(output_dir, fname)
        prefix = f"[{i}/{total}] {name}"

        if skip_existing and os.path.exists(out_path):
            print(f"{prefix} - exists, skipped")
            n_skip += 1
            results.append({
                "name": name, "status": "skipped", "original_shape": "",
                "final_shape": "", "augmented_count": "", "elapsed_sec": "",
                "message": "", "source": raw_path,
            })
            continue

        try:
            t0 = time.time()
            info = preprocess_one_from_nifti(
                raw_nifti_path=raw_path, output_dir=output_dir, name=name,
                atlas_path=atlas_path, normalization=normalization,
                target_shape=target_shape, bet_frac=bet_frac,
                augment_count=augment_count, seed=seed + i,
                keep_intermediate=keep_intermediate,
            )
            sample_elapsed = time.time() - t0
            print(f"{prefix} - done ({sample_elapsed:.1f}s, original {info['original_shape']} "
                  f"-> {info['final_shape']})")
            n_ok += 1
            results.append({
                "name": name, "status": "ok", "original_shape": info["original_shape"],
                "final_shape": info["final_shape"], "augmented_count": info["augmented_count"],
                "elapsed_sec": round(sample_elapsed, 1), "message": "", "source": raw_path,
            })
        except Exception as exc:
            print(f"{prefix} - failed: {exc}")
            n_fail += 1
            results.append({
                "name": name, "status": "failed", "original_shape": "", "final_shape": "",
                "augmented_count": "", "elapsed_sec": "", "message": str(exc), "source": raw_path,
            })

    total_elapsed = time.time() - t_start
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"preprocessing_log_ref21order_v2_{stamp}.csv")
    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["name", "status", "original_shape", "final_shape",
                           "augmented_count", "elapsed_sec", "message", "source"],
        )
        writer.writeheader()
        writer.writerows(results)

    manifest_path = write_run_manifest(output_dir, stamp, {
        "timestamp": stamp, "script": "preparing_ref21order_v2.py", "input_source": raw_nifti_dir,
        "output_dir": output_dir, "atlas_path": atlas_path, "normalization": normalization,
        "bias_correction_position": "after BET, before registration (always on)",
        "target_shape": list(target_shape), "bet_frac": bet_frac,
        "total": total, "success": n_ok, "failed": n_fail, "skipped": n_skip,
        "elapsed_sec": round(total_elapsed, 1), "csv_log": log_path,
    })

    print("\n===== Batch complete =====")
    print(f"success {n_ok} / failed {n_fail} / skipped {n_skip} (total {total})")
    print(f"elapsed: {total_elapsed:.1f}s")
    print(f"csv log: {log_path}")
    print(f"manifest: {manifest_path}")


def parse_shape(value):
    parts = value.lower().replace(",", "x").split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be like 56x56x56")
    return tuple(int(part) for part in parts)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "PPMI T2 preprocessing, v2 (N4-before-registration variant) "
            "(BET -> N4(항상 적용) -> ANTsPy affine+nonlinear registration -> normalize -> resize)."
        )
    )
    parser.add_argument(
        "--from-nifti-dir", required=True,
        help="이미 변환된 raw NIfTI 폴더(예: 01_raw_nifti)",
    )
    parser.add_argument("--output-dir", default=r"D:\new tensor\01_Preprocessing\전처리_ref21order_v2")
    parser.add_argument(
        "--atlas-path",
        default=os.environ.get("MNIPD25_ATLAS", ""),
        help="Path to MNIPD25-T1MPRAGE-1 atlas (참고문헌21). 로컬에 없으면 직접 받아서 지정.",
    )
    parser.add_argument(
        "--normalization", choices=list(NORMALIZATION_FUNCS.keys()), default="minmax",
        help="정규화 방식. 기본값 minmax(참고문헌21, 활성). zscore(참고문헌22)는 비활성 상태로 남겨둠.",
    )
    parser.add_argument("--target-shape", type=parse_shape, default=DEFAULT_TARGET_SHAPE)
    parser.add_argument("--bet-frac", type=float, default=0.5)
    parser.add_argument("--augment-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-keep-intermediate", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_batch_from_nifti(
        raw_nifti_dir=args.from_nifti_dir,
        output_dir=args.output_dir,
        atlas_path=args.atlas_path,
        normalization=args.normalization,
        target_shape=args.target_shape,
        skip_existing=not args.overwrite,
        bet_frac=args.bet_frac,
        augment_count=args.augment_count,
        seed=args.seed,
        keep_intermediate=not args.no_keep_intermediate,
    )
