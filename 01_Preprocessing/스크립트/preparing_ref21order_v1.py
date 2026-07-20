# pip install dicom2nifti nibabel scipy numpy antspyx
#
# External neuroimaging tools expected for full reproduction:
#   FSL: bet
#   ANTs: N4BiasFieldCorrection (CLI)
#   ANTsPy (antspyx, python package "ants"): registration
#
# [2026-07 신규 작성, 07-18 순서 재수정] preparing.py의 대안 버전
#
# 채택 근거: 주논문 Methods(p.4) 원문 그대로 - "brain extraction, registration,
# normalization and data augmentation as described in the reference21,22" -> 이 목록
# 순서(뇌추출->정합->정규화->증강)를 그대로 따름. 이 목록엔 bias/field correction 자체가
# 없으므로 N4는 기본 비활성화(--enable-bias-correction으로만 켤 수 있음).
#
# preparing.py(기존)와의 차이점:
#   1. 정합(Registration): FSL FLIRT affine-only + 범용 MNI152
#      -> ANTsPy(ants.registration) affine+nonlinear + MNIPD25-T1MPRAGE-1(PD 특화 아틀라스)
#         (참고문헌21 Table 3, Step 3 그대로: "Use Antspyx to align the image with the atlas".
#         정합 "도구" 자체는 참고문헌21을 따르기로 한 별도 결정이며, 아래 3번 순서 결정과는
#         독립적임 - 주논문 Methods는 정합 도구/아틀라스를 명시하지 않음)
#   2. 정규화(Normalization): z-score 단일 방식
#      -> min-max와 z-score 둘 다 구현하되, --normalization 인자로 토글.
#         기본값은 min-max 활성화(참고문헌21 Eq.1-3), z-score는 코드상 남겨두되 비활성.
#   3. Bias Field Correction: 기본 **비활성화**. 주논문 Methods 목록(뇌추출/정합/정규화/증강)에
#      bias correction이 아예 없기 때문 - "목록에 없으면 비활성화"라는 사용자 결정 반영.
#      함수는 남겨두고 --enable-bias-correction 플래그로만 켤 수 있게 함(정규화 이후 위치,
#      참고문헌21 순서 - 켤 경우에 한해서).
#   4. 리사이즈: 56x56x56 유지(주논문 명시값, 참고문헌21은 최종 크기를 규정하지 않음 -
#      참고문헌21은 이 지점에서 피질하부 분할로 이어지지만 우리는 전체 뇌 3D-CNN 입력이 목적이라
#      분할 단계는 채택하지 않음)
#
# 주논문 자체가 말하는 두 서로 다른 순서(참고, PREPROCESSING_DEVIATIONS.md 하단 표와 동일):
#   - Methods(p.4): 뇌추출 -> 정합 -> 정규화 -> 증강. bias correction 언급 없음.
#     -> 이 스크립트가 기본으로 따르는 순서.
#   - Results(p.9, ref.31=Smith 2002=BET 인용부): skull stripping과 field correction을
#     "초기 단계"로 묶어서 서술. -> 이 스크립트는 기본적으로 이 서술을 따르지 않음(N4 비활성).
#     기존 preparing.py는 반대로 이 Results 서술을 우선시해 "뇌추출 직후 N4"를 채택했었음.
#
# 실행 전 필요 사항(미충족 시 즉시 에러로 안내):
#   - pip install antspyx
#   - MNIPD25-T1MPRAGE-1 아틀라스 파일(로컬에 없음 - 사용자가 별도로 구해서 경로 지정 필요.
#     예: https://www.nitrc.org/projects/mni_pd25 참고)

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

# [2026-07-19 추가] stdout이 파일로 리다이렉트되면 기본적으로 완전 버퍼링되어 배치 실행 중
# 로그가 실시간으로 안 보이던 문제 - 진행상황을 즉시 확인할 수 있도록 라인 버퍼링으로 전환.
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


# [2026-07-19 추가] FSL(bet)/ANTs CLI(N4BiasFieldCorrection)가 네이티브 Windows PATH에
# 없는 경우(이 프로젝트 환경처럼 WSL 안에만 설치된 경우) 자동으로 WSL을 경유해서 실행.
# 또한 FSL/ANTs의 C 기반 바이너리는 경로에 공백이나 비-ASCII(한글 등) 문자가 있으면
# 셸 따옴표와 무관하게 자체적으로 실패하는 것을 확인함(예: "D:\new tensor\...전처리_0713...").
# 이를 우회하기 위해 입력을 WSL 내부의 공백/한글 없는 임시 경로(/tmp/ppmi_stage_<uuid>)로
# 복사한 뒤 처리하고, 결과만 원래(공백/한글 포함 가능) 위치로 복사해 돌려줌 - 프로젝트의
# 기존 폴더/파일명(한글, 공백)은 전혀 바꾸지 않고 FSL 호출 구간에서만 우회.
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
    # 주논문 Results(ref.31=Smith 2002=BET 인용)와 참고문헌22 모두와 일치하는 선택.
    # 참고문헌21은 ROBEX을 쓰지만, 뇌추출 도구는 이번 변경 범위에 포함되지 않음(기존 유지).
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

    # 네이티브 PATH에 bet이 없으면(이 프로젝트 환경) WSL 경유 + 클린 경로 스테이징으로 우회
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
    실패하는 것을 확인함(실제 배치 실행 중 발견 - 전처리_ref21order_v1 등 한글 폴더명에서
    전부 실패). 경로에 비-ASCII 문자가 있으면 임시 ASCII 경로로 복사해서 그 경로를 반환.
    공백은 문제 없음(테스트로 확인) - 비-ASCII 문자만 대상."""
    if not _has_non_ascii(os.path.abspath(path)):
        return path, None
    tmp_dir = tempfile.mkdtemp(prefix="ants_stage_")
    tmp_path = os.path.join(tmp_dir, "img.nii.gz")
    shutil.copy2(path, tmp_path)
    return tmp_path, tmp_dir


def run_registration_antspy(input_path, output_path, atlas_path):
    # 참고문헌21 Table 3, Step 3 그대로: ANTsPy로 MNIPD25-T1MPRAGE-1 아틀라스에
    # affine + non-linear 정합. type_of_transform='SyN'은 ANTsPy에서 rigid+affine+SyN
    # (비선형/deformable)을 순차 수행하는 표준 프리셋으로, 원문의 "affine and non-linear
    # registration techniques"에 해당하는 가장 근접한 매핑.
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
    # 기본 비활성(--enable-bias-correction일 때만 호출됨). bet과 동일하게 네이티브 우선,
    # 없으면 WSL 경유 + 클린 경로 스테이징.
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
    # 참고문헌21 Eq.1-3 그대로: (I - Min) / (Max - Min), 뇌 영역(0이 아닌 복셀) 기준.
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
    # 참고문헌22 방식(기존 preparing.py와 동일 공식). 현재 비활성 - 필요 시
    # --normalization zscore 로 전환.
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
    "zscore": normalize_zscore,   # 비활성(토글로만 사용) - 참고문헌22/기존 preparing.py
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
    # 논문/참고문헌 모두 증강 기법 미기재 - 기존 preparing.py와 동일(프로젝트 자체 결정,
    # DEVIATIONS.md 6번 항목 참조). 이 스크립트에서 바뀐 부분 아님.
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


def preprocess_one(
    dicom_dir, output_dir, name, atlas_path, normalization="minmax",
    enable_bias_correction=False, target_shape=DEFAULT_TARGET_SHAPE, bet_frac=0.5,
    augment_count=0, seed=42, keep_intermediate=True,
):
    ensure_dir(output_dir)
    work_dir = os.path.join(output_dir, "_work", name)
    ensure_dir(work_dir)

    # [2026-07-18 변경] 주논문 Methods 순서 그대로: 뇌추출 -> 정합(ANTsPy) -> 정규화 -> (증강,
    # 학습시 온더플라이) -> 리사이즈. bias correction은 이 목록에 없으므로 기본 비활성화.
    raw_nifti = os.path.join(work_dir, f"{name}_00_raw.nii.gz")
    brain_nifti = os.path.join(work_dir, f"{name}_01_bet.nii.gz")
    registered_nifti = os.path.join(work_dir, f"{name}_02_reg.nii.gz")
    normalized_nifti = os.path.join(work_dir, f"{name}_03_norm.nii.gz")
    corrected_nifti = os.path.join(work_dir, f"{name}_04_n4.nii.gz")
    final_nifti = os.path.join(output_dir, f"{name}.nii.gz")

    normalize_fn = NORMALIZATION_FUNCS[normalization]

    convert_dicom_to_nifti(dicom_dir, raw_nifti)
    original_shape = load_nifti(raw_nifti)[1].shape

    run_skull_stripping_bet(raw_nifti, brain_nifti, frac=bet_frac)
    run_registration_antspy(brain_nifti, registered_nifti, atlas_path)
    normalize_fn(registered_nifti, normalized_nifti)

    if enable_bias_correction:
        # 기본 비활성 - 켤 경우 참고문헌21 순서(정규화 이후)를 따름
        run_n4_field_correction(normalized_nifti, corrected_nifti)
        pre_resize_path = corrected_nifti
    else:
        pre_resize_path = normalized_nifti

    resize_nifti(pre_resize_path, final_nifti, target_shape=target_shape)
    augmented_outputs = save_augmented_resized(
        pre_resize_path, output_dir, name, target_shape=target_shape,
        augment_count=augment_count, seed=seed,
    )

    if not keep_intermediate:
        shutil.rmtree(work_dir, ignore_errors=True)

    return {
        "original_shape": "x".join(map(str, original_shape)),
        "final_shape": "x".join(map(str, target_shape)),
        "augmented_count": len(augmented_outputs),
    }


def preprocess_one_from_nifti(
    raw_nifti_path, output_dir, name, atlas_path, normalization="minmax",
    enable_bias_correction=False, target_shape=DEFAULT_TARGET_SHAPE, bet_frac=0.5,
    augment_count=0, seed=42, keep_intermediate=True,
):
    """preprocess_one과 동일하나 DICOM 대신 이미 변환된 raw NIfTI(예: 01_raw_nifti)에서
    시작 - 원본 DICOM 소스(E:\\ppmi_dti\\...)가 이 환경에 없을 때 사용."""
    ensure_dir(output_dir)
    work_dir = os.path.join(output_dir, "_work", name)
    ensure_dir(work_dir)

    brain_nifti = os.path.join(work_dir, f"{name}_01_bet.nii.gz")
    registered_nifti = os.path.join(work_dir, f"{name}_02_reg.nii.gz")
    normalized_nifti = os.path.join(work_dir, f"{name}_03_norm.nii.gz")
    corrected_nifti = os.path.join(work_dir, f"{name}_04_n4.nii.gz")
    final_nifti = os.path.join(output_dir, f"{name}.nii.gz")

    normalize_fn = NORMALIZATION_FUNCS[normalization]
    original_shape = load_nifti(raw_nifti_path)[1].shape

    run_skull_stripping_bet(raw_nifti_path, brain_nifti, frac=bet_frac)
    run_registration_antspy(brain_nifti, registered_nifti, atlas_path)
    normalize_fn(registered_nifti, normalized_nifti)

    if enable_bias_correction:
        run_n4_field_correction(normalized_nifti, corrected_nifti)
        pre_resize_path = corrected_nifti
    else:
        pre_resize_path = normalized_nifti

    resize_nifti(pre_resize_path, final_nifti, target_shape=target_shape)
    augmented_outputs = save_augmented_resized(
        pre_resize_path, output_dir, name, target_shape=target_shape,
        augment_count=augment_count, seed=seed,
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


def write_run_manifest(output_dir, stamp, manifest):
    """배치 실행 1회의 설정/요약을 JSON으로 남김(결과 CSV와 별도, 실행마다 새 파일 -
    train_ablation.py의 results/ 로그 패턴과 동일하게 덮어쓰지 않음)."""
    manifest_path = os.path.join(output_dir, f"run_manifest_{stamp}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path


def run_batch(
    ppmi_root, output_dir, atlas_path, normalization="minmax",
    enable_bias_correction=False, target_shape=DEFAULT_TARGET_SHAPE, skip_existing=True,
    bet_frac=0.5, augment_count=0, seed=42, keep_intermediate=True,
):
    ensure_dir(output_dir)
    series_dirs = find_dicom_series_dirs(ppmi_root)
    total = len(series_dirs)
    t_start = time.time()
    print(f"Found DICOM series folders: {total}\n")
    bias_step = "-> N4" if enable_bias_correction else "(N4 비활성 - 주논문 Methods 목록에 없음)"
    print(f"Normalization: {normalization} (active) | pipeline order: BET -> ANTsPy registration -> "
          f"normalize {bias_step} -> resize {target_shape}\n")

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
                "name": name, "status": "skipped", "original_shape": "",
                "final_shape": "", "augmented_count": "", "elapsed_sec": "",
                "message": "", "source": dicom_dir,
            })
            continue

        try:
            t0 = time.time()
            info = preprocess_one(
                dicom_dir=dicom_dir, output_dir=output_dir, name=name,
                atlas_path=atlas_path, normalization=normalization,
                enable_bias_correction=enable_bias_correction,
                target_shape=target_shape, bet_frac=bet_frac,
                augment_count=augment_count, seed=seed + i,
                keep_intermediate=keep_intermediate,
            )
            sample_elapsed = time.time() - t0
            print(f"{prefix} - done ({sample_elapsed:.1f}s, original {info['original_shape']} -> "
                  f"{info['final_shape']}, aug {info['augmented_count']})")
            n_ok += 1
            results.append({
                "name": name, "status": "ok", "original_shape": info["original_shape"],
                "final_shape": info["final_shape"], "augmented_count": info["augmented_count"],
                "elapsed_sec": round(sample_elapsed, 1), "message": "", "source": dicom_dir,
            })
        except Exception as exc:
            print(f"{prefix} - failed: {exc}")
            n_fail += 1
            results.append({
                "name": name, "status": "failed", "original_shape": "", "final_shape": "",
                "augmented_count": "", "elapsed_sec": "", "message": str(exc), "source": dicom_dir,
            })

    total_elapsed = time.time() - t_start
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"preprocessing_log_ref21order_{stamp}.csv")
    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["name", "status", "original_shape", "final_shape",
                           "augmented_count", "elapsed_sec", "message", "source"],
        )
        writer.writeheader()
        writer.writerows(results)

    manifest_path = write_run_manifest(output_dir, stamp, {
        "timestamp": stamp, "script": "preparing_ref21order_v1.py", "input_source": ppmi_root,
        "output_dir": output_dir, "atlas_path": atlas_path, "normalization": normalization,
        "enable_bias_correction": enable_bias_correction, "target_shape": list(target_shape),
        "bet_frac": bet_frac, "total": total, "success": n_ok, "failed": n_fail, "skipped": n_skip,
        "elapsed_sec": round(total_elapsed, 1), "csv_log": log_path,
    })

    print("\n===== Batch complete =====")
    print(f"success {n_ok} / failed {n_fail} / skipped {n_skip} (total {total})")
    print(f"elapsed: {total_elapsed:.1f}s")
    print(f"csv log: {log_path}")
    print(f"manifest: {manifest_path}")


def run_batch_from_nifti(
    raw_nifti_dir, output_dir, atlas_path, normalization="minmax",
    enable_bias_correction=False, target_shape=DEFAULT_TARGET_SHAPE, skip_existing=True,
    bet_frac=0.5, augment_count=0, seed=42, keep_intermediate=False,
):
    """run_batch와 동일하나 DICOM 대신 이미 변환된 raw NIfTI 폴더(예: 01_raw_nifti)에서
    시작 - 이 환경엔 원본 DICOM 소스가 없어 기존 01_raw_nifti를 재사용."""
    ensure_dir(output_dir)
    raw_files = sorted(f for f in os.listdir(raw_nifti_dir) if f.endswith(".nii.gz"))
    total = len(raw_files)
    t_start = time.time()
    print(f"Found raw NIfTI files: {total}\n")
    bias_step = "-> N4" if enable_bias_correction else "(N4 비활성 - 주논문 Methods 목록에 없음)"
    print(f"Normalization: {normalization} (active) | pipeline order: BET -> ANTsPy registration -> "
          f"normalize {bias_step} -> resize {target_shape}\n")

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
                enable_bias_correction=enable_bias_correction,
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
    log_path = os.path.join(output_dir, f"preprocessing_log_ref21order_{stamp}.csv")
    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["name", "status", "original_shape", "final_shape",
                           "augmented_count", "elapsed_sec", "message", "source"],
        )
        writer.writeheader()
        writer.writerows(results)

    manifest_path = write_run_manifest(output_dir, stamp, {
        "timestamp": stamp, "script": "preparing_ref21order_v1.py", "input_source": raw_nifti_dir,
        "output_dir": output_dir, "atlas_path": atlas_path, "normalization": normalization,
        "enable_bias_correction": enable_bias_correction, "target_shape": list(target_shape),
        "bet_frac": bet_frac, "total": total, "success": n_ok, "failed": n_fail, "skipped": n_skip,
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
            "PPMI T2 DICOM preprocessing, main-paper-Methods-order variant "
            "(BET -> ANTsPy affine+nonlinear registration -> normalize -> resize; "
            "N4 bias correction off by default, --enable-bias-correction to turn on)."
        )
    )
    parser.add_argument("--ppmi-root", default=r"E:\ppmi_dti\raw\data\PPMI")
    parser.add_argument(
        "--from-nifti-dir", default="",
        help=(
            "원본 DICOM(--ppmi-root) 대신 이미 변환된 raw NIfTI 폴더(예: 01_raw_nifti)에서 "
            "시작. 지정하면 --ppmi-root는 무시됨."
        ),
    )
    parser.add_argument("--output-dir", default=r"E:\ppmi_dti\preparing\nifti_ref21order_v1")
    parser.add_argument(
        "--atlas-path",
        default=os.environ.get("MNIPD25_ATLAS", ""),
        help="Path to MNIPD25-T1MPRAGE-1 atlas (참고문헌21). 로컬에 없으면 직접 받아서 지정.",
    )
    parser.add_argument(
        "--normalization", choices=list(NORMALIZATION_FUNCS.keys()), default="minmax",
        help="정규화 방식. 기본값 minmax(참고문헌21, 활성). zscore(참고문헌22)는 비활성 상태로 남겨둠.",
    )
    parser.add_argument(
        "--enable-bias-correction", action="store_true",
        help=(
            "N4 bias correction 활성화. 기본 비활성 - 주논문 Methods 목록(뇌추출/정합/정규화/증강)에 "
            "bias correction이 없기 때문. 켤 경우 정규화 이후에 적용(참고문헌21 순서)."
        ),
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
    common_kwargs = dict(
        output_dir=args.output_dir,
        atlas_path=args.atlas_path,
        normalization=args.normalization,
        enable_bias_correction=args.enable_bias_correction,
        target_shape=args.target_shape,
        skip_existing=not args.overwrite,
        bet_frac=args.bet_frac,
        augment_count=args.augment_count,
        seed=args.seed,
        keep_intermediate=not args.no_keep_intermediate,
    )
    if args.from_nifti_dir:
        run_batch_from_nifti(raw_nifti_dir=args.from_nifti_dir, **common_kwargs)
    else:
        run_batch(ppmi_root=args.ppmi_root, **common_kwargs)
