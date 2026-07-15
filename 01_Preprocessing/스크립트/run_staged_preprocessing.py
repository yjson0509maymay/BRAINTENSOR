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


# dicom2nifti는 슬라이스 간격이 미세하게 불균일하면 변환을 거부하는데,
# 실제 임상 DICOM에서는 흔한 일이라 이 검증을 꺼서 변환이 막히지 않게 한다.
dcm_settings.disable_validate_slice_increment()

# 전처리 파이프라인의 6단계 정의.
# 각 튜플은 (출력 폴더 이름, README에 쓰일 설명)이며,
# ensure_readmes()가 폴더 생성과 문서화에 사용하고
# process_sample()이 실제 단계별 파일 경로를 만드는 데 사용한다.
STAGES = [
    ("01_raw_nifti", "DICOM converted to a reoriented 3D NIfTI volume."),
    ("02_bet", "FSL BET brain extraction output."),
    ("03_n4", "ANTs N4 bias-field corrected brain volume."),
    ("04_mni152", "FSL FLIRT 12-DOF registration to the MNI152 template."),
    ("05_normalized", "Non-zero brain voxels z-score normalized and clipped to [-5, 5]."),
    ("06_resized", "Final model input resized to 56 x 56 x 56 voxels."),
]


def load_preparing(path):
    """--preparing-path로 지정된 .py 파일(BET/N4/FLIRT/정규화/리사이즈 함수 모음)을
    모듈 이름 없이 파일 경로만으로 동적으로 import한다.
    ProcessPoolExecutor의 각 워커 프로세스가 독립적으로 이 함수를 호출해
    자신만의 모듈 인스턴스를 갖는다(멀티프로세싱에서 pickle 문제를 피하기 위함)."""
    spec = importlib.util.spec_from_file_location("paper_preparing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_rows(path):
    """data.csv를 읽어 dict 리스트로 반환. utf-8-sig로 열어 BOM이 있는 CSV도 처리."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows, fields):
    """처리 결과를 CSV로 기록. extrasaction='ignore'로 fields에 없는 키는 무시한다."""
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_echo_time(path):
    """DICOM 파일 하나에서 EchoTime(TE) 태그만 읽는다.
    stop_before_pixels=True + specific_tags로 픽셀 데이터 전체를 읽지 않아 매우 빠르다.
    태그가 없거나 파일이 손상된 경우 None을 반환해 호출부에서 '미확인'으로 분류하게 한다."""
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
    """하나의 시리즈 폴더 안에 서로 다른 Echo Time(TE)의 슬라이스가 섞여 있는 경우
    (예: dual-echo 시퀀스에서 T2 강조를 얻으려 가장 긴 TE만 써야 하는 경우) 대응.
    TE 값별로 파일을 그룹핑하고, 그룹이 2개 이상이면 TE가 가장 큰(=T2 가중치가 가장 강한)
    그룹만 선택해서 반환한다. 그룹이 1개 이하면 필터링 없이 전체 파일을 그대로 반환."""
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
            # 부동소수점 오차로 같은 TE가 다른 그룹으로 갈리지 않도록 소수 4자리로 반올림
            echo_groups.setdefault(round(echo_time, 4), []).append(path)
    if len(echo_groups) <= 1:
        return files, next(iter(echo_groups), None), False
    selected_te = max(echo_groups)
    return echo_groups[selected_te], selected_te, True


def convert_t2_dicom(dicom_dir, output_path):
    """DICOM 시리즈 폴더 하나를 최종 01_raw_nifti/{sample_id}.nii.gz 하나로 변환한다.
    반환값은 (선택된 echo time, TE로 필터링했는지 여부, 최종 볼륨의 shape 문자열)."""
    selected_files, echo_time, echo_filtered = select_t2_echo_files(dicom_dir)
    input_dir = dicom_dir
    filtered_dir = None
    converted_dir = tempfile.mkdtemp(prefix="t2_nifti_")
    try:
        if echo_filtered:
            # dicom2nifti는 폴더 전체를 시리즈로 취급하므로, TE로 걸러낸 파일만
            # 별도 임시 폴더에 복사해 넣어야 원치 않는 echo가 섞여 들어가지 않는다.
            filtered_dir = tempfile.mkdtemp(prefix="t2_echo_")
            for index, source in enumerate(selected_files):
                target = os.path.join(filtered_dir, f"slice_{index:06d}.dcm")
                shutil.copy2(source, target)
            input_dir = filtered_dir

        # 폴더 안에 여러 시리즈가 섞여 있을 수 있으므로 convert_directory는
        # 시리즈별로 여러 개의 .nii(.gz) 파일을 만들어낼 수 있다.
        dicom2nifti.convert_directory(input_dir, converted_dir, compression=True, reorient=True)
        candidates = []
        for name in os.listdir(converted_dir):
            if not (name.endswith(".nii") or name.endswith(".nii.gz")):
                continue
            path = os.path.join(converted_dir, name)
            image = nib.load(path)
            data = image.get_fdata(dtype=np.float32)
            if data.ndim == 4:
                # 4D(예: 반복 측정/멀티 에코가 한 파일에 남은 경우)면
                # 마지막 3D 볼륨만 취해 3D로 축소한다.
                data = data[..., -1]
                image = nib.Nifti1Image(data.astype(np.float32), image.affine, image.header)
                collapsed = os.path.join(converted_dir, "collapsed_" + name)
                nib.save(image, collapsed)
                path = collapsed
            if data.ndim == 3:
                # 복셀 개수(shape의 곱)를 함께 저장해 두고, 여러 3D 후보 중
                # 가장 큰(=가장 완전한/고해상도) 볼륨을 최종 결과로 선택한다.
                candidates.append((int(np.prod(data.shape)), path, data.shape))
        if not candidates:
            raise RuntimeError("DICOM conversion produced no 3D NIfTI volume")
        _, selected_path, shape = max(candidates, key=lambda item: item[0])
        shutil.copy2(selected_path, output_path)
        return echo_time, echo_filtered, "x".join(map(str, shape))
    finally:
        # 성공하든 실패하든 임시 변환 폴더/echo 필터링 폴더는 반드시 정리한다.
        shutil.rmtree(converted_dir, ignore_errors=True)
        if filtered_dir:
            shutil.rmtree(filtered_dir, ignore_errors=True)


def ensure_readmes(output_root):
    """출력 루트 아래에 STAGES 각 단계 폴더 + logs/visualization 폴더를 만들고,
    각 폴더에 이 폴더가 뭘 하는지 설명하는 README.md를 생성한다(문서화 목적,
    처리 로직에는 영향 없음). main()에서 처리 시작 전에 한 번 호출된다."""
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
    """output_root/stage/sample_id.nii.gz 형태의 표준 경로를 만든다."""
    return str(Path(root) / stage / f"{sample_id}.nii.gz")


def platform_path(path):
    """data.csv에 Windows 드라이브 경로(예: E:\\ppmi_dti\\raw\\...)가 적혀 있어도,
    스크립트를 WSL/Linux에서 실행 중이면 /mnt/e/ppmi_dti/raw/... 형태로 변환해 준다.
    Windows(os.name == 'nt')에서 실행 중이면 원래 경로를 그대로 사용한다."""
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", path)
    if os.name != "nt" and match:
        drive = match.group(1).lower()
        tail = match.group(2).replace("\\", "/")
        return f"/mnt/{drive}/{tail}"
    return path


def process_sample(row, config):
    """샘플(피험자 스캔) 1건을 6단계 파이프라인 전부에 걸쳐 처리한다.
    ProcessPoolExecutor의 워커 프로세스에서 실행되며, 각 단계는 이미
    출력 파일이 존재하면(overwrite가 아닌 한) 건너뛰어 재실행 시 이어서
    진행할 수 있다. 어느 단계에서든 예외가 나면 잡아서 status='failed'로
    기록하고, 이미 만든 결과 dict를 그대로 반환한다(다른 샘플 처리에 영향 없음)."""
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
    # 6단계 각각의 최종 출력 경로를 미리 계산해 둔다 (예: paths['02_bet'] = .../02_bet/sub-01.nii.gz)
    paths = {stage: stage_path(config["output_root"], stage, sample_id) for stage, _ in STAGES}
    try:
        # 1단계: DICOM -> NIfTI 변환. 이미 결과 파일이 있고 --overwrite가 아니면 재사용(skip).
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
            # 이미 변환된 파일이 있으면 재변환하지 않고 shape 정보만 로그용으로 다시 읽는다.
            image = nib.load(paths["01_raw_nifti"])
            result["original_shape"] = "x".join(map(str, image.shape))

        # 2~6단계는 각 단계 함수가 (입력 경로, 출력 경로, ...) 형태로 통일되어 있어
        # 리스트로 정의해 놓고 순서대로 실행한다. 각 함수의 실제 구현은
        # --preparing-path로 지정한 외부 모듈(paper_preparing.py 등)에 있다.
        operations = [
            ("02_bet", preparing.run_skull_stripping_bet, (paths["01_raw_nifti"], paths["02_bet"])),
            ("03_n4", preparing.run_n4_field_correction, (paths["02_bet"], paths["03_n4"])),
            ("04_mni152", preparing.run_mni_registration_flirt, (paths["03_n4"], paths["04_mni152"], config["mni_template"])),
            ("05_normalized", preparing.normalize_intensity, (paths["04_mni152"], paths["05_normalized"])),
            ("06_resized", preparing.resize_nifti, (paths["05_normalized"], paths["06_resized"])),
        ]
        for stage, function, arguments in operations:
            # 각 단계도 마찬가지로 결과 파일이 이미 있으면 건너뛴다 (재실행 시 이어하기 지원).
            if config["overwrite"] or not os.path.exists(paths[stage]):
                started = time.time()
                function(*arguments)
                timings[stage[:2] + "_seconds"] = time.time() - started
    except Exception as exc:
        # 어느 단계에서 실패하든 전체 배치는 멈추지 않고, 이 샘플만 실패로 기록한다.
        result["status"] = "failed"
        result["message"] = str(exc)
    result.update({key: round(value, 3) for key, value in timings.items()})
    result["total_seconds"] = round(time.time() - start_total, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description="Staged paper-aligned T2 preprocessing.")
    parser.add_argument("--data-csv", required=True)       # sample_id, Subject, Image Data ID, Group, raw_dicom_dir 컬럼을 포함한 CSV
    parser.add_argument("--output-root", required=True)    # 6단계 결과 + logs/visualization이 생성될 루트 폴더
    parser.add_argument("--mni-template", required=True)   # FLIRT 등록에 쓸 MNI152 템플릿 파일 경로
    parser.add_argument("--preparing-path", required=True) # BET/N4/FLIRT/정규화/리사이즈 함수가 정의된 .py 파일
    parser.add_argument("--workers", type=int, default=3)  # 동시에 처리할 프로세스 수
    parser.add_argument("--limit", type=int, default=0)    # 0이면 전체, 양수면 앞에서부터 N건만 처리(테스트용)
    parser.add_argument("--overwrite", action="store_true")  # 이미 있는 출력도 다시 계산할지 여부
    args = parser.parse_args()

    rows = read_rows(args.data_csv)
    if args.limit > 0:
        rows = rows[: args.limit]
    ensure_readmes(args.output_root)  # 출력 폴더 구조 + README 먼저 생성
    config = {
        "output_root": args.output_root,
        "mni_template": args.mni_template,
        "preparing_path": args.preparing_path,
        "overwrite": args.overwrite,
    }
    results = []
    # 샘플별로 별도 프로세스에서 process_sample을 실행한다(전처리가 CPU 바운드이고
    # BET/N4/FLIRT 등 외부 도구 호출이 GIL과 무관하게 병렬화되어야 하므로 스레드가 아닌 프로세스 사용).
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
            # 매 샘플이 끝날 때마다 로그 CSV를 통째로 다시 써서, 중간에 중단되어도
            # 그때까지의 진행 상황이 preprocessing_log.csv에 남도록 한다.
            fields = sorted({key for item in results for key in item})
            write_rows(Path(args.output_root) / "logs" / "preprocessing_log.csv", results, fields)

    ok = sum(item["status"] == "ok" for item in results)
    failed = len(results) - ok
    print(f"Complete: success={ok}, failed={failed}, total={len(results)}")
    if failed:
        # 실패 건이 하나라도 있으면 CI/배치 스크립트 등에서 실패로 감지할 수 있도록 0이 아닌 코드로 종료.
        raise SystemExit(1)


if __name__ == "__main__":
    main()
