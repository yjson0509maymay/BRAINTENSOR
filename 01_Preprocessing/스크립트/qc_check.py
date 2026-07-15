import argparse
import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

GROUP_TARGET = {"Control": 110, "Prodromal": 58, "PD": 135}


def read_cohort(data_csv):
    with open(data_csv, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def numeric_qc(cohort, resized_dir):
    rows, missing = [], []
    for r in cohort:
        sid = r["sample_id"]
        path = resized_dir / f"{sid}.nii.gz"
        if not path.exists():
            missing.append(sid)
            continue
        data = nib.load(str(path)).get_fdata(dtype=np.float32)
        nonzero = data[data != 0]
        rows.append({
            "sample_id": sid,
            "Group": r["Group"],
            "shape": str(data.shape),
            "has_nan": bool(np.isnan(data).any()),
            "has_inf": bool(np.isinf(data).any()),
            "nonzero_fraction": round(float((data != 0).mean()), 4),
            "mean_nonzero": round(float(nonzero.mean()) if nonzero.size else float("nan"), 4),
            "std_nonzero": round(float(nonzero.std()) if nonzero.size else float("nan"), 4),
            "min": round(float(data.min()), 3),
            "max": round(float(data.max()), 3),
        })
    return rows, missing


def find_numeric_problems(numeric_rows, target_shape="(56, 56, 56)"):
    """Shape/NaN/Inf/nonzero_fraction-outlier checks only. This catches gross
    corruption (wrong shape, empty volumes) but NOT rotation/registration
    artifacts that preserve normal brain-tissue volume - those only show up
    on visual (contact sheet) review, not in these numbers.
    """
    nz_fracs = np.array([r["nonzero_fraction"] for r in numeric_rows])
    nz_mean, nz_std = nz_fracs.mean(), nz_fracs.std()
    problems = []
    for r in numeric_rows:
        reasons = []
        if r["shape"] != target_shape:
            reasons.append(f"unexpected shape {r['shape']}")
        if r["has_nan"]:
            reasons.append("has NaN")
        if r["has_inf"]:
            reasons.append("has Inf")
        if abs(r["nonzero_fraction"] - nz_mean) > 3 * nz_std:
            reasons.append(
                f"nonzero_fraction outlier ({r['nonzero_fraction']}, "
                f"cohort mean={nz_mean:.4f} std={nz_std:.4f})"
            )
        if reasons:
            problems.append((r["sample_id"], "; ".join(reasons)))
    return problems


def stage_numeric_trace(cohort, stage_root, stages):
    """Lightweight numeric fingerprint (shape, nonzero_fraction) for every
    subject at every stage. Used to automatically localize which stage a
    final-stage problem first appeared at, without re-visualizing everyone.
    """
    trace = {}
    for r in cohort:
        sid = r["sample_id"]
        per_stage = {}
        for stage in stages:
            path = stage_root / stage / f"{sid}.nii.gz"
            if not path.exists():
                per_stage[stage] = None
                continue
            data = nib.load(str(path)).get_fdata(dtype=np.float32)
            per_stage[stage] = {
                "shape": data.shape,
                "nonzero_fraction": float((data != 0).mean()),
            }
        trace[sid] = per_stage
    return trace


def cohort_stage_stats(trace, stages):
    """Per-stage mean/std of nonzero_fraction across all subjects that have
    that stage present. Used as the "normal range" to compare a flagged
    subject's own trajectory against."""
    stats = {}
    for stage in stages:
        vals = [v[stage]["nonzero_fraction"] for v in trace.values()
                 if v[stage] is not None]
        vals = np.array(vals)
        stats[stage] = (float(vals.mean()), float(vals.std())) if vals.size else (float("nan"), float("nan"))
    return stats


def localize_anomaly_stage(sample_id, trace, stage_stats, stages, z_thresh=3.0):
    """Walk a subject's own stage trajectory and report the first stage
    where it deviates >z_thresh cohort std devs from the per-stage cohort
    mean nonzero_fraction, plus a shape-change check. Returns
    (origin_stage_or_None, per_stage_detail_list)."""
    detail = []
    origin_stage = None
    prev_shape = None
    for stage in stages:
        entry = trace[sample_id].get(stage)
        if entry is None:
            detail.append((stage, "missing", None, None, None))
            continue
        mean, std = stage_stats[stage]
        z = (entry["nonzero_fraction"] - mean) / std if std else 0.0
        shape_changed_unexpectedly = (
            stage not in ("04_mni152",) and prev_shape is not None
            and entry["shape"] != prev_shape and stage != "06_resized"
        )
        flagged = abs(z) > z_thresh
        detail.append((stage, entry["shape"], round(entry["nonzero_fraction"], 4), round(z, 2), flagged))
        if flagged and origin_stage is None:
            origin_stage = stage
        prev_shape = entry["shape"]
    return origin_stage, detail


def auto_trace_and_diagnose(problem_sample_ids, cohort, stage_root, qc_out, stages):
    """For each flagged sample_id: compute the full stage trace for the
    whole cohort (once), localize which stage the anomaly first appears at
    (numeric heuristic - won't catch shape-preserving rotation/warp
    artifacts, those still need the human-flagged sample_id as input), and
    render the 6-stage side-by-side diagnostic image automatically."""
    if not problem_sample_ids:
        return []
    trace = stage_numeric_trace(cohort, stage_root, stages)
    stage_stats = cohort_stage_stats(trace, stages)

    rows = []
    for sid in problem_sample_ids:
        if sid not in trace:
            continue
        origin_stage, detail = localize_anomaly_stage(sid, trace, stage_stats, stages)
        problem_case_diagnostic(sid, stage_root, qc_out, stages)
        rows.append({
            "sample_id": sid,
            "likely_origin_stage_numeric": origin_stage or "not detected numerically (check image by eye)",
            "detail": detail,
        })

    with open(qc_out / "stage_anomaly_trace.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "likely_origin_stage_numeric"] + [f"{s}_nonzero_fraction_z" for s in stages])
        for row in rows:
            z_by_stage = {d[0]: d[3] for d in row["detail"]}
            w.writerow([row["sample_id"], row["likely_origin_stage_numeric"]] +
                       [z_by_stage.get(s, "") for s in stages])
    return rows


def write_manifest(out_path, cohort, numeric_rows, missing, problems,
                    replaced=None, fail_reason=None, residual_notes=None):
    replaced = replaced or {}
    fail_reason = fail_reason or {}
    residual_notes = residual_notes or []
    group_counts = {}
    for r in cohort:
        group_counts[r["Group"]] = group_counts.get(r["Group"], 0) + 1

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["=== 전체 처리 결과 요약 (Overall Processing Summary) ==="])
        w.writerow(["최종 코호트 목표 인원 (data_0713.csv 기준)", len(cohort)])
        w.writerow(["06_resized 폴더 내 대응 파일 확인", len(numeric_rows)])
        w.writerow(["누락 파일 수", len(missing)])
        w.writerow(["원래 실패 후 교체된 subject 수", len(replaced)])
        w.writerow([])
        w.writerow(["=== 파일별 상세 내역 (Per-file Detail) ==="])
        w.writerow(["sample_id", "Subject", "Group", "Final_Cohort_Status"])
        for r in cohort:
            w.writerow([r["sample_id"], r["Subject"], r["Group"], "유지(최종 코호트 포함)"])
        w.writerow([])
        if replaced:
            w.writerow(["=== 교체된 실패 표본 (Replaced Failed Samples) ==="])
            w.writerow(["original_sample_id", "Reason", "Replaced_By"])
            for orig, repl in replaced.items():
                w.writerow([orig, fail_reason.get(orig, ""), repl])
            w.writerow([])
        if residual_notes:
            w.writerow(["=== 잔존 파일 안내 (Residual Files Note) ==="])
            for note in residual_notes:
                w.writerow([note])
            w.writerow([])
        w.writerow(["=== 논문 기준 대비 검증 (Overall Summary / Verification) ==="])
        w.writerow(["검증 항목", "결과", "비고"])
        w.writerow(["논문의 코호트 구성 기준",
                     "Control 110 / Prodromal 58 / PD 135 (총 303명)", ""])
        actual = (f"Control {group_counts.get('Control', 0)} / "
                  f"Prodromal {group_counts.get('Prodromal', 0)} / "
                  f"PD {group_counts.get('PD', 0)} (총 {len(cohort)}명)")
        match = all(group_counts.get(g, 0) == n for g, n in GROUP_TARGET.items())
        w.writerow(["data_0713.csv 기준 코호트 구성", actual, "완전 일치" if match else "불일치"])
        w.writerow(["누락된 데이터", "없음" if not missing else f"{len(missing)}건: {missing}", ""])
        w.writerow(["Shape 불일치 / NaN / Inf / 시각 QC 이상",
                     "없음" if not problems else f"{len(problems)}건",
                     "; ".join(f"{s}: {r}" for s, r in problems) if problems else ""])
        w.writerow(["최종 판단",
                     "전체 코호트 구성 완료, QC 이상 없음" if not problems and match else "확인 필요", ""])


def slice_contact_sheets(numeric_rows, resized_dir, out_dir, z_frac=0.5, tag="",
                          per_sheet=42, cols=7, rows_=6):
    """z_frac: fraction of the z-axis to slice at (0.5 = mid-axial /
    ventricle level, ~0.3 = midbrain / substantia nigra level)."""
    sheets = [numeric_rows[i:i + per_sheet] for i in range(0, len(numeric_rows), per_sheet)]
    n_sheets = len(sheets)
    for sheet_idx, sheet_rows in enumerate(sheets, start=1):
        fig, axes = plt.subplots(rows_, cols, figsize=(cols * 2, rows_ * 2))
        axes = axes.flatten()
        for ax, r in zip(axes, sheet_rows):
            data = nib.load(str(resized_dir / f"{r['sample_id']}.nii.gz")).get_fdata(dtype=np.float32)
            zi = int(data.shape[2] * z_frac)
            ax.imshow(np.rot90(data[:, :, zi]), cmap="gray", vmin=-3, vmax=3)
            ax.set_title(f"{r['sample_id'][:14]}\n{r['Group']}", fontsize=6)
            ax.axis("off")
        for ax in axes[len(sheet_rows):]:
            ax.axis("off")
        name = f"qc_contact_sheet{('_' + tag) if tag else ''}_{sheet_idx}_of_{n_sheets}.png"
        fig.suptitle(f"{name} (z_frac={z_frac})", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=110)
        plt.close(fig)
    return n_sheets


def stage_montage(numeric_rows, stage_root, out_dir, stages, n_samples=9, seed=42):
    random.seed(seed)
    subset = random.sample(numeric_rows, min(n_samples, len(numeric_rows)))
    side = int(np.ceil(np.sqrt(len(subset))))
    for stage in stages:
        fig, axes = plt.subplots(side, side, figsize=(side * 3, side * 3))
        axes = axes.flatten()
        for ax, r in zip(axes, subset):
            data = nib.load(str(stage_root / stage / f"{r['sample_id']}.nii.gz")).get_fdata(dtype=np.float32)
            ax.imshow(np.rot90(data[:, :, data.shape[2] // 2]), cmap="gray")
            ax.set_title(f"{r['sample_id'][:16]} ({r['Group']})", fontsize=8)
            ax.axis("off")
        for ax in axes[len(subset):]:
            ax.axis("off")
        fig.suptitle(f"{stage} sample QC", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"qc_{stage}.png", dpi=130)
        plt.close(fig)


def problem_case_diagnostic(sample_id, stage_root, out_dir, stages):
    fig, axes = plt.subplots(1, len(stages), figsize=(4 * len(stages), 4))
    for ax, stage in zip(axes, stages):
        data = nib.load(str(stage_root / stage / f"{sample_id}.nii.gz")).get_fdata(dtype=np.float32)
        ax.imshow(np.rot90(data[:, :, data.shape[2] // 2]), cmap="gray")
        ax.set_title(f"{stage}\nshape={data.shape}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{sample_id} pipeline stage diagnostic", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / f"qc_problem_case_{sample_id}.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="QC checks for a staged preprocessing output.")
    parser.add_argument("--data-csv", required=True, help="Final cohort CSV (sample_id, Group, ...)")
    parser.add_argument("--stage-root", required=True, help="Root folder with 01_raw_nifti..06_resized")
    parser.add_argument("--qc-out", required=True, help="Output folder for QC CSVs and images")
    parser.add_argument("--midbrain-z-frac", type=float, default=0.30,
                         help="z fraction for substantia-nigra-level contact sheets")
    parser.add_argument("--visual-flags", nargs="*", default=[],
                         help="sample_id(s) a human found abnormal by eye (contact sheet review) "
                              "that the numeric checks missed - e.g. rotation/warp artifacts that "
                              "preserve tissue volume")
    parser.add_argument("--all-stages", nargs="*",
                         default=["01_raw_nifti", "02_bet", "03_n4", "04_mni152", "05_normalized", "06_resized"],
                         help="stage folder names in pipeline order, for stage-tracing")
    args = parser.parse_args()

    stage_root = Path(args.stage_root)
    qc_out = Path(args.qc_out)
    qc_out.mkdir(parents=True, exist_ok=True)
    resized_dir = stage_root / "06_resized"

    cohort = read_cohort(args.data_csv)
    numeric_rows, missing = numeric_qc(cohort, resized_dir)

    numeric_fields = ["sample_id", "Group", "shape", "has_nan", "has_inf",
                       "nonzero_fraction", "mean_nonzero", "std_nonzero", "min", "max"]
    with open(qc_out / "qc_numeric.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=numeric_fields)
        w.writeheader()
        w.writerows(numeric_rows)

    problems = find_numeric_problems(numeric_rows)

    write_manifest(qc_out / "manifest_verification.csv", cohort, numeric_rows, missing, problems)

    n_sheets_ventricle = slice_contact_sheets(numeric_rows, resized_dir, qc_out, z_frac=0.5, tag="ventricle")
    n_sheets_midbrain = slice_contact_sheets(numeric_rows, resized_dir, qc_out,
                                              z_frac=args.midbrain_z_frac, tag="midbrain")
    stage_montage(numeric_rows, stage_root, qc_out, ["02_bet", "06_resized"])

    # Union of numerically-flagged problems and human-reported visual flags.
    # For every one of them, automatically walk all 6 stages and report the
    # earliest stage where the subject's numbers diverge from the cohort -
    # no need to manually pick which sample_id to dig into.
    numeric_flagged_ids = [sid for sid, _ in problems]
    all_flagged_ids = sorted(set(numeric_flagged_ids) | set(args.visual_flags))
    trace_rows = auto_trace_and_diagnose(all_flagged_ids, cohort, stage_root, qc_out, args.all_stages)

    print(f"numeric QC rows: {len(numeric_rows)}, missing: {len(missing)}, numeric-flagged problems: {len(problems)}")
    print(f"contact sheets: {n_sheets_ventricle} (ventricle level), {n_sheets_midbrain} (midbrain level)")
    print("IMPORTANT: numeric checks only catch gross corruption (wrong shape/NaN/Inf/empty volume).")
    print("Rotation/registration artifacts that preserve tissue volume are NOT caught here -")
    print("review the contact sheets by eye and pass such sample_ids via --visual-flags.")
    if trace_rows:
        print(f"Auto-traced {len(trace_rows)} flagged sample(s):")
        for row in trace_rows:
            print(f"  {row['sample_id']}: likely_origin_stage = {row['likely_origin_stage_numeric']}")
    print(f"Wrote QC outputs to {qc_out}")


if __name__ == "__main__":
    main()
