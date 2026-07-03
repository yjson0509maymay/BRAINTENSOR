#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BrainPrep-equivalent Pipeline for PPMI T2 MRI Data
===================================================
Pure Python implementation (no FSL/ANTs CLI required).
Uses SimpleITK, nibabel, scipy, numpy.

Pipeline Steps:
  Step 0: DICOM → NIfTI conversion
  Step 1: Registration (affine to MNI152 T2 template)
  Step 2: Skull Stripping (intensity-based brain extraction)
  Step 3: Bias Field Correction (N4ITK via SimpleITK)
  Step 4: Enhancement (median filter + intensity rescaling)
  Step 5: Resize to 56×56×56

Author: BrainPrep adaptation for PPMI T2
"""

import os
import sys
import time
import json
import glob
import logging
import traceback
import numpy as np
import nibabel as nib
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    "raw_data_dir": r"D:\Brain_Tensor\RAW DATA\data\PPMI",
    "output_base_dir": r"D:\Brain_Tensor\brainprep_output",
    "template_path": None,  # Will be set after download/check
    "resize_target": (56, 56, 56),
    "n_workers": min(8, cpu_count()),  # Use 8 of 28 cores
    "bet_frac": 0.35,  # Skull stripping threshold for T2
    "n4_iterations": [100, 100, 60, 40],
    "n4_shrink_factor": 3,
    "enhancement_kernel": 3,
    "enhancement_percentiles": [0.5, 99.5],
    "enhancement_bins": 256,
}

# Output subdirectories for each step
STEP_DIRS = {
    "step0_nifti": "00_nifti",
    "step1_registered": "01_registered",
    "step2_skullstripped": "02_skullstripped",
    "step3_biascorrected": "03_biascorrected",
    "step4_enhanced": "04_enhanced",
    "step5_resized": "05_resized",
}

# ============================================================
# Logging Setup
# ============================================================
def setup_logging(output_dir):
    """Set up logging to both file and console."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file


# ============================================================
# Step 0: DICOM → NIfTI Conversion
# ============================================================
def step0_dicom_to_nifti(subject_id, dicom_dir, output_dir):
    """
    Convert DICOM series to NIfTI using pydicom + nibabel.
    This replaces dcm2niix for environments without it installed.
    """
    import pydicom

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{subject_id}_T2.nii.gz")

    if os.path.exists(output_path):
        logging.info(f"  [Step 0] Already exists: {output_path}")
        return output_path

    # Find all DICOM files recursively
    dcm_files = []
    for root, dirs, files in os.walk(dicom_dir):
        for f in files:
            if f.endswith('.dcm'):
                dcm_files.append(os.path.join(root, f))

    if not dcm_files:
        logging.warning(f"  [Step 0] No DICOM files found in {dicom_dir}")
        return None

    # Read all DICOM slices
    slices = []
    for dcm_file in dcm_files:
        try:
            ds = pydicom.dcmread(dcm_file)
            if hasattr(ds, 'pixel_array'):
                slices.append(ds)
        except Exception:
            continue

    if not slices:
        logging.warning(f"  [Step 0] No valid DICOM slices in {dicom_dir}")
        return None

    # Sort by Instance Number or Image Position
    try:
        slices.sort(key=lambda s: float(s.InstanceNumber))
    except (AttributeError, ValueError):
        try:
            slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
        except (AttributeError, ValueError):
            pass  # Keep original order

    # Build 3D volume
    pixel_data = np.stack([s.pixel_array.astype(np.float32) for s in slices], axis=-1)

    # Build affine matrix from DICOM headers
    try:
        ds0 = slices[0]
        ps = [float(x) for x in ds0.PixelSpacing]
        if len(slices) > 1:
            ds1 = slices[1]
            dz = abs(float(ds1.ImagePositionPatient[2]) - float(ds0.ImagePositionPatient[2]))
            if dz == 0:
                dz = float(getattr(ds0, 'SliceThickness', 1.0))
        else:
            dz = float(getattr(ds0, 'SliceThickness', 1.0))

        ipp = [float(x) for x in ds0.ImagePositionPatient]
        iop = [float(x) for x in ds0.ImageOrientationPatient]

        row_cos = np.array(iop[:3])
        col_cos = np.array(iop[3:])
        slice_cos = np.cross(row_cos, col_cos)

        affine = np.eye(4)
        affine[:3, 0] = row_cos * ps[0]
        affine[:3, 1] = col_cos * ps[1]
        affine[:3, 2] = slice_cos * dz
        affine[:3, 3] = ipp
    except (AttributeError, IndexError):
        # Fallback: identity affine with pixel spacing
        try:
            ps = [float(x) for x in slices[0].PixelSpacing]
            dz = float(getattr(slices[0], 'SliceThickness', 1.0))
        except (AttributeError, ValueError):
            ps = [1.0, 1.0]
            dz = 1.0
        affine = np.diag([ps[0], ps[1], dz, 1.0])

    # Save as NIfTI
    nii_img = nib.Nifti1Image(pixel_data, affine)
    nib.save(nii_img, output_path)

    logging.info(f"  [Step 0] Saved: {output_path} | Shape: {pixel_data.shape} | "
                 f"Voxel: ({affine[0,0]:.2f}, {affine[1,1]:.2f}, {affine[2,2]:.2f})mm")
    return output_path


# ============================================================
# Step 1: Registration (Affine to MNI152 T2 Template)
# ============================================================
def step1_registration(input_path, output_dir, template_path, subject_id):
    """
    Affine registration to MNI152 template using SimpleITK.
    Robust version to prevent C++ crashes under Windows.
    """
    import SimpleITK as sitk

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{subject_id}_T2_reg.nii.gz")

    if os.path.exists(output_path):
        logging.info(f"  [Step 1] Already exists: {output_path}")
        return output_path

    # Load images
    fixed_image = sitk.ReadImage(template_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(input_path, sitk.sitkFloat32)

    # Set up registration
    registration = sitk.ImageRegistrationMethod()

    # Use Mutual Information (Robust for T2)
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.05)

    # Optimizer settings (conservative steps to prevent divergence)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=0.01,
        numberOfIterations=200,
        gradientMagnitudeTolerance=1e-6
    )
    registration.SetOptimizerScalesFromPhysicalShift()

    # Linear interpolator is faster and more stable for initialization
    registration.SetInterpolator(sitk.sitkLinear)

    # Initial transform based on MOMENTS (much more robust to translation)
    try:
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image,
            sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.MOMENTS
        )
        registration.SetInitialTransform(initial_transform, inPlace=False)
    except Exception as e:
        logging.warning(f"  [Step 1] Centered transform initialization failed: {e}. Fallback to GEOMETRY.")
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image,
            sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        registration.SetInitialTransform(initial_transform, inPlace=False)

    # Gentle multi-resolution (2 levels instead of 3 for safety)
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Execute
    try:
        final_transform = registration.Execute(fixed_image, moving_image)

        # Final resample using BSpline for high quality
        resampled = sitk.Resample(
            moving_image, fixed_image, final_transform,
            sitk.sitkBSpline, 0.0, moving_image.GetPixelID()
        )

        sitk.WriteImage(resampled, output_path)
        metric_value = registration.GetMetricValue()
        logging.info(f"  [Step 1] Registered: {output_path} | Metric: {metric_value:.6f}")
        return output_path

    except Exception as e:
        logging.error(f"  [Step 1] Registration Execute failed: {e}")
        # Fallback 1: Resample using center-of-gravity (MOMENTS) only
        try:
            logging.info(f"  [Step 1] Attempting fallback registration for {subject_id}...")
            resampled = sitk.Resample(
                moving_image, fixed_image, initial_transform,
                sitk.sitkLinear, 0.0, moving_image.GetPixelID()
            )
            sitk.WriteImage(resampled, output_path)
            logging.info(f"  [Step 1] Fallback translation registration success.")
            return output_path
        except Exception as e_fallback:
            logging.error(f"  [Step 1] Fallback registration also failed: {e_fallback}")
            # Fallback 2: Identity mapping (just resample)
            identity = sitk.AffineTransform(3)
            resampled = sitk.Resample(
                moving_image, fixed_image, identity,
                sitk.sitkLinear, 0.0, moving_image.GetPixelID()
            )
            sitk.WriteImage(resampled, output_path)
            logging.warning(f"  [Step 1] Fallback 2: Identity transform used for {subject_id}")
            return output_path


# ============================================================
# Step 2: Skull Stripping (Brain Extraction)
# ============================================================
def step2_skull_stripping(input_path, output_dir, subject_id, frac=0.35):
    """
    Skull stripping using intensity-based thresholding + morphological operations.
    
    For T2 images:
    - CSF appears BRIGHT (high signal)
    - White matter appears DARK (low signal)
    - Gray matter appears intermediate
    
    frac parameter (0.0-1.0): Controls aggressiveness of brain extraction.
    - Lower frac → more brain preserved (risk of including skull)
    - Higher frac → less brain preserved (risk of removing brain tissue)
    - T2 recommended: 0.3-0.4 (lower than T1's typical 0.5)
    - Reason: T2 has different contrast, CSF is bright which can confuse
      standard threshold-based methods
    """
    import SimpleITK as sitk

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{subject_id}_T2_brain.nii.gz")
    mask_path = os.path.join(output_dir, f"{subject_id}_T2_brain_mask.nii.gz")

    if os.path.exists(output_path):
        logging.info(f"  [Step 2] Already exists: {output_path}")
        return output_path

    img = sitk.ReadImage(input_path, sitk.sitkFloat32)

    # Step 2a: Otsu threshold to get initial foreground
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(1)
    otsu_filter.SetOutsideValue(0)
    binary = otsu_filter.Execute(img)
    threshold = otsu_filter.GetThreshold()

    # Step 2b: Apply fractional threshold
    # For T2: use a lower threshold since CSF is bright
    adjusted_threshold = threshold * frac
    binary_mask = sitk.BinaryThreshold(img, adjusted_threshold, 1e10, 1, 0)

    # Step 2c: Morphological operations to clean mask
    # Fill holes
    binary_mask = sitk.BinaryFillhole(binary_mask)
    # Close small gaps
    binary_mask = sitk.BinaryMorphologicalClosing(binary_mask, [3, 3, 3])
    # Open to remove small fragments
    binary_mask = sitk.BinaryMorphologicalOpening(binary_mask, [2, 2, 2])

    # Step 2d: Keep largest connected component (the brain)
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    labeled = cc_filter.Execute(binary_mask)

    relabel = sitk.RelabelComponentImageFilter()
    relabel.SetMinimumObjectSize(1000)
    relabeled = relabel.Execute(labeled)

    # Keep only the largest component
    brain_mask = sitk.BinaryThreshold(relabeled, 1, 1, 1, 0)

    # Step 2e: Dilate slightly to ensure brain edges are included
    brain_mask = sitk.BinaryDilate(brain_mask, [1, 1, 1])

    # Apply mask
    brain_extracted = sitk.Mask(img, brain_mask)

    # Save
    sitk.WriteImage(brain_extracted, output_path)
    sitk.WriteImage(brain_mask, mask_path)

    # Quality metrics
    img_arr = sitk.GetArrayFromImage(img)
    mask_arr = sitk.GetArrayFromImage(brain_mask)
    brain_volume_voxels = int(np.sum(mask_arr > 0))
    total_voxels = int(np.prod(img_arr.shape))
    brain_ratio = brain_volume_voxels / total_voxels * 100

    logging.info(f"  [Step 2] Brain extracted: {output_path} | "
                 f"Brain voxels: {brain_volume_voxels:,} ({brain_ratio:.1f}%) | "
                 f"frac={frac}")
    return output_path


# ============================================================
# Step 3: Bias Field Correction (N4ITK)
# ============================================================
def step3_bias_correction(input_path, output_dir, subject_id):
    """
    N4 Bias Field Correction via SimpleITK.
    Equivalent to ANTs N4BiasFieldCorrection with:
      - iterations: [100, 100, 60, 40]
      - shrink_factor: 3
    
    N4ITK is modality-agnostic and works well on T2.
    """
    import SimpleITK as sitk

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{subject_id}_T2_n4.nii.gz")

    if os.path.exists(output_path):
        logging.info(f"  [Step 3] Already exists: {output_path}")
        return output_path

    input_image = sitk.ReadImage(input_path, sitk.sitkFloat32)

    # Create mask (non-zero region)
    mask_image = sitk.OtsuThreshold(input_image, 0, 1, 200)

    # N4 Bias Field Correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(CONFIG["n4_iterations"])
    corrector.SetConvergenceThreshold(0.001)

    # Shrink for speed
    shrink_factor = CONFIG["n4_shrink_factor"]
    input_shrunk = sitk.Shrink(input_image, [shrink_factor] * 3)
    mask_shrunk = sitk.Shrink(mask_image, [shrink_factor] * 3)

    try:
        corrected_shrunk = corrector.Execute(input_shrunk, mask_shrunk)

        # Get the bias field and apply to full resolution
        log_bias_field = corrector.GetLogBiasFieldAsImage(input_image)
        corrected = input_image / sitk.Exp(log_bias_field)

        sitk.WriteImage(corrected, output_path)

        logging.info(f"  [Step 3] Bias corrected: {output_path} | "
                     f"Iterations: {CONFIG['n4_iterations']}")
        return output_path

    except Exception as e:
        logging.error(f"  [Step 3] N4 correction failed: {e}")
        # Fallback: copy input
        sitk.WriteImage(input_image, output_path)
        logging.warning(f"  [Step 3] Fallback: copied input for {subject_id}")
        return output_path


# ============================================================
# Step 4: Enhancement (Denoise + Intensity Rescaling)
# ============================================================
def step4_enhancement(input_path, output_dir, subject_id):
    """
    Image enhancement: 3D median filtering + percentile-based intensity rescaling.
    Equivalent to BrainPrep's enhancement.py:
      - medfilt(volume, kernel_size=3)
      - rescale_intensity(volume, [0.5, 99.5], bins=256)
    """
    from scipy.ndimage import median_filter

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{subject_id}_T2_enhanced.nii.gz")

    if os.path.exists(output_path):
        logging.info(f"  [Step 4] Already exists: {output_path}")
        return output_path

    # Load
    nii = nib.load(input_path)
    data = nii.get_fdata().astype(np.float32)
    affine = nii.affine

    # Step 4a: 3D Median filter for denoising
    kernel = CONFIG["enhancement_kernel"]
    denoised = median_filter(data, size=kernel)

    # Step 4b: Intensity rescaling (percentile-based)
    percentiles = CONFIG["enhancement_percentiles"]
    bins = CONFIG["enhancement_bins"]

    obj_mask = denoised > 0
    if np.sum(obj_mask) == 0:
        logging.warning(f"  [Step 4] All-zero volume for {subject_id}")
        nib.save(nib.Nifti1Image(denoised, affine), output_path)
        return output_path

    obj_values = denoised[obj_mask]
    min_val = np.percentile(obj_values, percentiles[0])
    max_val = np.percentile(obj_values, percentiles[1])

    if max_val - min_val > 0:
        rescaled = np.zeros_like(denoised)
        rescaled[obj_mask] = np.round(
            (obj_values - min_val) / (max_val - min_val) * (bins - 1)
        )
        rescaled[rescaled < 1] = 1 * (rescaled[rescaled < 1] != 0).astype(float)
        rescaled[rescaled > (bins - 1)] = bins - 1
        # Keep background as 0
        rescaled[~obj_mask] = 0
    else:
        rescaled = denoised.copy()

    # Save
    nib.save(nib.Nifti1Image(rescaled.astype(np.float32), affine), output_path)

    logging.info(f"  [Step 4] Enhanced: {output_path} | "
                 f"Intensity range: [{min_val:.1f}, {max_val:.1f}] → [0, {bins-1}]")
    return output_path


# ============================================================
# Step 5: Resize to Target Shape (56×56×56)
# ============================================================
def step5_resize(input_path, output_dir, subject_id, target_shape=None):
    """
    Resize volume to target_shape using scipy zoom (cubic interpolation).
    Equivalent to BrainPrep's resize step.
    """
    from scipy.ndimage import zoom

    if target_shape is None:
        target_shape = CONFIG["resize_target"]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{subject_id}_T2_56.nii.gz")

    if os.path.exists(output_path):
        logging.info(f"  [Step 5] Already exists: {output_path}")
        return output_path

    # Load
    nii = nib.load(input_path)
    data = nii.get_fdata().astype(np.float32)
    affine = nii.affine

    # Calculate zoom factors
    current_shape = data.shape
    zoom_factors = [t / c for t, c in zip(target_shape, current_shape)]

    # Resize with cubic interpolation
    resized = zoom(data, zoom_factors, order=3)

    # Update affine for new voxel sizes
    new_affine = affine.copy()
    for i in range(3):
        new_affine[:3, i] = affine[:3, i] * (current_shape[i] / target_shape[i])

    # Save
    nib.save(nib.Nifti1Image(resized.astype(np.float32), new_affine), output_path)

    logging.info(f"  [Step 5] Resized: {output_path} | "
                 f"{current_shape} → {resized.shape}")
    return output_path


# ============================================================
# MNI152 T2 Template Download/Setup
# ============================================================
def setup_template(output_dir):
    """
    Download MNI152 T2 1mm template if not present.
    This is the standard T2 template from FSL.
    """
    import urllib.request

    os.makedirs(output_dir, exist_ok=True)
    template_path = os.path.join(output_dir, "MNI152_T2_1mm.nii.gz")

    if os.path.exists(template_path):
        logging.info(f"Template found: {template_path}")
        return template_path

    # Try to download from templateflow or other public sources
    urls = [
        "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T2w.nii.gz",
        "https://github.com/Washington-University/HCPpipelines/raw/master/global/templates/MNI152_T2_1mm.nii.gz",
    ]

    for url in urls:
        try:
            logging.info(f"Downloading T2 template from: {url}")
            urllib.request.urlretrieve(url, template_path)
            if os.path.exists(template_path) and os.path.getsize(template_path) > 1000:
                logging.info(f"Template downloaded: {template_path}")
                return template_path
        except Exception as e:
            logging.warning(f"Download failed from {url}: {e}")
            if os.path.exists(template_path):
                os.remove(template_path)
            continue

    # Fallback: Create a synthetic template from the first subject
    logging.warning("Could not download T2 template. Will use first subject as pseudo-template.")
    return None


# ============================================================
# Find Subject DICOM Directories
# ============================================================
def find_subjects(raw_dir, max_subjects=None):
    """Find PPMI subject directories containing DICOM files."""
    subjects = []
    ppmi_dir = raw_dir

    if not os.path.isdir(ppmi_dir):
        logging.error(f"PPMI directory not found: {ppmi_dir}")
        return subjects

    for item in sorted(os.listdir(ppmi_dir)):
        item_path = os.path.join(ppmi_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            # Find DICOM files recursively
            dcm_files = glob.glob(os.path.join(item_path, "**", "*.dcm"), recursive=True)
            if dcm_files:
                subjects.append({
                    "id": item,
                    "dicom_dir": item_path,
                    "n_files": len(dcm_files)
                })

        if max_subjects and len(subjects) >= max_subjects:
            break

    return subjects


# ============================================================
# Process Single Subject (Full Pipeline)
# ============================================================
def process_subject(subject_info, output_base, template_path):
    """Run full pipeline for a single subject."""
    subject_id = subject_info["id"]
    dicom_dir = subject_info["dicom_dir"]

    results = {
        "subject_id": subject_id,
        "steps": {},
        "total_time": 0,
        "success": True,
        "errors": []
    }

    total_start = time.time()

    try:
        # Step 0: DICOM → NIfTI
        t0 = time.time()
        nifti_path = step0_dicom_to_nifti(
            subject_id, dicom_dir,
            os.path.join(output_base, STEP_DIRS["step0_nifti"])
        )
        results["steps"]["step0_dicom_to_nifti"] = {
            "time": round(time.time() - t0, 2),
            "success": nifti_path is not None,
            "output": nifti_path
        }
        if nifti_path is None:
            results["success"] = False
            results["errors"].append("Step 0: DICOM conversion failed")
            return results

        # Step 1: Registration
        t1 = time.time()
        if template_path:
            reg_path = step1_registration(
                nifti_path,
                os.path.join(output_base, STEP_DIRS["step1_registered"]),
                template_path, subject_id
            )
        else:
            # Skip registration if no template
            reg_path = nifti_path
            logging.warning(f"  [Step 1] Skipped (no template) for {subject_id}")
        results["steps"]["step1_registration"] = {
            "time": round(time.time() - t1, 2),
            "success": reg_path is not None,
            "output": reg_path
        }
        if reg_path is None:
            results["success"] = False
            results["errors"].append("Step 1: Registration failed")
            return results

        # Step 2: Skull Stripping
        t2 = time.time()
        brain_path = step2_skull_stripping(
            reg_path,
            os.path.join(output_base, STEP_DIRS["step2_skullstripped"]),
            subject_id, frac=CONFIG["bet_frac"]
        )
        results["steps"]["step2_skull_stripping"] = {
            "time": round(time.time() - t2, 2),
            "success": brain_path is not None,
            "output": brain_path
        }
        if brain_path is None:
            results["success"] = False
            results["errors"].append("Step 2: Skull stripping failed")
            return results

        # Step 3: Bias Correction
        t3 = time.time()
        n4_path = step3_bias_correction(
            brain_path,
            os.path.join(output_base, STEP_DIRS["step3_biascorrected"]),
            subject_id
        )
        results["steps"]["step3_bias_correction"] = {
            "time": round(time.time() - t3, 2),
            "success": n4_path is not None,
            "output": n4_path
        }
        if n4_path is None:
            results["success"] = False
            results["errors"].append("Step 3: Bias correction failed")
            return results

        # Step 4: Enhancement
        t4 = time.time()
        enhanced_path = step4_enhancement(
            n4_path,
            os.path.join(output_base, STEP_DIRS["step4_enhanced"]),
            subject_id
        )
        results["steps"]["step4_enhancement"] = {
            "time": round(time.time() - t4, 2),
            "success": enhanced_path is not None,
            "output": enhanced_path
        }
        if enhanced_path is None:
            results["success"] = False
            results["errors"].append("Step 4: Enhancement failed")
            return results

        # Step 5: Resize
        t5 = time.time()
        resized_path = step5_resize(
            enhanced_path,
            os.path.join(output_base, STEP_DIRS["step5_resized"]),
            subject_id
        )
        results["steps"]["step5_resize"] = {
            "time": round(time.time() - t5, 2),
            "success": resized_path is not None,
            "output": resized_path
        }

    except Exception as e:
        results["success"] = False
        results["errors"].append(f"Unexpected error: {str(e)}")
        logging.error(f"  Pipeline error for {subject_id}: {traceback.format_exc()}")

    results["total_time"] = round(time.time() - total_start, 2)
    return results


# ============================================================
# Main Pipeline Runner
# ============================================================
def run_pipeline(test_mode=True, max_subjects=5):
    """Run the full BrainPrep pipeline."""
    output_base = CONFIG["output_base_dir"]
    log_file = setup_logging(output_base)

    logging.info("=" * 70)
    logging.info("BrainPrep Pipeline for PPMI T2 MRI Data")
    logging.info("=" * 70)
    logging.info(f"Raw data: {CONFIG['raw_data_dir']}")
    logging.info(f"Output: {output_base}")
    logging.info(f"Resize target: {CONFIG['resize_target']}")
    logging.info(f"CPU cores: {cpu_count()} (using {CONFIG['n_workers']})")
    logging.info(f"Test mode: {test_mode} (max {max_subjects} subjects)")
    logging.info("")

    # Setup template
    template_dir = os.path.join(output_base, "templates")
    template_path = setup_template(template_dir)
    CONFIG["template_path"] = template_path

    # Find subjects
    subjects = find_subjects(
        CONFIG["raw_data_dir"],
        max_subjects=max_subjects if test_mode else None
    )
    logging.info(f"Found {len(subjects)} subjects")

    if not subjects:
        logging.error("No subjects found! Check raw data directory.")
        return

    # Process subjects
    all_results = []
    pipeline_start = time.time()

    for i, subj in enumerate(subjects):
        logging.info(f"\n==================================================")
        logging.info(f"Processing [{i+1}/{len(subjects)}]: Subject {subj['id']} "
                     f"({subj['n_files']} DICOM files)")
        logging.info(f"==================================================")

        result = process_subject(subj, output_base, template_path)
        all_results.append(result)

        # Print step timings
        status = "SUCCESS" if result["success"] else "FAILED"
        logging.info(f"\n  [{status}] | Total: {result['total_time']:.1f}s")
        for step_name, step_info in result["steps"].items():
            s = "OK" if step_info["success"] else "FAIL"
            logging.info(f"    [{s}] {step_name}: {step_info['time']:.1f}s")
        if result["errors"]:
            for err in result["errors"]:
                logging.error(f"    [ERROR] {err}")

    # Summary
    total_time = time.time() - pipeline_start
    n_success = sum(1 for r in all_results if r["success"])
    n_failed = len(all_results) - n_success

    logging.info(f"\n{'='*70}")
    logging.info(f"PIPELINE SUMMARY")
    logging.info(f"{'='*70}")
    logging.info(f"Total subjects: {len(all_results)}")
    logging.info(f"Successful: {n_success}")
    logging.info(f"Failed: {n_failed}")
    logging.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logging.info(f"Avg per subject: {total_time/len(all_results):.1f}s")

    # Save results JSON
    results_path = os.path.join(output_base, "pipeline_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        # Clean results for JSON serialization
        clean_results = []
        for r in all_results:
            clean_r = {k: v for k, v in r.items()}
            clean_results.append(clean_r)
        json.dump(clean_results, f, indent=2, ensure_ascii=False)

    logging.info(f"Results saved to: {results_path}")
    logging.info(f"Log saved to: {log_file}")

    return all_results


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BrainPrep Pipeline for PPMI T2")
    parser.add_argument("--test", action="store_true", default=True,
                        help="Run in test mode (5 subjects)")
    parser.add_argument("--full", action="store_true",
                        help="Run on all subjects")
    parser.add_argument("--max-subjects", type=int, default=5,
                        help="Max subjects in test mode")
    args = parser.parse_args()

    if args.full:
        run_pipeline(test_mode=False)
    else:
        run_pipeline(test_mode=True, max_subjects=args.max_subjects)
