# Paper-aligned T2 preprocessing

Input: `E:/ppmi_dti/raw/data.csv` and raw DICOM folders.

Pipeline: DICOM to NIfTI, BET, N4, MNI152 registration, intensity normalization, and 56x56x56 resize. Each stage folder is the input to the next stage. FLAIR data is excluded from this cohort.
