# =====================================================================
# Brain_Tensor 폴더 재구성 스크립트 v2
# 작성일: 2026-07-03
#
# 목적: 1차 재구성(정리_작업_스크립트.ps1) 이후 "03_Model 안에 여러 단계가
# 섞여 있고 02(FeatureEngineering)가 03보다 먼저 실행되는 것처럼 헷갈린다"는
# 피드백을 반영하여, 폴더 번호를 "실제 실행 순서" 기준으로 재배치합니다.
#
# 실행 순서(파이프라인):
#   00 원본데이터 -> 01 전처리 -> 02 모델 정의 -> 03 모델 학습(특징 추출까지)
#   -> 04 특징 융합/최적화(CCA+WOA, 03에서 나온 특징을 입력으로 사용)
#   -> 05 모델 평가(분류기 학습+추론+k-fold 평가) -> 06 결과 -> 07 문서 -> 08 시각자료
#
# [원칙 - 이전 스크립트와 동일]
# 1. RAW DATA 폴더는 절대 이동/수정하지 않습니다.
# 2. 기존 파일은 삭제하지 않고 "지난파일" 폴더로 이동합니다.
# 3. 같은 이름이 이미 있으면 자동으로 _v2, _v3 버전 접미사를 붙입니다.
# 4. 먼저 $DryRun = $true 로 계획만 확인한 뒤, $false로 바꿔 실제 실행하세요.
# =====================================================================

$Root    = "D:\Brain_Tensor"
$DryRun  = $false
$LogPath = Join-Path $Root ("정리_작업_로그_v2_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

function Write-Log {
    param([string]$Message)
    Write-Host $Message
    Add-Content -Path $LogPath -Value $Message
}

function Move-Safe {
    param(
        [Parameter(Mandatory=$true)][string]$Source,
        [Parameter(Mandatory=$true)][string]$DestDir,
        [string]$NewName = $null
    )
    if (-not (Test-Path -LiteralPath $Source)) {
        Write-Log "  [건너뜀-원본없음] $Source"
        return
    }
    if (-not (Test-Path -LiteralPath $DestDir)) {
        if (-not $DryRun) { New-Item -ItemType Directory -Force -Path $DestDir | Out-Null }
    }
    $name = if ($NewName) { $NewName } else { Split-Path $Source -Leaf }
    $finalDest = Join-Path $DestDir $name
    $i = 2
    while (Test-Path -LiteralPath $finalDest) {
        $ext  = [System.IO.Path]::GetExtension($name)
        $base = [System.IO.Path]::GetFileNameWithoutExtension($name)
        $finalDest = Join-Path $DestDir ("{0}_v{1}{2}" -f $base, $i, $ext)
        $i++
    }
    if ($DryRun) {
        Write-Log "  [예정] $Source  ->  $finalDest"
    } else {
        Move-Item -LiteralPath $Source -Destination $finalDest
        Write-Log "  [이동완료] $Source  ->  $finalDest"
    }
}

function Copy-ContentInto {
    # 폴더 자체가 아니라 "내용물"만 옮기고 빈 폴더로 만들 때 사용
    param([string]$SrcDir, [string]$DestDir)
    if (-not (Test-Path -LiteralPath $SrcDir)) { return }
    if (-not $DryRun) { New-Item -ItemType Directory -Force -Path $DestDir | Out-Null }
    Get-ChildItem -LiteralPath $SrcDir -Force | ForEach-Object {
        Move-Safe -Source $_.FullName -DestDir $DestDir
    }
}

Write-Log "===================================================================="
Write-Log "Brain_Tensor 재구성 v2 시작 (DryRun=$DryRun) $(Get-Date)"
Write-Log "===================================================================="

# ---------------------------------------------------------------------
# 0. 신규 폴더 생성 (실행순서 기준 번호)
# ---------------------------------------------------------------------
Write-Log "`n[0] 신규 폴더 생성"
$NewDirs = @(
    "02_Model_Definition",
    "03_Model_Training",
    "03_Model_Training\checkpoints",
    "03_Model_Training\training_logs",
    "04_Feature_Engineering",
    "05_Model_Evaluation",
    "06_Result",
    "07_Document",
    "08_Visualization",
    "지난파일\모델링_이전버전\구버전_피처엔지니어링",
    "지난파일\기타\정리작업_이력_v1"
)
foreach ($d in $NewDirs) {
    $full = Join-Path $Root $d
    if (-not (Test-Path -LiteralPath $full)) {
        if (-not $DryRun) { New-Item -ItemType Directory -Force -Path $full | Out-Null }
        Write-Log "  생성: $full"
    }
}

# ---------------------------------------------------------------------
# 1. 02_Model_Definition - 순수 아키텍처 정의만
# ---------------------------------------------------------------------
Write-Log "`n[1] 02_Model_Definition (모델 구조 정의)"
Move-Safe -Source (Join-Path $Root "03_Model\models.py") -DestDir (Join-Path $Root "02_Model_Definition")

# ---------------------------------------------------------------------
# 2. 03_Model_Training - 학습 실행 관련 (기존 03_Model의 나머지)
# ---------------------------------------------------------------------
Write-Log "`n[2] 03_Model_Training (학습 실행)"
Move-Safe -Source (Join-Path $Root "03_Model\dataset.py") -DestDir (Join-Path $Root "03_Model_Training")
Move-Safe -Source (Join-Path $Root "03_Model\train.py") -DestDir (Join-Path $Root "03_Model_Training")
Move-Safe -Source (Join-Path $Root "03_Model\smoke_test.py") -DestDir (Join-Path $Root "03_Model_Training")
Move-Safe -Source (Join-Path $Root "03_Model\DEVIATIONS.md") -DestDir (Join-Path $Root "03_Model_Training")
Move-Safe -Source (Join-Path $Root "03_Model\모델링_작업리스트.txt") -DestDir (Join-Path $Root "03_Model_Training")

# 구버전 학습 스크립트(이미 새 train.py로 대체됨) -> 지난파일
Move-Safe -Source (Join-Path $Root "03_Model\train_variant3_pd25_affine.py") `
          -DestDir (Join-Path $Root "지난파일\모델링_이전버전\구버전_학습스크립트")

# classifiers.py는 05_Model_Evaluation으로 이동 (아래 [4]단계 내용을 여기서 먼저 처리하여
# 03_Model 폴더를 완전히 비운 뒤 정리해야 빈 폴더 이동 시 파일이 함께 묻히지 않습니다)
Move-Safe -Source (Join-Path $Root "03_Model\classifiers.py") `
          -DestDir (Join-Path $Root "05_Model_Evaluation") -NewName "ml_classifiers_kfold_eval.py"

# 03_Model 폴더는 이제 완전히 비어있을 것 -> 지난파일로 이동(빈 폴더 정리)
Move-Safe -Source (Join-Path $Root "03_Model") -DestDir (Join-Path $Root "지난파일\기타") -NewName "03_Model_구조체(빈폴더)"

# ---------------------------------------------------------------------
# 3. 04_Feature_Engineering - CCA 융합 + WOA 최적화 (파일명도 더 명확하게 변경)
# ---------------------------------------------------------------------
Write-Log "`n[3] 04_Feature_Engineering (CCA 융합 + WOA 특징선택, 03 학습 이후 실행)"
Move-Safe -Source (Join-Path $Root "02_FeatureEngineering\fusion.py") `
          -DestDir (Join-Path $Root "04_Feature_Engineering") -NewName "cca_feature_fusion.py"
Move-Safe -Source (Join-Path $Root "02_FeatureEngineering\feature_optimization.py") `
          -DestDir (Join-Path $Root "04_Feature_Engineering") -NewName "woa_feature_selection.py"

# 구버전 fusion_optimization.py(대체됨) -> 지난파일
Move-Safe -Source (Join-Path $Root "02_FeatureEngineering\fusion_optimization.py") `
          -DestDir (Join-Path $Root "지난파일\모델링_이전버전\구버전_피처엔지니어링")

# 02_FeatureEngineering 폴더 자체는 이제 비어있을 것 -> 지난파일로 이동
Move-Safe -Source (Join-Path $Root "02_FeatureEngineering") -DestDir (Join-Path $Root "지난파일\기타") -NewName "02_FeatureEngineering_구조체(빈폴더)"

# ---------------------------------------------------------------------
# 4. 05_Model_Evaluation - 분류기 추론 + k-fold 평가
#    (classifiers.py 이동은 위 [2]단계에서 03_Model을 비우는 과정에 이미 처리했습니다)
# ---------------------------------------------------------------------
Write-Log "`n[4] 05_Model_Evaluation (분류기 학습/추론/k-fold 평가) - 파일 이동은 [2]단계에서 완료"

# ---------------------------------------------------------------------
# 5. 06_Result (기존 04_Result 내용물 이관, 번호만 변경)
# ---------------------------------------------------------------------
Write-Log "`n[5] 06_Result (기존 04_Result -> 06_Result, 현재 비어있음)"
Copy-ContentInto -SrcDir (Join-Path $Root "04_Result") -DestDir (Join-Path $Root "06_Result")
Move-Safe -Source (Join-Path $Root "04_Result") -DestDir (Join-Path $Root "지난파일\기타") -NewName "04_Result_구조체(빈폴더)"

# ---------------------------------------------------------------------
# 6. 07_Document (기존 05_Document 내용물 이관, 번호만 변경)
# ---------------------------------------------------------------------
Write-Log "`n[6] 07_Document (기존 05_Document -> 07_Document)"
Copy-ContentInto -SrcDir (Join-Path $Root "05_Document") -DestDir (Join-Path $Root "07_Document")
Move-Safe -Source (Join-Path $Root "05_Document") -DestDir (Join-Path $Root "지난파일\기타") -NewName "05_Document_구조체(빈폴더)"

# ---------------------------------------------------------------------
# 7. 08_Visualization (기존 06_Visualization 내용물 이관, 번호만 변경)
# ---------------------------------------------------------------------
Write-Log "`n[7] 08_Visualization (기존 06_Visualization -> 08_Visualization)"
Copy-ContentInto -SrcDir (Join-Path $Root "06_Visualization") -DestDir (Join-Path $Root "08_Visualization")
Move-Safe -Source (Join-Path $Root "06_Visualization") -DestDir (Join-Path $Root "지난파일\기타") -NewName "06_Visualization_구조체(빈폴더)"

# ---------------------------------------------------------------------
# 8. 누락되었던 잔여물 정리
# ---------------------------------------------------------------------
Write-Log "`n[8] 1차 정리에서 누락된 잔여물 정리"
# val_sample_ids.txt가 1차 스크립트에서 누락되었음(발견된 버그) - 지난파일로 이동
Move-Safe -Source (Join-Path $Root "모델링\val_sample_ids.txt") `
          -DestDir (Join-Path $Root "지난파일\모델링_이전버전\특징벡터_npy")
# 이제 완전히 빈 껍데기만 남은 루트 레벨 구버전 폴더들 정리
Move-Safe -Source (Join-Path $Root "모델링") -DestDir (Join-Path $Root "지난파일\기타") -NewName "모델링_루트_구조체(빈폴더)"
Move-Safe -Source (Join-Path $Root "전처리") -DestDir (Join-Path $Root "지난파일\기타") -NewName "전처리_루트_구조체(빈폴더, 303_rerun_wsl 포함)"

# 1차 재구성 스크립트/로그/안내문 -> 이력 폴더로 보관
Move-Safe -Source (Join-Path $Root "정리_작업_스크립트.ps1") -DestDir (Join-Path $Root "지난파일\기타\정리작업_이력_v1")
Move-Safe -Source (Join-Path $Root "정리_작업_안내.md") -DestDir (Join-Path $Root "지난파일\기타\정리작업_이력_v1")
Get-ChildItem -LiteralPath $Root -Filter "정리_작업_로그_2*.txt" -File | ForEach-Object {
    Move-Safe -Source $_.FullName -DestDir (Join-Path $Root "지난파일\기타\정리작업_이력_v1")
}

# 01_Preprocessing의 리사이즈 폴더 이름 정리: 빈 "_최종" 폴더와 실제 데이터가 든 "_최종_v2"를 정리
# _최종(빈 폴더)는 지난파일로, _최종_v2(304개 파일, 진짜 최종본)는 이름에서 _v2를 제거
Move-Safe -Source (Join-Path $Root "01_Preprocessing\전처리06_리사이즈_최종") `
          -DestDir (Join-Path $Root "지난파일\기타") -NewName "전처리06_리사이즈_최종_빈폴더"
if (-not $DryRun) {
    $realFinal = Join-Path $Root "01_Preprocessing\전처리06_리사이즈_최종_v2"
    $cleanName = Join-Path $Root "01_Preprocessing\전처리06_리사이즈_최종"
    if ((Test-Path -LiteralPath $realFinal) -and (-not (Test-Path -LiteralPath $cleanName))) {
        Rename-Item -LiteralPath $realFinal -NewName "전처리06_리사이즈_최종"
        Write-Log "  [이름정리] 전처리06_리사이즈_최종_v2 -> 전처리06_리사이즈_최종 (304개 파일, 실제 최종본)"
    }
} else {
    Write-Log "  [예정-이름정리] 01_Preprocessing\전처리06_리사이즈_최종_v2 -> 전처리06_리사이즈_최종 (Rename, 실제 최종본 304개)"
}

Write-Log "`n===================================================================="
Write-Log "재구성 v2 완료 (DryRun=$DryRun). 로그 파일: $LogPath"
Write-Log "===================================================================="
