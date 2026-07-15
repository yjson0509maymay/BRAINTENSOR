# 03_Model_Training

3D-CNN/3D-ResNet 학습부터 CCA 융합·WOA 특징 선택·k-fold 평가까지 전체 파이프라인을 오케스트레이션합니다.

## 구성

| 파일 | 내용 |
|---|---|
| `train.py` | 메인 파이프라인. 3D-CNN·3D-ResNet 학습 → 특징 추출(FV-3, FC-4) → CCA 융합(Model-3) → WOA 특징 선택(Model-4) → 5/10/15-fold 평가까지 한 번에 실행 |
| `dataset.py` | `PPMIT2Dataset`, `get_holdout_split()`(70/15/15 hold-out, 피험자 단위 분할), `get_kfold_splits()`(k=5/10/15), `augment_volume_3d()`(회전/뒤집기/스케일/노이즈, train에만 on-the-fly 적용) |
| `train_ablation.py` | Ablation 후보(현재는 `CNN3D_Base` 8계층)만 별도로 학습 |
| `smoke_test.py` | 소규모 샘플·1epoch로 코드 동작만 빠르게 검증 |
| `DEVIATIONS.md` | 이 폴더 코드에서 논문에 없어 자체 결정한 값들의 근거 |
| `full_run_result.json` | 실제 303명 전체 학습 결과 (재현 정확도 66~69%대) |
| `smoke_test_result.json` | smoke test 실행 결과 |
| `모델링_작업리스트.txt` | 작업 메모 |

## 실행 방법

```bash
python train.py \
  --csv_path ../01_Preprocessing/data_0713.csv \
  --image_dir ../01_Preprocessing/전처리_0713/06_resized \
  --epochs 30 --batch_size 8 --effective_batch_size 64
```

물리 배치(`--batch_size`)와 gradient accumulation으로 논문 Table 3(Variant3 확인값 64)의 유효 배치를 재현합니다. GPU 없이 코드만 검증하려면 `--smoke_test` 플래그를 사용하세요.

## 참고

`split.py`는 이전에 쓰던 스크래치 코드이며, 지금은 `dataset.py`의 `get_holdout_split()`이 동일 로직을 정식으로 대체해서 `train.py`가 그쪽을 사용합니다.
