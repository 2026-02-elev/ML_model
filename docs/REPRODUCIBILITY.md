# 재현

이 저장소만으로 재현할 수 있는 것은 저장된 모델의 오프라인 추론이다. 학습과 테스트 점수 재계산은 공개되지 않은 실측 파일이 필요하다.

## 이 저장소에서 확인하는 추론

저장소 루트에서 실행한다.

```powershell
python scripts/verify_models.py
python scripts/run_demo.py
python -m pytest tests
```

`examples/`의 CSV는 스키마를 맞춘 합성 데이터다. 그 출력은 실측 테스트 점수가 아니다. 같은 파일을 두 번 넣으면 예측은 같다. 비교 기준은 `examples/expected_output.json`이다.

시드 20261111은 학습 때 사용한 값이다. 이 저장소의 추론은 난수를 쓰지 않는다.

## 공개 테스트 점수는 다시 계산하지 못한다

`python scripts/evaluate.py --data <evaluation-data-path> --output outputs/evaluation`는 호출자가 준 CSV에 Ridge를 적용한다. 실측 테스트 분할 파일은 이 저장소에 없다. 따라서 보고서에 적힌 MAE, onset, 매크로 F1을 이 clone만으로 다시 만들지 못한다.

## 학습을 다시 실행하는 조건

학습 코드와 실측 zip은 이 저장소에 없다. 비공개 작업 디렉터리에서 사용한 명령은 다음과 같다.

```powershell
cd D:\data-generator\cellguard-real-data
..\.venv\Scripts\python.exe scripts\run_all.py --config configs\pipeline.yaml --phase all
```

그 명령을 실행하려면 실측 번들, Python 3.12.10, 그리고 학습 당시 라이브러리(numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, torch 2.15.0+cpu)가 필요하다. 기록된 전체 실행은 약 725초였다. 이번 공개 패키지를 실행하는 데는 그 학습을 다시 할 필요가 없다.

학습으로 고른 모델 중 HistGradientBoosting과 Autoencoder는 가중치가 디스크에 남지 않았다. 없는 파일을 맞추려고 테스트 데이터로 다시 맞추지 않았다.
