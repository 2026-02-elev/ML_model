# CellGuard 실측 모델

슈퍼커패시터 4셀 직렬 팩의 실측 기록으로 고른 전압 예측, 잔차 경보, 부하 강하 판정을 재학습 없이 실행하는 저장소다. 고장 분류기와 오토인코더는 평가까지 끝났지만 가중치 파일이 없어 여기서 실행하지 않는다.

## Quick Start

Windows PowerShell:

```powershell
git clone https://github.com/2026-02-elev/ML_model.git
cd ML_model
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/run_demo.py
```

Linux, macOS:

```bash
git clone https://github.com/2026-02-elev/ML_model.git
cd ML_model
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_demo.py
```

데모가 끝나면 `outputs/demo_predictions.csv`가 생긴다. 예제 입력은 합성 데이터이고, 실측 테스트 점수가 아니다.

## 1. CellGuard 프로젝트 소개

CellGuard는 슈퍼커패시터 4셀 직렬 팩의 이상을 보는 시스템이다. 측정과 고정 임계 보호는 Arduino Nano, 엣지 계산은 Raspberry Pi Zero 2 WH를 전제로 한다. 전류 센서 INA219의 측정값은 이번 데이터에 없다.

이 저장소의 모델은 그 장치에서 모은 파일로 오프라인 학습한 결과다. 합성 데이터 모델의 가중치와 임계값은 쓰지 않았다. 라벨은 명령된 ESR, LEAK, HEAT와 정상 운전이며, 물리 고장이 확인된 행은 없다.

AI 역할은 네 가지로 나뉜다.

- ② 다음 1초 셀 전압, 그리고 잔차와 온도 상승으로 만든 경보
- ③ 30초 창의 명령 조건 분류
- ⑥ 정상 창만으로 학습한 오토인코더
- 부하 강하 ESR: 휴지 전압과 부하 전압의 셀별 차이

## 2. 시스템 아키텍처

```text
Arduino Nano  --UART-->  Raspberry Pi Zero 2 WH  --MQTT local-->  server
  ADC, NTC,               ② Ridge, persistence/EWMA,              ⑥ autoencoder
  fixed protection        ③ command classifier                    (weights not saved)
CGP load sample ----------------------------------------------->  load-drop ESR
```

셀 순서는 cell_1=H_, cell_2=H8, cell_3=H7, cell_4=H1이다. CGT 1 Hz 시계열과 CGP 부하 샘플은 한 표로 합치지 않는다.

## 3. 최종 모델

수치는 정상 1초 쌍 또는 명령 이벤트 기준이다. 정의는 [모델 비교](docs/MODEL_COMPARISON.md)에 있다. 선정 이유는 [모델 선정](docs/MODEL_SELECTION.md)에 있다.

| 모델 | 역할 | 파일 | 성능 | 실행 |
|---|---|---|---|---|
| ② Ridge alpha 0.1 | 다음 1초 전압 | `models/regression/ridge_numpy.npz` | 검증 MAE 0.0036587928360879915 V, 테스트 MAE 0.004108472472556147 V | 이 저장소에서 실행 |
| ② persistence, rise, alpha 0.2 | 잔차 EWMA와 온도 상승 경보 | `models/anomaly/thresholds.json` | 테스트 onset ESR 2/19, LEAK 0/10, HEAT 9/17. 정상 onset 1회 / 4.26051481280062시간 | 이 저장소에서 실행 |
| ③ HistGradientBoosting, 30초 | 명령 조건 분류 | 가중치 없음 | 검증 매크로 F1 0.6369627307090966, 테스트 매크로 F1 0.7575336905637502 | 실행할 수 없음 |
| ⑥ Autoencoder, 16 step | 재구성 오차 | 가중치 없음 | 테스트 AUROC 0.4946321057345614, PR-AUC 0.26279260444494035 | 실행할 수 없음 |
| 부하 강하 z-score | 휴지 대비 부하 전압 | `models/load_esr/parameters.json` | 테스트 19/19, 셀 일치 1.0, 오경보 0/169 | CGP와 휴지 전압이 있을 때만 |

로지스틱 회귀 파일 `models/classifier/logistic_reference.npz`는 선정 분류기가 아니다. 검증 매크로 F1은 0.5698663683972636이다. `--model logistic_reference`로만 실행한다.

## 4. 모델 성능 비교

비교표, 이벤트 onset, 클래스별 recall, 그림은 [docs/MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md)에 있다.

전압 모델은 검증 MAE가 가장 낮은 Ridge를 골랐다. 경보는 검증 정상 onset을 먼저 줄이고, 그다음 명령 HEAT와 LEAK onset을 보았다. 분류기는 검증 매크로 F1이 가장 높은 HistGradientBoosting을 골랐다. 테스트 점수는 그 선택을 바꾼 데 쓰지 않았다.

검출기의 테스트 HEAT 평균 지연은 17.202529774771797초이고, 분류기 HEAT 평균 지연은 2.7031579157885384초다. 분류기 ESR 셀 일치는 onset 성공 건 기준 0.375다. 이 분류기는 실행 파일이 없다.

선행 보고서의 검출기 HEAT 100%, ESR 0% 같은 숫자는 이번 모델의 점수가 아니다. `results/previous_experiment_comparison.csv`로만 구분한다.

## 5. 프로젝트 폴더 구조

```text
ML_model/
├── README.md
├── requirements.txt
├── pytest.ini
├── .gitignore
├── configs/
│   ├── inference.yaml
│   ├── model_registry.json
│   └── input_schema.json
├── docs/
│   ├── FINAL_REPORT.md
│   ├── MODEL_COMPARISON.md
│   ├── MODEL_SELECTION.md
│   ├── DATA_LIMITATIONS.md
│   └── REPRODUCIBILITY.md
├── models/
│   ├── checksums.json
│   ├── regression/ridge_numpy.npz
│   ├── anomaly/thresholds.json
│   ├── classifier/logistic_reference.npz
│   ├── classifier/WEIGHTS_NOT_PERSISTED.json
│   ├── autoencoder/WEIGHTS_NOT_PERSISTED.json
│   └── load_esr/parameters.json
├── src/
├── scripts/
│   ├── predict.py
│   ├── run_demo.py
│   ├── evaluate.py
│   └── verify_models.py
├── examples/
├── results/
└── tests/
```

라이선스 파일은 없다. 원 프로젝트에서 확인된 라이선스가 없어 새로 선언하지 않았다.

## 6. 설치 방법

추론에 필요한 라이브러리는 NumPy 2.5.3, pandas 3.0.5, PyYAML 6.0.1이다. 테스트에는 pytest 9.1.1이 있다. scikit-learn과 torch는 추론에서 import하지 않는다. 학습 당시 환경은 Python 3.12.10, scikit-learn 1.9.1, torch 2.15.0+cpu였다.

이 문서의 명령은 Python 3.12.10에서 확인했다.

모델 파일은 수 KB라 Git LFS 없이 저장소에 들어 있다. 별도 다운로드는 없다.

## 7. 가장 빠른 실행 방법

저장소를 clone한 디렉터리에서:

```powershell
python scripts/run_demo.py
```

하는 일:

1. 체크섬이 맞는 모델 파일을 읽는다.
2. `examples/example_timeseries.csv`와 `examples/example_load_esr.csv`를 읽는다.
3. 1초 특징을 만들고 Ridge와 persistence 경보를 계산한다.
4. 부하 강하 점수와, 선정되지 않은 로지스틱 참고 예측을 계산한다.
5. `outputs/`에 CSV를 쓴다.

HistGradientBoosting과 Autoencoder는 파일이 없다고 출력하고, 다른 모델로 대체하지 않는다.

## 8. 모델별 실행 방법

공통 시계열:

```powershell
python scripts/predict.py --model regression --input examples/example_timeseries.csv --output outputs/regression_result.csv
python scripts/predict.py --model anomaly --input examples/example_timeseries.csv --output outputs/anomaly_result.csv
python scripts/predict.py --model best --input examples/example_timeseries.csv --output outputs/predictions.csv
```

`best`는 Ridge 전압과 persistence 경보를 한 파일에 쓴다. 고장 유형 열은 만들지 않는다.

선정 분류기와 오토인코더는 종료 코드 2로 끝난다.

```powershell
python scripts/predict.py --model classifier --input examples/example_timeseries.csv --output outputs/classification_result.csv
python scripts/predict.py --model autoencoder --input examples/example_timeseries.csv --output outputs/anomaly_ae_result.csv
```

선정되지 않은 로지스틱 참고 모델만 따로 실행할 수 있다. 이 출력의 클래스 이름은 보고서의 테스트 매크로 F1 0.7575와 연결하지 않는다.

```powershell
python scripts/predict.py --model logistic_reference --input examples/example_timeseries.csv --output outputs/logistic_reference.csv
```

부하 강하는 1 Hz CGT 파일이 아니라, 이미 짝을 맞춘 휴지 전압과 부하 전압이다.

```powershell
python scripts/predict.py --model load_esr --input examples/example_load_esr.csv --output outputs/esr_result.csv
```

하드웨어 제어, MQTT 발행, 릴레이 조작은 하지 않는다.

## 9. 입력 데이터 형식

시계열 필수 열: `run_id`, `cell_1_voltage_v`, `cell_2_voltage_v`, `cell_3_voltage_v`, `cell_4_voltage_v`.

시간: `t_s`(초) 또는 `timestamp`(ISO-8601 또는 초). 샘플 간격은 약 1 Hz다. 5 Hz 원본은 없었고, 보간하지 않는다.

선택 열: `cell_1_temperature_c`부터 `cell_4_temperature_c`, `cell_voltage_spread_v`. 온도가 없으면 학습 중앙값으로 채운 뒤 표준화한다. spread가 없으면 네 셀 전압의 최대값에서 최소값을 뺀다. 단위는 전압 V, 온도 °C, spread V다.

전압이 비어 있거나 0 이하이면 오류로 멈춘다. 학습 표도 그런 행을 넣지 않았다. 전류 열은 받지 않는다.

같은 run에서 직전 샘플과의 간격이 0.5초 초과 2.0초 이하일 때만 1초 예측을 만든다. 첫 샘플은 `insufficient_history`, 그 밖은 `gap_outside_pair_window`다. 직전 간격도 그 범위일 때만 전압 변화량 `dv`를 쓰고, 아니면 `dv`와 `dt_prev`는 0이다.

30초 로지스틱 참고 창은 샘플이 8개 이상이고 창 길이가 22.5초 이상일 때만 만든다. 더 짧으면 행이 없는 CSV를 쓴다.

셀 순서는 H_, H8, H7, H1이다. 예제는 `examples/example_input.json`, `examples/example_timeseries.csv`다.

부하 ESR 필수 열은 `rest_cell_1_voltage_v`부터 `rest_cell_4_voltage_v`, `load_cell_1_voltage_v`부터 `load_cell_4_voltage_v`다. 스키마는 `configs/input_schema.json`이다.

## 10. 출력 데이터 해석

`best`와 데모 CSV:

| 열 | 의미 |
|---|---|
| `predicted_voltage_cell_*_v` | Ridge가 예측한 해당 시각의 셀 전압 |
| `ridge_abs_residual_cell_*_v` | 그 예측과 실제 전압의 절대 오차. 경보에는 쓰지 않는다 |
| `persistence_residual_cell_*_v` | 직전 전압과 현재 전압의 절대차. 경보의 전압 잔차 |
| `ewma_cell_*`, `ewma_score` | 잔차의 시간 EWMA와 네 셀 중 최댓값. alpha 0.2, 간격이 2.5초를 넘기면 초기화 |
| `temp_rise_cell_*_c`, `temp_rise_score_c` | 300초 기준선 대비 온도 상승. 음수는 0 |
| `anomaly_detected` | EWMA 최댓값이 0.025206015466180683 V 이상이거나 온도 상승 최댓값이 0.7 °C 이상 |
| `anomaly_cell` | 온도 경보면 온도 상승이 가장 큰 셀, 아니면 EWMA가 가장 큰 셀. 1부터 4 |
| `row_status` | `scored`, `insufficient_history`, `gap_outside_pair_window` |
| `model_version` | `measured-local-0` |

점수가 계산되지 않은 행의 경보 칸은 비어 있다. 경보 시각은 그 행의 `t_s` 또는 `timestamp`다.

로지스틱 참고 CSV의 `predicted_fault_type`은 NORMAL, ESR, LEAK, HEAT 중 하나이고 `prediction_confidence`는 해당 클래스의 softmax 값이다. 합성 예제에서 나온 클래스 이름은 실측 고장이 아니다.

부하 ESR CSV의 `score`는 네 셀 z-score의 최댓값이고, `alarm`은 score가 3.4429849433264446 이상일 때 참이다. `pred_cell`은 z가 가장 큰 셀이다.

## 11. 모델 성능 재현 방법

실측 평가 파일을 가지고 있다면 Ridge 예측만 다시 쓸 수 있다.

```powershell
python scripts/evaluate.py --data <evaluation-data-path> --output outputs/evaluation
```

인자 없이 실행하면 종료 코드 2와 함께, 공개 테스트 점수는 이 저장소만으로 다시 계산할 수 없다고 출력한다. 합성 예제로 실행한 MAE는 보고서의 테스트 MAE가 아니다.

이벤트 onset, 분류 매크로 F1, 오토인코더 AUROC, 부하 ESR 19/19는 비공개 실측 분할로 이미 계산된 값이다. 그 분할은 시간 순 14개 run 중 마지막 3개다.

## 12. 전체 모델 재학습 방법

데모를 실행하려고 다시 학습할 필요는 없다.

재학습은 실측 학습 데이터, 비공개 `cellguard-real-data` 학습 코드, Python 3.12와 학습 당시 라이브러리가 있을 때만 가능하다. 로컬에서 사용한 명령은 `docs/REPRODUCIBILITY.md`에 있다. 기록된 전체 실행은 약 725초였다. 결과는 그 프로젝트의 `results/`에 저장된다. 그 코드와 데이터는 이 GitHub 저장소에 없다.

## 13. Raspberry Pi Zero 2 WH 배포 관련 사항

Ridge 파일은 2050바이트, 로지스틱 참고 파일은 4488바이트다. 추론은 NumPy 행렬 곱이다. 입력 순서는 `configs/inference.yaml`의 14개 점 특징과 셀 순서 H_, H8, H7, H1이다.

이 저장소에는 OTA 패키지, 장치 주소, 펌웨어 변경이 없다. Raspberry Pi에서 지연을 측정하지 않았다. 상태는 EDGE_BENCHMARK_PENDING이다. PC에서 Ridge 64개 벡터의 NumPy 출력 차이는 0이었고, 로지스틱 참고 모델의 클래스 불일치는 0이었다. 그 PC 확인을 Pi 실측으로 적지 않는다.

NFR-01의 5초는 감지, 판단, 보호를 포함한 시간이다. 이 저장소의 행렬 곱 시간은 그 측정이 아니다.

## 14. 데이터 및 모델의 한계

물리 고장 라벨은 검증되지 않았다. 수집은 팩 1개, run 14개, 전압은 대략 0.8 V 이하, 교정은 NOMINAL이다. INA219 전류는 없다. 한 팩에서 고른 임계값이 다른 팩으로 이어진다고 보지 않는다.

데모 CSV는 1초 간격의 합성 전압이다. 25초부터 35초까지 4번 셀을 0.04 V씩 낮춰 경보 분기가 지나가게 만든 입력이며, 실측 평가 데이터가 아니다. 자세한 한계는 [docs/DATA_LIMITATIONS.md](docs/DATA_LIMITATIONS.md)에 있다.

운영 배포 준비가 된 모델은 없다.

## 15. 최종보고서

[CellGuard 최종보고서](docs/FINAL_REPORT.md)

보고서의 문장과 수치는 그대로 두었다. 선행 실험과 이번 Phase 1~6 결과는 보고서 3절과 `results/previous_experiment_comparison.csv`에서 구분한다.

## 16. 개발 환경 및 재현성

- 학습: Python 3.12.10, numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, matplotlib 3.11.2, joblib 1.6.0, torch 2.15.0+cpu. LightGBM은 설치되지 않아 학습하지 않았다. GPU는 쓰지 않았다.
- 이 저장소의 추론 확인: Python 3.12.10, numpy 2.5.3, pandas 3.0.5, PyYAML 6.0.1, pytest 9.1.1.
- 모델 버전 `measured-local-0`. 시드 20261111. 분할은 시간 순 6개 학습, 5개 검증, 3개 테스트 run이다.
- Ridge 파일 SHA-256 `115bcb1a80ae8e8059d6aca4c9a54a7949ee006cb5a99e0b0e7f3095906b04fd`. 이 값은 Phase 6에 기록된 Ridge 파일과 같다.
- 나머지 파일의 SHA-256은 `models/checksums.json`에 있다. 로지스틱 참고 파일은 스케일러를 포함하므로 Phase 6의 스케일러 없는 파일과 해시가 다르다.
- 체크섬이 다르면 로더가 예외를 내고 예측하지 않는다. pickle, joblib 파일은 읽지 않는다.

## 17. 라이선스 및 프로젝트 정보

원 프로젝트와 데이터에 대해 확인된 라이선스 문구가 없다. 이 저장소도 라이선스를 새로 붙이지 않았다. 실측 원본의 재배포 권한은 확인되지 않았으므로 포함하지 않았다.

모델 파일은 이 저장소에 올린 체크섬과 일치할 때만 사용한다. 다른 경로의 pickle 모델을 불러오지 않는다.
