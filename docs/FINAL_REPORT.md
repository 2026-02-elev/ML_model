# CellGuard 실측 데이터 학습·평가 보고

작성 시각: 2026-11-11T13:42:10Z

이 문서의 수치는 `cellguard-real-data/results`에 저장된 이번 실행 결과다. `2026-11-11_measured.md`의 수치는 선행 실험으로만 인용한다. 합성 데이터 모델의 가중치와 임계값은 쓰지 않았다.

## 1. 프로젝트 개요

CellGuard는 슈퍼커패시터 4셀 직렬 팩의 이상을 보는 시스템이다. 측정과 고정 임계 보호는 Arduino Nano, 엣지 계산은 Raspberry Pi Zero 2 WH를 전제로 한다. 이번 작업은 그 장치에서 받은 파일을 오프라인으로 학습하고 평가한 것이다. 펌웨어, 릴레이, MOSFET은 바꾸지 않았다.

- 모델 ②: 다음 1초 셀 전압 예측, 잔차, 온도 점수. 안전 판단을 대신한다고 보지 않고, 오프라인 경보만 계산했다.
- 모델 ③: 검증에서 고른 창(이번 실행은 30초)으로 명령된 NORMAL/ESR/LEAK/HEAT를 구분한다.
- 모델 ⑥: 정상 창만으로 학습한 오토인코더. 서버용 검토 모델이다.
- 부하 강하 ESR: CGP 전압과 직전 CGT 휴지 전압의 차를 셀 간에 비교한다. 전류가 없어 옴 단위 저항은 계산하지 않았다.

## 2. 실측 데이터

- 독립 수집 run: 14개. manifest 일치: True.
- 텔레메트리: 80020행. 라벨 파일: 80020행.
- 학습 후보 feature 행: 72321. 제외: 7699.
- CGP 부하 측정: 1194행. CGT 시계열에 합치지 않았다.
- 장치: {'PCB2-20260918-R2': 80020}. 펌웨어: {'0.1.9-RUNTIME': 80020}.
- 셀 순서: cell_1=H_, cell_2=H8, cell_3=H7, cell_4=H1. 학습 입력도 이 순서다.
- 샘플 간격 중앙값: 0.998984초. 결론: 1Hz_only_no_5Hz_raw_stream. 5 Hz 원본은 없어서 보간으로 만들지 않았다.
- 전류: feature 표의 비결측 전류 0개. null을 0 A로 바꾸지 않았다.
- 교정: 관측 calibration source {'NOMINAL': 79980, 'None': 40}. 진단 문자열 CAL_NOMINAL 횟수 28.
- 라벨: physical_fault_verified가 참인 행은 0개. 신뢰도는 COMMANDED_ONLY다.

정상 1초 예측에 실제로 쓴 쌍 수: train 39739, validation 5783, test 10331.

## 3. 선행 실험

선행 보고서(테스트 3세션)는 ② 과열 100%, ESR 0%, 누설 0%, ③ ESR 10.5%, 누설 90%, 과열 100%, 매크로 F1 0.659, ⑥ AUROC 0.60, 부하 ESR 100%를 적었다. 그 코드(`app/core/measured.py`, `load_check.py`, `load_esr.py`)는 이 작업 공간에 없다. 그래서 이번 숫자는 재구현 결과이고, 차이를 개선율로 쓰지 않는다. 비교표는 `PREVIOUS_EXPERIMENT_COMPARISON.csv`다.

## 4. Phase 1~3

분할은 시간 순 14개 run을 6/5/3으로 고정했다. 학습 run ['run-20261107T075840Z-834d92569d0d', 'run-20261107T130212Z-a8d08382d9e7', 'run-20261107T141531Z-26c396f21530', 'run-20261108T022249Z-216d6c95a048', 'run-20261108T042957Z-fc9dd84ae8f8', 'run-20261108T083002Z-3df6154898a4']. 검증 run ['run-20261108T145459Z-39afda733aa3', 'run-20261108T151639Z-e6c3eb8c9e09', 'run-20261109T074950Z-587050bf183f', 'run-20261109T081421Z-e490f418113d', 'run-20261109T081646Z-89af6b0cec8b']. 테스트 run ['run-20261109T092358Z-b8051bb2d2ef', 'run-20261109T123029Z-335884aca822', 'run-20261110T001039Z-63d2979f11a9'].

Phase 3 기준 검출기(검증에서 선택, 테스트는 한 번):

- 모델 `persistence`, 온도 방식 `rise`, alpha `0.2`
- 전압 임계 0.025206 V, 온도 임계 0.7000 °C
- 테스트 정상 경보 onset 1회 / 정상 시간 4.261시간
- 테스트 명령 ESR onset recall 0.1053 (2/19), 평균 지연 3.00초
- 테스트 명령 LEAK onset recall 0.0000 (0/10)
- 테스트 명령 HEAT onset recall 0.5294 (9/17), 평균 지연 17.20초

ECM은 전류가 없어 BLOCKED다.

## 5. Phase 4

1초 정상 전압, 검증 MAE가 가장 낮은 모델은 `ridge`다. 테스트(선택 후 1회) MAE 0.004108 V, RMSE 0.005988 V, R² 0.9982.

검증 MAE (V): persistence=0.005098, ridge=0.003659, hist_gradient_boosting=0.004177, random_forest=0.004893, bagging=0.005177, extra_trees=0.005010, mlp_sklearn=0.011839, torch_mlp=0.013791.

분류 창 30.0초, 가중치 `sqrt`. 검증 매크로 F1로 고른 분류기는 `hist_gradient_boosting`다. 테스트 매크로 F1 0.7575, 정상 창을 고장으로 본 비율 0.0086.

명령 이벤트 onset recall (테스트, 이미 켜져 있던 경보는 성공에서 제외):

- ESR 0.8421 (16/19), 셀 일치 0.3750
- LEAK 0.8000 (8/10), 셀 일치 0.7500
- HEAT 1.0000 (17/17), 셀 일치 0.4118

셀 머리: {'status': 'COMPLETED', 'meaning': 'commanded cell id on fault windows, not a verified physical location', 'validation': {'n': 333, 'accuracy': 0.7447447447447447}, 'test': {'n': 1247, 'accuracy': 0.7177225340817963}}.

오토인코더 상태 COMPLETED. 테스트 AUROC 0.4946, PR-AUC 0.2628, 정상 창 경보율 0.0062, ESR/LEAK/HEAT 창 recall 0.1711 / 0.0000 / 0.3931.

부하 ESR 상태 COMPLETED. 짝지은 부하 1194, 짝 시각차 중앙값 494.0 ms. 임계값 3.4430 (검증 정상 최대 + 1e-6). 테스트 이벤트 recall 1.0000 (19/19), 셀 일치 1.0000, 정상 부하 오경보 0/169.

열화·RUL: DATA_INSUFFICIENT / DATA_INSUFFICIENT. The sessions are short commanded experiments on one pack. There is no measured capacity label, no end-of-life label, and no valid current for coulomb counting. Cycle ids count injection cycles, not lifetime aging cycles.

LightGBM은 이 환경에 설치되어 있지 않아 학습하지 않았다.

## 6. 비교에서 읽을 것

전압 예측의 테스트 오차는 위 MAE다. 분류의 창 단위 점수는 `CLASSIFICATION_COMPARISON.csv`, 이벤트 단위 점수는 `EVENT_DETECTION_COMPARISON.csv`에 있다. 샘플 재현율과 이벤트 onset을 같은 이름으로 부르지 않았다.

선행 수치와 이번 수치의 평가 코드가 같지 않으므로 향상률은 계산하지 않았다.

## 7. 배포

Raspberry Pi Zero 2 WH에서 시간을 재지 못했다. `EDGE_BENCHMARK_PENDING`. PC에서 NumPy로 다시 구현한 Ridge와 로지스틱의 출력 일치는 `results/phase6/parity_test.json`에 있다. 운영 배포 준비 완료로 표시하지 않는다. NFR-01의 5초는 추론 시간만이 아니다.

## 8. 한계

한 리그, 한 팩, 수집 run 14개, 테스트 run 3개다. 전압은 약 0.8 V 이하다. 교정은 NOMINAL이다. 전류가 없다. 라벨은 명령이다. 장기 열화 정답이 없다.

## 9. 결론

1. 독립 실험 run은 14개다.
2. 파일에서 확인된 것은 명령된 ESR, LEAK, HEAT와 정상 운전이다. 센서·UART 고장 클래스는 학습 후보에 없다.
3. 물리적 고장 분류, 전류 ECM, RUL, LightGBM은 학습하지 않았다.
4. 검증 MAE 기준 정상 전압 모델은 `ridge`다.
5. 잔차·온도 경보의 검증 선택 모델은 `persistence` / `rise`다. 테스트 HEAT onset recall은 0.5294다.
6. 그 검출기의 테스트 정상 onset은 1회다. 분류기의 테스트 정상 창 오분류율은 0.0086다.
7. 명령 ESR은 부하 강하 판정의 테스트 recall 1.0000가 1초 잔차보다 이 데이터에서 역할이 크다. 잔차 검출기 ESR onset은 0.1053다.
8. 명령 LEAK의 분류기 onset recall은 0.8000다.
9. 명령 HEAT는 평가했다. 검출기 onset 0.5294, 분류기 onset 1.0000. 물리 발열 검증은 아니다.
10. 명령 조건 분류의 검증 1위는 `hist_gradient_boosting`다.
11. 고장 창의 명령 셀 구분 테스트 정확도는 셀 머리 결과 {'status': 'COMPLETED', 'meaning': 'commanded cell id on fault windows, not a verified physical location', 'validation': {'n': 333, 'accuracy': 0.7447447447447447}, 'test': {'n': 1247, 'accuracy': 0.7177225340817963}} 안에 있다. 이벤트 셀 일치율은 위 onset 성공 건에 대해서만 계산했다.
12. 지연은 이벤트 표의 mean_delay_s다. 검출기 HEAT 17.20초, 분류기 HEAT 2.70초.
13. 5초 보호 목표는 추론만으로 충족했다고 말하지 않는다. 장치 종단 지연을 재지 않았다.
14. Pi에 올리기 쉬운 형태는 Ridge와 로지스틱이다. 검증 점수가 더 높은 트리·신경망은 변환 비용이 따로 있다.
15. 1초 Ridge는 특징이 작고 행렬 곱으로 끝난다. 창 분류기는 과거 창을 모아야 해서 입력이 늦다. 정확도와 그 지연을 한 숫자로 합치지 않았다.
16. 큰 제한은 교정되지 않은 전압, 없는 전류, 짧은 명령 주입, 한 팩이다.
17. 합성 실험(Phase 3D-1)은 셀 전압이 더 높고 전류와 고장 정답이 있다. 그 MAE와 이번 MAE는 같은 운전이 아니다.
18. 추가로 필요한 것은 교정된 전압, 유효한 전류, 독립적으로 확인한 물리 고장, 더 높은 전압 구간, 다른 팩, 수명 데이터, Pi 위 지연 측정이다.
19. 운영 배포 준비가 된 모델은 없다.
20. ②③⑥과 부하 ESR은 모두 명령 조건에 대한 잠정 결과다. 물리 고장 성능은 PHYSICAL_FAULT_VALIDATION_PENDING이다.

라이브러리: {'python': '3.12.10', 'numpy': '2.5.3', 'pandas': '3.0.5', 'sklearn': '1.9.1', 'matplotlib': '3.11.2', 'joblib': '1.6.0', 'torch': '2.15.0+cpu', 'lightgbm': 'not_installed'}.
