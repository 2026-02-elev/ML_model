# 학습·추론 비용

수치는 Phase 4 `model_registry.csv`와 Phase 6 `inference_benchmark.csv`에 기록된 값이다. 이 저장소에서 다시 측정하지 않았다.

학습 시간은 그 실행의 CPU 경과 시간이다. 오토인코더와 부하 ESR은 등록표의 `train_seconds`가 비어 있어 여기에 숫자를 적지 않는다.

| 모델 | 작업 | train_seconds |
|---|---|---:|
| ridge | 1초 전압 | 0.1303094000031706 |
| persistence | 1초 전압 기준 | 0.0 |
| hist_gradient_boosting | 30초 명령 조건 분류 | 18.735877200000687 |
| logistic | 30초 명령 조건 분류, 선정되지 않은 참고 모델 | 5.514803600002779 |

PC에서 이미 스케일된 64개 벡터를 200번 반복한 평균이다. Raspberry Pi 측정이 아니다. 상태 `EDGE_BENCHMARK_PENDING`.

| 모델 | 이 PC에서 벡터 1개당 초 |
|---|---:|
| ridge | 1.49190624995299e-05 |
| logistic | 5.1855703122782874e-06 |

원본 표: `results/phase6/inference_benchmark.csv`. NumPy 출력 일치는 `results/phase6/parity_test.json`이다. Ridge 최대 절대차 0, 로지스틱 클래스 불일치 0, 각 64개 벡터.
