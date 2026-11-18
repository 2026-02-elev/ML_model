# Model selection

Choices used validation only. The test block in the metrics file was written after these choices and was not used to change them.

- Voltage model: `ridge`, lowest validation MAE among completed 1-step models.
- Detector: `persistence`, temperature mode `rise`, alpha `0.2`, voltage quantile `0.999`.
  The grid minimised validation normal alarm onsets, then maximised commanded HEAT onset recall, then LEAK onset recall.
- Classifier window: 30.0 s, highest validation macro F1 among 10/20/30 s.
- Class weight: `sqrt`, higher validation macro F1 between sqrt and balanced.
- Classifier: `hist_gradient_boosting`, highest validation macro F1, then lower normal-window fault rate.

These scores describe commanded experiment conditions. They are not verified physical-fault accuracy.

## 이 저장소에 들어 있는 가중치

선정 결과와 저장된 파일은 같다.

| 선정 모델 | 저장 여부 |
|---|---|
| Ridge, alpha 0.1 | `models/regression/ridge_numpy.npz`. Phase 6 패키지와 SHA-256이 같다. |
| persistence / rise | `models/anomaly/thresholds.json`. 학습 가중치가 없는 규칙이다. |
| HistGradientBoosting, 30초, sqrt | 가중치 파일이 없다. `models/classifier/WEIGHTS_NOT_PERSISTED.json`. 다시 학습하지 않았다. |
| Autoencoder | 가중치 파일이 없다. `models/autoencoder/WEIGHTS_NOT_PERSISTED.json`. 다시 학습하지 않았다. |
| 부하 강하 ESR | `models/load_esr/parameters.json`의 mu, sigma, 임계값. |

`models/classifier/logistic_reference.npz`는 같은 실험에서 저장된 로지스틱 회귀다. 검증 매크로 F1은 0.5698663683972636이고, 선정 분류기가 아니다.
