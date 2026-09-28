```markdown
# AWG Digital Twin: Phase 1

Data-driven prototype that predicts water output (litres) of an Atmospheric Water Generator from environmental and operating parameters.

## Data
~35,000 rows: solar radiation, ambient temperature, humidity, wind speed, absolute pressure, dew point, and system temperatures (Tevp, Thot, Tcold, Th,TEC). Duplicates removed, missing values interpolated, and physically invalid readings (e.g., humidity above 100%, extreme output outliers) corrected or excluded.

## Method
- Split: 70% train / 15% validation / 15% test (chronological)
- Baseline: Linear Regression
- Primary model: XGBoost Regressor, tuned via randomized search (n_estimators, max_depth, learning_rate, subsample, plus regularisation)

## Results (Test Set)
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.0766 | 0.1555 | 0.349 |
| XGBoost | 0.0089 | 0.0151 | 0.994 |

## Key Drivers
Solar radiation, absolute pressure, and humidity are the most influential factors (permutation importance).

## Outputs
Trained model (`awg_xgb_model.json`), metrics table, feature importance chart, predicted-vs-actual plot, and `model_metadata.json` for future frontend integration.

## Limitations
- Time features (Month, Day, Solar Hours) were unusable due to missing values
- Pressure importance likely reflects a time/weather proxy
- Validated on held-out data from the same period; seasonal generalisation is untested

## Phase 2 (Planned)
Hybrid digital twin: thermodynamic condensation model, heat-transfer calculations, dew-point estimation, and XGBoost error correction for "what-if" simulation.
```
