### Issue Overview
Based on the recent peer review action list, the following areas required resolution:
1. **RUL Plot Timeframe**: Model evaluation and plotting were restricted to single window visualizations rather than reconstructing a continuous timeline over a complete engine lifecycle.
2. **Gradient Stability**: The training loop lacked gradient clipping, leaving the model vulnerable to exploding gradients and performance spikes (e.g., epoch 11).
3. **Feature Redundancy**: Sensors `s6`, `s14`, `s20`, and `s21` were identified as providing highly collinear and redundant data based on SHAP feature importance.

### Fixes Applied (Merged)
- **Data Engineering**: Modified `02_preprocessing.py` to add `s6`, `s14`, `s20`, and `s21` to the list of dropped sensors. 
- **Architecture**: Adjusted input sizes across `05_model.py` (MultiTaskModel) and FastAPI endpoints (`api/main.py`) from 18 features down to exactly 14 features.
- **Model Training**: Inserted `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before optimizer steps in `06_train.py` to enforce training stability.
- **Evaluation & Visualization**: Updated logical iteration in `07_evaluate.py` to run predictions over chronological cycle steps of the longest engine, properly plotting True vs. Predicted RUL overall timelines.
- **Outcome**: The whole pipeline was successfully retrained and validated with the new stable RMSE metrics and smooth learning curves.