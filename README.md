# HR Performance Rating Predictor

Predicts employee performance ratings (3 = Good, 4 = Excellent) from the IBM HR Analytics dataset using 7 ML models.

## Models

| Model | Accuracy |
|---|---|
| Random Forest | 84.69% |
| SVM (RBF) | 77.21% |
| Gradient Boosting | 74.49% |
| Logistic Regression | 57.48% |
| Naive Bayes | 57.48% |
| KNN (k=5) | 56.80% |
| Decision Tree | 52.38% |

## Setup

```bash
pip install -r requirements.txt
```

## Run

### Streamlit UI (interactive predictor)
```bash
streamlit run streamlit_app.py
```

### CLI training script (prints results to terminal)
```bash
python mlproj_fixed.py
```

## Files

- `streamlit_app.py` — Interactive web UI for predicting performance ratings
- `mlproj_fixed.py` — Standalone training script (no UI, prints results)
- `requirements.txt` — Python dependencies

## Key Design Decisions

1. **Removed `PercentSalaryHike`** — It's directly determined by `PerformanceRating` in this dataset (data leakage).
2. **Train/test split before resampling** — Upsampling only happens on training data to prevent leakage.
3. **StandardScaler fitted on training data only** — Test data is transformed, not fitted.
