"""
validate.py — Run AFTER the Singapore GP 2026 race to score model accuracy.
Compares predicted finish order vs actual results.
"""
import os
import joblib
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from data_loader import extract_weekend_features

FEATURE_COLS = [
    'Quali_Position',
    'Quali_Time_Delta',
    'GridPosition',
    'Sprint_Position',
    'Sprint_Pos_Gained',
    'Driver_Season_AvgFinish',
    'Team_Season_AvgFinish',
    'Driver_Cumulative_Points',
    'Driver_Singapore_AvgFinish',
    'Team_Singapore_AvgFinish',
    'Air_Temp',
    'Humidity',
]

def validate_singapore_2026():
    """
    Fetches the actual Singapore GP 2026 race results and scores the saved model.
    Run this after the race has concluded and data is published to the FastF1 API.
    """
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_path = os.path.join(models_dir, 'f1_race_model.joblib')

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file missing. Run 'python src/train.py' first.")

    artifacts = joblib.load(model_path)
    model = artifacts['model']

    print("Fetching Singapore GP 2026 actual race results...")
    actual_df = extract_weekend_features(2026, 'Singapore')

    if actual_df is None:
        print("Could not retrieve session data. Try again once the race data is published.")
        return

    if actual_df['Target_Race_Position'].isna().all():
        print("Race results are not yet available. Run this script after the race.")
        return

    # Drop rows with missing actual finish (DNF/DSQ with no position)
    actual_df = actual_df.dropna(subset=['Target_Race_Position'])

    # Predict
    X = actual_df[FEATURE_COLS]
    actual_df = actual_df.copy()
    actual_df['Predicted_Score'] = model.predict(X)
    actual_df['Predicted_Rank'] = actual_df['Predicted_Score'].rank(method='min').astype(int)
    actual_df['Actual_Rank'] = actual_df['Target_Race_Position'].rank(method='min').astype(int)
    actual_df['Error'] = (actual_df['Predicted_Rank'] - actual_df['Actual_Rank']).abs()

    # Metrics
    mae = mean_absolute_error(actual_df['Actual_Rank'], actual_df['Predicted_Rank'])
    corr, p_value = spearmanr(actual_df['Predicted_Rank'], actual_df['Actual_Rank'])

    # Output
    output = actual_df[['Abbreviation', 'Actual_Rank', 'Predicted_Rank', 'Error']].sort_values('Actual_Rank').reset_index(drop=True)

    print("\n========= SINGAPORE GP 2026 — PREDICTED vs ACTUAL =========")
    print(output.to_string(index=False))
    print(f"\n  MAE:                 {mae:.2f} positions")
    print(f"  Spearman Rank Corr:  {corr:.3f}  (p={p_value:.4f})")
    if corr >= 0.8:
        print("  ✅ Excellent predictive accuracy!")
    elif corr >= 0.6:
        print("  ✅ Good predictive accuracy.")
    elif corr >= 0.4:
        print("  ⚠️  Moderate predictive accuracy.")
    else:
        print("  ❌ Weak correlation — consider retraining with more data.")

if __name__ == '__main__':
    validate_singapore_2026()
