import os
import joblib
import pandas as pd
from data_loader import extract_prediction_features

def predict_singapore_2026():
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_path = os.path.join(models_dir, 'f1_race_model.joblib')

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file missing. Run 'python src/train.py' first.")

    artifacts = joblib.load(model_path)
    model = artifacts['model']
    feature_cols = artifacts['features']

    # Extract Singapore GP weekend data (Use event name string for FastF1 lookup)
    print("Extracting Singapore GP 2026 session data...")
    event_df = extract_prediction_features(2026, 'Singapore')
    
    if event_df is None:
        print("Could not retrieve session data for Singapore GP 2026.")
        return

    # Calculate model score predictions
    X_today = event_df[feature_cols]
    event_df['Predicted_Score'] = model.predict(X_today)
    
    # Rank continuous score output into integer finishing ranks
    event_df['Predicted_Rank'] = event_df['Predicted_Score'].rank(method='min').astype(int)

    # Output top results
    output = event_df[['Predicted_Rank', 'Abbreviation', 'Quali_Position', 'Sprint_Position', 'GridPosition']]
    output = output.sort_values(by='Predicted_Rank').reset_index(drop=True)

    print("\n================ SINGAPORE GP 2026 PREDICTED FINISH ================")
    print(output.to_string(index=False))

if __name__ == '__main__':
    predict_singapore_2026()