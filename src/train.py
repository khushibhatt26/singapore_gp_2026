import os
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from data_loader import build_historical_dataset

FEATURE_COLS = [
    # Core session features
    'Quali_Position',
    'Quali_Time_Delta',
    'GridPosition',
    'Sprint_Position',
    'Sprint_Pos_Gained',
    # Form / momentum
    'Driver_Season_AvgFinish',      # Rolling avg finish this season (prior races)
    'Team_Season_AvgFinish',        # Constructor rolling avg finish
    'Driver_Cumulative_Points',     # Championship points going into the race
    # Track-specific history
    'Driver_Singapore_AvgFinish',   # Driver's historical avg at Singapore
    'Team_Singapore_AvgFinish',     # Constructor's historical avg at Singapore
    # Weather (from qualifying session)
    'Air_Temp',                     # Air temperature (°C)
    'Humidity',                     # Relative humidity (%)
]

def train():
    print("Fetching historical F1 race data...")
    df = build_historical_dataset(seasons=[2023, 2024, 2025, 2026])
    
    # Filter valid finishing positions
    clean_df = df.dropna(subset=['Target_Race_Position']).copy()

    # Impute any residual NaN in features with column medians (safety net)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(clean_df[FEATURE_COLS]), columns=FEATURE_COLS)
    y = clean_df['Target_Race_Position'].reset_index(drop=True)

    # Initialize models
    gbr = GradientBoostingRegressor(n_estimators=120, learning_rate=0.05, max_depth=3, random_state=42)
    lr = LinearRegression()

    # Train models
    gbr.fit(X, y)
    lr.fit(X, y)

    # Evaluate error metrics
    gbr_mae = mean_absolute_error(y, gbr.predict(X))
    lr_mae = mean_absolute_error(y, lr.predict(X))

    print(f"Training Complete.")
    print(f"Gradient Boosting Regressor MAE: {gbr_mae:.3f} positions")
    print(f"Linear Regression MAE:            {lr_mae:.3f} positions")

    # Save model and feature definitions
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    best_model = gbr if gbr_mae <= lr_mae else lr
    model_path = os.path.join(models_dir, 'f1_race_model.joblib')
    joblib.dump({'model': best_model, 'features': FEATURE_COLS}, model_path)
    print(f"Saved trained model to {model_path}")

if __name__ == '__main__':
    train()