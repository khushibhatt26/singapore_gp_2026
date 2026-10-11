# 🏎️ Singapore Grand Prix 2026 — F1 Finishing Position Predictor

An end-to-end Machine Learning pipeline predicting driver finishing positions for the **Singapore Grand Prix 2026** at the Marina Bay Street Circuit, built using [FastF1](https://github.com/theOehrly/Fast-F1), `pandas`, and `scikit-learn`.

---

## 📌 Overview

Street circuits like Marina Bay demand extreme mechanical precision, high downforce, and flawless qualifying execution. This project extracts multi-season telemetry and timing data from the official Formula 1 timing feeds, engineers track-specific and momentum features, trains regression models, and produces a ranked prediction of the final race classification.

### 🌟 Key Highlights
- **Automated Data Ingestion**: Extracts historical qualifying, sprint, race, and weather data across 2023–2026 seasons using FastF1.
- **Smart Local Caching**: Minimizes API requests with session-level caching (`f1_cache/`) and preprocessed feature dataset caching (`data/historical_races.csv`).
- **Rich Feature Engineering**: Incorporates 12 distinct performance, momentum, environmental, and track-specific variables.
- **Benchmarked Models**: Trains Gradient Boosting Regressor (GBR) alongside Linear Regression baseline to select the best-performing model.
- **Pre-Race Prediction & Post-Race Validation**: Dedicated workflows to generate race predictions prior to lights out, and evaluate accuracy (MAE, Spearman rank correlation) after the chequered flag.

---

## 🏗️ Project Architecture

```
singapore_gp_2026/
├── data/
│   └── historical_races.csv       # Preprocessed historical feature dataset
├── f1_cache/                      # FastF1 disk cache for raw session timing & telemetry
├── models/
│   └── f1_race_model.joblib       # Serialized best model & feature metadata
├── src/
│   ├── data_loader.py             # Data fetching, session merging, and feature pipeline
│   ├── train.py                   # Model training, metric evaluation, and artifact export
│   ├── predict.py                 # Pre-race prediction script for Singapore GP 2026
│   └── validate.py                # Post-race accuracy scoring (MAE & Spearman rank)
├── requirements.txt               # Project dependencies
├── .gitignore                     # Git ignore rules (caches, models, environments)
└── README.md                      # Project documentation
```

---

## 📊 Feature Set (12 Features)

The model trains on 12 features designed to capture single-lap pace, race-weekend progression, championship momentum, and Marina Bay historical affinity:

| Category | Feature Name | Description |
|---|---|---|
| **Qualifying Pace** | `Quali_Position` | Driver's official qualifying classification (P1–P20) |
| | `Quali_Time_Delta` | Lap time delta (in seconds) relative to the pole position lap |
| **Grid Position** | `GridPosition` | Starting grid slot after accounting for engine/grid penalties |
| **Sprint Session** | `Sprint_Position` | Finishing position in the sprint race (defaults to qualifying if no sprint) |
| | `Sprint_Pos_Gained` | Positions gained or lost during the sprint session |
| **Season Momentum** | `Driver_Season_AvgFinish` | Rolling expanding average finish for the driver earlier in the current season *(strictly prior races, no future data leakage)* |
| | `Team_Season_AvgFinish` | Rolling expanding average finish for constructor earlier in the current season |
| | `Driver_Cumulative_Points`| Championship points accumulated by the driver entering the race weekend |
| **Track Heritage** | `Driver_Singapore_AvgFinish` | Driver's historical average finish at the Singapore Grand Prix |
| | `Team_Singapore_AvgFinish` | Constructor's historical average finish at the Singapore Grand Prix |
| **Track Conditions**| `Air_Temp` | Average air temperature (°C) during qualifying session |
| | `Humidity` | Average relative humidity (%) during qualifying session |

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- **Python 3.10+**
- Virtual environment (`venv` recommended)

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone <repo-url>
cd singapore_gp_2026

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS / Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

### 1. Train the Model
Fetches multi-season data (2023–2026), compiles historical features, trains the regression models, and exports the top performer to `models/f1_race_model.joblib`:

```bash
python src/train.py
```

**Example Output:**
```text
Fetching historical F1 race data...
Loading dataset from local cache: /.../data/historical_races.csv
Training Complete.
Gradient Boosting Regressor MAE: 2.856 positions
Linear Regression MAE:            3.140 positions
Saved trained model to /.../models/f1_race_model.joblib
```

> **Note**: To force a re-fetch or re-computation of features, delete `data/historical_races.csv` before running `python src/train.py`.

---

### 2. Predict Singapore GP 2026
Once qualifying and sprint sessions for the Singapore Grand Prix 2026 have completed, run the prediction script:

```bash
python src/predict.py
```

**Example Output:**
```text
Extracting Singapore GP 2026 session data...

================ SINGAPORE GP 2026 PREDICTED FINISH ================
 Predicted_Rank Abbreviation  Quali_Position  Sprint_Position  GridPosition
              1          VER               1                1             1
              2          NOR               2                2             2
              3          PIA               3                3             3
              4          LEC               4                4             4
              5          RUS               5                5             5
              6          HAM               6                6             6
              ...
```

---

### 3. Validate After the Race
Once official race results for Singapore GP 2026 are published, evaluate model performance:

```bash
python src/validate.py
```

**Metrics Calculated:**
- **MAE (Mean Absolute Error)**: Average deviation between predicted finishing rank and actual finishing rank.
- **Spearman Rank Correlation ($\rho$)**: Measures monotonic rank ordering accuracy (ranges from -1.0 to +1.0).

---

## 💡 Technical Notes & API Rate Limits

- **FastF1 & Ergast API Rate Limiting**: The underlying F1 timing data provider enforces an hourly rate limit (typically 500 calls/hour).
  - All downloaded session files are automatically cached in `f1_cache/` so identical sessions are never requested twice.
  - If you encounter `⚠️ API rate limit reached (500 calls/hour)`, data gathered up to that point is preserved. You can wait 30–60 minutes or continue with locally cached sessions.
- **Data Leakage Prevention**: In `data_loader.py`, rolling averages (`Driver_Season_AvgFinish` and `Team_Season_AvgFinish`) and points (`Driver_Cumulative_Points`) strictly use `shift(1)` within each season to avoid lookahead leakage from the race being evaluated.
- **Missing Data Handling**: Median imputation via `SimpleImputer(strategy='median')` is applied to handle edge-case NaNs in track debutants or unclassified grid positions.

---

## 🛠️ Tech Stack

- **Data Ingestion**: [FastF1](https://docs.fastf1.dev/)
- **Data Processing**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning**: [scikit-learn](https://scikit-learn.org/) (`GradientBoostingRegressor`, `LinearRegression`, `SimpleImputer`)
- **Evaluation**: [SciPy](https://scipy.org/) (`spearmanr`), `mean_absolute_error`
- **Serialization**: [Joblib](https://joblib.readthedocs.io/)
