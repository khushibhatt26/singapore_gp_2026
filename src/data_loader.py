import os
import time
import pandas as pd
import fastf1
from fastf1.exceptions import RateLimitExceededError

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'f1_cache')
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

def extract_weekend_features(year: int, round_identifier):
    try:
        # Respect API rate limits with a small pause between loads
        time.sleep(1.0)

        # Load main qualifying (with weather for temp/humidity features)
        q_session = fastf1.get_session(year, round_identifier, 'Q')
        q_session.load(telemetry=False, weather=True)
        
        q_df = q_session.results[['DriverNumber', 'Abbreviation', 'TeamName', 'Position', 'Q3', 'Q2', 'Q1']].copy()
        q_df['BestLap'] = q_df['Q3'].fillna(q_df['Q2']).fillna(q_df['Q1'])
        q_df['BestLapSec'] = q_df['BestLap'].dt.total_seconds()
        min_q = q_df['BestLapSec'].min()
        q_df['Quali_Time_Delta'] = q_df['BestLapSec'] - min_q
        q_df.rename(columns={'Position': 'Quali_Position'}, inplace=True)

        # Load sprint session if present
        has_sprint = True
        try:
            s_session = fastf1.get_session(year, round_identifier, 'S')
            s_session.load(telemetry=False, weather=False)
            s_df = s_session.results[['DriverNumber', 'Position', 'GridPosition']].copy()
            s_df.rename(columns={'Position': 'Sprint_Position', 'GridPosition': 'Sprint_Grid'}, inplace=True)
            s_df['Sprint_Pos_Gained'] = s_df['Sprint_Grid'] - s_df['Sprint_Position']
        except Exception:
            has_sprint = False

        # Load main race session
        r_session = fastf1.get_session(year, round_identifier, 'R')
        r_session.load(telemetry=False, weather=False)
        r_df = r_session.results[['DriverNumber', 'Position', 'GridPosition', 'Status', 'Points']].copy()
        r_df.rename(columns={'Position': 'Target_Race_Position', 'Points': 'Race_Points'}, inplace=True)

        # Merge results
        df = pd.merge(
            q_df[['DriverNumber', 'Abbreviation', 'TeamName', 'Quali_Position', 'Quali_Time_Delta']],
            r_df[['DriverNumber', 'GridPosition', 'Target_Race_Position', 'Status', 'Race_Points']],
            on='DriverNumber',
            how='inner'
        )

        if has_sprint:
            df = pd.merge(df, s_df[['DriverNumber', 'Sprint_Position', 'Sprint_Pos_Gained']], on='DriverNumber', how='left')
        else:
            df['Sprint_Position'] = df['Quali_Position']
            df['Sprint_Pos_Gained'] = 0

        df['Sprint_Position'] = df['Sprint_Position'].fillna(20)
        df['Sprint_Pos_Gained'] = df['Sprint_Pos_Gained'].fillna(0)
        max_delta = df['Quali_Time_Delta'].max()
        df['Quali_Time_Delta'] = df['Quali_Time_Delta'].fillna(max_delta + 1.0 if pd.notnull(max_delta) else 2.0)
        df['Race_Points'] = df['Race_Points'].fillna(0)

        # ── Weather features from qualifying session (same weekend conditions) ──
        try:
            w = q_session.weather_data
            df['Air_Temp'] = round(w['AirTemp'].mean(), 1) if not w.empty else 30.0
            df['Humidity'] = round(w['Humidity'].mean(), 1) if not w.empty else 75.0
        except Exception:
            df['Air_Temp'] = 30.0
            df['Humidity'] = 75.0

        # Tag with season/round for rolling feature computation
        df['Year'] = year
        df['RoundNumber'] = int(round_identifier) if isinstance(round_identifier, int) else 0

        return df

    except RateLimitExceededError:
        print("\n⚠️ API rate limit reached (500 calls/hour).")
        print("Data loaded so far will be saved locally. Wait a bit or continue training with cached sessions.\n")
        return None
    except Exception as e:
        print(f"Skipping {year} Round {round_identifier}: {e}")
        return None


def build_historical_dataset(seasons=[2023, 2024]):
    """
    Builds or loads cached CSV to avoid redundant API hits.
    """
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'historical_races.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Use saved CSV dataset if available
    if os.path.exists(csv_path):
        print(f"Loading dataset from local cache: {csv_path}")
        return pd.read_csv(csv_path)

    dataset = []
    limit_hit = False

    for year in seasons:
        schedule = fastf1.get_event_schedule(year)
        for _, row in schedule.iterrows():
            rnd = row['RoundNumber']
            if rnd == 0:
                continue
            
            df = extract_weekend_features(year, rnd)
            if df is not None:
                df['EventName'] = str(row.get('EventName', ''))
                dataset.append(df)
            else:
                # Check if rate limit was hit
                limit_hit = True

    if not dataset:
        raise ValueError("No historical race data retrieved. Wait for rate limit reset (~30-60 mins).")

    full_df = pd.concat(dataset, ignore_index=True)

    # ── Sort chronologically ──
    full_df = full_df.sort_values(['Year', 'RoundNumber']).reset_index(drop=True)

    # ── 1. Rolling season avg finish (no leakage) ──
    full_df['Driver_Season_AvgFinish'] = (
        full_df.groupby(['Year', 'Abbreviation'])['Target_Race_Position']
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    full_df['Driver_Season_AvgFinish'] = (
        full_df['Driver_Season_AvgFinish']
        .fillna(full_df['GridPosition'])
        .fillna(full_df['Quali_Position'])
        .fillna(11.0)
    )
    full_df['Team_Season_AvgFinish'] = (
        full_df.groupby(['Year', 'TeamName'])['Target_Race_Position']
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    full_df['Team_Season_AvgFinish'] = (
        full_df['Team_Season_AvgFinish']
        .fillna(full_df['GridPosition'])
        .fillna(full_df['Quali_Position'])
        .fillna(11.0)
    )

    # ── 2. Driver championship momentum: cumulative points before each race ──
    full_df['Race_Points'] = pd.to_numeric(full_df.get('Race_Points', 0), errors='coerce').fillna(0)
    full_df['Driver_Cumulative_Points'] = (
        full_df.groupby(['Year', 'Abbreviation'])['Race_Points']
        .transform(lambda x: x.shift(1).expanding().sum())
    ).fillna(0)

    # ── 3. Singapore-specific historical performance ──
    sing_mask = full_df.get('EventName', pd.Series(dtype=str)).str.contains('Singapore', case=False, na=False)
    driver_sing_avg = full_df[sing_mask].groupby('Abbreviation')['Target_Race_Position'].mean()
    team_sing_avg   = full_df[sing_mask].groupby('TeamName')['Target_Race_Position'].mean()
    full_df['Driver_Singapore_AvgFinish'] = (
        full_df['Abbreviation'].map(driver_sing_avg)
        .fillna(full_df['Driver_Season_AvgFinish'])
        .fillna(11.0)
    )
    full_df['Team_Singapore_AvgFinish'] = (
        full_df['TeamName'].map(team_sing_avg)
        .fillna(full_df['Team_Season_AvgFinish'])
        .fillna(11.0)
    )

    full_df.to_csv(csv_path, index=False)
    print(f"Saved {len(full_df)} rows to {csv_path}")

    return full_df


def extract_prediction_features(year: int, round_identifier):
    """
    Extracts qualifying + sprint features for a FUTURE race where race results
    are not yet available. GridPosition defaults to Quali_Position.
    """
    try:
        time.sleep(1.0)

        # Load qualifying
        q_session = fastf1.get_session(year, round_identifier, 'Q')
        q_session.load(telemetry=False, weather=False)

        q_df = q_session.results[['DriverNumber', 'Abbreviation', 'TeamName', 'Position', 'Q3', 'Q2', 'Q1']].copy()
        q_df['BestLap'] = q_df['Q3'].fillna(q_df['Q2']).fillna(q_df['Q1'])
        q_df['BestLapSec'] = q_df['BestLap'].dt.total_seconds()
        min_q = q_df['BestLapSec'].min()
        q_df['Quali_Time_Delta'] = q_df['BestLapSec'] - min_q
        q_df.rename(columns={'Position': 'Quali_Position'}, inplace=True)

        # Grid position = qualifying position (race hasn't happened yet)
        q_df['GridPosition'] = q_df['Quali_Position']

        # Try sprint session
        has_sprint = True
        try:
            s_session = fastf1.get_session(year, round_identifier, 'S')
            s_session.load(telemetry=False, weather=False)
            s_df = s_session.results[['DriverNumber', 'Position', 'GridPosition']].copy()
            s_df.rename(columns={'Position': 'Sprint_Position', 'GridPosition': 'Sprint_Grid'}, inplace=True)
            s_df['Sprint_Pos_Gained'] = s_df['Sprint_Grid'] - s_df['Sprint_Position']
        except Exception:
            has_sprint = False

        df = q_df[['DriverNumber', 'Abbreviation', 'TeamName', 'Quali_Position', 'Quali_Time_Delta', 'GridPosition']].copy()

        if has_sprint:
            df = pd.merge(df, s_df[['DriverNumber', 'Sprint_Position', 'Sprint_Pos_Gained']], on='DriverNumber', how='left')
        else:
            df['Sprint_Position'] = df['Quali_Position']
            df['Sprint_Pos_Gained'] = 0

        df['Sprint_Position'] = df['Sprint_Position'].fillna(20)
        df['Sprint_Pos_Gained'] = df['Sprint_Pos_Gained'].fillna(0)
        max_delta = df['Quali_Time_Delta'].max()
        df['Quali_Time_Delta'] = df['Quali_Time_Delta'].fillna(max_delta + 1.0 if pd.notnull(max_delta) else 2.0)

        # ── Weather from qualifying (Singapore sprint weekend = same conditions) ──
        try:
            w = q_session.weather_data
            df['Air_Temp'] = round(w['AirTemp'].mean(), 1) if not w.empty else 30.0
            df['Humidity'] = round(w['Humidity'].mean(), 1) if not w.empty else 75.0
        except Exception:
            df['Air_Temp'] = 30.0
            df['Humidity'] = 75.0

        # ── Look up all enriched features from historical data ──
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'historical_races.csv')
        if os.path.exists(csv_path):
            hist_df = pd.read_csv(csv_path)
            # Most recent row per driver = best proxy for current form
            latest = (
                hist_df.sort_values(['Year', 'RoundNumber'])
                .groupby('Abbreviation').last()
                [['Driver_Season_AvgFinish', 'Team_Season_AvgFinish',
                  'Driver_Cumulative_Points', 'Driver_Singapore_AvgFinish',
                  'Team_Singapore_AvgFinish']]
                .reset_index()
            )
            df = pd.merge(df, latest, on='Abbreviation', how='left')
        else:
            df['Driver_Season_AvgFinish']    = df['GridPosition']
            df['Team_Season_AvgFinish']      = df['GridPosition']
            df['Driver_Cumulative_Points']   = 0
            df['Driver_Singapore_AvgFinish'] = 11.0
            df['Team_Singapore_AvgFinish']   = 11.0

        # Fill any remaining NaN (new/unknown drivers)
        df['Driver_Season_AvgFinish']    = df['Driver_Season_AvgFinish'].fillna(df['GridPosition'])
        df['Team_Season_AvgFinish']      = df['Team_Season_AvgFinish'].fillna(df['GridPosition'])
        df['Driver_Cumulative_Points']   = df['Driver_Cumulative_Points'].fillna(0)
        df['Driver_Singapore_AvgFinish'] = df['Driver_Singapore_AvgFinish'].fillna(df['GridPosition'])
        df['Team_Singapore_AvgFinish']   = df['Team_Singapore_AvgFinish'].fillna(df['GridPosition'])

        return df

    except RateLimitExceededError:
        print("\n⚠️ API rate limit reached (500 calls/hour). Try again later.\n")
        return None
    except Exception as e:
        print(f"Error extracting prediction features for {year} {round_identifier}: {e}")
        return None