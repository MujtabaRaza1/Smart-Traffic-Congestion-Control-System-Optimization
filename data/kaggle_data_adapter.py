#!/usr/bin/env python3
"""
Kaggle Traffic Data Adapter

This script adapts the Kaggle xtraffic dataset to our format
and saves it as synthetic_traffic_data.csv.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def fetch_kaggle_data(file_path=None):
    """
    Fetch the Kaggle traffic dataset.
    
    Args:
        file_path: Specific file path within the dataset
        
    Returns:
        DataFrame: The loaded dataset
    """
    try:
        # If no file_path specified, try to load the main files
        if not file_path:
            # Try common filenames for traffic datasets
            possible_files = [
                "traffic_data.csv", "traffic.csv", "traffic_dataset.csv", 
                "xtraffic.csv", "traffic_volume.csv", "Traffic.csv"
            ]
            
            # Try each possible file
            for file in possible_files:
                try:
                    print(f"Trying to load {file}...")
                    df = kagglehub.load_dataset(
                        KaggleDatasetAdapter.PANDAS,
                        "gpxlcj/xtraffic",
                        file
                    )
                    print(f"Successfully loaded {file}")
                    return df
                except Exception as e:
                    print(f"Failed to load {file}: {str(e)}")
            
            # If no specific file works, try loading without a filename
            print("Trying to load dataset without specific filename...")
            df = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                "gpxlcj/xtraffic"
            )
            return df
        else:
            # Load the specified file
            df = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                "gpxlcj/xtraffic",
                file_path
            )
            return df
    except Exception as e:
        print(f"Error loading Kaggle dataset: {str(e)}")
        raise

def adapt_kaggle_data(df):
    """
    Adapt the Kaggle dataset to match our expected format.
    
    Args:
        df: The raw Kaggle dataset
        
    Returns:
        DataFrame: The adapted dataset
    """
    print("Original dataset columns:", df.columns.tolist())
    print("Original dataset sample:\n", df.head())
    
    # Create a copy to avoid modifying the original
    adapted_df = df.copy()
    
    # Adapting columns based on likely column names in the dataset
    # This needs to be adjusted based on the actual columns in the Kaggle dataset
    column_mapping = {}
    
    # Identify time-related columns
    time_cols = [col for col in df.columns if any(term in col.lower() for term in ['time', 'date', 'hour', 'day'])]
    if time_cols:
        column_mapping[time_cols[0]] = 'timestamp'
    
    # Identify road/location identifiers
    road_cols = [col for col in df.columns if any(term in col.lower() for term in ['road', 'street', 'location', 'junction', 'id'])]
    if road_cols:
        column_mapping[road_cols[0]] = 'road_id'
    
    # Identify volume/count columns
    volume_cols = [col for col in df.columns if any(term in col.lower() for term in ['volume', 'count', 'traffic', 'vehicles'])]
    if volume_cols:
        column_mapping[volume_cols[0]] = 'volume'
    
    # Identify speed columns
    speed_cols = [col for col in df.columns if any(term in col.lower() for term in ['speed', 'velocity'])]
    if speed_cols:
        column_mapping[speed_cols[0]] = 'speed'
    
    # Rename columns based on mapping
    if column_mapping:
        adapted_df = adapted_df.rename(columns=column_mapping)
        print("Renamed columns using mapping:", column_mapping)
    
    # Check required columns and add missing ones with generated data
    required_columns = [
        'timestamp', 'road_id', 'hour', 'day_of_week', 'is_weekend',
        'is_rush_hour', 'road_type', 'road_lanes', 'road_importance',
        'volume', 'speed', 'congestion', 'precipitation', 'incident'
    ]
    
    # Add missing columns
    for col in required_columns:
        if col not in adapted_df.columns:
            print(f"Adding missing column: {col}")
            
            if col == 'timestamp' and 'timestamp' not in adapted_df.columns:
                # Use an existing time column if available or generate timestamps
                if time_cols:
                    try:
                        adapted_df['timestamp'] = pd.to_datetime(df[time_cols[0]])
                    except:
                        # Generate dates for the last 30 days, hourly
                        end_date = pd.Timestamp.now().normalize()
                        start_date = end_date - pd.Timedelta(days=30)
                        timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
                        
                        # Repeat timestamps for each road
                        n_roads = len(adapted_df['road_id'].unique()) if 'road_id' in adapted_df.columns else 20
                        all_timestamps = np.tile(timestamps, n_roads)
                        adapted_df['timestamp'] = np.resize(all_timestamps, len(adapted_df))
                else:
                    # Generate dates for the last 30 days, hourly
                    end_date = pd.Timestamp.now().normalize()
                    start_date = end_date - pd.Timedelta(days=30)
                    timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
                    
                    # Repeat timestamps for each road
                    n_roads = len(adapted_df['road_id'].unique()) if 'road_id' in adapted_df.columns else 20
                    all_timestamps = np.tile(timestamps, n_roads)
                    adapted_df['timestamp'] = np.resize(all_timestamps, len(adapted_df))
            
            elif col == 'road_id' and 'road_id' not in adapted_df.columns:
                # Generate road IDs (1 to 20)
                n_timestamps = len(adapted_df['timestamp'].unique()) if 'timestamp' in adapted_df.columns else 720
                road_ids = np.repeat(np.arange(1, 21), n_timestamps)
                adapted_df['road_id'] = np.resize(road_ids, len(adapted_df))
            
            elif col == 'hour':
                if 'timestamp' in adapted_df.columns:
                    adapted_df['hour'] = pd.to_datetime(adapted_df['timestamp']).dt.hour
                else:
                    adapted_df['hour'] = np.random.randint(0, 24, len(adapted_df))
            
            elif col == 'day_of_week':
                if 'timestamp' in adapted_df.columns:
                    adapted_df['day_of_week'] = pd.to_datetime(adapted_df['timestamp']).dt.dayofweek
                else:
                    adapted_df['day_of_week'] = np.random.randint(0, 7, len(adapted_df))
            
            elif col == 'is_weekend':
                if 'day_of_week' in adapted_df.columns:
                    adapted_df['is_weekend'] = (adapted_df['day_of_week'] >= 5).astype(int)
                else:
                    adapted_df['is_weekend'] = np.random.randint(0, 2, len(adapted_df))
            
            elif col == 'is_rush_hour':
                if 'hour' in adapted_df.columns:
                    morning_rush = (adapted_df['hour'] >= 7) & (adapted_df['hour'] <= 9)
                    evening_rush = (adapted_df['hour'] >= 16) & (adapted_df['hour'] <= 18)
                    adapted_df['is_rush_hour'] = (morning_rush | evening_rush).astype(int)
                else:
                    adapted_df['is_rush_hour'] = np.random.randint(0, 2, len(adapted_df))
            
            elif col == 'road_type':
                road_types = ['residential', 'arterial', 'highway']
                adapted_df['road_type'] = np.random.choice(road_types, len(adapted_df))
            
            elif col == 'road_lanes':
                adapted_df['road_lanes'] = np.random.randint(1, 5, len(adapted_df))
            
            elif col == 'road_importance':
                adapted_df['road_importance'] = np.random.randint(1, 5, len(adapted_df))
            
            elif col == 'volume' and 'volume' not in adapted_df.columns:
                # Generate reasonable traffic volumes (vehicles per hour)
                adapted_df['volume'] = np.random.randint(50, 200, len(adapted_df))
                
                # Adjust volume based on time of day if hour exists
                if 'hour' in adapted_df.columns and 'is_rush_hour' in adapted_df.columns:
                    # Increase volume during rush hours
                    rush_hour_factor = 1.5
                    adapted_df.loc[adapted_df['is_rush_hour'] == 1, 'volume'] *= rush_hour_factor
                    
                    # Decrease volume during night hours (10 PM - 5 AM)
                    night_hours = (adapted_df['hour'] >= 22) | (adapted_df['hour'] <= 5)
                    night_factor = 0.5
                    adapted_df.loc[night_hours, 'volume'] *= night_factor
            
            elif col == 'speed' and 'speed' not in adapted_df.columns:
                # Generate reasonable speeds (km/h or mph)
                # Base speeds by road type
                if 'road_type' in adapted_df.columns:
                    adapted_df['speed'] = 30  # Default
                    adapted_df.loc[adapted_df['road_type'] == 'residential', 'speed'] = np.random.uniform(15, 30, sum(adapted_df['road_type'] == 'residential'))
                    adapted_df.loc[adapted_df['road_type'] == 'arterial', 'speed'] = np.random.uniform(25, 45, sum(adapted_df['road_type'] == 'arterial'))
                    adapted_df.loc[adapted_df['road_type'] == 'highway', 'speed'] = np.random.uniform(45, 70, sum(adapted_df['road_type'] == 'highway'))
                else:
                    adapted_df['speed'] = np.random.uniform(15, 70, len(adapted_df))
                
                # Adjust speed based on volume (higher volume = lower speed)
                if 'volume' in adapted_df.columns:
                    # Normalize volume to 0-1 range for adjustment
                    vol_min = adapted_df['volume'].min()
                    vol_max = adapted_df['volume'].max()
                    vol_range = vol_max - vol_min
                    
                    if vol_range > 0:
                        norm_vol = (adapted_df['volume'] - vol_min) / vol_range
                        speed_factor = 1 - (norm_vol * 0.7)  # Higher volume reduces speed by up to 70%
                        adapted_df['speed'] = adapted_df['speed'] * speed_factor
            
            elif col == 'congestion':
                # Calculate congestion as a function of volume and speed if available
                if 'volume' in adapted_df.columns and 'speed' in adapted_df.columns:
                    # Normalize volume and speed
                    vol_min = adapted_df['volume'].min()
                    vol_max = adapted_df['volume'].max()
                    speed_min = adapted_df['speed'].min()
                    speed_max = adapted_df['speed'].max()
                    
                    # Avoid division by zero
                    vol_range = vol_max - vol_min
                    speed_range = speed_max - speed_min
                    
                    if vol_range > 0 and speed_range > 0:
                        norm_vol = (adapted_df['volume'] - vol_min) / vol_range
                        norm_speed = (adapted_df['speed'] - speed_min) / speed_range
                        
                        # Congestion increases with volume and decreases with speed
                        adapted_df['congestion'] = (0.7 * norm_vol + 0.3 * (1 - norm_speed))
                        
                        # Constrain to 0-1 range
                        adapted_df['congestion'] = adapted_df['congestion'].clip(0, 1)
                    else:
                        adapted_df['congestion'] = np.random.uniform(0, 1, len(adapted_df))
                else:
                    adapted_df['congestion'] = np.random.uniform(0, 1, len(adapted_df))
            
            elif col == 'precipitation':
                # Random precipitation values (mm of rain)
                adapted_df['precipitation'] = np.random.uniform(0, 2, len(adapted_df))
                # 80% of values should be 0 (no rain)
                adapted_df.loc[np.random.rand(len(adapted_df)) < 0.8, 'precipitation'] = 0
            
            elif col == 'incident':
                # Random incidents (boolean, mostly false)
                adapted_df['incident'] = np.random.choice([0, 1], len(adapted_df), p=[0.95, 0.05])
    
    # Ensure consistent data types
    if 'timestamp' in adapted_df.columns:
        adapted_df['timestamp'] = pd.to_datetime(adapted_df['timestamp'])
    
    if 'road_id' in adapted_df.columns:
        adapted_df['road_id'] = adapted_df['road_id'].astype(int)
    
    print("Adapted dataset columns:", adapted_df.columns.tolist())
    print("Adapted dataset sample:\n", adapted_df.head())
    
    return adapted_df

def generate_network_data(adapted_df):
    """
    Generate road network data based on the adapted traffic data.
    
    Args:
        adapted_df: The adapted traffic dataset
        
    Returns:
        DataFrame: The network dataset
    """
    # Get unique road IDs
    if 'road_id' not in adapted_df.columns:
        raise ValueError("Adapted dataset must contain 'road_id' column")
    
    road_ids = adapted_df['road_id'].unique()
    n_roads = len(road_ids)
    
    # Create network edges (connections between roads)
    # Each road connects to 2-4 other roads
    network_data = []
    
    for source_road in road_ids:
        # Determine number of connections
        n_connections = np.random.randint(2, min(5, n_roads))
        
        # Select target roads (excluding self)
        potential_targets = [r for r in road_ids if r != source_road]
        if len(potential_targets) > 0:
            target_roads = np.random.choice(potential_targets, 
                                           size=min(n_connections, len(potential_targets)), 
                                           replace=False)
            
            # Generate distances between roads
            for target_road in target_roads:
                network_data.append({
                    'source_road': source_road,
                    'target_road': target_road,
                    'distance': np.random.uniform(0.5, 5.0)  # in km
                })
    
    return pd.DataFrame(network_data)

def main():
    """Main function to fetch, adapt and save Kaggle data."""
    print("Fetching Kaggle traffic data...")
    try:
        # Try to fetch the data
        raw_df = fetch_kaggle_data()
        
        # Adapt the data to our format
        print("Adapting data to required format...")
        adapted_df = adapt_kaggle_data(raw_df)
        
        # Generate network data
        print("Generating road network data...")
        network_df = generate_network_data(adapted_df)
        
        # Save the adapted datasets
        data_dir = Path(__file__).parent
        adapted_df.to_csv(data_dir / "synthetic_traffic_data.csv", index=False)
        network_df.to_csv(data_dir / "road_network_data.csv", index=False)
        
        print("Kaggle data saved successfully as synthetic_traffic_data.csv")
        print("Network data saved successfully as road_network_data.csv")
        
    except Exception as e:
        print(f"Error processing Kaggle data: {str(e)}")
        print("Generating realistic traffic data instead...")
        
        # Generate realistic traffic data
        realistic_df = generate_realistic_traffic_data()
        
        # Generate network data
        network_df = generate_network_data(realistic_df)
        
        # Save the datasets
        data_dir = Path(__file__).parent
        realistic_df.to_csv(data_dir / "synthetic_traffic_data.csv", index=False)
        network_df.to_csv(data_dir / "road_network_data.csv", index=False)
        
        print("Realistic traffic data saved successfully as synthetic_traffic_data.csv")
        print("Network data saved successfully as road_network_data.csv")

def generate_realistic_traffic_data():
    """
    Generate realistic traffic data that resembles real-world patterns.
    
    Returns:
        DataFrame: Realistic traffic dataset
    """
    print("Generating realistic traffic data with real-world patterns...")
    
    # Number of days to simulate
    days = 30
    
    # Number of roads
    n_roads = 20
    
    # Create timestamps (hourly for 30 days)
    end_date = pd.Timestamp.now().normalize()
    start_date = end_date - pd.Timedelta(days=days)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
    n_hours = len(timestamps)
    
    # Road types and distributions
    road_types = ['residential', 'arterial', 'highway']
    road_type_probs = [0.5, 0.3, 0.2]  # 50% residential, 30% arterial, 20% highway
    road_types_assigned = np.random.choice(road_types, n_roads, p=road_type_probs)
    
    # Road characteristics
    road_characteristics = []
    for i in range(n_roads):
        road_type = road_types_assigned[i]
        
        if road_type == 'residential':
            lanes = np.random.choice([1, 2], p=[0.3, 0.7])
            importance = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
        elif road_type == 'arterial':
            lanes = np.random.choice([2, 3, 4], p=[0.2, 0.6, 0.2])
            importance = np.random.choice([2, 3, 4], p=[0.2, 0.5, 0.3])
        else:  # highway
            lanes = np.random.choice([2, 3, 4], p=[0.1, 0.3, 0.6])
            importance = np.random.choice([3, 4], p=[0.4, 0.6])
        
        road_characteristics.append({
            'road_id': i+1,
            'road_type': road_type,
            'road_lanes': lanes,
            'road_importance': importance
        })
    
    road_chars_df = pd.DataFrame(road_characteristics)
    
    # Create a list to store all traffic records
    traffic_data = []
    
    # Define time-based patterns
    # 1. Day-of-week patterns
    weekday_factors = {
        0: 1.2,   # Monday: higher than average
        1: 1.1,   # Tuesday: higher than average
        2: 1.1,   # Wednesday: higher than average
        3: 1.1,   # Thursday: higher than average
        4: 1.3,   # Friday: highest weekday
        5: 0.7,   # Saturday: lower
        6: 0.6    # Sunday: lowest
    }
    
    # 2. Hour-of-day patterns
    hour_factors = {
        0: 0.3,  1: 0.2,  2: 0.1,  3: 0.1,  # 12am-4am: very low
        4: 0.2,  5: 0.5,  6: 0.8,  7: 1.5,  # 4am-8am: morning increase
        8: 1.8,  9: 1.3, 10: 1.0, 11: 1.0,  # 8am-12pm: morning peak then decrease
        12: 1.1, 13: 1.0, 14: 1.0, 15: 1.1,  # 12pm-4pm: midday
        16: 1.7, 17: 1.9, 18: 1.6, 19: 1.2,  # 4pm-8pm: evening peak
        20: 0.8, 21: 0.6, 22: 0.5, 23: 0.4   # 8pm-12am: evening decrease
    }
    
    # Generate data for each timestamp and road
    for ts in timestamps:
        hour = ts.hour
        day_of_week = ts.dayofweek
        is_weekend = 1 if day_of_week >= 5 else 0
        is_rush_hour = 1 if hour in [7, 8, 9, 16, 17, 18] else 0
        
        # Weather pattern - precipitation more likely in certain months
        # More precipitation in winter/fall months
        month = ts.month
        is_rainy_season = month in [1, 2, 3, 10, 11, 12]
        precip_prob = 0.3 if is_rainy_season else 0.1
        
        for road_id in range(1, n_roads + 1):
            road_info = road_chars_df[road_chars_df['road_id'] == road_id].iloc[0]
            road_type = road_info['road_type']
            lanes = road_info['road_lanes']
            importance = road_info['road_importance']
            
            # Base volume depends on road type and importance
            if road_type == 'residential':
                base_volume = np.random.randint(30, 70)
            elif road_type == 'arterial':
                base_volume = np.random.randint(70, 150)
            else:  # highway
                base_volume = np.random.randint(150, 300)
            
            # Adjust volume based on time patterns
            time_factor = weekday_factors[day_of_week] * hour_factors[hour]
            volume = int(base_volume * time_factor * (1 + 0.1 * np.random.randn()))
            
            # Ensure volume is positive and reasonable
            volume = max(10, min(500, volume))
            
            # Speed depends on road type, adjusted by congestion
            if road_type == 'residential':
                max_speed = 30
            elif road_type == 'arterial':
                max_speed = 50
            else:  # highway
                max_speed = 70
                
            # Congestion calculated based on volume relative to capacity
            capacity = lanes * (50 if road_type == 'residential' else 100 if road_type == 'arterial' else 200)
            raw_congestion = volume / capacity
            
            # Add randomness to congestion
            congestion = min(1.0, max(0.0, raw_congestion * (1 + 0.1 * np.random.randn())))
            
            # Calculate speed based on congestion (more congestion = lower speed)
            speed = max_speed * (1 - 0.7 * congestion) * (1 + 0.1 * np.random.randn())
            speed = max(5, min(max_speed * 1.1, speed))
            
            # Precipitation (mm of rain)
            precipitation = 0
            if np.random.random() < precip_prob:
                precipitation = np.random.exponential(0.5)  # exponential distribution for rainfall
            
            # Incidents more likely during rush hour, bad weather, and high congestion
            incident_prob = 0.01  # base probability
            if is_rush_hour:
                incident_prob *= 2
            if precipitation > 0:
                incident_prob *= 1.5
            if congestion > 0.8:
                incident_prob *= 2
                
            incident = 1 if np.random.random() < incident_prob else 0
            
            # If there's an incident, increase congestion and decrease speed
            if incident:
                congestion = min(1.0, congestion * 1.5)
                speed = max(5, speed * 0.6)
            
            # Create record
            traffic_data.append({
                'timestamp': ts,
                'road_id': road_id,
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'is_rush_hour': is_rush_hour,
                'road_type': road_type,
                'road_lanes': lanes,
                'road_importance': importance,
                'volume': volume,
                'speed': speed,
                'congestion': congestion,
                'precipitation': precipitation,
                'incident': incident
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(traffic_data)
    
    print(f"Generated {len(df)} traffic data points for {n_roads} roads over {days} days")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Sample data:\n{df.head()}")
    
    return df

if __name__ == "__main__":
    main() 