import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_traffic_data(days=30, roads=10, save_path=None):
    """
    Generate synthetic traffic data for a specified number of days and roads.
    
    Parameters:
    -----------
    days : int
        Number of days to generate data for
    roads : int
        Number of roads to simulate
    save_path : str, optional
        Path to save the generated data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the synthetic traffic data
    """
    print(f"Generating traffic data for {days} days across {roads} roads...")
    
    # Time features
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=h) for h in range(24*days)]
    
    data = []
    for road_id in range(1, roads+1):
        # Define road characteristics
        road_capacity = 150 * (0.8 + 0.4 * road_id/roads)
        road_importance = np.random.randint(1, 5)  # 1-4 scale of importance
        road_lanes = np.random.choice([1, 2, 3, 4])
        road_type = np.random.choice(['residential', 'arterial', 'highway'])
        
        for date in dates:
            # Base patterns
            hour = date.hour
            weekday = date.weekday()
            
            # Rush hour patterns (7-9AM, 4-6PM on weekdays)
            is_rush_hour = (weekday < 5 and ((7 <= hour <= 9) or (16 <= hour <= 18)))
            
            # Base volume with daily/weekly patterns
            volume_base = 100 + 50 * np.sin(hour/24 * 2 * np.pi) 
            volume_base *= 0.7 + 0.3 * np.cos(weekday/7 * 2 * np.pi)
            
            # Add rush hour effect
            if is_rush_hour:
                volume_base *= 1.5
                
            # Random variation
            volume = int(volume_base * (0.9 + 0.2 * np.random.random()))
            
            # Speed depends on volume relative to capacity and road type
            speed_cap = 30 if road_type == 'residential' else (60 if road_type == 'arterial' else 100)
            congestion = min(1.0, volume / road_capacity)
            speed = max(5, speed_cap * (1 - 0.8 * congestion) + 5 * np.random.randn())
            
            # Weather effects
            precipitation = np.random.exponential(0.5) if np.random.random() < 0.3 else 0
            if precipitation > 0:
                speed *= max(0.7, 1 - 0.1 * precipitation)
            
            # Incident probability increases with congestion
            incident_prob = 0.01 + 0.08 * congestion
            incident = 1 if np.random.random() < incident_prob else 0
            
            if incident:
                speed *= 0.5 + 0.3 * np.random.random()  # Significant speed reduction
            
            data.append({
                'timestamp': date,
                'road_id': road_id,
                'hour': hour,
                'day_of_week': weekday,
                'is_weekend': 1 if weekday >= 5 else 0,
                'is_rush_hour': 1 if is_rush_hour else 0,
                'road_type': road_type,
                'road_lanes': road_lanes,
                'road_importance': road_importance,
                'volume': volume,
                'speed': speed,
                'congestion': congestion,
                'precipitation': precipitation,
                'incident': incident
            })
    
    # Convert to DataFrame
    traffic_df = pd.DataFrame(data)
    
    # Sort by timestamp and road_id
    traffic_df = traffic_df.sort_values(['timestamp', 'road_id']).reset_index(drop=True)
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        traffic_df.to_csv(save_path, index=False)
        print(f"Data saved to {save_path}")
    
    return traffic_df

def add_spatial_relationships(traffic_df, num_roads, save_path=None):
    """
    Add spatial relationship data between roads.
    
    Parameters:
    -----------
    traffic_df : pandas.DataFrame
        Traffic data DataFrame
    num_roads : int
        Number of roads in the dataset
    save_path : str, optional
        Path to save the road network data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing road connections
    """
    # Create road connection matrix
    connections = []
    
    # Simple linear network with some branching
    for i in range(1, num_roads):
        # Connect to previous road
        connections.append({
            'source_road': i,
            'target_road': i + 1,
            'distance': np.random.uniform(0.5, 2.0),
            'connection_type': 'sequential'
        })
        
        # Add some random connections for more complex network
        if np.random.random() < 0.3:
            target = np.random.randint(1, num_roads + 1)
            if target != i and target != i + 1:
                connections.append({
                    'source_road': i,
                    'target_road': target,
                    'distance': np.random.uniform(1.0, 5.0),
                    'connection_type': 'intersection'
                })
    
    # Convert to DataFrame
    network_df = pd.DataFrame(connections)
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        network_df.to_csv(save_path, index=False)
        print(f"Network data saved to {save_path}")
    
    return network_df

if __name__ == "__main__":
    # Generate data for 30 days across 20 roads
    traffic_data = generate_traffic_data(
        days=30, 
        roads=20, 
        save_path='../data/synthetic_traffic_data.csv'
    )
    
    # Add spatial relationships
    network_data = add_spatial_relationships(
        traffic_data, 
        num_roads=20,
        save_path='../data/road_network_data.csv'
    )
    
    print(f"Generated {len(traffic_data)} traffic data points")
    print(f"Generated {len(network_data)} road connections")
    print("\nSample traffic data:")
    print(traffic_data.head())
    print("\nSample network data:")
    print(network_data.head()) 