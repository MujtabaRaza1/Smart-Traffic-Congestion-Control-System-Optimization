#!/usr/bin/env python3
"""
Traffic Data Visualizer

This script provides visualizations for the traffic data,
including time-based patterns, congestion analysis, and feature relationships.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import MinMaxScaler

# Add the project directory to the path
sys.path.append(str(Path(__file__).parent.parent))

def load_traffic_data():
    """Load the traffic data."""
    data_path = Path(__file__).parent.parent / "data" / "synthetic_traffic_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Traffic data file not found at {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df

def visualize_traffic_patterns(df, output_dir=None):
    """
    Create visualizations of traffic patterns.
    
    Args:
        df: DataFrame containing traffic data
        output_dir: Directory to save visualizations
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set up the style
    sns.set(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16,
        'figure.figsize': (12, 8)
    })
    
    # Custom colormap for traffic congestion (green to red)
    colors = [(0, 0.6, 0), (0.8, 0.8, 0), (0.8, 0, 0)]  # green, yellow, red
    traffic_cmap = LinearSegmentedColormap.from_list('traffic_cmap', colors, N=100)
    
    # 1. Daily Pattern - Average congestion by hour of day
    print("Creating daily traffic pattern visualization...")
    plt.figure(figsize=(14, 8))
    
    hourly_congestion = df.groupby(['hour', 'road_type'])['congestion'].mean().reset_index()
    
    sns.lineplot(
        data=hourly_congestion,
        x='hour',
        y='congestion',
        hue='road_type',
        marker='o',
        linewidth=2.5,
        markersize=8
    )
    
    plt.title('Average Traffic Congestion by Hour of Day', fontsize=16, pad=20)
    plt.xlabel('Hour of Day', fontsize=14)
    plt.ylabel('Average Congestion Level', fontsize=14)
    plt.xticks(range(0, 24))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(Path(output_dir) / 'daily_congestion_pattern.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Weekly Pattern - Average congestion by day of week
    print("Creating weekly traffic pattern visualization...")
    plt.figure(figsize=(14, 8))
    
    # Map day of week to names
    day_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                   4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    
    weekly_data = df.copy()
    weekly_data['day_name'] = weekly_data['day_of_week'].map(day_mapping)
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_congestion = weekly_data.groupby(['day_name', 'road_type'])['congestion'].mean().reset_index()
    
    sns.barplot(
        data=weekly_congestion,
        x='day_name',
        y='congestion',
        hue='road_type',
        order=day_order,
        palette='viridis'
    )
    
    plt.title('Average Traffic Congestion by Day of Week', fontsize=16, pad=20)
    plt.xlabel('Day of Week', fontsize=14)
    plt.ylabel('Average Congestion Level', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(Path(output_dir) / 'weekly_congestion_pattern.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Rush Hour vs. Non-Rush Hour Comparison
    print("Creating rush hour comparison visualization...")
    plt.figure(figsize=(14, 8))
    
    rush_hour_comparison = df.groupby(['is_rush_hour', 'road_type'])[['congestion', 'speed', 'volume']].mean().reset_index()
    rush_hour_comparison['is_rush_hour'] = rush_hour_comparison['is_rush_hour'].map({0: 'Non-Rush Hour', 1: 'Rush Hour'})
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['congestion', 'speed', 'volume']
    titles = ['Congestion', 'Speed', 'Volume']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        sns.barplot(
            data=rush_hour_comparison,
            x='road_type',
            y=metric,
            hue='is_rush_hour',
            ax=axes[i],
            palette=['#5cb85c', '#d9534f']
        )
        
        axes[i].set_title(f'Average {title} - Rush Hour vs. Non-Rush Hour', fontsize=14)
        axes[i].set_xlabel('Road Type', fontsize=12)
        axes[i].set_ylabel(f'Average {title}', fontsize=12)
        axes[i].grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(Path(output_dir) / 'rush_hour_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Congestion Heatmap by Hour and Day
    print("Creating congestion heatmap visualization...")
    
    # Prepare data for heatmap
    heatmap_data = df.copy()
    heatmap_data['day_name'] = heatmap_data['day_of_week'].map(day_mapping)
    
    pivot_data = heatmap_data.pivot_table(
        values='congestion',
        index='hour',
        columns='day_name',
        aggfunc='mean'
    )
    
    # Reorder columns for correct day order
    pivot_data = pivot_data[day_order]
    
    plt.figure(figsize=(14, 10))
    
    ax = sns.heatmap(
        pivot_data,
        cmap=traffic_cmap,
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={'label': 'Average Congestion Level'}
    )
    
    plt.title('Traffic Congestion Heatmap by Hour and Day', fontsize=16, pad=20)
    plt.ylabel('Hour of Day', fontsize=14)
    plt.xlabel('Day of Week', fontsize=14)
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(Path(output_dir) / 'congestion_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Correlation Matrix
    print("Creating feature correlation visualization...")
    
    # Select numeric columns for correlation
    numeric_cols = ['hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 
                    'road_lanes', 'road_importance', 'volume', 'speed', 
                    'congestion', 'precipitation', 'incident']
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        mask=mask,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation Coefficient'}
    )
    
    plt.title('Correlation Matrix of Traffic Features', fontsize=16, pad=20)
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(Path(output_dir) / 'feature_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Volume vs Congestion Scatter Plot
    print("Creating volume vs congestion visualization...")
    plt.figure(figsize=(14, 8))
    
    scatter_data = df.sample(min(1000, len(df)))  # Sample to avoid overcrowding
    
    sns.scatterplot(
        data=scatter_data,
        x='volume',
        y='congestion',
        hue='road_type',
        size='road_lanes',
        sizes=(50, 200),
        alpha=0.7
    )
    
    plt.title('Relationship Between Traffic Volume and Congestion', fontsize=16, pad=20)
    plt.xlabel('Traffic Volume', fontsize=14)
    plt.ylabel('Congestion Level', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(Path(output_dir) / 'volume_congestion_relationship.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Road Type Comparison
    print("Creating road type comparison visualization...")
    plt.figure(figsize=(16, 12))
    
    road_type_metrics = df.groupby('road_type')[
        ['congestion', 'speed', 'volume', 'incident']
    ].agg(['mean', 'max']).reset_index()
    
    road_type_metrics.columns = [
        '_'.join(col).strip() if col[1] else col[0] for col in road_type_metrics.columns.values
    ]
    
    # Melt the dataframe for easier plotting
    melted_data = pd.melt(
        road_type_metrics,
        id_vars=['road_type'],
        value_vars=[c for c in road_type_metrics.columns if c != 'road_type']
    )
    
    # Split the variable column into metric and statistic
    melted_data[['metric', 'statistic']] = melted_data['variable'].str.split('_', expand=True)
    
    # Create subplot for each metric
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    metrics = ['congestion', 'speed', 'volume', 'incident']
    titles = ['Congestion Levels', 'Speed (km/h)', 'Traffic Volume', 'Incident Rate']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        metric_data = melted_data[melted_data['metric'] == metric]
        
        sns.barplot(
            data=metric_data,
            x='road_type',
            y='value',
            hue='statistic',
            ax=axes[i],
            palette=['#3498db', '#e74c3c']
        )
        
        axes[i].set_title(f'{title} by Road Type', fontsize=14)
        axes[i].set_xlabel('Road Type', fontsize=12)
        axes[i].set_ylabel(title, fontsize=12)
        axes[i].grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(Path(output_dir) / 'road_type_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Combined Time Series Visualization
    print("Creating time series visualization...")
    
    # Sample data for a specific road to show time series
    sample_road_id = 1
    sample_data = df[df['road_id'] == sample_road_id].copy()
    sample_data = sample_data.sort_values('timestamp')
    
    # Limit to a week of data to avoid overcrowding
    one_week = pd.Timedelta(days=7)
    start_date = sample_data['timestamp'].min()
    end_date = start_date + one_week
    
    sample_data = sample_data[
        (sample_data['timestamp'] >= start_date) & 
        (sample_data['timestamp'] <= end_date)
    ]
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # Plot congestion
    ax1.plot(sample_data['timestamp'], sample_data['congestion'], 
             color='red', linewidth=2, label='Congestion')
    ax1.set_ylabel('Congestion Level', fontsize=12)
    ax1.set_title(f'Traffic Metrics Over Time (Road ID: {sample_road_id}, Road Type: {sample_data["road_type"].iloc[0]})',
                 fontsize=16, pad=20)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Plot speed
    ax2.plot(sample_data['timestamp'], sample_data['speed'], 
             color='blue', linewidth=2, label='Speed')
    ax2.set_ylabel('Speed (km/h)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Plot volume
    ax3.plot(sample_data['timestamp'], sample_data['volume'], 
             color='green', linewidth=2, label='Volume')
    ax3.set_ylabel('Traffic Volume', fontsize=12)
    ax3.set_xlabel('Time', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend()
    
    # Format x-axis ticks
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()
    
    # Mark incidents with red dots
    incidents = sample_data[sample_data['incident'] == 1]
    if not incidents.empty:
        ax1.scatter(incidents['timestamp'], incidents['congestion'], 
                   color='red', s=100, marker='*', label='Incident')
        ax1.legend()
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(Path(output_dir) / 'time_series_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("All visualizations created successfully!")

def main():
    """Main function to create traffic visualizations."""
    try:
        # Load the traffic data
        print("Loading traffic data...")
        traffic_df = load_traffic_data()
        
        # Set output directory
        output_dir = Path(__file__).parent.parent / "visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualizations
        visualize_traffic_patterns(traffic_df, output_dir)
        
        print(f"Visualizations saved to {output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 