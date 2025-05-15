#!/usr/bin/env python3
"""
Smart Traffic Congestion Control System

This script orchestrates the complete pipeline:
1. Data generation/loading
2. Data preprocessing
3. Model training
4. Knowledge graph construction
5. Traffic optimization
6. Visualization
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import project modules
from data.synthetic_data_generator import generate_traffic_data, add_spatial_relationships
from preprocessing.data_processor import TrafficDataProcessor
from models.congestion_predictor import CongestionPredictor
from optimization.traffic_optimizer import TrafficKnowledgeGraph, DEFAULT_RULES

def create_directories():
    """Create necessary directories for outputs."""
    dirs = [
        Path('data'),
        Path('models/saved_models'),
        Path('models/outputs'),
        Path('optimization/saved_graphs'),
        Path('optimization/outputs'),
        Path('visualization/outputs')
    ]
    
    for directory in dirs:
        os.makedirs(Path(__file__).parent / directory, exist_ok=True)
        print(f"Created directory: {directory}")

def generate_data(args):
    """Generate synthetic data if requested."""
    if args.generate_data:
        print("\n--- Generating Synthetic Traffic Data ---")
        data_path = Path(__file__).parent / "data" / "synthetic_traffic_data.csv"
        network_path = Path(__file__).parent / "data" / "road_network_data.csv"
        
        # Generate traffic data
        traffic_data = generate_traffic_data(
            days=args.days,
            roads=args.roads,
            save_path=str(data_path)
        )
        
        # Generate network data
        network_data = add_spatial_relationships(
            traffic_data,
            num_roads=args.roads,
            save_path=str(network_path)
        )
        
        print(f"Generated {len(traffic_data)} traffic data points for {args.roads} roads over {args.days} days")
        return traffic_data, network_data
    else:
        # Try to load existing data
        data_path = Path(__file__).parent / "data" / "synthetic_traffic_data.csv"
        network_path = Path(__file__).parent / "data" / "road_network_data.csv"
        
        if data_path.exists() and network_path.exists():
            print("\n--- Loading Existing Traffic Data ---")
            traffic_data = pd.read_csv(data_path)
            network_data = pd.read_csv(network_path)
            print(f"Loaded {len(traffic_data)} traffic data points")
            return traffic_data, network_data
        else:
            print("\n--- No Existing Data Found, Generating Synthetic Data ---")
            return generate_data(argparse.Namespace(generate_data=True, days=30, roads=20))

def preprocess_data(traffic_data):
    """Preprocess the traffic data."""
    print("\n--- Preprocessing Traffic Data ---")
    
    processor = TrafficDataProcessor()
    processed_data = processor.fit_transform(
        traffic_data,
        save_path=str(Path(__file__).parent / "data" / "processed_traffic_data.csv")
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = processor.train_test_split(processed_data)
    
    print(f"Preprocessed data shape: {processed_data.shape}")
    print(f"Training features shape: {X_train.shape}")
    print(f"Testing features shape: {X_test.shape}")
    
    return processed_data, processor, (X_train, X_test, y_train, y_test)

def train_models(data_splits):
    """Train the prediction models."""
    print("\n--- Training Prediction Models ---")
    
    X_train, X_test, y_train, y_test = data_splits
    
    # Train congestion model
    congestion_model = CongestionPredictor(model_type='random_forest')
    congestion_model.train(X_train, y_train, target='congestion')
    
    # Evaluate model
    congestion_metrics = congestion_model.evaluate(X_test, y_test, target='congestion')
    
    # Plot feature importance
    congestion_model.plot_feature_importance(
        save_path=str(Path(__file__).parent / "models" / "outputs" / "congestion_feature_importance.png"),
        top_n=15
    )
    
    # Save model
    model_path = Path(__file__).parent / "models" / "saved_models" / "congestion_rf_model.pkl"
    congestion_model.save_model(str(model_path))
    
    print(f"Model saved to {model_path}")
    
    # Optionally train models for speed and incident prediction
    if 'speed' in y_train.columns:
        speed_model = CongestionPredictor(model_type='gradient_boosting')
        speed_model.train(X_train, y_train, target='speed')
        speed_model.evaluate(X_test, y_test, target='speed')
        speed_model.save_model(str(Path(__file__).parent / "models" / "saved_models" / "speed_gb_model.pkl"))
    
    if 'incident' in y_train.columns:
        # For incident, we'd typically use a classifier, but for simplicity we're using the same interface
        incident_model = CongestionPredictor(model_type='random_forest')
        incident_model.train(X_train, y_train, target='incident')
        incident_model.evaluate(X_test, y_test, target='incident')
        incident_model.save_model(str(Path(__file__).parent / "models" / "saved_models" / "incident_rf_model.pkl"))
    
    return congestion_model

def build_knowledge_graph(traffic_data, network_data):
    """Build the traffic knowledge graph."""
    print("\n--- Building Traffic Knowledge Graph ---")
    
    # Convert timestamp to datetime if needed
    if 'timestamp' in traffic_data.columns and not pd.api.types.is_datetime64_any_dtype(traffic_data['timestamp']):
        traffic_data['timestamp'] = pd.to_datetime(traffic_data['timestamp'])
    
    # Get the latest timestamp
    latest_timestamp = traffic_data['timestamp'].max()
    
    # Create knowledge graph
    kg = TrafficKnowledgeGraph()
    
    # Load road network
    network_path = Path(__file__).parent / "data" / "road_network_data.csv"
    kg.load_network_data(str(network_path))
    
    # Add traffic data
    kg.add_traffic_data(traffic_data, timestamp=latest_timestamp)
    
    # Add default rules
    kg.add_rules(DEFAULT_RULES)
    
    # Get congestion summary
    summary = kg.get_congestion_summary()
    print("\nCongestion Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Save the knowledge graph
    kg_path = Path(__file__).parent / "optimization" / "saved_graphs" / "traffic_knowledge_graph.pkl"
    kg.save(str(kg_path))
    
    return kg

def optimize_traffic(kg):
    """Generate traffic optimization recommendations."""
    print("\n--- Generating Traffic Optimization Recommendations ---")
    
    # Visualize the network
    kg.visualize_network(
        save_path=str(Path(__file__).parent / "optimization" / "outputs" / "traffic_network.png")
    )
    
    # Get optimization recommendations
    recommendations = kg.optimize_traffic_flow()
    
    print(f"\nGenerated {len(recommendations)} traffic optimization recommendations")
    
    # Print sample recommendations
    sample_count = min(5, len(recommendations))
    if sample_count > 0:
        print("\nSample Recommendations:")
        for i, (road, rec) in enumerate(list(recommendations.items())[:sample_count]):
            print(f"  Road {road}: {rec['action']} (Congestion: {rec['congestion']:.2f})")
    
    return recommendations

def main():
    """Main function to run the traffic congestion control system."""
    parser = argparse.ArgumentParser(description='Smart Traffic Congestion Control System')
    parser.add_argument('--generate-data', action='store_true', help='Generate new synthetic data')
    parser.add_argument('--days', type=int, default=30, help='Number of days to generate data for')
    parser.add_argument('--roads', type=int, default=20, help='Number of roads to simulate')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    args = parser.parse_args()
    
    # Create directories
    create_directories()
    
    # Generate or load data
    traffic_data, network_data = generate_data(args)
    
    # Preprocess data
    processed_data, processor, data_splits = preprocess_data(traffic_data)
    
    # Train models (unless skipped)
    if not args.skip_training:
        model = train_models(data_splits)
    
    # Build knowledge graph
    kg = build_knowledge_graph(traffic_data, network_data)
    
    # Generate optimization recommendations
    recommendations = optimize_traffic(kg)
    
    print("\n--- Smart Traffic Congestion Control System Completed Successfully ---")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 