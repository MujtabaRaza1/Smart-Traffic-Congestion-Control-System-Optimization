#!/usr/bin/env python3
"""
Run the Smart Traffic Congestion Control Dashboard

This script first checks if data exists, runs data generation if needed,
and then launches the dashboard.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project directory to the path
sys.path.append(str(Path(__file__).parent))

def main():
    """Main function to run the dashboard."""
    parser = argparse.ArgumentParser(description='Run the Smart Traffic Dashboard')
    parser.add_argument('--port', type=int, default=8051, help='Port to run the dashboard on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--generate-data', action='store_true', help='Generate new data before running')
    args = parser.parse_args()
    
    # Check if data exists, if not, run data generation
    data_dir = Path(__file__).parent / "data"
    traffic_file = data_dir / "synthetic_traffic_data.csv"
    network_file = data_dir / "road_network_data.csv"
    
    if args.generate_data or not (traffic_file.exists() and network_file.exists()):
        print("Generating data before launching dashboard...")
        from main import main as main_process
        # Run with data generation enabled
        main_process_args = argparse.Namespace(
            generate_data=True,
            days=30,
            roads=20,
            skip_training=False
        )
        main_process(main_process_args)
    
    # Now run the dashboard
    from visualization.dashboard import run_dashboard
    print(f"Launching dashboard on port {args.port}...")
    run_dashboard(debug=args.debug, port=args.port)

if __name__ == "__main__":
    main() 