import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pickle
import sys

class TrafficKnowledgeGraph:
    """
    A graph-based representation of traffic knowledge for optimization.
    
    Integrates traffic data, road network, and traffic rules into a knowledge graph.
    """
    def __init__(self):
        self.G = nx.DiGraph()
        self.road_data = {}
        self.congestion_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
        self.rules = []
    
    def load_network_data(self, network_file):
        """
        Load road network data to build the graph.
        
        Parameters:
        -----------
        network_file : str
            Path to the network data file
            
        Returns:
        --------
        self
            The knowledge graph instance
        """
        print(f"Loading network data from {network_file}...")
        network_df = pd.read_csv(network_file)
        
        # Create nodes for all unique roads
        roads = set(network_df['source_road'].unique()) | set(network_df['target_road'].unique())
        for road in roads:
            self.G.add_node(road, type='road')
        
        # Add edges between roads
        for _, row in network_df.iterrows():
            self.G.add_edge(
                row['source_road'],
                row['target_road'],
                distance=row['distance'],
                connection_type=row['connection_type']
            )
        
        print(f"Knowledge graph constructed with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        return self
    
    def add_traffic_data(self, traffic_df, timestamp=None):
        """
        Add traffic data to the graph nodes.
        
        Parameters:
        -----------
        traffic_df : pandas.DataFrame
            Traffic data
        timestamp : datetime, optional
            Specific timestamp to filter data by
            
        Returns:
        --------
        self
            The knowledge graph instance
        """
        # Filter by timestamp if provided
        if timestamp is not None:
            df = traffic_df[traffic_df['timestamp'] == timestamp].copy()
        else:
            # Use the most recent timestamp if available
            if 'timestamp' in traffic_df.columns:
                latest_timestamp = traffic_df['timestamp'].max()
                df = traffic_df[traffic_df['timestamp'] == latest_timestamp].copy()
            else:
                df = traffic_df.copy()
        
        # Store traffic data by road ID
        for _, row in df.iterrows():
            road_id = row['road_id']
            
            # Skip if road not in graph
            if road_id not in self.G:
                continue
            
            # Add attributes to graph node
            for col in df.columns:
                if col != 'road_id':
                    self.G.nodes[road_id][col] = row[col]
        
        print(f"Added traffic data to {len(df['road_id'].unique())} roads")
        return self
    
    def add_rules(self, rules):
        """
        Add traffic management rules to the knowledge base.
        
        Parameters:
        -----------
        rules : list
            List of rule dictionaries
            
        Returns:
        --------
        self
            The knowledge graph instance
        """
        self.rules = rules
        return self
    
    def visualize_network(self, save_path=None, with_congestion=True):
        """
        Visualize the road network with congestion levels.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the visualization
        with_congestion : bool
            Whether to color nodes by congestion level
            
        Returns:
        --------
        None
        """
        plt.figure(figsize=(12, 10))
        
        # Create positions using spring layout
        pos = nx.spring_layout(self.G, seed=42)
        
        # Create node colors based on congestion
        node_colors = []
        if with_congestion:
            for node in self.G.nodes():
                if 'congestion' in self.G.nodes[node]:
                    congestion = self.G.nodes[node]['congestion']
                    if congestion >= self.congestion_thresholds['high']:
                        node_colors.append('red')
                    elif congestion >= self.congestion_thresholds['medium']:
                        node_colors.append('orange')
                    elif congestion >= self.congestion_thresholds['low']:
                        node_colors.append('yellow')
                    else:
                        node_colors.append('green')
                else:
                    node_colors.append('gray')
        else:
            node_colors = ['blue'] * self.G.number_of_nodes()
        
        # Draw the graph
        nx.draw(
            self.G, 
            pos, 
            with_labels=True, 
            node_color=node_colors,
            node_size=500,
            font_size=10,
            font_weight='bold',
            arrowsize=15
        )
        
        # Add edge labels
        edge_labels = {(u, v): f"{d['distance']:.1f}" for u, v, d in self.G.edges(data=True)}
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title('Traffic Network with Congestion Levels')
        
        # Add legend
        if with_congestion:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Low Congestion'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Medium Congestion'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='High Congestion'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Severe Congestion'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='No Data')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Network visualization saved to {save_path}")
        
        plt.show()
    
    def optimize_traffic_flow(self):
        """
        Apply optimization rules to suggest traffic control actions.
        
        Returns:
        --------
        dict
            Dictionary of road IDs and recommended actions
        """
        recommendations = {}
        
        # Loop through all roads
        for road in self.G.nodes():
            # Skip roads without congestion data
            if 'congestion' not in self.G.nodes[road]:
                continue
            
            congestion = self.G.nodes[road]['congestion']
            
            # Apply rules based on congestion level
            if congestion >= self.congestion_thresholds['high']:
                # Check for alternative routes
                alternative_routes = self._find_alternative_routes(road)
                
                if alternative_routes:
                    recommendations[road] = {
                        'action': 'reroute',
                        'congestion': congestion,
                        'alternatives': alternative_routes
                    }
                else:
                    recommendations[road] = {
                        'action': 'increase_capacity',
                        'congestion': congestion
                    }
            
            elif congestion >= self.congestion_thresholds['medium']:
                # Check if this is a significant road
                if 'road_importance' in self.G.nodes[road] and self.G.nodes[road]['road_importance'] >= 3:
                    recommendations[road] = {
                        'action': 'adjust_signals',
                        'congestion': congestion
                    }
            
            # Rules for incident management
            if 'incident' in self.G.nodes[road] and self.G.nodes[road]['incident'] == 1:
                recommendations[road] = {
                    'action': 'manage_incident',
                    'congestion': congestion
                }
        
        print(f"Generated optimization recommendations for {len(recommendations)} roads")
        return recommendations
    
    def _find_alternative_routes(self, road, max_alternatives=3):
        """
        Find alternative routes to avoid a congested road.
        
        Parameters:
        -----------
        road : int
            Road ID to find alternatives for
        max_alternatives : int
            Maximum number of alternatives to find
            
        Returns:
        --------
        list
            List of alternative route dictionaries
        """
        alternatives = []
        
        # Get predecessors and successors of the road
        predecessors = list(self.G.predecessors(road))
        successors = list(self.G.successors(road))
        
        if not predecessors or not successors:
            return alternatives
        
        # Try to find paths that don't include the congested road
        for source in predecessors:
            for target in successors:
                try:
                    # Remove the congested road temporarily
                    self.G.remove_node(road)
                    
                    # Find alternative paths
                    paths = list(nx.shortest_simple_paths(self.G, source, target, weight='distance'))
                    
                    # Limit the number of alternatives
                    for i, path in enumerate(paths[:max_alternatives]):
                        # Calculate expected congestion along this path
                        avg_congestion = 0
                        count = 0
                        
                        for node in path:
                            if 'congestion' in self.G.nodes[node]:
                                avg_congestion += self.G.nodes[node]['congestion']
                                count += 1
                        
                        if count > 0:
                            avg_congestion /= count
                        
                        alternatives.append({
                            'path': path,
                            'length': len(path),
                            'avg_congestion': avg_congestion
                        })
                    
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    # No alternative path exists
                    pass
                finally:
                    # Add the congested road back
                    self.G.add_node(road)
                    for pred in predecessors:
                        self.G.add_edge(pred, road)
                    for succ in successors:
                        self.G.add_edge(road, succ)
        
        # Sort alternatives by average congestion
        return sorted(alternatives, key=lambda x: x['avg_congestion'])
    
    def get_congestion_summary(self):
        """
        Get a summary of congestion levels in the network.
        
        Returns:
        --------
        dict
            Dictionary of congestion statistics
        """
        congestion_values = []
        
        for node, attrs in self.G.nodes(data=True):
            if 'congestion' in attrs:
                congestion_values.append(attrs['congestion'])
        
        if not congestion_values:
            return {
                'min': None,
                'max': None,
                'avg': None,
                'severe_count': 0,
                'total_roads': self.G.number_of_nodes()
            }
        
        congestion_array = np.array(congestion_values)
        
        return {
            'min': congestion_array.min(),
            'max': congestion_array.max(),
            'avg': congestion_array.mean(),
            'severe_count': sum(congestion_array >= self.congestion_thresholds['high']),
            'total_roads': self.G.number_of_nodes()
        }
    
    def save(self, filepath):
        """
        Save the knowledge graph to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the knowledge graph
            
        Returns:
        --------
        None
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Knowledge graph saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load a knowledge graph from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved knowledge graph
            
        Returns:
        --------
        TrafficKnowledgeGraph
            Loaded knowledge graph instance
        """
        with open(filepath, 'rb') as f:
            kg = pickle.load(f)
        print(f"Knowledge graph loaded from {filepath}")
        return kg

# Defined functions for rule conditions
def condition_high_congestion(road, attrs):
    return attrs.get('congestion', 0) >= 0.8

def condition_incident(road, attrs):
    return attrs.get('incident', 0) == 1

def condition_medium_congestion(road, attrs):
    return 0.6 <= attrs.get('congestion', 0) < 0.8

def condition_precipitation(road, attrs):
    return attrs.get('precipitation', 0) > 1.0

# Example traffic rules
DEFAULT_RULES = [
    {
        'name': 'high_congestion_rule',
        'condition': condition_high_congestion,
        'action': 'reroute',
        'priority': 3
    },
    {
        'name': 'incident_rule',
        'condition': condition_incident,
        'action': 'manage_incident',
        'priority': 4
    },
    {
        'name': 'medium_congestion_rule',
        'condition': condition_medium_congestion,
        'action': 'adjust_signals',
        'priority': 2
    },
    {
        'name': 'precipitation_rule',
        'condition': condition_precipitation,
        'action': 'reduce_speed_limit',
        'priority': 1
    }
]

if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    import sys
    
    # Add parent directory to path to import data
    sys.path.append(str(Path(__file__).parent.parent))
    
    # Get the paths to the data files
    data_dir = Path(__file__).parent.parent / "data"
    traffic_file = data_dir / "synthetic_traffic_data.csv"
    network_file = data_dir / "road_network_data.csv"
    
    if traffic_file.exists() and network_file.exists():
        # Load the data
        traffic_data = pd.read_csv(traffic_file)
        
        # Convert timestamp to datetime
        traffic_data['timestamp'] = pd.to_datetime(traffic_data['timestamp'])
        
        # Get the latest timestamp
        latest_timestamp = traffic_data['timestamp'].max()
        
        # Create knowledge graph
        kg = TrafficKnowledgeGraph()
        
        # Load network data
        kg.load_network_data(network_file)
        
        # Add traffic data for the latest timestamp
        kg.add_traffic_data(traffic_data, timestamp=latest_timestamp)
        
        # Add rules
        kg.add_rules(DEFAULT_RULES)
        
        # Visualize the network
        kg.visualize_network(
            save_path=str(Path(__file__).parent / "outputs" / "traffic_network.png")
        )
        
        # Get congestion summary
        summary = kg.get_congestion_summary()
        print("\nCongestion Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Optimize traffic flow
        recommendations = kg.optimize_traffic_flow()
        
        print("\nTraffic Optimization Recommendations:")
        for road, rec in recommendations.items():
            print(f"  Road {road}: {rec['action']} (Congestion: {rec['congestion']:.2f})")
        
        # Save the knowledge graph
        kg.save(str(Path(__file__).parent / "saved_graphs" / "traffic_knowledge_graph.pkl"))
    else:
        print(f"Error: Could not find data files at {traffic_file} or {network_file}")
        print("Please run the synthetic_data_generator.py script first.") 