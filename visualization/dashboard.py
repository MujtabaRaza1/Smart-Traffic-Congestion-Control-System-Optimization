#!/usr/bin/env python3
"""
Traffic Dashboard

A Dash-based web dashboard for visualizing traffic data and recommendations.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import project modules
from optimization.traffic_optimizer import TrafficKnowledgeGraph

# Initialize the Dash app
app = dash.Dash(__name__, title="Smart Traffic Control Dashboard")

# Sidebar style
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# Content style
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

# Define the app layout
app.layout = html.Div([
    # Sidebar
    html.Div([
        html.H2("Traffic Control", className="display-4"),
        html.Hr(),
        html.P("Dashboard for traffic monitoring and optimization", className="lead"),
        html.Hr(),
        dcc.Dropdown(
            id='timestamp-dropdown',
            placeholder="Select Timestamp",
        ),
        html.Hr(),
        html.Div([
            html.Button('Load Data', id='load-data-button', n_clicks=0),
            html.Div(id='data-load-status')
        ]),
        html.Hr(),
        html.Div([
            html.Button('Generate Recommendations', id='generate-recommendations-button', n_clicks=0, disabled=True),
            html.Div(id='recommendation-status')
        ]),
    ], style=SIDEBAR_STYLE),
    
    # Main content
    html.Div([
        html.H1("Smart Traffic Congestion Control System", style={"text-align": "center"}),
        
        # Store for data
        dcc.Store(id='traffic-data-store'),
        dcc.Store(id='network-data-store'),
        dcc.Store(id='recommendation-store'),
        
        # Traffic overview
        html.Div([
            html.H2("Traffic Overview"),
            dcc.Graph(id='traffic-overview-graph')
        ]),
        
        # Network visualization
        html.Div([
            html.H2("Road Network"),
            dcc.Graph(id='network-graph', style={"height": "700px"})
        ]),
        
        # Recommendations
        html.Div([
            html.H2("Traffic Optimization Recommendations"),
            html.Div(id='recommendations-table')
        ]),
        
    ], style=CONTENT_STYLE)
])

@app.callback(
    [Output('traffic-data-store', 'data'),
     Output('network-data-store', 'data'),
     Output('timestamp-dropdown', 'options'),
     Output('data-load-status', 'children'),
     Output('generate-recommendations-button', 'disabled')],
    Input('load-data-button', 'n_clicks'),
    prevent_initial_call=True
)
def load_data(n_clicks):
    """Load traffic and network data."""
    if n_clicks <= 0:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Data paths
    data_dir = Path(__file__).parent.parent / "data"
    traffic_file = data_dir / "synthetic_traffic_data.csv"
    network_file = data_dir / "road_network_data.csv"
    
    if not traffic_file.exists() or not network_file.exists():
        return dash.no_update, dash.no_update, dash.no_update, "Error: Data files not found.", True
    
    # Load data
    traffic_data = pd.read_csv(traffic_file)
    network_data = pd.read_csv(network_file)
    
    # Convert timestamp to datetime
    traffic_data['timestamp'] = pd.to_datetime(traffic_data['timestamp'])
    
    # Get unique timestamps
    timestamps = traffic_data['timestamp'].unique()
    timestamp_options = [{'label': str(ts), 'value': str(ts)} for ts in timestamps]
    
    # For storage, convert timestamps to string
    traffic_data['timestamp'] = traffic_data['timestamp'].astype(str)
    
    return traffic_data.to_dict('records'), network_data.to_dict('records'), timestamp_options, "Data loaded successfully.", False

@app.callback(
    Output('traffic-overview-graph', 'figure'),
    [Input('traffic-data-store', 'data'),
     Input('timestamp-dropdown', 'value')],
    prevent_initial_call=True
)
def update_traffic_overview(traffic_data, selected_timestamp):
    """Update traffic overview graph."""
    if not traffic_data:
        return go.Figure().update_layout(title="No data available")
    
    # Convert to DataFrame
    df = pd.DataFrame(traffic_data)
    
    # Filter by timestamp if selected
    if selected_timestamp:
        df = df[df['timestamp'] == selected_timestamp]
    
    # Calculate average congestion by road type
    road_type_congestion = df.groupby('road_type')['congestion'].mean().reset_index()
    
    # Create bar chart
    fig = px.bar(
        road_type_congestion, 
        x='road_type', 
        y='congestion',
        title='Average Congestion by Road Type',
        labels={'congestion': 'Congestion Level', 'road_type': 'Road Type'},
        color='congestion',
        color_continuous_scale='RdYlGn_r'
    )
    
    return fig

@app.callback(
    Output('network-graph', 'figure'),
    [Input('network-data-store', 'data'),
     Input('traffic-data-store', 'data'),
     Input('timestamp-dropdown', 'value')],
    prevent_initial_call=True
)
def update_network_graph(network_data, traffic_data, selected_timestamp):
    """Update network graph visualization."""
    if not network_data or not traffic_data:
        return go.Figure().update_layout(title="No data available")
    
    # Convert to DataFrames
    network_df = pd.DataFrame(network_data)
    traffic_df = pd.DataFrame(traffic_data)
    
    # Filter by timestamp if selected
    if selected_timestamp:
        traffic_df = traffic_df[traffic_df['timestamp'] == selected_timestamp]
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Get unique road IDs
    unique_roads = set(network_df['source_road'].unique()) | set(network_df['target_road'].unique())
    
    # Add nodes
    for road in unique_roads:
        G.add_node(road)
    
    # Add edges
    for _, row in network_df.iterrows():
        G.add_edge(
            row['source_road'],
            row['target_road'],
            weight=row['distance']
        )
    
    # Set node attributes from traffic data
    road_attrs = {}
    for _, row in traffic_df.iterrows():
        road_id = row['road_id']
        if road_id in unique_roads:
            road_attrs[road_id] = {
                'congestion': row['congestion'],
                'speed': row['speed'],
                'volume': row['volume'],
                'road_type': row['road_type']
            }
    
    # Create node positions using Kamada-Kawai layout
    pos = nx.kamada_kawai_layout(G)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create node trace
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    # Prepare node colors based on congestion
    node_colors = []
    node_sizes = []
    node_text = []
    
    for node in G.nodes():
        if node in road_attrs:
            attrs = road_attrs[node]
            congestion = attrs['congestion']
            node_colors.append(congestion)
            node_sizes.append(20 + attrs['volume'] / 10)
            road_type = attrs['road_type']
            node_text.append(f"Road ID: {node}<br>Congestion: {congestion:.2f}<br>Type: {road_type}<br>Speed: {attrs['speed']:.1f}<br>Volume: {attrs['volume']}")
        else:
            node_colors.append(0)
            node_sizes.append(15)
            node_text.append(f"Road ID: {node}<br>No data")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='RdYlGn_r',
            color=node_colors,
            size=node_sizes,
            colorbar=dict(
                title='Congestion',
                thickness=15,
                tickvals=[0, 0.5, 1],
                ticktext=['Low', 'Medium', 'High']
            ),
            line=dict(width=2)
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
         layout=go.Layout(
            title='Traffic Network Visualization',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    
    return fig

@app.callback(
    [Output('recommendation-store', 'data'),
     Output('recommendation-status', 'children')],
    Input('generate-recommendations-button', 'n_clicks'),
    [State('traffic-data-store', 'data'),
     State('network-data-store', 'data'),
     State('timestamp-dropdown', 'value')],
    prevent_initial_call=True
)
def generate_recommendations(n_clicks, traffic_data, network_data, selected_timestamp):
    """Generate traffic optimization recommendations."""
    if n_clicks <= 0 or not traffic_data or not network_data:
        return dash.no_update, dash.no_update
    
    # Convert to DataFrames
    traffic_df = pd.DataFrame(traffic_data)
    network_df = pd.DataFrame(network_data)
    
    # Convert timestamp back to datetime
    traffic_df['timestamp'] = pd.to_datetime(traffic_df['timestamp'])
    
    # Filter by timestamp if selected
    if selected_timestamp:
        traffic_df = traffic_df[traffic_df['timestamp'] == selected_timestamp]
    
    # Create knowledge graph
    kg = TrafficKnowledgeGraph()
    
    # Save network data to temp file
    temp_network_file = Path(__file__).parent / "temp_network.csv"
    network_df.to_csv(temp_network_file, index=False)
    
    # Load network data
    kg.load_network_data(str(temp_network_file))
    
    # Add traffic data
    kg.add_traffic_data(traffic_df)
    
    # Get recommendations
    recommendations = kg.optimize_traffic_flow()
    
    # Clean up temp file
    if temp_network_file.exists():
        temp_network_file.unlink()
    
    return recommendations, f"Generated {len(recommendations)} recommendations"

@app.callback(
    Output('recommendations-table', 'children'),
    Input('recommendation-store', 'data'),
    prevent_initial_call=True
)
def update_recommendations_table(recommendations):
    """Update recommendations table."""
    if not recommendations:
        return "No recommendations available."
    
    # Create table
    header = html.Tr([
        html.Th("Road ID"),
        html.Th("Action"),
        html.Th("Congestion"),
        html.Th("Details")
    ])
    
    rows = []
    for road_id, rec in recommendations.items():
        rows.append(html.Tr([
            html.Td(road_id),
            html.Td(rec['action']),
            html.Td(f"{rec['congestion']:.2f}"),
            html.Td(get_recommendation_details(rec))
        ]))
    
    table = html.Table([header] + rows, className="table table-striped table-hover")
    return table

def get_recommendation_details(recommendation):
    """Format recommendation details for display."""
    if recommendation['action'] == 'reroute' and 'alternatives' in recommendation:
        alternatives = recommendation['alternatives']
        if alternatives:
            alt_info = alternatives[0]  # Just show the first alternative
            return f"Alternative route: {' → '.join(map(str, alt_info['path']))} (avg congestion: {alt_info['avg_congestion']:.2f})"
    
    actions = {
        'reroute': "Reroute traffic to alternative roads",
        'increase_capacity': "Increase road capacity (add lanes or adjust signals)",
        'adjust_signals': "Optimize traffic signal timing",
        'manage_incident': "Clear incident and manage traffic flow",
        'reduce_speed_limit': "Reduce speed limit due to conditions"
    }
    
    return actions.get(recommendation['action'], "No additional details")

def run_dashboard(debug=True, port=8050):
    """Run the dashboard."""
    # Create output directory
    output_dir = Path(__file__).parent / "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the app
    app.run_server(debug=debug, port=port)

if __name__ == "__main__":
    run_dashboard() 