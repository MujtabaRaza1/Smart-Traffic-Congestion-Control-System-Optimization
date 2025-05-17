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
import json

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
    "box-shadow": "2px 0 5px rgba(0,0,0,0.1)",
    "z-index": "1000"
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
        html.Div([
            html.H2("Traffic Control", className="display-4", style={
                "color": "#2c3e50",
                "font-size": "1.8rem",
                "font-weight": "600",
                "margin-bottom": "1rem",
                "text-align": "center"
            }),
            html.P("Dashboard for traffic monitoring and optimization", style={
                "color": "#7f8c8d",
                "font-size": "0.9rem",
                "text-align": "center",
                "margin-bottom": "1.5rem"
            }),
        ], style={"margin-bottom": "2rem"}),
        
        html.Div([
            html.H4("Data Source", style={
                "color": "#2c3e50",
                "font-size": "1.1rem",
                "margin-bottom": "0.5rem"
            }),
            dcc.RadioItems(
                id='dataset-selector-radio',
                options=[
                    {'label': 'Enhanced Synthetic Data', 'value': 'enhanced'},
                    {'label': 'Original Synthetic Data', 'value': 'original'},
                ],
                value='enhanced',
                labelStyle={
                    'display': 'block',
                    'padding': '0.5rem 1rem',
                    'margin': '0.3rem 0',
                    'border-radius': '4px',
                    'cursor': 'pointer',
                    'transition': 'background-color 0.2s'
                },
                style={'margin-bottom': '1.5rem'}
            ),
        ], style={"margin-bottom": "1.5rem"}),
        
        html.Div([
            html.H4("Time Selection", style={
                "color": "#2c3e50",
                "font-size": "1.1rem",
                "margin-bottom": "0.5rem"
            }),
            dcc.Dropdown(
                id='timestamp-dropdown',
                placeholder="Select Timestamp",
                style={
                    'margin-bottom': '1.5rem',
                    'border-radius': '4px'
                }
            ),
        ], style={"margin-bottom": "1.5rem"}),
        
        html.Div([
            html.Button(
                'Load Data',
                id='load-data-button',
                n_clicks=0,
                style={
                    'width': '100%',
                    'padding': '0.5rem',
                    'background-color': '#3498db',
                    'color': 'white',
                    'border': 'none',
                    'border-radius': '4px',
                    'cursor': 'pointer',
                    'margin-bottom': '0.5rem',
                    'transition': 'background-color 0.2s'
                }
            ),
            html.Div(
                id='data-load-status',
                style={
                    'font-size': '0.9rem',
                    'color': '#7f8c8d',
                    'margin-top': '0.5rem',
                    'text-align': 'center'
                }
            )
        ], style={"margin-bottom": "1.5rem"}),
        
        html.Div([
            html.Button(
                'Generate Recommendations',
                id='generate-recommendations-button',
                n_clicks=0,
                disabled=True,
                style={
                    'width': '100%',
                    'padding': '0.5rem',
                    'background-color': '#2ecc71',
                    'color': 'white',
                    'border': 'none',
                    'border-radius': '4px',
                    'cursor': 'pointer',
                    'margin-bottom': '0.5rem',
                    'transition': 'background-color 0.2s',
                    'opacity': '0.7'
                }
            ),
            html.Div(
                id='recommendation-status',
                style={
                    'font-size': '0.9rem',
                    'color': '#7f8c8d',
                    'margin-top': '0.5rem',
                    'text-align': 'center'
                }
            )
        ])
    ], style=SIDEBAR_STYLE),
    
    # Main content
    html.Div([
        html.H1("Smart Traffic Congestion Control System", style={
            "text-align": "center",
            "color": "#2c3e50",
            "margin-bottom": "2rem",
            "font-weight": "600"
        }),
        
        # Store for data
        dcc.Store(id='enhanced-traffic-data-store'),
        dcc.Store(id='original-traffic-data-store'),
        dcc.Store(id='network-data-store'),
        dcc.Store(id='recommendation-store'),
        
        # Add tabs for different dashboard sections
        dcc.Tabs([
            # Main dashboard tab
            dcc.Tab(label="Dashboard", children=[
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
            ]),
            
            # Traffic pattern analysis tab
            dcc.Tab(label="Traffic Pattern Analysis", children=[
                html.Div([
                    html.H2("Traffic Pattern Analysis", style={"text-align": "center"}),
                    
                    # Daily patterns
                    html.Div([
                        html.H3("Daily Traffic Patterns"),
                        html.Img(src="/assets/daily_congestion_pattern.png", style={"width": "100%"})
                    ], style={"margin-bottom": "30px"}),
                    
                    # Weekly patterns
                    html.Div([
                        html.H3("Weekly Traffic Patterns"),
                        html.Img(src="/assets/weekly_congestion_pattern.png", style={"width": "100%"})
                    ], style={"margin-bottom": "30px"}),
                    
                    # Rush hour comparison
                    html.Div([
                        html.H3("Rush Hour vs. Non-Rush Hour"),
                        html.Img(src="/assets/rush_hour_comparison.png", style={"width": "100%"})
                    ], style={"margin-bottom": "30px"}),
                    
                    # Congestion heatmap
                    html.Div([
                        html.H3("Congestion Heatmap"),
                        html.Img(src="/assets/congestion_heatmap.png", style={"width": "100%"})
                    ], style={"margin-bottom": "30px"})
                ])
            ]),
            
            # Feature analysis tab
            dcc.Tab(label="Feature Analysis", children=[
                html.Div([
                    html.H2("Traffic Feature Analysis", style={"text-align": "center"}),
                    
                    # Feature correlation
                    html.Div([
                        html.H3("Feature Correlation Matrix"),
                        html.Img(src="/assets/feature_correlation.png", style={"width": "100%"})
                    ], style={"margin-bottom": "30px"}),
                    
                    # Volume vs congestion
                    html.Div([
                        html.H3("Volume vs Congestion Relationship"),
                        html.Img(src="/assets/volume_congestion_relationship.png", style={"width": "100%"})
                    ], style={"margin-bottom": "30px"}),
                    
                    # Road type comparison
                    html.Div([
                        html.H3("Road Type Comparison"),
                        html.Img(src="/assets/road_type_comparison.png", style={"width": "100%"})
                    ], style={"margin-bottom": "30px"}),
                    
                    # Time series
                    html.Div([
                        html.H3("Traffic Metrics Over Time"),
                        html.Div([
                            html.Div([
                                html.H4("Congestion Over Time"),
                                dcc.Graph(id='congestion-time-series')
                            ], style={"width": "100%", "margin-bottom": "30px"}),
                            html.Div([
                                html.H4("Speed Over Time"),
                                dcc.Graph(id='speed-time-series')
                            ], style={"width": "100%", "margin-bottom": "30px"}),
                            html.Div([
                                html.H4("Volume Over Time"),
                                dcc.Graph(id='volume-time-series')
                            ], style={"width": "100%", "margin-bottom": "30px"})
                        ])
                    ], style={"margin-bottom": "30px"})
                ])
            ])
        ])
    ], style=CONTENT_STYLE)
])

@app.callback(
    [Output('enhanced-traffic-data-store', 'data'),
     Output('original-traffic-data-store', 'data'),
     Output('network-data-store', 'data'),
     Output('timestamp-dropdown', 'options'),
     Output('data-load-status', 'children'),
     Output('generate-recommendations-button', 'disabled')],
    [Input('load-data-button', 'n_clicks'),
     Input('dataset-selector-radio', 'value')],
    prevent_initial_call=True
)
def load_data(n_clicks, selected_dataset):
    """Load traffic and network data."""
    if n_clicks <= 0:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Data paths
    data_dir = Path(__file__).parent.parent / "data"
    original_traffic_file = data_dir / "synthetic_traffic_data.csv"
    enhanced_traffic_file = data_dir / "enhanced_synthetic_traffic_data.csv"
    network_file = data_dir / "road_network_data.csv"
    
    original_data_dict = None
    enhanced_data_dict = None
    network_data_dict = None
    all_timestamps = set()
    status_messages = []
    disable_recommendations = True

    if network_file.exists():
        try:
            network_df = pd.read_csv(network_file)
            network_data_dict = network_df.to_dict('records')
            status_messages.append(f"Loaded {network_file.name}.")
        except Exception as e:
            status_messages.append(f"Error loading {network_file.name}: {e}")
            # If network data fails to load, we probably can't do much.
            return None, None, None, [], ", ".join(status_messages), True
    else:
        status_messages.append(f"Error: {network_file.name} not found.")
        return None, None, None, [], ", ".join(status_messages), True

    # Load data based on selected dataset type
    if selected_dataset == 'enhanced':
        if enhanced_traffic_file.exists():
            try:
                df_enhanced = pd.read_csv(enhanced_traffic_file)
                df_enhanced['timestamp'] = pd.to_datetime(df_enhanced['timestamp'])
                all_timestamps.update(df_enhanced['timestamp'].unique())
                df_enhanced['timestamp'] = df_enhanced['timestamp'].astype(str) # For storage
                enhanced_data_dict = df_enhanced.to_dict('records')
                status_messages.append(f"Loaded {enhanced_traffic_file.name}.")
                disable_recommendations = False
            except Exception as e:
                status_messages.append(f"Error loading {enhanced_traffic_file.name}: {e}")
        else:
            status_messages.append(f"{enhanced_traffic_file.name} not found.")
    else:  # original dataset
        if original_traffic_file.exists():
            try:
                df_original = pd.read_csv(original_traffic_file)
                df_original['timestamp'] = pd.to_datetime(df_original['timestamp'])
                all_timestamps.update(df_original['timestamp'].unique())
                df_original['timestamp'] = df_original['timestamp'].astype(str) # For storage
                original_data_dict = df_original.to_dict('records')
                status_messages.append(f"Loaded {original_traffic_file.name}.")
                disable_recommendations = False
            except Exception as e:
                status_messages.append(f"Error loading {original_traffic_file.name}: {e}")
        else:
            status_messages.append(f"{original_traffic_file.name} not found.")

    if not all_timestamps:
        status_messages.append("No timestamps found in any dataset.")
        timestamp_options = []
        disable_recommendations = True # No data, so disable
    else:
        sorted_timestamps = sorted(list(all_timestamps))
        # Ensure consistent string format for value, compatible with pd.to_datetime
        timestamp_options = [{'label': str(pd.Timestamp(ts)), 'value': str(pd.Timestamp(ts))} for ts in sorted_timestamps] 
        status_messages.append("Timestamps updated.")

    final_status = ", ".join(status_messages)
    if not original_data_dict and not enhanced_data_dict:
        final_status += " Critical: No traffic data loaded."
        disable_recommendations = True
    
    return enhanced_data_dict, original_data_dict, network_data_dict, timestamp_options, final_status, disable_recommendations

@app.callback(
    Output('traffic-overview-graph', 'figure'),
    [Input('enhanced-traffic-data-store', 'data'),
     Input('original-traffic-data-store', 'data'),
     Input('dataset-selector-radio', 'value'),
     Input('timestamp-dropdown', 'value')],
    prevent_initial_call=True
)
def update_traffic_overview(enhanced_traffic_data, original_traffic_data, selected_dataset, selected_timestamp):
    """Update traffic overview graph."""
    traffic_data_to_use = None
    if selected_dataset == 'enhanced':
        traffic_data_to_use = enhanced_traffic_data
    elif selected_dataset == 'original':
        traffic_data_to_use = original_traffic_data

    if not traffic_data_to_use:
        return go.Figure().update_layout(title="No data available for selected dataset")
    
    # Convert to DataFrame
    df = pd.DataFrame(traffic_data_to_use)
    
    # Filter by timestamp if selected
    if selected_timestamp:
        # Ensure selected_timestamp is comparable with df['timestamp']
        # Timestamps in store are strings, selected_timestamp from dropdown is also string
        df = df[df['timestamp'] == str(pd.Timestamp(selected_timestamp))] 
    
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
     Input('enhanced-traffic-data-store', 'data'),
     Input('original-traffic-data-store', 'data'),
     Input('dataset-selector-radio', 'value'),
     Input('timestamp-dropdown', 'value')],
    prevent_initial_call=True
)
def update_network_graph(network_data, enhanced_traffic_data, original_traffic_data, selected_dataset, selected_timestamp):
    """Update network graph visualization."""
    traffic_data_to_use = None
    if selected_dataset == 'enhanced':
        traffic_data_to_use = enhanced_traffic_data
    elif selected_dataset == 'original':
        traffic_data_to_use = original_traffic_data

    if not network_data or not traffic_data_to_use:
        return go.Figure().update_layout(title="No data available for selected dataset")
    
    # Convert to DataFrames
    network_df = pd.DataFrame(network_data)
    traffic_df = pd.DataFrame(traffic_data_to_use)
    
    # Filter by timestamp if selected
    if selected_timestamp:
        traffic_df = traffic_df[traffic_df['timestamp'] == str(pd.Timestamp(selected_timestamp))]
    
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
    [State('enhanced-traffic-data-store', 'data'),
     State('original-traffic-data-store', 'data'),
     State('dataset-selector-radio', 'value'),
     State('network-data-store', 'data'),
     State('timestamp-dropdown', 'value')],
    prevent_initial_call=True
)
def generate_recommendations(n_clicks, enhanced_traffic_data, original_traffic_data, selected_dataset, network_data, selected_timestamp):
    """Generate traffic optimization recommendations."""
    traffic_data_to_use = None
    if selected_dataset == 'enhanced':
        traffic_data_to_use = enhanced_traffic_data
    elif selected_dataset == 'original':
        traffic_data_to_use = original_traffic_data
        
    if n_clicks <= 0 or not traffic_data_to_use or not network_data:
        return dash.no_update, dash.no_update
    
    try:
        # Convert to DataFrames
        traffic_df = pd.DataFrame(traffic_data_to_use)
        
        # Filter by timestamp if selected
        if selected_timestamp:
            traffic_df = traffic_df[traffic_df['timestamp'] == str(pd.Timestamp(selected_timestamp))]
        
        # Find roads with high congestion
        high_congestion_roads = traffic_df[traffic_df['congestion'] > 0.7]
        
        # Create recommendations based on congestion, road type, and other factors
        recommendations = {}
        
        for _, road in high_congestion_roads.iterrows():
            road_id = str(road['road_id'])
            congestion = float(road['congestion'])
            road_type = str(road['road_type'])
            has_incident = bool(road['incident'])
            
            # Determine appropriate action based on conditions
            if has_incident:
                action = "manage_incident"
            elif congestion > 0.9:
                action = "reroute"
            elif congestion > 0.8:
                action = "adjust_signals"
            elif congestion > 0.7:
                action = "increase_capacity"
            
            # Create a recommendation entry
            recommendation = {
                'road_id': road_id,
                'congestion': congestion,
                'action': action,
                'road_type': road_type
            }
            
            # Add to recommendations
            recommendations[road_id] = recommendation
        
        # If no high congestion roads found
        if not recommendations:
            # Get top 3 most congested roads as fallback
            top_congested = traffic_df.sort_values('congestion', ascending=False).head(3)
            for _, road in top_congested.iterrows():
                road_id = str(road['road_id'])
                congestion = float(road['congestion'])
                road_type = str(road['road_type'])
                
                # Create a recommendation entry
                recommendations[road_id] = {
                    'road_id': road_id,
                    'congestion': congestion,
                    'action': "monitor" if congestion < 0.7 else "adjust_signals",
                    'road_type': road_type
                }
        
        return recommendations, f"Generated {len(recommendations)} recommendations"
    
    except Exception as e:
        import traceback
        print(f"Error generating recommendations: {str(e)}")
        print(traceback.format_exc())
        return {}, f"Error generating recommendations: {str(e)}"

@app.callback(
    Output('recommendations-table', 'children'),
    Input('recommendation-store', 'data'),
    prevent_initial_call=True
)
def update_recommendations_table(recommendations):
    """Update recommendations table."""
    if not recommendations:
        return html.Div("No recommendations available. Try selecting a different timestamp.", 
                       style={'color': 'red', 'margin': '20px', 'font-weight': 'bold'})
    
    # Create table
    header = html.Tr([
        html.Th("Road ID"),
        html.Th("Action"),
        html.Th("Congestion"),
        html.Th("Details")
    ])
    
    rows = []
    try:
        for road_id, rec in recommendations.items():
            rows.append(html.Tr([
                html.Td(road_id),
                html.Td(rec.get('action', 'Unknown')),
                html.Td(f"{rec.get('congestion', 0):.2f}"),
                html.Td(get_recommendation_details(rec))
            ]))
        
        if not rows:
            return html.Div("No specific recommendations for the selected timestamp. "
                          "Try selecting a different timestamp with higher congestion levels.",
                          style={'color': 'blue', 'margin': '20px'})
        
        table = html.Table([header] + rows, className="table table-striped table-hover")
        return table
    except Exception as e:
        return html.Div(f"Error displaying recommendations: {str(e)}", 
                       style={'color': 'red', 'margin': '20px'})

def get_recommendation_details(recommendation):
    """Format recommendation details for display."""
    if not recommendation:
        return "No details available"
        
    action = recommendation.get('action', '')
    road_type = recommendation.get('road_type', '')
    congestion = recommendation.get('congestion', 0)
    
    actions = {
        'reroute': f"Reroute traffic from this {road_type} road (congestion: {congestion:.2f})",
        'increase_capacity': f"Increase road capacity on this {road_type} road",
        'adjust_signals': f"Optimize traffic signal timing for {road_type} road",
        'manage_incident': f"Clear incident and manage traffic flow on {road_type} road",
        'reduce_speed_limit': f"Reduce speed limit due to high congestion ({congestion:.2f})",
        'monitor': f"Monitor traffic conditions on this {road_type} road"
    }
    
    return actions.get(action, "No additional details")

@app.callback(
    [Output('congestion-time-series', 'figure'),
     Output('speed-time-series', 'figure'),
     Output('volume-time-series', 'figure')],
    [Input('enhanced-traffic-data-store', 'data'),
     Input('original-traffic-data-store', 'data'),
     Input('dataset-selector-radio', 'value')],
    prevent_initial_call=True
)
def update_time_series(enhanced_traffic_data, original_traffic_data, selected_dataset):
    """Update time series visualizations."""
    traffic_data_to_use = None
    if selected_dataset == 'enhanced':
        traffic_data_to_use = enhanced_traffic_data
    elif selected_dataset == 'original':
        traffic_data_to_use = original_traffic_data

    if not traffic_data_to_use:
        empty_fig = go.Figure().update_layout(title="No data available for selected dataset")
        return empty_fig, empty_fig, empty_fig
    
    # Convert to DataFrame
    df = pd.DataFrame(traffic_data_to_use)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate average metrics over time
    time_series = df.groupby('timestamp').agg({
        'congestion': 'mean',
        'speed': 'mean',
        'volume': 'mean'
    }).reset_index()
    
    # Create congestion figure
    congestion_fig = go.Figure()
    congestion_fig.add_trace(
        go.Scatter(
            x=time_series['timestamp'],
            y=time_series['congestion'],
            name="Congestion",
            line=dict(color='red')
        )
    )
    congestion_fig.update_layout(
        title="Congestion Level Over Time",
        xaxis_title="Time",
        yaxis_title="Congestion Level",
        hovermode="x unified",
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=30, l=60, r=30, b=30),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=False
        )
    )
    
    # Create speed figure
    speed_fig = go.Figure()
    speed_fig.add_trace(
        go.Scatter(
            x=time_series['timestamp'],
            y=time_series['speed'],
            name="Speed",
            line=dict(color='green')
        )
    )
    speed_fig.update_layout(
        title="Average Speed Over Time",
        xaxis_title="Time",
        yaxis_title="Speed (km/h)",
        hovermode="x unified",
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=30, l=60, r=30, b=30),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=False
        )
    )
    
    # Create volume figure
    volume_fig = go.Figure()
    volume_fig.add_trace(
        go.Scatter(
            x=time_series['timestamp'],
            y=time_series['volume'],
            name="Volume",
            line=dict(color='blue')
        )
    )
    volume_fig.update_layout(
        title="Traffic Volume Over Time",
        xaxis_title="Time",
        yaxis_title="Volume (vehicles)",
        hovermode="x unified",
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=30, l=60, r=30, b=30),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=False
        )
    )
    
    return congestion_fig, speed_fig, volume_fig

def run_dashboard(debug=True, port=8050):
    """Run the dashboard."""
    # Create output directory
    output_dir = Path(__file__).parent / "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the app
    app.run(debug=debug, port=port)

if __name__ == "__main__":
    run_dashboard() 