#!/usr/bin/env python3
"""
Enhanced Traffic Congestion Analysis for Smart Traffic Control System
=====================================================================

This script performs comprehensive analysis of traffic congestion data,
implementing multiple visualizations and models that directly address 
the project's objectives of using machine learning for congestion prediction
and knowledge representation for traffic optimization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           roc_curve, auc, precision_recall_curve)

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
colors = plt.cm.tab10.colors

# Load the synthetic traffic data
print("Loading traffic data...")
traffic_data = pd.read_csv('data/synthetic_traffic_data.csv')

# Display basic information about the dataset
print(f"Dataset shape: {traffic_data.shape}")
print(f"Columns: {traffic_data.columns.tolist()}")

# Create a figure for multiple plots
plt.figure(figsize=(18, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.4)

# Convert timestamp to datetime for time analysis
traffic_data['timestamp'] = pd.to_datetime(traffic_data['timestamp'])
traffic_data['hour_of_day'] = traffic_data['timestamp'].dt.hour
traffic_data['day_name'] = traffic_data['timestamp'].dt.day_name()

# Convert congestion to binary classes (high/low congestion) for classification
# Let's consider congestion > 0.7 as high congestion
traffic_data['congestion_class'] = (traffic_data['congestion'] > 0.7).astype(int)

# ========================== DATA EXPLORATION ==========================

# PLOT 1: Distribution of congestion by time of day (Objective 1)
plt.subplot(2, 3, 1)
hourly_congestion = traffic_data.groupby('hour_of_day')['congestion'].mean()
sns.lineplot(x=hourly_congestion.index, y=hourly_congestion.values, linewidth=2.5, color=colors[0])
plt.title('Average Congestion by Hour of Day', fontweight='bold')
plt.xlabel('Hour of Day')
plt.ylabel('Average Congestion')
plt.xticks(range(0, 24, 4))
plt.grid(True, linestyle='--', alpha=0.7)

# PLOT 2: Congestion levels by road type (Objective 1 & 2)
plt.subplot(2, 3, 2)
road_congestion = traffic_data.groupby('road_type')['congestion'].mean().sort_values(ascending=False)
sns.barplot(x=road_congestion.index, y=road_congestion.values, palette='viridis')
plt.title('Average Congestion by Road Type', fontweight='bold')
plt.xlabel('Road Type')
plt.ylabel('Average Congestion')
plt.xticks(rotation=45)
plt.ylim(0, 1)

# PLOT 3: Effect of incidents on congestion (Objective 1 & 2)
plt.subplot(2, 3, 3)
incident_congestion = traffic_data.groupby('incident')['congestion'].mean()
sns.barplot(x=['No Incident', 'Incident'], y=incident_congestion.values, palette='coolwarm')
plt.title('Effect of Incidents on Congestion', fontweight='bold')
plt.xlabel('Incident Status')
plt.ylabel('Average Congestion')
plt.ylim(0, 1)

# ========================== MACHINE LEARNING ANALYSIS ==========================

# Select features for prediction
features = [
    'hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 
    'volume', 'speed', 'precipitation', 'incident', 
    'road_lanes', 'road_importance'
]

# Create feature matrix X and target vector y
X = traffic_data[features]
y = traffic_data['congestion_class']

# Add road_type as a categorical feature
X_with_cat = pd.concat([
    X, 
    pd.get_dummies(traffic_data['road_type'], prefix='road_type')
], axis=1)

# Split the data into training and testing sets (70-30 split)
X_train, X_test, y_train, y_test = train_test_split(
    X_with_cat, y, test_size=0.3, random_state=42, stratify=y
)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PLOT 4: k-NN Varying number of neighbors (Objective 1)
# Range of k values to test
k_values = range(1, 9)  # Testing k from 1 to 8

# Lists to store training and testing accuracy
train_accuracy = []
test_accuracy = []

# Train and evaluate k-NN models with different values of k
print("Training k-NN models with different numbers of neighbors...")
for k in k_values:
    # Create and train k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    # Calculate training accuracy
    y_train_pred = knn.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_accuracy.append(train_acc)
    
    # Calculate testing accuracy
    y_test_pred = knn.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_accuracy.append(test_acc)
    
    print(f"k = {k}: Training Accuracy = {train_acc:.4f}, Testing Accuracy = {test_acc:.4f}")

# Plot k-NN results
plt.subplot(2, 3, 4)
plt.plot(k_values, test_accuracy, 'b-', linewidth=2.5, label='Testing Accuracy')
plt.plot(k_values, train_accuracy, color='#FF8C00', linewidth=2.5, label='Training Accuracy')

# Customize plot appearance
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('k-NN Varying Number of Neighbors', fontweight='bold')
plt.xticks(k_values)
plt.legend(frameon=True, facecolor='white', edgecolor='gray')
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0.65, 1.01)  # Match the y-axis scale with the reference image

# Get best k value
best_k = k_values[np.argmax(test_accuracy)]
max_test_acc = max(test_accuracy)

# ========================== FEATURE IMPORTANCE ANALYSIS ==========================

# PLOT 5: Feature importance using Random Forest (Objective 1 & 2)
plt.subplot(2, 3, 5)
# Train Random Forest for feature importance
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

# Plot top 10 features
top_features = feature_importance.head(10)
sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
plt.title('Top 10 Features for Congestion Prediction', fontweight='bold')
plt.xlabel('Importance')
plt.tight_layout()

# ========================== CONGESTION OPTIMIZATION ANALYSIS ==========================

# PLOT 6: Congestion reduction potential (Objective 2)
plt.subplot(2, 3, 6)

# Calculate average congestion by road type and number of lanes
congestion_by_type_lanes = traffic_data.groupby(['road_type', 'road_lanes'])['congestion'].mean().reset_index()
congestion_pivot = congestion_by_type_lanes.pivot(index='road_type', columns='road_lanes', values='congestion')

# Plot heatmap for optimization potential
sns.heatmap(congestion_pivot, annot=True, cmap='YlOrRd', fmt='.2f', cbar_kws={'label': 'Average Congestion'})
plt.title('Congestion by Road Type and Number of Lanes', fontweight='bold')
plt.xlabel('Number of Lanes')
plt.ylabel('Road Type')

# Save the multi-plot figure
plt.tight_layout()
plt.savefig('traffic_congestion_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# ========================== MODEL EVALUATION DETAILS ==========================

# Train the best k-NN model
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_scaled, y_train)
y_pred = best_knn.predict(X_test_scaled)
y_pred_proba = best_knn.predict_proba(X_test_scaled)[:, 1]

# Create confusion matrix visualization
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for k-NN Traffic Congestion Prediction', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.xticks([0.5, 1.5], ['Low Congestion', 'High Congestion'], fontsize=12)
plt.yticks([0.5, 1.5], ['Low Congestion', 'High Congestion'], fontsize=12, rotation=90, va='center')
plt.tight_layout()
plt.savefig('congestion_confusion_matrix.png', dpi=300)
plt.close()

# Create ROC curve
plt.figure(figsize=(8, 8))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve for Congestion Prediction', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('congestion_roc_curve.png', dpi=300)
plt.close()

# ========================== FINAL SUMMARY ==========================

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Low Congestion', 'High Congestion']))

print("\nROC AUC Score:", roc_auc)

print("\nTop 5 Important Features for Congestion Prediction:")
for i, (feature, importance) in enumerate(zip(top_features['Feature'].head(5), top_features['Importance'].head(5))):
    print(f"{i+1}. {feature}: {importance:.4f}")

print("\nCongestion Patterns Summary:")
print(f"- Highest congestion road type: {road_congestion.index[0]} (avg: {road_congestion.values[0]:.2f})")
print(f"- Lowest congestion road type: {road_congestion.index[-1]} (avg: {road_congestion.values[-1]:.2f})")
print(f"- Congestion increase during incidents: {(incident_congestion[1] - incident_congestion[0]) / incident_congestion[0]:.1%}")
print(f"- Peak congestion hour: {hourly_congestion.idxmax()} (avg: {hourly_congestion.max():.2f})")
print(f"- Lowest congestion hour: {hourly_congestion.idxmin()} (avg: {hourly_congestion.min():.2f})")

print("\nSmart Traffic Congestion Control System - Findings Summary:")
print("=" * 80)
print("1. OBJECTIVE 1: Machine Learning for Congestion Prediction")
print(f"   - Best model: k-NN with {best_k} neighbors achieving {max_test_acc:.2%} accuracy")
print(f"   - Most predictive features: {', '.join(top_features['Feature'].head(3))}")
print(f"   - High congestion prediction precision: {classification_report(y_test, y_pred, output_dict=True)['1']['precision']:.2f}")
print()
print("2. OBJECTIVE 2: Knowledge Representation & Optimization")
print(f"   - Critical congestion factors: {', '.join(top_features['Feature'].head(5))}")
print(f"   - Optimal traffic management for {road_congestion.index[0]} roads should be prioritized")
print(f"   - Incident management could reduce congestion by up to {(incident_congestion[1] - incident_congestion[0]) / incident_congestion[0]:.1%}")
print(f"   - Time-based signal optimization should focus on hour {hourly_congestion.idxmax()}")
print("=" * 80)

print("\nMultiple professional visualizations have been saved:")
print("1. traffic_congestion_analysis.png - Comprehensive multi-plot analysis")
print("2. congestion_confusion_matrix.png - Detailed model evaluation") 
print("3. congestion_roc_curve.png - Model performance curve")
print("\nThese visualizations directly address the research objectives by demonstrating")
print("both the machine learning prediction capabilities and the knowledge-based optimization")
print("potential of the Smart Traffic Congestion Control System.") 