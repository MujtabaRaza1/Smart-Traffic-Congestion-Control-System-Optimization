#!/usr/bin/env python3
"""
k-NN Model Analysis for Traffic Congestion Prediction

This script implements a k-NN classifier for predicting traffic congestion
using the synthetic traffic data. It tests different values of k (number of neighbors)
and plots the training and testing accuracy for each value.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the synthetic traffic data
print("Loading traffic data...")
traffic_data = pd.read_csv('data/synthetic_traffic_data.csv')

# Display basic information about the dataset
print(f"Dataset shape: {traffic_data.shape}")
print(f"Columns: {traffic_data.columns.tolist()}")

# Convert congestion to binary classes (high/low congestion) for classification
# Let's consider congestion > 0.7 as high congestion
traffic_data['congestion_class'] = (traffic_data['congestion'] > 0.7).astype(int)

# Select features for prediction
features = [
    'hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 
    'volume', 'speed', 'precipitation'
]

# Create feature matrix X and target vector y
X = traffic_data[features]
y = traffic_data['congestion_class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(k_values, test_accuracy, 'b-', label='Testing Accuracy')
plt.plot(k_values, train_accuracy, 'orange', label='Training accuracy')
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.title('k-NN Varying number of neighbors')
plt.legend()
plt.grid(False)
plt.ylim(0.65, 1.01)  # Match the y-axis scale with the reference image
plt.savefig('knn_accuracy_plot.png')
plt.show()

# Analyze best k value
best_k = k_values[np.argmax(test_accuracy)]
print(f"\nBest k value based on testing accuracy: {best_k}")
print(f"Best testing accuracy: {max(test_accuracy):.4f}")

# Feature importance analysis (via correlation with congestion class)
print("\nFeature importance (correlation with congestion class):")
correlation = X.corrwith(traffic_data['congestion_class']).sort_values(ascending=False)
for feature, corr in correlation.items():
    print(f"{feature}: {corr:.4f}")

print("\nDataset Analysis Summary:")
print(f"Total samples: {len(traffic_data)}")
print(f"High congestion samples: {sum(y)}")
print(f"Low congestion samples: {len(y) - sum(y)}")
print(f"Class balance: {sum(y)/len(y):.2%} high congestion") 