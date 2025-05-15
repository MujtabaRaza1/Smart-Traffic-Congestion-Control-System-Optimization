import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path

class CongestionPredictor:
    """
    Class for predicting traffic congestion using various ML models.
    
    Supports multiple regression models for predicting congestion levels.
    """
    def __init__(self, model_type='random_forest'):
        """
        Initialize the congestion predictor.
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('random_forest', 'gradient_boosting', 'linear')
        """
        self.model_type = model_type
        self.model = self._create_model(model_type)
        self.feature_importance = None
    
    def _create_model(self, model_type):
        """
        Create and return the specified model.
        
        Parameters:
        -----------
        model_type : str
            Type of model to create
            
        Returns:
        --------
        sklearn model
            The initialized model
        """
        if model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif model_type == 'linear':
            return LinearRegression()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train(self, X_train, y_train, target='congestion'):
        """
        Train the congestion prediction model.
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training features
        y_train : pandas.DataFrame
            Training targets
        target : str
            Target variable to predict ('congestion', 'speed', 'incident')
            
        Returns:
        --------
        self
            Trained model instance
        """
        print(f"Training {self.model_type} model to predict {target}...")
        
        # Extract the target variable
        y = y_train[target]
        
        # Train the model
        self.model.fit(X_train, y)
        
        # Store feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
        return self
    
    def predict(self, X):
        """
        Make predictions with the trained model.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input features
            
        Returns:
        --------
        numpy.ndarray
            Predicted values
        """
        if not hasattr(self.model, 'predict'):
            raise ValueError("Model has not been trained. Call train() first.")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test, target='congestion'):
        """
        Evaluate the model performance.
        
        Parameters:
        -----------
        X_test : pandas.DataFrame
            Test features
        y_test : pandas.DataFrame
            Test targets
        target : str
            Target variable to evaluate ('congestion', 'speed', 'incident')
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        # Extract the target variable
        y_true = y_test[target]
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        print(f"Model evaluation metrics for {target}:")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        return metrics
    
    def plot_feature_importance(self, save_path=None, top_n=20):
        """
        Plot feature importance if available.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        top_n : int
            Number of top features to display
            
        Returns:
        --------
        None
        """
        if self.feature_importance is None:
            print("Feature importance not available for this model")
            return
        
        # Plot top N features
        plt.figure(figsize=(10, 8))
        
        # Limit to top N features
        data = self.feature_importance.head(top_n)
        
        # Plot horizontal bar chart
        plt.barh(data['feature'], data['importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importance for {self.model_type.title()} Model')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
            
        Returns:
        --------
        None
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        CongestionPredictor
            Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model

if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    import sys
    
    # Add parent directory to path to import preprocessing
    sys.path.append(str(Path(__file__).parent.parent))
    from preprocessing.data_processor import TrafficDataProcessor
    
    # Get the path to the synthetic data
    data_dir = Path(__file__).parent.parent / "data"
    traffic_file = data_dir / "synthetic_traffic_data.csv"
    
    if traffic_file.exists():
        # Load the synthetic traffic data
        traffic_data = pd.read_csv(traffic_file)
        
        # Process the data
        processor = TrafficDataProcessor()
        processed_data = processor.fit_transform(traffic_data)
        
        # Split the data
        X_train, X_test, y_train, y_test = processor.train_test_split(processed_data)
        
        # Create and train the model
        model = CongestionPredictor(model_type='random_forest')
        model.train(X_train, y_train, target='congestion')
        
        # Evaluate the model
        metrics = model.evaluate(X_test, y_test, target='congestion')
        
        # Plot feature importance
        model.plot_feature_importance(
            save_path=str(Path(__file__).parent / "outputs" / "feature_importance.png"),
            top_n=15
        )
        
        # Save the model
        model.save_model(str(Path(__file__).parent / "saved_models" / "congestion_rf_model.pkl"))
    else:
        print(f"Error: Could not find traffic data file at {traffic_file}")
        print("Please run the synthetic_data_generator.py script first.") 