import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

class TrafficDataProcessor:
    """
    Class for preprocessing traffic data for machine learning models.
    
    Handles cleaning, feature engineering, normalization, and train-test splitting.
    """
    def __init__(self):
        self.preprocessor = None
        self.cat_features = ['road_type', 'is_weekend', 'is_rush_hour']
        self.num_features = ['hour', 'day_of_week', 'road_lanes', 'road_importance', 
                            'volume', 'precipitation']
    
    def fit_transform(self, df, save_processor=False, save_path=None):
        """
        Fit the preprocessing pipeline and transform the data.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input traffic data
        save_processor : bool
            Whether to save the fitted preprocessor
        save_path : str, optional
            Path to save the processed data
            
        Returns:
        --------
        pandas.DataFrame
            Processed DataFrame ready for model training
        """
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # Add time-based features
        data = self._add_temporal_features(data)
        
        # Create preprocessing pipeline
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, self.num_features),
                ('cat', cat_transformer, self.cat_features)
            ]
        )
        
        # Fit and transform the data
        features_processed = self.preprocessor.fit_transform(data[self.num_features + self.cat_features])
        
        # Get feature names after one-hot encoding
        cat_feature_names = self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.cat_features)
        feature_names = np.concatenate([self.num_features, cat_feature_names])
        
        # Create processed DataFrame
        processed_df = pd.DataFrame(
            features_processed, 
            columns=feature_names,
            index=data.index
        )
        
        # Add target variables back
        processed_df['congestion'] = data['congestion']
        processed_df['speed'] = data['speed']
        processed_df['incident'] = data['incident']
        
        # Add identifiers
        processed_df['timestamp'] = data['timestamp']
        processed_df['road_id'] = data['road_id']
        
        # Save processed data if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            processed_df.to_csv(save_path, index=False)
            print(f"Processed data saved to {save_path}")
        
        return processed_df
    
    def transform(self, df, save_path=None):
        """
        Transform data using the fitted preprocessor.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input traffic data
        save_path : str, optional
            Path to save the processed data
            
        Returns:
        --------
        pandas.DataFrame
            Processed DataFrame ready for model training
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted. Call fit_transform first.")
        
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # Add time-based features
        data = self._add_temporal_features(data)
        
        # Transform the data
        features_processed = self.preprocessor.transform(data[self.num_features + self.cat_features])
        
        # Get feature names after one-hot encoding
        cat_feature_names = self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.cat_features)
        feature_names = np.concatenate([self.num_features, cat_feature_names])
        
        # Create processed DataFrame
        processed_df = pd.DataFrame(
            features_processed, 
            columns=feature_names,
            index=data.index
        )
        
        # Add target variables back if they exist
        for col in ['congestion', 'speed', 'incident']:
            if col in data.columns:
                processed_df[col] = data[col]
        
        # Add identifiers
        processed_df['timestamp'] = data['timestamp']
        processed_df['road_id'] = data['road_id']
        
        # Save processed data if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            processed_df.to_csv(save_path, index=False)
            print(f"Processed data saved to {save_path}")
        
        return processed_df
    
    def _add_temporal_features(self, df):
        """
        Add additional temporal features to the dataset.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input traffic data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with additional features
        """
        data = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Extract more temporal features
        data['month'] = data['timestamp'].dt.month
        data['day_of_month'] = data['timestamp'].dt.day
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        # Add to numerical features list
        self.num_features.extend(['month', 'day_of_month', 'hour_sin', 'hour_cos', 
                                 'day_of_week_sin', 'day_of_week_cos'])
        
        return data
    
    def train_test_split(self, df, test_size=0.2, by_time=True):
        """
        Split the data into training and testing sets.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input processed data
        test_size : float
            Proportion of the data to include in test split
        by_time : bool
            Whether to split by time (last portion) or randomly
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test) split
        """
        if by_time:
            # Sort by timestamp
            data = df.sort_values('timestamp')
            
            # Split by time
            train_size = int(len(data) * (1 - test_size))
            train_data = data.iloc[:train_size]
            test_data = data.iloc[train_size:]
        else:
            # Random split
            train_data = df.sample(frac=(1-test_size), random_state=42)
            test_data = df.drop(train_data.index)
        
        # Define target variables
        target_cols = ['congestion', 'speed', 'incident']
        feature_cols = [col for col in train_data.columns if col not in target_cols + ['timestamp', 'road_id']]
        
        # Create train/test splits
        X_train = train_data[feature_cols]
        y_train = train_data[target_cols]
        X_test = test_data[feature_cols]
        y_test = test_data[target_cols]
        
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    # Get the path to the synthetic data
    data_dir = Path("../data")
    traffic_file = data_dir / "synthetic_traffic_data.csv"
    
    if traffic_file.exists():
        # Load the synthetic traffic data
        traffic_data = pd.read_csv(traffic_file)
        
        # Create a data processor
        processor = TrafficDataProcessor()
        
        # Process the data
        processed_data = processor.fit_transform(
            traffic_data,
            save_path=str(data_dir / "processed_traffic_data.csv")
        )
        
        # Create train/test split
        X_train, X_test, y_train, y_test = processor.train_test_split(processed_data)
        
        print(f"Processed data shape: {processed_data.shape}")
        print(f"Training features shape: {X_train.shape}")
        print(f"Testing features shape: {X_test.shape}")
        print("\nSample processed data:")
        print(processed_data.head())
    else:
        print(f"Error: Could not find traffic data file at {traffic_file}")
        print("Please run the synthetic_data_generator.py script first.") 