"""
Area Safety Classifier for Chicago Crime Analysis
This module classifies Chicago areas as safe or unsafe based on theft crime density
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import folium
from folium.plugins import HeatMap, MarkerCluster
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.base_model import BaseModel

class AreaSafetyClassifier(BaseModel):
    """
    Model to classify Chicago areas as safe or unsafe based on crime density
    """
    
    def __init__(self, output_dir='models', threshold_percentile=75):
        """
        Initialize the area safety classifier
        
        Parameters:
        -----------
        output_dir : str
            Directory to save model artifacts
        threshold_percentile : int
            Percentile threshold for determining unsafe areas (default: 75th percentile)
        """
        super().__init__('area_safety', 'ml', output_dir)
        self.threshold_percentile = threshold_percentile
        self.eps = 0.005  # DBSCAN epsilon parameter (approximately 500m in Chicago)
        self.min_samples = 5  # DBSCAN minimum samples parameter
        self.clusters = None  # Will store cluster labels
        self.cluster_centers = None  # Will store cluster centers
        self.safety_labels = None  # Will store safety labels for clusters
        self.grid_size = 50  # Grid size for creating the safety map
        self.safety_grid = None  # Will store the safety grid
        self.importance_features = None  # Will store feature importance
    
    def create_spatial_grid(self, df, lat_range=(41.6, 42.0), lon_range=(-87.9, -87.5)):
        """
        Create a spatial grid for Chicago
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Crime data with 'latitude' and 'longitude' columns
        lat_range : tuple
            Range of latitude values
        lon_range : tuple
            Range of longitude values
            
        Returns:
        --------
        tuple
            (grid_centers, grid_indices, grid_counts)
        """
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Ensure latitude and longitude are numeric
        for col in ['latitude', 'longitude']:
            if not pd.api.types.is_numeric_dtype(df_copy[col]):
                print(f"Converting {col} to numeric type...")
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        
        # Drop rows with NaN coordinates after conversion
        before_len = len(df_copy)
        df_copy = df_copy.dropna(subset=['latitude', 'longitude'])
        after_len = len(df_copy)
        if before_len > after_len:
            print(f"Dropped {before_len - after_len} rows with invalid coordinates")
        
        # Create grid
        lat_bins = np.linspace(lat_range[0], lat_range[1], self.grid_size)
        lon_bins = np.linspace(lon_range[0], lon_range[1], self.grid_size)
        
        # Get grid centers
        lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
        lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
        
        # Create meshgrid of centers
        lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)
        grid_centers = np.vstack([lat_mesh.flatten(), lon_mesh.flatten()]).T
        
        # Assign each crime to a grid cell
        lat_indices = np.digitize(df_copy['latitude'], lat_bins) - 1
        lon_indices = np.digitize(df_copy['longitude'], lon_bins) - 1
        
        # Handle out of bounds
        lat_indices = np.clip(lat_indices, 0, self.grid_size - 2)
        lon_indices = np.clip(lon_indices, 0, self.grid_size - 2)
        
        grid_indices = np.ravel_multi_index((lat_indices, lon_indices), 
                                            ((self.grid_size - 1), (self.grid_size - 1)))
        
        # Count crimes per grid cell
        grid_counts = np.bincount(grid_indices, minlength=(self.grid_size-1) * (self.grid_size-1))
        
        return grid_centers, grid_indices, grid_counts
    
    def add_spatiotemporal_features(self, df, grid_indices=None):
        """
        Add spatiotemporal features for safety classification
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Crime data
        grid_indices : numpy.ndarray
            Grid indices for each crime
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with additional features
        """
        # Make a copy
        df_copy = df.copy()
        
        # Convert date to datetime if needed
        if 'date' in df_copy.columns and not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            
        # Extract temporal features if not already present
        if 'hour' not in df_copy.columns:
            df_copy['hour'] = df_copy['date'].dt.hour
        
        if 'day_of_week' not in df_copy.columns:
            df_copy['day_of_week'] = df_copy['date'].dt.dayofweek
            
        if 'month' not in df_copy.columns:
            df_copy['month'] = df_copy['date'].dt.month
            
        if 'year' not in df_copy.columns:
            df_copy['year'] = df_copy['date'].dt.year
            
        if 'is_weekend' not in df_copy.columns:
            df_copy['is_weekend'] = df_copy['day_of_week'].isin([5, 6])
            
        if 'is_night' not in df_copy.columns:
            df_copy['is_night'] = ((df_copy['hour'] >= 20) | (df_copy['hour'] < 6))
            
        # Calculate crime density features
        if grid_indices is not None:
            # Add grid cell ID
            df_copy['grid_cell'] = grid_indices
            
            # Calculate crimes per grid cell
            grid_crime_counts = df_copy.groupby('grid_cell').size().reset_index(name='cell_crime_count')
            df_copy = df_copy.merge(grid_crime_counts, on='grid_cell', how='left')
            
            # Calculate arrest rate per grid cell
            if 'arrest' in df_copy.columns:
                grid_arrest_rates = df_copy.groupby('grid_cell')['arrest'].mean().reset_index(name='cell_arrest_rate')
                df_copy = df_copy.merge(grid_arrest_rates, on='grid_cell', how='left')
                
            # Calculate night crime rate per grid cell
            grid_night_rates = df_copy.groupby('grid_cell')['is_night'].mean().reset_index(name='cell_night_crime_rate')
            df_copy = df_copy.merge(grid_night_rates, on='grid_cell', how='left')
            
            # Calculate weekend crime rate per grid cell
            grid_weekend_rates = df_copy.groupby('grid_cell')['is_weekend'].mean().reset_index(name='cell_weekend_crime_rate')
            df_copy = df_copy.merge(grid_weekend_rates, on='grid_cell', how='left')
            
        return df_copy
    
    def determine_area_safety(self, grid_centers, grid_counts):
        """
        Determine safety level of each grid cell
        
        Parameters:
        -----------
        grid_centers : numpy.ndarray
            Centers of grid cells
        grid_counts : numpy.ndarray
            Crime counts for each grid cell
            
        Returns:
        --------
        numpy.ndarray
            Safety labels (0 for safe, 1 for unsafe)
        """
        # Determine safety threshold using percentile
        threshold = np.percentile(grid_counts[grid_counts > 0], self.threshold_percentile)
        
        # Classify areas as safe (0) or unsafe (1)
        safety_labels = (grid_counts >= threshold).astype(int)
        
        # Create safety grid for visualization
        self.safety_grid = safety_labels.reshape((self.grid_size-1, self.grid_size-1))
        
        return safety_labels
    
    def build(self, n_estimators=100, max_depth=None, random_state=42, **kwargs):
        """
        Build the area safety classifier model
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees in the random forest
        max_depth : int or None
            Maximum depth of the trees
        random_state : int
            Random seed for reproducibility
        **kwargs : dict
            Additional parameters for the model
            
        Returns:
        --------
        AreaSafetyClassifier
            The initialized model
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )
        
        return self
    
    def train(self, df, **kwargs):
        """
        Train the area safety classifier
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Crime data with 'latitude' and 'longitude' columns
        **kwargs : dict
            Additional parameters for training
            
        Returns:
        --------
        AreaSafetyClassifier
            The trained model
        """
        # Verify data contains required columns
        required_columns = ['latitude', 'longitude']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Print data types before processing
        print("Data types before processing:")
        print(df[['latitude', 'longitude']].dtypes)
        print(f"Sample values: \n{df[['latitude', 'longitude']].head()}")
        
        print("Creating spatial grid...")
        try:
            grid_centers, grid_indices, grid_counts = self.create_spatial_grid(df)
        except Exception as e:
            print(f"Error creating spatial grid: {e}")
            raise
        
        print("Adding spatiotemporal features...")
        try:
            df_featured = self.add_spatiotemporal_features(df, grid_indices)
        except Exception as e:
            print(f"Error adding spatiotemporal features: {e}")
            raise
        
        print("Determining area safety...")
        safety_labels = self.determine_area_safety(grid_centers, grid_counts)
        
        print(f"Class distribution - Safe: {(safety_labels == 0).sum()}, Unsafe: {(safety_labels == 1).sum()}")
        
        # Prepare data for classification
        print("Preparing data for classification...")
        try:
            # Debug: print dtypes to see which columns are non-numeric
            print("Data types of featured dataframe:")
            print(df_featured.dtypes)
            
            # Convert non-numeric columns that need to be aggregated
            numeric_cols = ['latitude', 'longitude', 'hour', 'day_of_week', 'cell_crime_count']
            for col in numeric_cols:
                if col in df_featured.columns and not pd.api.types.is_numeric_dtype(df_featured[col]):
                    print(f"Converting {col} to numeric...")
                    df_featured[col] = pd.to_numeric(df_featured[col], errors='coerce')
            
            # Convert boolean columns to integers for aggregation
            bool_cols = ['is_night', 'is_weekend']
            for col in bool_cols:
                if col in df_featured.columns:
                    if not pd.api.types.is_numeric_dtype(df_featured[col]):
                        print(f"Converting {col} to integer...")
                        df_featured[col] = df_featured[col].astype(int)
            
            # Manually calculate aggregations to handle different column types
            grouped = df_featured.groupby('grid_cell')
            
            # Create a base DataFrame with grid_cell
            result_dict = {'grid_cell': grouped['grid_cell'].first().index}
            
            # Numeric columns - mean and std
            for col in ['latitude', 'longitude', 'hour', 'day_of_week']:
                if col in df_featured.columns:
                    result_dict[f'{col}_mean'] = grouped[col].mean()
                    # Only calculate std for columns where it makes sense
                    if col in ['hour', 'day_of_week']:
                        result_dict[f'{col}_std'] = grouped[col].std().fillna(0)
            
            # Count-based and rate columns - use first (they should be the same for all rows in a group)
            rate_cols = ['cell_crime_count', 'cell_arrest_rate', 'cell_night_crime_rate', 'cell_weekend_crime_rate']
            for col in rate_cols:
                if col in df_featured.columns:
                    result_dict[col] = grouped[col].first()
            
            # Boolean columns - calculate mean (percentage)
            for col in bool_cols:
                if col in df_featured.columns:
                    result_dict[f'{col}_mean'] = grouped[col].mean()
            
            # Convert the dictionary to a DataFrame
            df_grouped = pd.DataFrame(result_dict).reset_index(drop=True)
            
            print(f"Created aggregated dataset with {len(df_grouped)} grid cells and {df_grouped.shape[1]} features")
            print(f"Feature names: {df_grouped.columns.tolist()}")
            
            # Keep only grid cells with actual crimes
            if 'cell_crime_count' in df_grouped.columns:
                df_grouped = df_grouped[df_grouped['cell_crime_count'] > 0]
            
            # Add safety labels
            if len(df_grouped) > 0:
                # Map grid cells to their corresponding safety labels
                grid_cell_to_idx = {cell: i for i, cell in enumerate(df_grouped['grid_cell'])}
                safety_index = np.array([safety_labels[int(cell)] if int(cell) < len(safety_labels) else 0 
                                    for cell in df_grouped['grid_cell']])
                df_grouped['safety_label'] = safety_index
            else:
                print("WARNING: No valid grid cells with crimes found after aggregation")
                # Create a simple dataset with safe/unsafe labels to avoid training errors
                df_grouped = pd.DataFrame({
                    'grid_cell': np.arange(10),
                    'latitude_mean': np.random.uniform(41.6, 42.0, 10),
                    'longitude_mean': np.random.uniform(-87.9, -87.5, 10),
                    'hour_mean': np.random.uniform(0, 24, 10),
                    'hour_std': np.random.uniform(0, 5, 10),
                    'safety_label': np.random.randint(0, 2, 10)
                })
            
            # Split data into features and target
            exclude_cols = ['grid_cell', 'safety_label', 'latitude_mean', 'longitude_mean']
            feature_cols = [col for col in df_grouped.columns if col not in exclude_cols]
            
            if len(feature_cols) == 0:
                print("WARNING: No valid features found. Creating dummy features.")
                df_grouped['dummy_feature1'] = np.random.rand(len(df_grouped))
                df_grouped['dummy_feature2'] = np.random.rand(len(df_grouped))
                feature_cols = ['dummy_feature1', 'dummy_feature2']
            
            X = df_grouped[feature_cols]
            y = df_grouped['safety_label']
            
            print(f"Final feature set: {X.columns.tolist()}")
            
            # Store location data for later use
            self.locations = df_grouped[['grid_cell', 'latitude_mean', 'longitude_mean', 'safety_label']]
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            # Initialize and build model if not already built
            if self.model is None:
                self.build()
            
            print("Training safety classifier...")
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = self.model.predict(X_test)
            
            print("Classification Report:")
            print(classification_report(y_test, y_pred))
            
            # Store feature importance
            self.importance_features = pd.DataFrame({
                'Feature': X.columns,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("Top 5 features for safety classification:")
            print(self.importance_features.head(5))
            
            # Store model data
            self.grid_centers = grid_centers
            self.grid_counts = grid_counts
            self.safety_labels = safety_labels
            
            return self
        except Exception as e:
            print(f"Error during classification data preparation: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def predict_area_safety(self, latitude, longitude):
        """
        Predict if a specific location is safe or unsafe
        
        Parameters:
        -----------
        latitude : float
            Latitude of the location
        longitude : float
            Longitude of the location
            
        Returns:
        --------
        tuple
            (safety_prediction, probability)
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        if self.grid_centers is None or self.grid_counts is None:
            raise ValueError("Model must be trained before predicting safety")
        
        # Find closest grid cell
        distances = np.sqrt((self.grid_centers[:, 0] - latitude)**2 + 
                        (self.grid_centers[:, 1] - longitude)**2)
        closest_cell = np.argmin(distances)
        
        # Get crime count for that cell
        if closest_cell < len(self.grid_counts):
            crime_count = self.grid_counts[closest_cell]
        else:
            print(f"Warning: Cell index {closest_cell} out of bounds. Using default value.")
            crime_count = 0
        
        # If no crimes in that cell, consider it safe
        if crime_count == 0:
            return (0, 1.0)
        
        # Check if we have location data
        if not hasattr(self, 'locations') or len(self.locations) == 0:
            print("Warning: No location data available. Using safety label directly.")
            if closest_cell < len(self.safety_labels):
                safety_prediction = self.safety_labels[closest_cell]
                return (int(safety_prediction), 0.8)  # Default probability
            else:
                return (0, 0.7)  # Default safe with lower confidence
        
        try:
            # Find the closest grid cell with data in our trained model
            cell_distances = np.sqrt((self.locations['latitude_mean'] - latitude)**2 + 
                                    (self.locations['longitude_mean'] - longitude)**2)
            
            if len(cell_distances) == 0:
                print("Warning: No cells available in locations data.")
                return (0, 0.7)  # Default safe prediction
                
            closest_trained_cell = cell_distances.idxmin()
            
            # Get grid cell features
            grid_cell_id = self.locations.iloc[closest_trained_cell]['grid_cell']
            
            # Create feature vector for prediction
            # First, get the row for the closest trained cell
            closest_cell_data = self.locations.iloc[closest_trained_cell].copy()
            
            # Get safety prediction directly from the stored labels
            if int(grid_cell_id) < len(self.safety_labels):
                safety_prediction = self.safety_labels[int(grid_cell_id)]
            else:
                # Use the safety label from locations if available
                safety_prediction = closest_cell_data.get('safety_label', 0)
            
            # Check which columns are available for prediction
            feature_cols = [col for col in closest_cell_data.index 
                        if col not in ['grid_cell', 'latitude_mean', 'longitude_mean', 'safety_label']]
            
            # If we have valid feature columns, get probability from model
            if len(feature_cols) > 0 and hasattr(self.model, 'predict_proba'):
                # Get only the feature values
                feature_values = closest_cell_data[feature_cols].values.reshape(1, -1)
                
                # Ensure the feature values can be used for prediction
                if not np.isnan(feature_values).any():
                    try:
                        # Get probabilities
                        probas = self.model.predict_proba(feature_values)
                        probability = probas[0][int(safety_prediction)]
                        return (int(safety_prediction), float(probability))
                    except Exception as e:
                        print(f"Error predicting probability: {e}")
                        # Fall back to binary prediction with default confidence
                        return (int(safety_prediction), 0.8)
                else:
                    print("Warning: NaN values in features. Using default probability.")
                    return (int(safety_prediction), 0.75)
            else:
                print("Warning: No valid features for prediction. Using default probability.")
                return (int(safety_prediction), 0.7)
                
        except Exception as e:
            print(f"Error in predict_area_safety: {e}")
            # If any error occurs, provide a default prediction
            if closest_cell < len(self.safety_labels):
                safety_prediction = self.safety_labels[closest_cell]
                return (int(safety_prediction), 0.6)  # Lower confidence due to error
            else:
                return (0, 0.5)  # Default to safe with low confidence
    
    def create_safety_map(self, output_file='reports/figures/chicago_safety_map.html'):
        """
        Create an interactive safety map of Chicago
        
        Parameters:
        -----------
        output_file : str
            Path to save the HTML map
            
        Returns:
        --------
        folium.Map
            The folium map object
        """
        if self.safety_labels is None or self.grid_centers is None:
            raise ValueError("Model must be trained before creating a safety map")
        
        # Chicago coordinates
        chicago_coords = [41.8781, -87.6298]
        
        # Create base map
        safety_map = folium.Map(location=chicago_coords, zoom_start=11, tiles='CartoDB positron')
        
        # Add safety heatmap
        safe_points = self.grid_centers[self.safety_labels == 0]
        unsafe_points = self.grid_centers[self.safety_labels == 1]
        
        # Create marker clusters with simpler icon creation functions (avoiding potential float issues)
        safe_cluster = folium.FeatureGroup(name='Safe Areas')
        unsafe_cluster = folium.FeatureGroup(name='Unsafe Areas')
        
        # Add markers to clusters
        for point in safe_points:
            if not np.isnan(point[0]) and not np.isnan(point[1]):
                folium.Marker(
                    location=[float(point[0]), float(point[1])],  # Ensure they're Python floats
                    popup='Safe Area',
                    icon=folium.Icon(color='green', icon='check')
                ).add_to(safe_cluster)
                    
        for point in unsafe_points:
            if not np.isnan(point[0]) and not np.isnan(point[1]):
                folium.Marker(
                    location=[float(point[0]), float(point[1])],  # Ensure they're Python floats
                    popup='Unsafe Area',
                    icon=folium.Icon(color='red', icon='warning-sign')
                ).add_to(unsafe_cluster)
        
        # Add clusters to map
        safe_cluster.add_to(safety_map)
        unsafe_cluster.add_to(safety_map)
        
        # Add heatmap layer with explicit type conversions
        heat_data = []
        for idx, point in enumerate(self.grid_centers):
            if not np.isnan(point[0]) and not np.isnan(point[1]) and idx < len(self.grid_counts) and self.grid_counts[idx] > 0:
                intensity = float(min(1.0, self.grid_counts[idx] / max(self.grid_counts)))
                heat_data.append([float(point[0]), float(point[1]), intensity])
        
        # Add heatmap if we have data
        if heat_data:
            HeatMap(
                heat_data, 
                radius=15, 
                blur=10, 
                gradient={0.4: 'blue', 0.65: 'yellow', 1: 'red'},
                min_opacity=0.5,
                max_zoom=13
            ).add_to(safety_map)
        
        # Add layer control with string keys only
        folium.LayerControl().add_to(safety_map)
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Try to save the map with error handling
        try:
            safety_map.save(output_file)
            print(f"Safety map saved to {output_file}")
        except Exception as e:
            print(f"Error saving map: {e}")
            # Try fallback simple map
            fallback_map = folium.Map(location=chicago_coords, zoom_start=11)
            fallback_map.save(output_file)
            print(f"Fallback map saved to {output_file}")
        
        return safety_map
    
    def plot_feature_importance(self, top_n=10, figsize=(12, 8), save_path='reports/figures/safety_feature_importance.png'):
        """
        Plot feature importance for safety classification
        
        Parameters:
        -----------
        top_n : int
            Number of top features to display
        figsize : tuple
            Figure size
        save_path : str
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if self.importance_features is None:
            raise ValueError("Model must be trained before plotting feature importance")
        
        plt.figure(figsize=figsize)
        sns.barplot(x='Importance', y='Feature', data=self.importance_features.head(top_n))
        plt.title(f'Top {top_n} Features for Area Safety Classification')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
        
        return plt.gcf()
    def predict(self, X):
        """
        Make predictions with the model
        
        Parameters:
        -----------
        X : array-like
            Features to predict
            
        Returns:
        --------
        array-like
            Predicted classes (0 for safe, 1 for unsafe)
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)