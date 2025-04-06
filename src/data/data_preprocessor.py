"""
Data preprocessing module for Chicago crime data
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

class CrimeDataPreprocessor:
    """
    Class to preprocess Chicago crime data for analysis and modeling
    """
    
    def __init__(self):
        """Initialize the preprocessor"""
        self.scaler = None
    
    def preprocess_classification_data(self, df, target='arrest', test_size=0.2, random_state=42):
        """
        Preprocess data for classification tasks (e.g., predicting arrests)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Raw crime data
        target : str
            Target variable for classification
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test, feature_names)
        """
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Convert date to datetime
        if 'date' in df_copy.columns:
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            
            # Extract datetime features
            df_copy['hour'] = df_copy['date'].dt.hour
            df_copy['day_of_week'] = df_copy['date'].dt.day_name()
            df_copy['month'] = df_copy['date'].dt.month
            df_copy['year'] = df_copy['date'].dt.year
            df_copy['day'] = df_copy['date'].dt.day
            df_copy['is_weekend'] = df_copy['date'].dt.dayofweek >= 5
        
        # Convert coordinates to float
        for col in ['latitude', 'longitude', 'x_coordinate', 'y_coordinate']:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        
        # Drop records with missing coordinates
        if 'latitude' in df_copy.columns and 'longitude' in df_copy.columns:
            df_copy = df_copy.dropna(subset=['latitude', 'longitude'])
        
        # Convert target variable to binary
        if target in df_copy.columns:
            df_copy[target] = df_copy[target].map({'true': 1, 'false': 0, True: 1, False: 0})
        
        # One-hot encode categorical variables
        categorical_cols = ['day_of_week', 'location_description', 'district', 'beat']
        categorical_cols = [col for col in categorical_cols if col in df_copy.columns]
        
        if categorical_cols:
            df_encoded = pd.get_dummies(df_copy, columns=categorical_cols, drop_first=True)
        else:
            df_encoded = df_copy.copy()
        
        # Select features and target
        drop_cols = ['id', 'case_number', 'date', 'block', 'iucr', 'fbi_code', 
                     'updated_on', 'primary_type', 'description', 'location']
        drop_cols = [col for col in drop_cols if col in df_encoded.columns]
        
        if target in df_encoded.columns:
            X = df_encoded.drop(columns=drop_cols + [target])
            y = df_encoded[target]
        else:
            raise ValueError(f"Target column '{target}' not found in the dataset")
        
        # Get feature names
        feature_names = X.columns.tolist()
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Handle missing values - separately for numeric and non-numeric columns
        numeric_cols = X_train.select_dtypes(include=['number']).columns
        non_numeric_cols = X_train.select_dtypes(exclude=['number']).columns
        
        # For numeric columns, fill with median
        if not numeric_cols.empty:
            X_train[numeric_cols] = X_train[numeric_cols].fillna(X_train[numeric_cols].median())
            X_test[numeric_cols] = X_test[numeric_cols].fillna(X_train[numeric_cols].median())
        
        # For non-numeric columns, fill with most frequent value (mode)
        if not non_numeric_cols.empty:
            for col in non_numeric_cols:
                most_frequent = X_train[col].mode()[0]
                X_train[col] = X_train[col].fillna(most_frequent)
                X_test[col] = X_test[col].fillna(most_frequent)
        
        # Scale numerical features
        self.scaler = StandardScaler()
        X_train[numeric_cols] = pd.DataFrame(self.scaler.fit_transform(X_train[numeric_cols]), 
                                            columns=numeric_cols, index=X_train.index)
        X_test[numeric_cols] = pd.DataFrame(self.scaler.transform(X_test[numeric_cols]), 
                                            columns=numeric_cols, index=X_test.index)
        
        return X_train, X_test, y_train, y_test, feature_names
    
    def preprocess_time_series_data(self, df, freq='D', seq_length=8):
        """
        Preprocess data for time series tasks (e.g., predicting crime counts)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Raw crime data
        freq : str
            Frequency for resampling ('D' for daily, 'W' for weekly, 'M' for monthly)
        seq_length : int
            Sequence length for time series prediction
            
        Returns:
        --------
        dict
            Dictionary containing preprocessed time series data
        """
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Convert date to datetime
        if 'date' in df_copy.columns:
            df_copy['date'] = pd.to_datetime(df_copy['date'])
        else:
            raise ValueError("'date' column not found in the dataset")
        
        # Resample data to get crime counts at specified frequency
        df_copy = df_copy.set_index('date')
        crime_counts = df_copy.resample(freq).size().reset_index(name='crime_count')
        
        # Ensure data is sorted by date
        crime_counts = crime_counts.sort_values('date')
        
        # Create sequences for time series prediction
        X, y = [], []
        for i in range(len(crime_counts) - seq_length):
            X.append(crime_counts['crime_count'].values[i:i+seq_length])
            y.append(crime_counts['crime_count'].values[i+seq_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Normalize data
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
        y_scaled = self.scaler.transform(y.reshape(-1, 1)).flatten()
        
        # Split into train and test sets (80% train, 20% test)
        train_size = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
        y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]
        
        # Reshape for LSTM input [samples, time steps, features]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': self.scaler,
            'dates': crime_counts['date'].values[seq_length:],
            'original_data': crime_counts
        }


if __name__ == "__main__":
    import os
    import glob
    
    # Find the most recent data file
    data_dir = 'data'
    data_files = glob.glob(os.path.join(data_dir, "chicago_theft_data_*.csv"))
    
    if data_files:
        latest_file = max(data_files, key=os.path.getctime)
        print(f"Loading data from {latest_file}")
        
        df = pd.read_csv(latest_file)
        
        preprocessor = CrimeDataPreprocessor()
        
        # Preprocess for classification
        X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_classification_data(df)
        print("\nClassification data:")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"Number of features: {len(feature_names)}")
        
        # Preprocess for time series
        ts_data = preprocessor.preprocess_time_series_data(df, freq='W')
        print("\nTime series data:")
        print(f"X_train shape: {ts_data['X_train'].shape}")
        print(f"X_test shape: {ts_data['X_test'].shape}")
    else:
        print(f"No data files found in {data_dir}")