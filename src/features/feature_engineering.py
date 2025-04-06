"""
Feature engineering for Chicago crime data
"""
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

class CrimeFeatureEngineering:
    """
    Class for creating advanced features from Chicago crime data
    """
    
    def __init__(self):
        """Initialize the feature engineering class"""
        # Known high-crime locations in Chicago (for example purposes)
        self.high_crime_locations = [
            (41.881832, -87.623177),  # Downtown Chicago
            (41.878876, -87.629839),  # Chicago Loop
            (41.867046, -87.606890),  # Near South Side
            (41.925130, -87.652130)   # Lincoln Park
        ]
    
    def add_temporal_features(self, df):
        """
        Add temporal features based on datetime
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Crime data with 'date' column
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with additional temporal features
        """
        df_copy = df.copy()
        
        if 'date' not in df_copy.columns:
            raise ValueError("'date' column not found in the dataset")
        
        # Convert date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
            df_copy['date'] = pd.to_datetime(df_copy['date'])
        
        # Basic temporal features
        df_copy['hour'] = df_copy['date'].dt.hour
        df_copy['day_of_week'] = df_copy['date'].dt.day_name()
        df_copy['day_of_week_num'] = df_copy['date'].dt.dayofweek
        df_copy['month'] = df_copy['date'].dt.month
        df_copy['year'] = df_copy['date'].dt.year
        df_copy['day'] = df_copy['date'].dt.day
        df_copy['quarter'] = df_copy['date'].dt.quarter
        
        # Time of day categories
        conditions = [
            (df_copy['hour'] >= 5) & (df_copy['hour'] < 12),
            (df_copy['hour'] >= 12) & (df_copy['hour'] < 17),
            (df_copy['hour'] >= 17) & (df_copy['hour'] < 22),
            (df_copy['hour'] >= 22) | (df_copy['hour'] < 5)
        ]
        categories = ['Morning', 'Afternoon', 'Evening', 'Night']
        df_copy['time_of_day'] = np.select(conditions, categories)
        
        # Is weekend
        df_copy['is_weekend'] = df_copy['day_of_week_num'] >= 5
        
        # Is holiday (simplified, you might want to use a proper holiday calendar)
        holidays = ['01-01', '07-04', '12-25', '12-31']  # New Year, Independence Day, Christmas, New Year's Eve
        df_copy['date_mmdd'] = df_copy['date'].dt.strftime('%m-%d')
        df_copy['is_holiday'] = df_copy['date_mmdd'].isin(holidays)
        
        # Season
        season_map = {
            1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring',
            5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'
        }
        df_copy['season'] = df_copy['month'].map(season_map)
        
        return df_copy
    
    def add_spatial_features(self, df):
        """
        Add spatial features based on geographical coordinates
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Crime data with 'latitude' and 'longitude' columns
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with additional spatial features
        """
        df_copy = df.copy()
        
        if 'latitude' not in df_copy.columns or 'longitude' not in df_copy.columns:
            raise ValueError("'latitude' or 'longitude' columns not found in the dataset")
        
        # Convert coordinates to float if they're not already
        for col in ['latitude', 'longitude']:
            if not pd.api.types.is_float_dtype(df_copy[col]):
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        
        # Drop rows with missing coordinates
        df_copy = df_copy.dropna(subset=['latitude', 'longitude'])
        
        # Distance to high-crime locations
        crime_coords = df_copy[['latitude', 'longitude']].values
        for i, loc in enumerate(self.high_crime_locations):
            distances = cdist(crime_coords, [loc], 'euclidean')
            df_copy[f'dist_to_hotspot_{i+1}'] = distances
        
        # Minimum distance to any high-crime location
        df_copy['min_dist_to_hotspot'] = df_copy[[f'dist_to_hotspot_{i+1}' for i in range(len(self.high_crime_locations))]].min(axis=1)
        
        # Create crime density features by district (if available)
        if 'district' in df_copy.columns:
            district_counts = df_copy['district'].value_counts()
            df_copy['district_crime_density'] = df_copy['district'].map(district_counts)
        
        return df_copy
    
    def add_crime_specific_features(self, df):
        """
        Add features specific to the type of crime
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Crime data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with additional crime-specific features
        """
        df_copy = df.copy()
        
        # For theft crimes, create binary feature for pocket-picking vs purse-snatching
        if 'description' in df_copy.columns:
            df_copy['is_pocket_picking'] = df_copy['description'].str.contains('POCKET-PICKING', case=False)
            df_copy['is_purse_snatching'] = df_copy['description'].str.contains('PURSE-SNATCHING', case=False)
        
        # Create features based on location description
        if 'location_description' in df_copy.columns:
            # Public transit related
            transit_keywords = ['CTA', 'PLATFORM', 'TRAIN', 'BUS', 'STATION']
            df_copy['is_transit_related'] = df_copy['location_description'].apply(
                lambda x: any(keyword in str(x).upper() for keyword in transit_keywords)
            )
            
            # Commercial location
            commercial_keywords = ['STORE', 'RESTAURANT', 'BAR', 'TAVERN', 'GROCERY', 'RETAIL']
            df_copy['is_commercial'] = df_copy['location_description'].apply(
                lambda x: any(keyword in str(x).upper() for keyword in commercial_keywords)
            )
            
            # Residential location
            residential_keywords = ['APARTMENT', 'RESIDENCE', 'HOUSE', 'BUILDING', 'DWELLING']
            df_copy['is_residential'] = df_copy['location_description'].apply(
                lambda x: any(keyword in str(x).upper() for keyword in residential_keywords)
            )
        
        return df_copy
    
    def create_all_features(self, df):
        """
        Apply all feature engineering steps
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Raw crime data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with all additional features
        """
        df_temp = self.add_temporal_features(df)
        df_spatial = self.add_spatial_features(df_temp)
        df_final = self.add_crime_specific_features(df_spatial)
        
        return df_final


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
        
        # Convert date to datetime for testing
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Apply feature engineering
        feature_eng = CrimeFeatureEngineering()
        df_featured = feature_eng.create_all_features(df)
        
        # Print info about the new features
        new_columns = set(df_featured.columns) - set(df.columns)
        print(f"\nAdded {len(new_columns)} new features:")
        for col in sorted(new_columns):
            print(f"- {col}")
        
        # Save engineered features
        output_file = os.path.join(data_dir, "chicago_theft_data_featured.csv")
        df_featured.to_csv(output_file, index=False)
        print(f"\nEnhanced dataset saved to {output_file}")
    else:
        print(f"No data files found in {data_dir}")