"""
Data loader module to fetch crime data from Chicago Data Portal
"""
import argparse
import json
import os
import requests
import pandas as pd
from datetime import datetime

class ChicagoCrimeDataLoader:
    """
    Class to load Chicago crime data from the Chicago Data Portal API
    """
    
    def __init__(self, data_dir='data'):
        """
        Initialize the data loader
        
        Parameters:
        -----------
        data_dir : str
            Directory to save the downloaded data
        """
        self.base_url = "https://data.cityofchicago.org/resource/crimes.json"
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    def fetch_theft_data(self, limit=70000, save=True):
        """
        Fetch theft crime data (pocket-picking and purse-snatching)
        
        Parameters:
        -----------
        limit : int
            Number of records to fetch
        save : bool
            Whether to save the data to a file
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the fetched crime data
        """
        # Parameters to filter for theft crimes
        params = {
            "$where": "primary_type='THEFT' AND (description LIKE '%POCKET-PICKING%' OR description LIKE '%PURSE-SNATCHING%')",
            "$order": "date DESC",
            "$limit": limit
        }
        
        print(f"Fetching {limit} theft records from Chicago Data Portal...")
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            data = response.json()
            print(f"Successfully fetched {len(data)} records")
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.data_dir, f"chicago_theft_data_{timestamp}.csv")
                df = df.dropna(subset=['latitude', 'longitude'])
                df.to_csv(filename, index=False)
                print(f"Data saved to {filename}")
            
            return df
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None
    
    def load_data_from_file(self, filename):
        """
        Load data from a CSV file
        
        Parameters:
        -----------
        filename : str
            Path to the CSV file
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the loaded data
        """
        try:
            return pd.read_csv(filename)
        except Exception as e:
            print(f"Error loading data from {filename}: {e}")
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch Chicago theft crime data')
    parser.add_argument('--limit', type=int, default=5000, help='Number of records to fetch')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to save data')
    args = parser.parse_args()
    
    loader = ChicagoCrimeDataLoader(data_dir=args.data_dir)
    df = loader.fetch_theft_data(limit=args.limit)
    
    if df is not None:
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {', '.join(df.columns)}")