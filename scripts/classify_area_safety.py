#!/usr/bin/env python
"""
Script to classify Chicago areas as safe or unsafe based on crime data
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_loader import ChicagoCrimeDataLoader
from src.models.area_safety_classifier import AreaSafetyClassifier
from src.visualization.visualize import CrimeDataVisualization

def main():
    parser = argparse.ArgumentParser(description='Classify Chicago areas as safe or unsafe based on crime data')
    parser.add_argument('--data_file', type=str, help='Path to the crime data file')
    parser.add_argument('--fetch_data', action='store_true', 
                        help='Fetch new data from Chicago Data Portal')
    parser.add_argument('--limit', type=int, default=50000,
                        help='Number of records to fetch (used with --fetch_data)')
    parser.add_argument('--threshold', type=int, default=75, 
                       help='Percentile threshold for determining unsafe areas (default: 75)')
    parser.add_argument('--output_dir', type=str, default='reports/figures',
                       help='Directory to save output files')
    parser.add_argument('--eval_location', type=str, default='41.8781,-87.6298',
                       help='Location to evaluate (format: lat,lon)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load or fetch data
    if args.fetch_data:
        print(f"Fetching {args.limit} theft records from Chicago Data Portal...")
        loader = ChicagoCrimeDataLoader()
        df = loader.fetch_theft_data(limit=args.limit)
    elif args.data_file:
        print(f"Loading data from {args.data_file}")
        df = pd.read_csv(args.data_file)
    else:
        # Find the most recent data file
        data_dir = 'data'
        data_files = glob.glob(os.path.join(data_dir, "chicago_theft_data_*.csv"))
        
        if not data_files:
            raise FileNotFoundError(f"No data files found in {data_dir}. Use --fetch_data to download new data.")
        
        data_file = max(data_files, key=os.path.getctime)
        print(f"Loading data from {data_file}")
        df = pd.read_csv(data_file)
    
    # Convert date to datetime if needed
    if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Check if coordinates are available
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        print("Error: Data must include 'latitude' and 'longitude' columns for spatial analysis.")
        sys.exit(1)
    
    # Handle missing coordinates
    df = df.dropna(subset=['latitude', 'longitude'])
    print(f"Working with {len(df)} records with valid coordinates.")
    
    # Initialize and train the model
    print(f"Training area safety classifier with {args.threshold}th percentile threshold...")
    safety_classifier = AreaSafetyClassifier(threshold_percentile=args.threshold)
    safety_classifier.train(df)
    
    # Create safety map
    print("Creating safety map...")
    map_file = os.path.join(args.output_dir, 'chicago_safety_map.html')
    safety_classifier.create_safety_map(output_file=map_file)
    
    # Plot feature importance
    print("Creating feature importance plot...")
    importance_file = os.path.join(args.output_dir, 'safety_feature_importance.png')
    safety_classifier.plot_feature_importance(save_path=importance_file)
    
    # Save model
    print("Saving model...")
    model_file = safety_classifier.save()
    print(f"Model saved to {model_file}")
    
    # Evaluate specified location
    try:
        lat, lon = map(float, args.eval_location.split(','))
        safety, probability = safety_classifier.predict_area_safety(lat, lon)
        
        print(f"\nSafety prediction for location ({lat}, {lon}):")
        print(f"Classification: {'Unsafe' if safety == 1 else 'Safe'}")
        print(f"Confidence: {probability:.2f}")
    except ValueError:
        print(f"Invalid location format: {args.eval_location}. Use format 'latitude,longitude'")
    
    print("\nArea safety classification complete!")
    print(f"Safety map saved to: {map_file}")
    print(f"Feature importance plot saved to: {importance_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())