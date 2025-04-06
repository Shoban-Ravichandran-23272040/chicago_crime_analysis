"""
Visualization utilities for Chicago crime data
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

class CrimeDataVisualization:
    """
    Class for visualizing Chicago crime data
    """
    
    def __init__(self, output_dir='reports/figures'):
        """
        Initialize the visualization class
        
        Parameters:
        -----------
        output_dir : str
            Directory to save visualizations
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set default plot style
        sns.set(style="whitegrid")
        plt.rcParams.update({'font.size': 12})
    
    def plot_crime_over_time(self, df, time_unit='month', title=None, figsize=(12, 6), save=True):
        """
        Plot crime counts over time
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Crime data with a 'date' column
        time_unit : str
            Time unit for resampling ('day', 'week', 'month', 'year')
        title : str, optional
            Plot title
        figsize : tuple
            Figure size
        save : bool
            Whether to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Convert date to datetime if it's not already
        if 'date' in df_copy.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
                df_copy['date'] = pd.to_datetime(df_copy['date'])
        else:
            raise ValueError("'date' column not found in the dataset")
        
        # Set date as index
        df_copy = df_copy.set_index('date')
        
        # Resample based on time unit
        if time_unit == 'day':
            crime_counts = df_copy.resample('D').size()
            date_format = '%Y-%m-%d'
        elif time_unit == 'week':
            crime_counts = df_copy.resample('W').size()
            date_format = '%Y-%m-%d'
        elif time_unit == 'month':
            crime_counts = df_copy.resample('M').size()
            date_format = '%Y-%m'
        elif time_unit == 'year':
            crime_counts = df_copy.resample('Y').size()
            date_format = '%Y'
        else:
            raise ValueError("Invalid time_unit. Choose from 'day', 'week', 'month', 'year'")
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot line chart
        ax.plot(crime_counts.index, crime_counts.values, marker='o', linestyle='-')
        
        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Crimes')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Theft Crimes by {time_unit.capitalize()}')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        plt.xticks(rotation=45)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Save plot if requested
        if save:
            filename = os.path.join(self.output_dir, f'crime_over_{time_unit}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filename}")
        
        return fig
    
    def plot_crime_by_hour(self, df, figsize=(12, 6), save=True):
        """
        Plot crime counts by hour of day
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Crime data with a 'date' column
        figsize : tuple
            Figure size
        save : bool
            Whether to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Convert date to datetime and extract hour
        if 'date' in df_copy.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
                df_copy['date'] = pd.to_datetime(df_copy['date'])
            
            df_copy['hour'] = df_copy['date'].dt.hour
        else:
            raise ValueError("'date' column not found in the dataset")
        
        # Count crimes by hour
        hour_counts = df_copy.groupby('hour').size().reset_index(name='count')
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot bar chart
        sns.barplot(x='hour', y='count', data=hour_counts, ax=ax, color='steelblue')
        
        # Set labels and title
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Number of Crimes')
        ax.set_title('Theft Crimes by Hour of Day')
        
        # Set x-ticks
        ax.set_xticks(range(24))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Tight layout
        plt.tight_layout()
        
        # Save plot if requested
        if save:
            filename = os.path.join(self.output_dir, 'crime_by_hour.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filename}")
        
        return fig
    
    def plot_crime_by_day_of_week(self, df, figsize=(12, 6), save=True):
        """
        Plot crime counts by day of week
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Crime data with a 'date' column
        figsize : tuple
            Figure size
        save : bool
            Whether to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Convert date to datetime and extract day of week
        if 'date' in df_copy.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
                df_copy['date'] = pd.to_datetime(df_copy['date'])
            
            df_copy['day_of_week'] = df_copy['date'].dt.day_name()
        else:
            raise ValueError("'date' column not found in the dataset")
        
        # Count crimes by day of week
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = df_copy.groupby('day_of_week').size().reindex(days_order).reset_index(name='count')
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot bar chart
        sns.barplot(x='day_of_week', y='count', data=day_counts, ax=ax, color='steelblue')
        
        # Set labels and title
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Number of Crimes')
        ax.set_title('Theft Crimes by Day of Week')
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Tight layout
        plt.tight_layout()
        
        # Save plot if requested
        if save:
            filename = os.path.join(self.output_dir, 'crime_by_day_of_week.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filename}")
        
        return fig
    
    def plot_crime_heatmap(self, df, zoom_start=11, title='Chicago Theft Crime Heatmap', save=True):
        """
        Create a heatmap of crime locations
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Crime data with 'latitude' and 'longitude' columns
        zoom_start : int
            Initial zoom level for the map
        title : str
            Title for the HTML file
        save : bool
            Whether to save the heatmap
            
        Returns:
        --------
        folium.Map
            The folium map object
        """
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Check if latitude and longitude columns exist
        if 'latitude' not in df_copy.columns or 'longitude' not in df_copy.columns:
            raise ValueError("'latitude' or 'longitude' columns not found in the dataset")
        
        # Convert coordinates to float if they're not already
        for col in ['latitude', 'longitude']:
            if not pd.api.types.is_float_dtype(df_copy[col]):
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        
        # Drop rows with missing coordinates
        df_copy = df_copy.dropna(subset=['latitude', 'longitude'])
        
        # Chicago coordinates
        chicago_coords = [41.8781, -87.6298]
        
        # Create map
        crime_map = folium.Map(location=chicago_coords, zoom_start=zoom_start, tiles='CartoDB positron')
        
        # Add heatmap
        heat_data = df_copy[['latitude', 'longitude']].values.tolist()
        HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(crime_map)
        
        # Add title
        title_html = f"""
            <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
        """
        crime_map.get_root().html.add_child(folium.Element(title_html))
        
        # Save map if requested
        if save:
            filename = os.path.join(self.output_dir, 'crime_heatmap.html')
            crime_map.save(filename)
            print(f"Heatmap saved to {filename}")
        
        return crime_map
    
    def plot_arrest_distribution(self, df, figsize=(10, 6), save=True):
        """
        Plot the distribution of arrests vs non-arrests
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Crime data with an 'arrest' column
        figsize : tuple
            Figure size
        save : bool
            Whether to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Check if arrest column exists
        if 'arrest' not in df_copy.columns:
            raise ValueError("'arrest' column not found in the dataset")
        
        # Convert arrest to boolean if it's not already
        df_copy['arrest'] = df_copy['arrest'].map({'true': True, 'false': False, True: True, False: False})
        
        # Count arrests
        arrest_counts = df_copy['arrest'].value_counts().reset_index()
        arrest_counts.columns = ['Arrest', 'Count']
        arrest_counts['Arrest'] = arrest_counts['Arrest'].map({True: 'Arrested', False: 'Not Arrested'})
        
        # Calculate percentages
        total = arrest_counts['Count'].sum()
        arrest_counts['Percentage'] = (arrest_counts['Count'] / total * 100).round(1)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot pie chart
        wedges, texts, autotexts = ax.pie(
            arrest_counts['Count'],
            labels=arrest_counts['Arrest'],
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'edgecolor': 'w'},
            textprops={'fontsize': 12}
        )
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        
        # Set title
        ax.set_title('Arrest Distribution for Theft Crimes')
        
        # Tight layout
        plt.tight_layout()
        
        # Save plot if requested
        if save:
            filename = os.path.join(self.output_dir, 'arrest_distribution.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filename}")
        
        return fig
    
    def plot_feature_importances(self, importances_df, top_n=20, figsize=(12, 8), save=True):
        """
        Plot feature importances
        
        Parameters:
        -----------
        importances_df : pandas.DataFrame
            DataFrame with 'Feature' and 'Importance' columns
        top_n : int
            Number of top features to show
        figsize : tuple
            Figure size
        save : bool
            Whether to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        # Get top N features
        top_features = importances_df.head(top_n).copy()
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot horizontal bar chart
        sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax, color='steelblue')
        
        # Set labels and title
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.set_title(f'Top {top_n} Feature Importances')
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='x')
        
        # Tight layout
        plt.tight_layout()
        
        # Save plot if requested
        if save:
            filename = os.path.join(self.output_dir, 'feature_importances.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filename}")
        
        return fig
    
    def plot_model_comparison(self, models_metrics, metric_name='f1_score', figsize=(10, 6), save=True):
        """
        Plot model comparison
        
        Parameters:
        -----------
        models_metrics : dict
            Dictionary with model metrics
        metric_name : str
            Name of the metric to compare
        figsize : tuple
            Figure size
        save : bool
            Whether to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        # Extract metrics for comparison
        models = []
        metrics = []
        
        for model_name, results in models_metrics.items():
            if metric_name in results['metrics']:
                models.append(model_name)
                metrics.append(results['metrics'][metric_name])
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot bar chart
        sns.barplot(x=models, y=metrics, ax=ax, palette='viridis')
        
        # Set labels and title
        ax.set_xlabel('Model')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.set_title(f'Model Comparison: {metric_name.replace("_", " ").title()}')
        
        # Add values on top of bars
        for i, metric in enumerate(metrics):
            ax.text(i, metric + 0.01, f'{metric:.4f}', ha='center', va='bottom', fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Tight layout
        plt.tight_layout()
        
        # Save plot if requested
        if save:
            filename = os.path.join(self.output_dir, f'model_comparison_{metric_name}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filename}")
        
        return fig
    
    def plot_lstm_predictions(self, actual, predictions, dates=None, figsize=(12, 6), save=True):
        """
        Plot LSTM predictions vs actual values
        
        Parameters:
        -----------
        actual : array-like
            Actual values
        predictions : array-like
            Predicted values
        dates : array-like, optional
            Date values for x-axis
        figsize : tuple
            Figure size
        save : bool
            Whether to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot line chart
        if dates is not None:
            ax.plot(dates, actual, marker='o', linestyle='-', label='Actual')
            ax.plot(dates, predictions, marker='x', linestyle='--', label='Predicted')
            
            # Format x-axis if dates are provided
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        else:
            ax.plot(actual, marker='o', linestyle='-', label='Actual')
            ax.plot(predictions, marker='x', linestyle='--', label='Predicted')
        
        # Set labels and title
        ax.set_xlabel('Time' if dates is None else 'Date')
        ax.set_ylabel('Crime Count')
        ax.set_title('LSTM Predictions vs Actual Crime Counts')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Save plot if requested
        if save:
            filename = os.path.join(self.output_dir, 'lstm_predictions.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filename}")
        
        return fig


if __name__ == "__main__":
    import argparse
    import os
    import glob
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate visualizations for Chicago crime data')
    parser.add_argument('--data_file', type=str, help='Path to the data file')
    args = parser.parse_args()
    
    # Find the most recent data file if not specified
    if args.data_file:
        data_file = args.data_file
    else:
        data_dir = 'data'
        data_files = glob.glob(os.path.join(data_dir, "chicago_theft_data_*.csv"))
        
        if not data_files:
            raise FileNotFoundError(f"No data files found in {data_dir}")
        
        data_file = max(data_files, key=os.path.getctime)
    
    print(f"Loading data from {data_file}")
    
    df = pd.read_csv(data_file)
    
    # Convert date to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Create visualizations
    visualizer = CrimeDataVisualization()
    
    # Generate various plots
    visualizer.plot_crime_over_time(df, time_unit='month')
    visualizer.plot_crime_by_hour(df)
    visualizer.plot_crime_by_day_of_week(df)
    visualizer.plot_arrest_distribution(df)
    
    # Create heatmap if coordinates are available
    if 'latitude' in df.columns and 'longitude' in df.columns:
        visualizer.plot_crime_heatmap(df)
    
    print("All visualizations have been generated!")