"""
Metrics for evaluating model performance
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error, mean_absolute_error,
    r2_score
)

def classification_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate classification metrics
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_prob : array-like, optional
        Predicted probabilities for the positive class
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    # Calculate AUC if probabilities are provided
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics['auc'] = np.nan
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    metrics['true_positives'] = tp
    
    return metrics

def regression_metrics(y_true, y_pred):
    """
    Calculate regression metrics
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
    # Mean Absolute Percentage Error (MAPE)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    metrics['mape'] = mape
    
    return metrics

def error_analysis(y_true, y_pred, X, feature_names=None):
    """
    Perform error analysis
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    X : array-like
        Features
    feature_names : list, optional
        Names of the features
        
    Returns:
    --------
    dict
        Dictionary with error analysis results
    """
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    # Find misclassified samples
    errors = y_true != y_pred
    error_indices = np.where(errors)[0]
    
    # Analyze false positives and false negatives
    false_positives = np.where((y_pred == 1) & (y_true == 0))[0]
    false_negatives = np.where((y_pred == 0) & (y_true == 1))[0]
    
    results = {
        'error_rate': errors.mean(),
        'num_errors': errors.sum(),
        'num_false_positives': len(false_positives),
        'num_false_negatives': len(false_negatives)
    }
    
    # If there are errors, analyze their characteristics
    if len(error_indices) > 0:
        # Convert X to numpy array if it's a DataFrame
        X_array = X.values if hasattr(X, 'values') else X
        
        # Get statistics for each feature in correctly and incorrectly classified samples
        correct_indices = np.where(~errors)[0]
        
        feature_stats = {}
        for i, name in enumerate(feature_names):
            feature_stats[name] = {
                'correct_mean': X_array[correct_indices, i].mean(),
                'correct_std': X_array[correct_indices, i].std(),
                'error_mean': X_array[error_indices, i].mean(),
                'error_std': X_array[error_indices, i].std(),
                'fp_mean': X_array[false_positives, i].mean() if len(false_positives) > 0 else np.nan,
                'fn_mean': X_array[false_negatives, i].mean() if len(false_negatives) > 0 else np.nan
            }
        
        results['feature_stats'] = feature_stats
    
    return results

def compare_models(models, X_test, y_test):
    """
    Compare multiple models
    
    Parameters:
    -----------
    models : list
        List of trained model objects
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
        
    Returns:
    --------
    dict
        Dictionary with comparison results
    """
    results = {}
    
    for model in models:
        # Make predictions
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        
        # Calculate metrics
        if np.issubdtype(y_test.dtype, np.number) and len(np.unique(y_test)) > 10:
            # Regression task
            metrics = regression_metrics(y_test, y_pred)
        else:
            # Classification task
            metrics = classification_metrics(y_test, y_pred)
        
        # Add model info
        model_info = model.get_model_info()
        model_info['inference_time_test'] = inference_time
        
        results[model.model_name] = {
            'metrics': metrics,
            'model_info': model_info
        }
    
    return results


if __name__ == "__main__":
    import argparse
    import pandas as pd
    import os
    import glob
    import joblib
    import time
    import matplotlib.pyplot as plt
    import seaborn as sns
    from src.data.data_loader import ChicagoCrimeDataLoader
    from src.data.data_preprocessor import CrimeDataPreprocessor
    from src.models.ml_models import RandomForestModel, GradientBoostingModel
    from src.models.dl_models import MLPModel, LSTMModel
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate and compare models for Chicago crime analysis')
    parser.add_argument('--model', type=str, choices=['rf', 'gb', 'mlp', 'lstm', 'all'], default='all',
                        help='Model to evaluate (rf: Random Forest, gb: Gradient Boosting, mlp: MLP, lstm: LSTM, all: all)')
    args = parser.parse_args()
    
    # Load data
    data_dir = 'data'
    data_files = glob.glob(os.path.join(data_dir, "chicago_theft_data_*.csv"))
    
    if not data_files:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    
    latest_file = max(data_files, key=os.path.getctime)
    print(f"Loading data from {latest_file}")
    
    df = pd.read_csv(latest_file)
    
    # Check if models directory exists
    models_dir = 'models'
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory '{models_dir}' not found")
    
    # List of models to evaluate
    models_to_evaluate = []
    
    if args.model in ['rf', 'all']:
        # Find Random Forest model file
        rf_files = glob.glob(os.path.join(models_dir, "random_forest_ml.joblib"))
        if rf_files:
            print("Loading Random Forest model")
            rf_model = RandomForestModel()
            rf_model.load(rf_files[0])
            models_to_evaluate.append(rf_model)
    
    if args.model in ['gb', 'all']:
        # Find Gradient Boosting model file
        gb_files = glob.glob(os.path.join(models_dir, "gradient_boosting_ml.joblib"))
        if gb_files:
            print("Loading Gradient Boosting model")
            gb_model = GradientBoostingModel()
            gb_model.load(gb_files[0])
            models_to_evaluate.append(gb_model)
    
    # Check if any ML models were loaded
    if models_to_evaluate:
        # Preprocess data for classification
        preprocessor = CrimeDataPreprocessor()
        X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_classification_data(df)
        
        # Compare ML models
        comparison_results = compare_models(models_to_evaluate, X_test, y_test)
        
        # Print comparison results
        print("\nML Model Comparison:")
        for model_name, results in comparison_results.items():
            print(f"\n{model_name}:")
            
            print("Metrics:")
            for metric_name, metric_value in results['metrics'].items():
                if metric_name not in ['true_negatives', 'false_positives', 'false_negatives', 'true_positives']:
                    print(f"  {metric_name}: {metric_value:.4f}")
            
            print("Model Info:")
            for info_name, info_value in results['model_info'].items():
                if isinstance(info_value, (int, float)):
                    print(f"  {info_name}: {info_value:.6f}")
                else:
                    print(f"  {info_name}: {info_value}")
            
            # Perform error analysis for one model (e.g., Random Forest)
            if model_name == 'random_forest':
                print("\nError Analysis:")
                y_pred = models_to_evaluate[0].predict(X_test)
                error_results = error_analysis(y_test, y_pred, X_test, feature_names)
                
                print(f"  Error rate: {error_results['error_rate']:.4f}")
                print(f"  Number of errors: {error_results['num_errors']}")
                print(f"  False positives: {error_results['num_false_positives']}")
                print(f"  False negatives: {error_results['num_false_negatives']}")
    
    # Evaluate LSTM model separately if requested
    if args.model in ['lstm', 'all']:
        # Find LSTM model file
        lstm_files = glob.glob(os.path.join(models_dir, "lstm_dl.pth"))
        if lstm_files:
            print("\nEvaluating LSTM model for time series prediction")
            
            # Preprocess data for time series
            preprocessor = CrimeDataPreprocessor()
            ts_data = preprocessor.preprocess_time_series_data(df, freq='W')
            
            # Load LSTM model
            lstm_model = LSTMModel()
            lstm_model.load(lstm_files[0], input_size=1, hidden_size=64, num_layers=2)
            lstm_model.set_scaler(ts_data['scaler'])
            
            # Make predictions
            X_test = ts_data['X_test']
            y_test = ts_data['y_test']
            
            scaled_predictions = lstm_model.predict(X_test)
            predictions = lstm_model.scaler.inverse_transform(scaled_predictions)
            actual = lstm_model.scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # Calculate metrics
            metrics = regression_metrics(actual.flatten(), predictions.flatten())
            
            print("\nLSTM Model Evaluation:")
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name}: {metric_value:.4f}")
            
            # Plot predictions vs actual
            plt.figure(figsize=(12, 6))
            plt.plot(actual, label='Actual')
            plt.plot(predictions, label='Predicted')
            plt.title('LSTM Predictions vs Actual Crime Counts')
            plt.xlabel('Time Step')
            plt.ylabel('Crime Count')
            plt.legend()
            plt.grid(True)