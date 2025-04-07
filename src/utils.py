"""
Utility functions for Chicago crime analysis project
"""
import pandas as pd
import numpy as np

def align_features(model, X_test):
    """
    Align features between model's training data and test data
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X_test : pandas DataFrame
        Test features
    
    Returns:
    --------
    X_test_aligned : aligned DataFrame
    """
    # Get feature names used during training
    if hasattr(model, 'feature_names_in_'):
        train_features = model.feature_names_in_
    else:
        raise ValueError("Cannot retrieve feature names from the model")
    
    # Identify common features
    test_features = X_test.columns.tolist()
    
    # Find missing and extra features
    missing_features = set(train_features) - set(test_features)
    extra_features = set(test_features) - set(train_features)
    
    # Print out feature mismatches for debugging
    if missing_features:
        print("Features in training but missing in test:")
        print(missing_features)
    
    if extra_features:
        print("Features in test but not in training:")
        print(extra_features)
    
    # Create aligned test dataset
    X_test_aligned = X_test[list(set(train_features) & set(test_features))]
    
    # Add missing columns with zeros if needed
    for feature in missing_features:
        X_test_aligned[feature] = 0
    
    # Reorder columns to match training order
    X_test_aligned = X_test_aligned[train_features]
    
    return X_test_aligned