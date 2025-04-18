"""
Machine learning models for Chicago crime analysis with SMOTE sampling and cost-sensitive learning
"""
import time
import argparse
import numpy as np
import pandas as pd
import os
import glob
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
# Import SMOTE for handling class imbalance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.models.base_model import BaseModel
from src.data.data_loader import ChicagoCrimeDataLoader
from src.data.data_preprocessor import CrimeDataPreprocessor
from src.evaluation.metrics import classification_metrics

class RandomForestModel(BaseModel):
    """
    Random Forest classifier for crime prediction with SMOTE sampling
    """
    
    def __init__(self, output_dir='models', use_smote=True, class_weight=None):
        """
        Initialize the Random Forest model
        
        Parameters:
        -----------
        output_dir : str
            Directory to save model artifacts
        use_smote : bool
            Whether to use SMOTE for handling class imbalance
        class_weight : dict or 'balanced', optional
            Class weights for cost-sensitive learning
        """
        super().__init__('random_forest', 'ml', output_dir)
        self.use_smote = use_smote
        self.class_weight = class_weight
        self.pipeline = None
    
    def build(self, n_estimators=100, max_depth=None, min_samples_split=2, random_state=42, **kwargs):
        """
        Build the Random Forest model
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees in the forest
        max_depth : int or None
            Maximum depth of the trees
        min_samples_split : int
            Minimum samples required to split an internal node
        random_state : int
            Random seed for reproducibility
        **kwargs : dict
            Additional parameters for the model
            
        Returns:
        --------
        RandomForestModel
            The initialized model
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            class_weight=self.class_weight,
            **kwargs
        )
        
        # Create a pipeline with SMOTE if requested
        if self.use_smote:
            self.pipeline = ImbPipeline([
                ('smote', SMOTE(random_state=random_state)),
                ('classifier', self.model)
            ])
        
        return self
    
    def train(self, X_train, y_train, **kwargs):
        """
        Train the Random Forest model
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        **kwargs : dict
            Additional parameters for training
            
        Returns:
        --------
        RandomForestModel
            The trained model
        """
        if self.model is None:
            self.build()
        
        start_time = time.time()
        
        if self.use_smote and self.pipeline is not None:
            print("Training with SMOTE sampling...")
            self.pipeline.fit(X_train, y_train)
        else:
            print("Training without SMOTE sampling...")
            self.model.fit(X_train, y_train)
            
        end_time = time.time()
        
        self.training_time = end_time - start_time
        
        return self
    
    def predict(self, X):
        """
        Make predictions with the Random Forest model
        
        Parameters:
        -----------
        X : array-like
            Features to predict
            
        Returns:
        --------
        array-like
            Predicted classes
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        if self.use_smote and self.pipeline is not None:
            return self.pipeline.predict(X)
        else:
            return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Parameters:
        -----------
        X : array-like
            Features to predict
            
        Returns:
        --------
        array-like
            Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        if self.use_smote and self.pipeline is not None:
            return self.pipeline.predict_proba(X)
        else:
            return self.model.predict_proba(X)
    
    def get_feature_importances(self, feature_names=None):
        """
        Get feature importances
        
        Parameters:
        -----------
        feature_names : list, optional
            Names of the features
            
        Returns:
        --------
        pandas.DataFrame
            Feature importances sorted in descending order
        """
        if self.model is None:
            raise ValueError("Model must be trained before getting feature importances")
        
        if self.use_smote and self.pipeline is not None:
            importances = self.pipeline.named_steps['classifier'].feature_importances_
        else:
            importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        return importance_df.sort_values('Importance', ascending=False)


class GradientBoostingModel(BaseModel):
    """
    Gradient Boosting classifier for crime prediction with SMOTE sampling
    """
    
    def __init__(self, output_dir='models', use_smote=True, class_weight=None):
        """
        Initialize the Gradient Boosting model
        
        Parameters:
        -----------
        output_dir : str
            Directory to save model artifacts
        use_smote : bool
            Whether to use SMOTE for handling class imbalance
        class_weight : dict or 'balanced', optional
            Class weights for cost-sensitive learning
        """
        super().__init__('gradient_boosting', 'ml', output_dir)
        self.use_smote = use_smote
        self.class_weight = class_weight
        self.pipeline = None
    
    def build(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, **kwargs):
        """
        Build the Gradient Boosting model
        
        Parameters:
        -----------
        n_estimators : int
            Number of boosting stages
        learning_rate : float
            Learning rate
        max_depth : int
            Maximum depth of the trees
        random_state : int
            Random seed for reproducibility
        **kwargs : dict
            Additional parameters for the model
            
        Returns:
        --------
        GradientBoostingModel
            The initialized model
        """
        # Note: GradientBoostingClassifier doesn't directly support class_weight,
        # but we can use sample_weight during fitting to achieve similar effect
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )
        
        # Create a pipeline with SMOTE if requested
        if self.use_smote:
            self.pipeline = ImbPipeline([
                ('smote', SMOTE(random_state=random_state)),
                ('classifier', self.model)
            ])
        
        return self
    
    def train(self, X_train, y_train, **kwargs):
        """
        Train the Gradient Boosting model
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        **kwargs : dict
            Additional parameters for training
            
        Returns:
        --------
        GradientBoostingModel
            The trained model
        """
        if self.model is None:
            self.build()
        
        start_time = time.time()
        
        # Implement cost-sensitive learning through sample weights if class_weight is provided
        sample_weight = None
        if self.class_weight and not self.use_smote:
            if self.class_weight == 'balanced':
                # Calculate balanced weights
                class_counts = np.bincount(y_train)
                weight_per_class = 1. / class_counts
                sample_weight = weight_per_class[y_train]
            elif isinstance(self.class_weight, dict):
                # Use provided weights
                sample_weight = np.array([self.class_weight.get(c, 1.0) for c in y_train])
        
        if self.use_smote and self.pipeline is not None:
            print("Training with SMOTE sampling...")
            self.pipeline.fit(X_train, y_train)
        else:
            print("Training without SMOTE sampling...")
            if sample_weight is not None:
                print("Using cost-sensitive learning with sample weights...")
                self.model.fit(X_train, y_train, sample_weight=sample_weight)
            else:
                self.model.fit(X_train, y_train)
                
        end_time = time.time()
        
        self.training_time = end_time - start_time
        
        return self
    
    def predict(self, X):
        """
        Make predictions with the Gradient Boosting model
        
        Parameters:
        -----------
        X : array-like
            Features to predict
            
        Returns:
        --------
        array-like
            Predicted classes
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        if self.use_smote and self.pipeline is not None:
            return self.pipeline.predict(X)
        else:
            return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Parameters:
        -----------
        X : array-like
            Features to predict
            
        Returns:
        --------
        array-like
            Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        if self.use_smote and self.pipeline is not None:
            return self.pipeline.predict_proba(X)
        else:
            return self.model.predict_proba(X)
    
    def get_feature_importances(self, feature_names=None):
        """
        Get feature importances
        
        Parameters:
        -----------
        feature_names : list, optional
            Names of the features
            
        Returns:
        --------
        pandas.DataFrame
            Feature importances sorted in descending order
        """
        if self.model is None:
            raise ValueError("Model must be trained before getting feature importances")
        
        if self.use_smote and self.pipeline is not None:
            importances = self.pipeline.named_steps['classifier'].feature_importances_
        else:
            importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        return importance_df.sort_values('Importance', ascending=False)


def train_model(model_name, param_grid=None, cv=5, use_smote=True, class_weight='balanced'):
    """
    Train a machine learning model with optional hyperparameter tuning
    
    Parameters:
    -----------
    model_name : str
        Name of the model to train ('rf' for Random Forest, 'gb' for Gradient Boosting)
    param_grid : dict, optional
        Parameter grid for GridSearchCV
    cv : int
        Number of cross-validation folds
    use_smote : bool
        Whether to use SMOTE for handling class imbalance
    class_weight : str or dict
        Class weights for cost-sensitive learning
        
    Returns:
    --------
    tuple
        (trained_model, X_test, y_test, feature_names)
    """
    # Load data
    data_dir = 'data'
    data_files = glob.glob(os.path.join(data_dir, "chicago_theft_data_*.csv"))
    
    if not data_files:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    
    latest_file = max(data_files, key=os.path.getctime)
    print(f"Loading data from {latest_file}")
    
    df = pd.read_csv(latest_file)
    
    # Preprocess data
    preprocessor = CrimeDataPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_classification_data(df)
    
    # Print class distribution
    train_class_counts = np.bincount(y_train.astype(int))
    test_class_counts = np.bincount(y_test.astype(int))
    print(f"Training set class distribution: {train_class_counts}")
    print(f"Test set class distribution: {test_class_counts}")
    print(f"Training set arrest rate: {train_class_counts[1] / len(y_train):.4f}")
    
    # Initialize model
    if model_name.lower() == 'rf':
        model_class = RandomForestModel
        if param_grid is None:
            param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5, 10]
            }
    elif model_name.lower() == 'gb':
        model_class = GradientBoostingModel
        if param_grid is None:
            param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7]
            }
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Initialize and build the model with SMOTE and class weights
    model = model_class(use_smote=use_smote, class_weight=class_weight).build()
    
    # Perform hyperparameter tuning if param_grid is provided
    if param_grid and use_smote:
        print(f"Performing grid search for {model.model_name} with SMOTE...")
        # When using SMOTE with pipeline, we need to prefix parameters with 'classifier__'
        grid_search = GridSearchCV(model.pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1)
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        end_time = time.time()
        
        model.pipeline = grid_search.best_estimator_
        model.model = grid_search.best_estimator_.named_steps['classifier']
        model.training_time = end_time - start_time
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    elif param_grid and not use_smote:
        print(f"Performing grid search for {model.model_name} without SMOTE...")
        grid_search = GridSearchCV(model.model, param_grid, cv=cv, scoring='f1', n_jobs=-1)
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        end_time = time.time()
        
        model.model = grid_search.best_estimator_
        model.training_time = end_time - start_time
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    else:
        # Train the model with default parameters
        model.train(X_train, y_train)
    
    # Measure inference time
    model.measure_inference_time(X_test)
    
    # Save the model
    model_path = model.save()
    print(f"Model saved to {model_path}")
    
    return model, X_test, y_test, feature_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train machine learning models for Chicago crime analysis')
    parser.add_argument('--model', type=str, choices=['rf', 'gb', 'all'], default='all',
                        help='Model to train (rf: Random Forest, gb: Gradient Boosting, all: both)')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--smote', action='store_true', default=True, help='Use SMOTE for handling class imbalance')
    parser.add_argument('--class_weight', type=str, default='balanced', 
                        help='Class weights for cost-sensitive learning (balanced, none, or custom)')
    args = parser.parse_args()
    
    # Parse class_weight argument
    if args.class_weight == 'balanced':
        class_weight = 'balanced'
    elif args.class_weight == 'none':
        class_weight = None
    else:
        try:
            # Try to parse as a dictionary, e.g., "{0:1, 1:10}"
            class_weight = eval(args.class_weight)
        except:
            print(f"Invalid class_weight: {args.class_weight}, using 'balanced' instead")
            class_weight = 'balanced'
    
    if args.model == 'all':
        models_to_train = ['rf', 'gb']
    else:
        models_to_train = [args.model]
    
    for model_name in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_name.upper()} model")
        print(f"SMOTE: {'Enabled' if args.smote else 'Disabled'}")
        print(f"Class Weight: {class_weight}")
        print(f"{'='*50}")
        
        model, X_test, y_test, feature_names = train_model(
            model_name,
            param_grid=None if not args.tune else {},
            use_smote=args.smote,
            class_weight=class_weight
        )
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        metrics = classification_metrics(y_test, y_pred)
        
        print("\nModel Evaluation:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        
        # Print confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Print feature importances
        importances = model.get_feature_importances(feature_names)
        print("\nTop 10 Feature Importances:")
        print(importances.head(10))
        
        # Print model info
        model_info = model.get_model_info()
        print("\nModel Info:")
        print(f"Training time: {model_info['training_time']:.4f} seconds")
