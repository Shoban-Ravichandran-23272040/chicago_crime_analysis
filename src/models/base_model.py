"""
Base model interface for Chicago crime analysis
"""
from abc import ABC, abstractmethod
import time
import numpy as np
import joblib
import os

class BaseModel(ABC):
    """
    Abstract base class for all models
    """
    
    def __init__(self, model_name, model_type, output_dir='models'):
        """
        Initialize the base model
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        model_type : str
            Type of the model ('ml' or 'dl')
        output_dir : str
            Directory to save model artifacts
        """
        self.model_name = model_name
        self.model_type = model_type
        self.output_dir = output_dir
        self.model = None
        self.training_time = None
        self.inference_time = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    @abstractmethod
    def build(self, **kwargs):
        """
        Build the model architecture
        
        Parameters:
        -----------
        **kwargs : dict
            Additional parameters for model building
        """
        pass
    
    @abstractmethod
    def train(self, X_train, y_train, **kwargs):
        """
        Train the model
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        **kwargs : dict
            Additional parameters for training
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
        -----------
        X : array-like
            Features to predict
            
        Returns:
        --------
        array-like
            Predictions
        """
        pass
    
    def measure_inference_time(self, X, n_repeats=10):
        """
        Measure the inference time
        
        Parameters:
        -----------
        X : array-like
            Features to predict
        n_repeats : int
            Number of times to repeat for averaging
            
        Returns:
        --------
        float
            Average inference time in seconds
        """
        if self.model is None:
            raise ValueError("Model must be trained before measuring inference time")
        
        # Warm-up run
        _ = self.predict(X)
        
        # Measure inference time
        times = []
        for _ in range(n_repeats):
            start_time = time.time()
            _ = self.predict(X)
            end_time = time.time()
            times.append(end_time - start_time)
        
        self.inference_time = np.mean(times)
        return self.inference_time
    
    def save(self, filename=None):
        """
        Save the model to disk
        
        Parameters:
        -----------
        filename : str, optional
            Name of the file to save the model
            
        Returns:
        --------
        str
            Path to the saved model
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        if filename is None:
            filename = f"{self.model_name}_{self.model_type}.joblib"
        
        filepath = os.path.join(self.output_dir, filename)
        joblib.dump(self.model, filepath)
        
        return filepath
    
    def load(self, filepath):
        """
        Load the model from disk
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        object
            The loaded model
        """
        self.model = joblib.load(filepath)
        return self.model
    
    def get_model_info(self):
        """
        Get information about the model
        
        Returns:
        --------
        dict
            Model information
        """
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'training_time': self.training_time,
            'inference_time': self.inference_time
        }