"""
Deep learning models for Chicago crime analysis
"""
import time
import argparse
import numpy as np
import pandas as pd
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from src.models.base_model import BaseModel
from src.data.data_loader import ChicagoCrimeDataLoader
from src.data.data_preprocessor import CrimeDataPreprocessor
from src.evaluation.metrics import classification_metrics, regression_metrics

# For reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class MLPModel(BaseModel):
    """
    Multi-Layer Perceptron for crime classification
    """
    
    def __init__(self, output_dir='models'):
        """
        Initialize the MLP model
        
        Parameters:
        -----------
        output_dir : str
            Directory to save model artifacts
        """
        super().__init__('mlp', 'dl', output_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def build(self, input_dim, hidden_dims=[128, 64], dropout_rate=0.3, **kwargs):
        """
        Build the MLP model
        
        Parameters:
        -----------
        input_dim : int
            Dimension of input features
        hidden_dims : list
            Dimensions of hidden layers
        dropout_rate : float
            Dropout rate for regularization
        **kwargs : dict
            Additional parameters for the model
            
        Returns:
        --------
        MLPModel
            The initialized model
        """
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers).to(self.device)
        
        return self
    
    def train(self, X_train, y_train, batch_size=64, epochs=100, learning_rate=0.001, **kwargs):
        """
        Train the MLP model
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        batch_size : int
            Batch size for training
        epochs : int
            Number of epochs to train
        learning_rate : float
            Learning rate for the optimizer
        **kwargs : dict
            Additional parameters for training
            
        Returns:
        --------
        MLPModel
            The trained model
        """
        if self.model is None:
            raise ValueError("Model must be built before training")
        
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X_train.values if isinstance(X_train, pd.DataFrame) else X_train)
        y_tensor = torch.FloatTensor(y_train.values if isinstance(y_train, pd.Series) else y_train).reshape(-1, 1)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        start_time = time.time()
        
        for epoch in range(epochs):
            running_loss = 0.0
            
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                # Forward pass
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            # Print epoch statistics
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}')
        
        end_time = time.time()
        self.training_time = end_time - start_time
        
        return self
    
    def predict(self, X):
        """
        Make predictions with the MLP model
        
        Parameters:
        -----------
        X : array-like
            Features to predict
            
        Returns:
        --------
        array-like
            Predicted classes (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X.values if isinstance(X, pd.DataFrame) else X).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = (outputs >= 0.5).float().cpu().numpy().flatten()
        
        return predictions
    
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
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X.values if isinstance(X, pd.DataFrame) else X).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(X_tensor).cpu().numpy()
        
        # Convert to probabilities for binary classification
        probas = np.hstack([1 - outputs, outputs])
        
        return probas
    
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
            filename = f"{self.model_name}_{self.model_type}.pth"
        
        filepath = os.path.join(self.output_dir, filename)
        torch.save(self.model.state_dict(), filepath)
        
        return filepath
    
    def load(self, filepath, input_dim, hidden_dims=[128, 64], dropout_rate=0.3):
        """
        Load the model from disk
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        input_dim : int
            Dimension of input features
        hidden_dims : list
            Dimensions of hidden layers
        dropout_rate : float
            Dropout rate for regularization
            
        Returns:
        --------
        MLPModel
            The loaded model
        """
        self.build(input_dim, hidden_dims, dropout_rate)
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()
        
        return self


class LSTMModel(BaseModel):
    """
    LSTM model for crime count prediction
    """
    
    class LSTMNet(nn.Module):
        """
        LSTM Neural Network architecture
        """
        def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
            super(LSTMModel.LSTMNet, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            # Initialize hidden state with zeros
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            
            # Forward propagate LSTM
            out, _ = self.lstm(x, (h0, c0))  # out: batch_size, seq_length, hidden_size
            
            # Decode the hidden state of the last time step
            out = self.fc(out[:, -1, :])
            return out
    
    def __init__(self, output_dir='models'):
        """
        Initialize the LSTM model
        
        Parameters:
        -----------
        output_dir : str
            Directory to save model artifacts
        """
        super().__init__('lstm', 'dl', output_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = None
    
    def build(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2, **kwargs):
        """
        Build the LSTM model
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        hidden_size : int
            Number of features in the hidden state
        num_layers : int
            Number of stacked LSTM layers
        output_size : int
            Number of output features
        dropout : float
            Dropout rate
        **kwargs : dict
            Additional parameters for the model
            
        Returns:
        --------
        LSTMModel
            The initialized model
        """
        self.model = self.LSTMNet(input_size, hidden_size, num_layers, output_size, dropout).to(self.device)
        return self
    
    def train(self, X_train, y_train, batch_size=16, epochs=100, learning_rate=0.001, **kwargs):
        """
        Train the LSTM model
        
        Parameters:
        -----------
        X_train : array-like
            Training features with shape [samples, sequence_length, features]
        y_train : array-like
            Training targets
        batch_size : int
            Batch size for training
        epochs : int
            Number of epochs to train
        learning_rate : float
            Learning rate for the optimizer
        **kwargs : dict
            Additional parameters for training
            
        Returns:
        --------
        LSTMModel
            The trained model
        """
        if self.model is None:
            raise ValueError("Model must be built before training")
        
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        start_time = time.time()
        
        for epoch in range(epochs):
            running_loss = 0.0
            
            for X_batch, y_batch in dataloader:
                # Forward pass
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            # Print epoch statistics
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}')
        
        end_time = time.time()
        self.training_time = end_time - start_time
        
        return self
    
    def predict(self, X):
        """
        Make predictions with the LSTM model
        
        Parameters:
        -----------
        X : array-like
            Features to predict with shape [samples, sequence_length, features]
            
        Returns:
        --------
        array-like
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions
    
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
            filename = f"{self.model_name}_{self.model_type}.pth"
        
        filepath = os.path.join(self.output_dir, filename)
        torch.save(self.model.state_dict(), filepath)
        
        return filepath
    
    def load(self, filepath, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        """
        Load the model from disk
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        input_size : int
            Number of input features
        hidden_size : int
            Number of features in the hidden state
        num_layers : int
            Number of stacked LSTM layers
        output_size : int
            Number of output features
        dropout : float
            Dropout rate
            
        Returns:
        --------
        LSTMModel
            The loaded model
        """
        self.build(input_size, hidden_size, num_layers, output_size, dropout)
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()
        
        return self
    
    def set_scaler(self, scaler):
        """
        Set the scaler for inverse transformation
        
        Parameters:
        -----------
        scaler : object
            Scaler object with inverse_transform method
        """
        self.scaler = scaler
    
    def predict_and_inverse_transform(self, X):
        """
        Make predictions and inverse transform them
        
        Parameters:
        -----------
        X : array-like
            Features to predict
            
        Returns:
        --------
        array-like
            Inverse transformed predictions
        """
        predictions = self.predict(X)
        
        if self.scaler is not None:
            predictions = self.scaler.inverse_transform(predictions)
        
        return predictions


def train_mlp_model(X_train, y_train, X_test, y_test, input_dim, hidden_dims=[128, 64], dropout_rate=0.3,
                   batch_size=64, epochs=100, learning_rate=0.001):
    """
    Train an MLP model
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    input_dim : int
        Dimension of input features
    hidden_dims : list
        Dimensions of hidden layers
    dropout_rate : float
        Dropout rate for regularization
    batch_size : int
        Batch size for training
    epochs : int
        Number of epochs to train
    learning_rate : float
        Learning rate for the optimizer
        
    Returns:
    --------
    tuple
        (trained_model, test_predictions)
    """
    # Initialize and build the model
    model = MLPModel().build(input_dim, hidden_dims, dropout_rate)
    
    # Train the model
    model.train(X_train, y_train, batch_size, epochs, learning_rate)
    
    # Measure inference time
    model.measure_inference_time(X_test)
    
    # Make predictions
    test_predictions = model.predict(X_test)
    
    # Save the model
    model_path = model.save()
    print(f"Model saved to {model_path}")
    
    return model, test_predictions


def train_lstm_model(ts_data, hidden_size=64, num_layers=2, dropout=0.2,
                     batch_size=16, epochs=100, learning_rate=0.001):
    """
    Train an LSTM model for time series prediction
    
    Parameters:
    -----------
    ts_data : dict
        Time series data from preprocess_time_series_data
    hidden_size : int
        Number of features in the hidden state
    num_layers : int
        Number of stacked LSTM layers
    dropout : float
        Dropout rate
    batch_size : int
        Batch size for training
    epochs : int
        Number of epochs to train
    learning_rate : float
        Learning rate for the optimizer
        
    Returns:
    --------
    tuple
        (trained_model, test_predictions, actual_values)
    """
    X_train = ts_data['X_train']
    y_train = ts_data['y_train']
    X_test = ts_data['X_test']
    y_test = ts_data['y_test']
    
    # Get input size from data
    input_size = X_train.shape[2]
    
    # Initialize and build the model
    model = LSTMModel().build(input_size, hidden_size, num_layers, output_size=1, dropout=dropout)
    
    # Train the model
    model.train(X_train, y_train, batch_size, epochs, learning_rate)
    
    # Measure inference time
    model.measure_inference_time(X_test)
    
    # Set the scaler for inverse transformation
    model.set_scaler(ts_data['scaler'])
    
    # Make predictions
    scaled_predictions = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    predictions = model.scaler.inverse_transform(scaled_predictions)
    actual = model.scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Save the model
    model_path = model.save()
    print(f"Model saved to {model_path}")
    
    return model, predictions, actual


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train deep learning models for Chicago crime analysis')
    parser.add_argument('--model', type=str, choices=['mlp', 'lstm', 'all'], default='all',
                        help='Model to train (mlp: Multi-Layer Perceptron, lstm: LSTM, all: both)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    args = parser.parse_args()
    
    # Load data
    data_dir = 'data'
    data_files = glob.glob(os.path.join(data_dir, "chicago_theft_data_*.csv"))
    
    if not data_files:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    
    latest_file = max(data_files, key=os.path.getctime)
    print(f"Loading data from {latest_file}")
    
    df = pd.read_csv(latest_file)
    
    if args.model in ['mlp', 'all']:
        print(f"\n{'='*50}")
        print(f"Training MLP model for classification")
        print(f"{'='*50}")
        
        # Preprocess data for classification
        preprocessor = CrimeDataPreprocessor()
        X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_classification_data(df)
        
        # Train MLP model
        model, y_pred = train_mlp_model(
            X_train, y_train, X_test, y_test,
            input_dim=X_train.shape[1],
            epochs=args.epochs
        )
        
        # Evaluate the model
        metrics = classification_metrics(y_test, y_pred)
        
        print("\nModel Evaluation:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        
        # Print model info
        model_info = model.get_model_info()
        print("\nModel Info:")
        print(f"Training time: {model_info['training_time']:.4f} seconds")
        print(f"Inference time: {model_info['inference_time']:.6f} seconds")
    
    if args.model in ['lstm', 'all']:
        print(f"\n{'='*50}")
        print(f"Training LSTM model for time series prediction")
        print(f"{'='*50}")
        
        # Preprocess data for time series
        preprocessor = CrimeDataPreprocessor()
        ts_data = preprocessor.preprocess_time_series_data(df, freq='W')
        
        # Train LSTM model
        model, predictions, actual = train_lstm_model(
            ts_data,
            epochs=args.epochs
        )
        
        # Evaluate the model
        metrics = regression_metrics(actual.flatten(), predictions.flatten())
        
        print("\nModel Evaluation:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        
        # Print model info
        model_info = model.get_model_info()
        print("\nModel Info:")
        print(f"Training time: {model_info['training_time']:.4f} seconds")
        print(f"Inference time: {model_info['inference_time']:.6f} seconds")