"""
Tests for the model modules
"""
import unittest
import numpy as np
import pandas as pd
import os
import torch
from src.models.ml_models import RandomForestModel, GradientBoostingModel
from src.models.dl_models import MLPModel, LSTMModel

class TestMLModels(unittest.TestCase):
    """Test cases for ML models"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a simple dataset for testing
        np.random.seed(42)
        self.X_train = np.random.rand(100, 5)
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = np.random.rand(20, 5)
        self.y_test = np.random.randint(0, 2, 20)
        
        # Create test output directory
        os.makedirs('test_models', exist_ok=True)
    
    def tearDown(self):
        """Tear down test fixtures"""
        # Clean up test files
        for filename in os.listdir('test_models'):
            os.remove(os.path.join('test_models', filename))
        os.rmdir('test_models')
    
    def test_random_forest_model(self):
        """Test RandomForestModel"""
        # Initialize and build the model
        rf_model = RandomForestModel(output_dir='test_models')
        rf_model.build(n_estimators=10, random_state=42)
        
        # Train the model
        rf_model.train(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = rf_model.predict(self.X_test)
        
        # Check if predictions have the right shape
        self.assertEqual(len(y_pred), len(self.y_test))
        
        # Check if probabilities can be computed
        probas = rf_model.predict_proba(self.X_test)
        self.assertEqual(probas.shape, (len(self.y_test), 2))
        
        # Check if feature importances can be computed
        importances = rf_model.get_feature_importances()
        self.assertEqual(len(importances), 5)
        
        # Check if model can be saved and loaded
        model_path = rf_model.save(filename='rf_test.joblib')
        self.assertTrue(os.path.exists(model_path))
        
        # Load the model
        new_model = RandomForestModel(output_dir='test_models')
        new_model.load(model_path)
        
        # Make predictions with the loaded model
        new_pred = new_model.predict(self.X_test)
        
        # Check if predictions are the same
        np.testing.assert_array_equal(y_pred, new_pred)
    
    def test_gradient_boosting_model(self):
        """Test GradientBoostingModel"""
        # Initialize and build the model
        gb_model = GradientBoostingModel(output_dir='test_models')
        gb_model.build(n_estimators=10, random_state=42)
        
        # Train the model
        gb_model.train(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = gb_model.predict(self.X_test)
        
        # Check if predictions have the right shape
        self.assertEqual(len(y_pred), len(self.y_test))
        
        # Check if probabilities can be computed
        probas = gb_model.predict_proba(self.X_test)
        self.assertEqual(probas.shape, (len(self.y_test), 2))
        
        # Check if feature importances can be computed
        importances = gb_model.get_feature_importances()
        self.assertEqual(len(importances), 5)
        
        # Check if model can be saved and loaded
        model_path = gb_model.save(filename='gb_test.joblib')
        self.assertTrue(os.path.exists(model_path))


class TestDLModels(unittest.TestCase):
    """Test cases for DL models"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a simple dataset for testing classification
        np.random.seed(42)
        self.X_train = np.random.rand(100, 5)
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = np.random.rand(20, 5)
        self.y_test = np.random.randint(0, 2, 20)
        
        # Create a simple time series dataset
        seq_length = 4
        self.ts_X_train = np.random.rand(50, seq_length, 1)
        self.ts_y_train = np.random.rand(50, 1)
        self.ts_X_test = np.random.rand(10, seq_length, 1)
        self.ts_y_test = np.random.rand(10, 1)
        
        # Create test output directory
        os.makedirs('test_models', exist_ok=True)
    
    def tearDown(self):
        """Tear down test fixtures"""
        # Clean up test files
        for filename in os.listdir('test_models'):
            os.remove(os.path.join('test_models', filename))
        os.rmdir('test_models')
    
    def test_mlp_model(self):
        """Test MLPModel"""
        # Initialize and build the model
        mlp_model = MLPModel(output_dir='test_models')
        mlp_model.build(input_dim=5, hidden_dims=[10, 5], dropout_rate=0.2)
        
        # Convert data to tensors
        X_train_tensor = torch.FloatTensor(self.X_train)
        y_train_tensor = torch.FloatTensor(self.y_train)
        
        # Train the model with minimal epochs for testing
        mlp_model.train(X_train_tensor, y_train_tensor, batch_size=16, epochs=2)
        
        # Make predictions
        y_pred = mlp_model.predict(self.X_test)
        
        # Check if predictions have the right shape
        self.assertEqual(len(y_pred), len(self.y_test))
        
        # Check if model can be saved and loaded
        model_path = mlp_model.save(filename='mlp_test.pth')
        self.assertTrue(os.path.exists(model_path))
    
    def test_lstm_model(self):
        """Test LSTMModel"""
        # Initialize and build the model
        lstm_model = LSTMModel(output_dir='test_models')
        lstm_model.build(input_size=1, hidden_size=10, num_layers=1)
        
        # Train the model with minimal epochs for testing
        lstm_model.train(self.ts_X_train, self.ts_y_train, batch_size=16, epochs=2)
        
        # Make predictions
        y_pred = lstm_model.predict(self.ts_X_test)
        
        # Check if predictions have the right shape
        self.assertEqual(y_pred.shape, self.ts_y_test.shape)
        
        # Check if model can be saved and loaded
        model_path = lstm_model.save(filename='lstm_test.pth')
        self.assertTrue(os.path.exists(model_path))


if __name__ == '__main__':
    unittest.main()