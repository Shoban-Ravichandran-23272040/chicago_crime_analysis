"""
Tests for the data preprocessor module
"""
import unittest
import pandas as pd
import numpy as np
from src.data.data_preprocessor import CrimeDataPreprocessor

class TestCrimeDataPreprocessor(unittest.TestCase):
    """Test cases for CrimeDataPreprocessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = CrimeDataPreprocessor()
        
        # Create a mock DataFrame
        self.df = pd.DataFrame({
            'id': ['1', '2', '3', '4', '5'],
            'case_number': ['HZ123456', 'HZ123457', 'HZ123458', 'HZ123459', 'HZ123460'],
            'date': pd.date_range(start='2023-01-01', periods=5, freq='D'),
            'primary_type': ['THEFT'] * 5,
            'description': ['POCKET-PICKING', 'PURSE-SNATCHING', 'POCKET-PICKING', 'PURSE-SNATCHING', 'POCKET-PICKING'],
            'arrest': ['false', 'true', 'false', 'true', 'false'],
            'latitude': [41.8781, 41.8782, 41.8783, 41.8784, np.nan],
            'longitude': [-87.6298, -87.6299, -87.6300, -87.6301, np.nan],
            'location_description': ['STREET', 'CTA PLATFORM', 'RESIDENCE', 'RESTAURANT', 'GROCERY']
        })
    
    def test_preprocess_classification_data(self):
        """Test preprocessing for classification tasks"""
        X_train, X_test, y_train, y_test, feature_names = self.preprocessor.preprocess_classification_data(
            self.df, target='arrest', test_size=0.4, random_state=42
        )
        
        # Check shapes
        self.assertEqual(len(X_train) + len(X_test), 4)  # One row dropped due to missing coordinates
        self.assertEqual(X_train.shape[1], X_test.shape[1])
        
        # Check if target is binary
        self.assertTrue(set(y_train.unique()).issubset({0, 1}))
        self.assertTrue(set(y_test.unique()).issubset({0, 1}))
        
        # Check if feature names exist
        self.assertGreater(len(feature_names), 0)
    
    def test_preprocess_time_series_data(self):
        """Test preprocessing for time series tasks"""
        ts_data = self.preprocessor.preprocess_time_series_data(self.df, freq='D', seq_length=2)
        
        # Check keys
        expected_keys = ['X_train', 'X_test', 'y_train', 'y_test', 'scaler', 'dates', 'original_data']
        for key in expected_keys:
            self.assertIn(key, ts_data)
        
        # Check shapes
        self.assertEqual(ts_data['X_train'].shape[2], 1)  # Feature dimension
        self.assertEqual(ts_data['X_train'].shape[1], 2)  # Sequence length
        
        # Check if scaler exists
        self.assertIsNotNone(ts_data['scaler'])
        
        # Check original data
        self.assertEqual(len(ts_data['original_data']), 5)  # 5 days


if __name__ == '__main__':
    unittest.main()