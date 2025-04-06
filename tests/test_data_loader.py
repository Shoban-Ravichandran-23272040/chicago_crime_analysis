"""
Tests for the data loader module
"""
import unittest
import pandas as pd
from src.data.data_loader import ChicagoCrimeDataLoader

class TestChicagoCrimeDataLoader(unittest.TestCase):
    """Test cases for ChicagoCrimeDataLoader"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.loader = ChicagoCrimeDataLoader(data_dir='test_data')
    
    def test_initialization(self):
        """Test initialization of the data loader"""
        self.assertEqual(self.loader.base_url, "https://data.cityofchicago.org/resource/crimes.json")
        self.assertEqual(self.loader.data_dir, 'test_data')
    
    def test_load_data_from_file(self):
        """Test loading data from file"""
        # Create a mock CSV file
        df = pd.DataFrame({
            'id': ['1', '2', '3'],
            'case_number': ['HZ123456', 'HZ123457', 'HZ123458'],
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'primary_type': ['THEFT', 'THEFT', 'THEFT'],
            'description': ['POCKET-PICKING', 'PURSE-SNATCHING', 'POCKET-PICKING'],
            'arrest': ['false', 'true', 'false']
        })
        
        # Save the mock data
        import os
        os.makedirs('test_data', exist_ok=True)
        test_file = 'test_data/test_data.csv'
        df.to_csv(test_file, index=False)
        
        # Load the data
        loaded_df = self.loader.load_data_from_file(test_file)
        
        # Check if the data was loaded correctly
        self.assertIsNotNone(loaded_df)
        self.assertEqual(len(loaded_df), 3)
        self.assertEqual(loaded_df.iloc[1]['case_number'], 'HZ123457')
        
        # Clean up
        os.remove(test_file)
        os.rmdir('test_data')


if __name__ == '__main__':
    unittest.main()