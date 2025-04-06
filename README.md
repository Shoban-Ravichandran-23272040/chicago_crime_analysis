# Chicago Crime Analysis

## Overview
This project analyzes theft crimes (specifically pocket-picking and purse-snatching) in Chicago using machine learning and deep learning methods. The analysis focuses on predicting crime counts and arrest probabilities using data from the Chicago Data Portal.

## Features
- Data extraction from Chicago Data Portal API
- Comprehensive data preprocessing and feature engineering
- Exploratory data analysis with visualizations
- Implementation of multiple machine learning models:
  - Random Forest Classifier
  - Gradient Boosting Classifier
- Implementation of deep learning models:
  - Multi-Layer Perceptron (MLP)
  - LSTM for time series prediction
- Model evaluation and comparison

## Requirements
- Python 3.8+
- Required packages listed in requirements.txt

## Installation
```bash
git clone https://github.com/yourusername/chicago-crime-analysis.git
cd chicago-crime-analysis
pip install -r requirements.txt
```

## Usage
1. Data collection:
```bash
python -m src.data.data_loader --limit 70000
```

2. Training models:
```bash
python -m src.models.ml_models --model rf  # For Random Forest
python -m src.models.ml_models --model gb  # For Gradient Boosting
python -m src.models.dl_models --model mlp  # For MLP
python -m src.models.dl_models --model lstm  # For LSTM
```

3. Run evaluation:
```bash
python -m src.evaluation.metrics --model all
```

## Folder Structure
- `src/`: Source code
  - `data/`: Data loading and preprocessing
  - `features/`: Feature engineering
  - `models/`: ML and DL model implementations
  - `evaluation/`: Metrics and evaluation tools
  - `visualization/`: Visualization utilities
- `notebooks/`: Jupyter notebooks for exploration and results
- `tests/`: Unit tests
- `reports/`: Generated reports and figures

## License
This project is licensed under the MIT License - see the LICENSE file for details.
