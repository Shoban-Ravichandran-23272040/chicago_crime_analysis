from setuptools import setup, find_packages

setup(
    name="chicago_crime_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.23.2",
        "requests>=2.24.0",
        "torch>=1.7.0",
        "tensorflow>=2.4.0",
        "tqdm>=4.50.0",
    ],
    author="Shoban Ravichandran",
    author_email="x23272040@student.ncirl.ie",
    description="Analysis of Chicago theft crimes using ML and DL approaches",
    keywords="crime-analysis, machine-learning, deep-learning",
    url="https://github.com/yourusername/chicago-crime-analysis",
)