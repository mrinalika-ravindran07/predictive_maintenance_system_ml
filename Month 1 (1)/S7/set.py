import os
from pathlib import Path

def create_ml_project_structure(base_path="my_ml_project"):
    """
    Creates a standard, scalable directory structure for an ML project.
    """
    # Define the core directories
    directories = [
        "data/raw",              # Immutable, original data
        "data/processed",        # Cleaned, finalized datasets
        "notebooks",             # Jupyter notebooks for exploration (named 01_eda.ipynb, etc.)
        "src/data",              # Scripts to fetch or generate data
        "src/features",          # Scripts to turn raw data into features
        "src/models",            # Scripts to train models and make predictions
        "models",                # Saved model artifacts (e.g., .pkl files)
        "reports/figures"        # Generated analysis files and graphics
    ]

    # Define standard starting files
    files = {
        "README.md": "# Project Title\n\nDescription of the project.",
        "requirements.txt": "scikit-learn\npandas\nnumpy\nmatplotlib",
        ".gitignore": "data/raw/\nmodels/*.pkl\n__pycache__/\n.env",
        "src/__init__.py": ""
    }

    print(f"--- Creating ML Project Structure at '{base_path}/' ---")
    
    # Create directories
    for dir_path in directories:
        path = Path(base_path) / dir_path
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")

    # Create initial files
    for file_name, content in files.items():
        file_path = Path(base_path) / file_name
        with open(file_path, "w") as f:
            f.write(content)
        print(f"Created file:      {file_path}")
        
    print("\nProject structure initialized successfully!")

if __name__ == "__main__":
    # Run this to generate the folder structure in your current working directory
    create_ml_project_structure()