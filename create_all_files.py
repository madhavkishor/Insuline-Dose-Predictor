#!/usr/bin/env python3
"""
Script to create ALL DiabeDose project files at once
"""

import os
import sys
from pathlib import Path

def create_directory_structure():
    """Create all necessary directories"""
    directories = [
        'data',
        'models',
        'notebooks',
        'utils',
        'tests',
        'docs',
        'monitoring',
        'logs',
        'reports'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created {directory}/")
    
    return True

def create_requirements():
    """Create requirements.txt"""
    content = """streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
plotly==5.17.0
joblib==1.3.2
matplotlib==3.7.2
seaborn==0.12.2
python-dotenv==1.0.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(content)
    
    print("✓ Created requirements.txt")
    return True

def create_readme():
    """Create README.md"""