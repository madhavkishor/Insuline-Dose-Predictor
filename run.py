
import sys
import os
import argparse
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'scikit-learn',
        'plotly',
        'joblib',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def setup_environment():
    """Set up the project environment"""
    print("Setting up DiabeDose environment...")
    
    # Create necessary directories
    directories = ['data', 'models', 'logs', 'reports']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✓ Created {directory}/ directory")
    
    # Check if data exists
    data_file = Path('data/dummy_data.csv')
    if not data_file.exists():
        print("  ⚠️  No data file found. Generating sample data...")
        from utils.data_generator import DiabetesDataGenerator
        generator = DiabetesDataGenerator()
        df = generator.generate_patient_data(1000)
        generator.save_to_csv(df, 'data/dummy_data.csv')
        print("  ✓ Generated sample data")
    
    # Check if model exists
    model_file = Path('models/insulin_predictor.pkl')
    if not model_file.exists():
        print("  ⚠️  No model file found. Training model...")
        try:
            from models.model_training import InsulinDosePredictor
            import pandas as pd
            
            # Load data
            df = pd.read_csv('data/dummy_data.csv')
            
            # Train model
            predictor = InsulinDosePredictor()
            X = df[['glucose', 'hba1c', 'weight', 'age', 'diabetes_type', 'carbs', 'bmi', 'diabetes_duration']]
            y = df['recommended_dose']
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            results, best_model = predictor.train_models(X_train, X_test, y_train, y_test)
            predictor.save_model()
            
            print("  ✓ Trained and saved model")
        except Exception as e:
            print(f"  ✗ Error training model: {e}")
            print("  ⚠️  Will use rule-based calculations only")
    
    print("✓ Environment setup complete")

def run_web_app(port=8501, host='localhost'):
    """Run the Streamlit web application"""
    print(f"\nStarting DiabeDose web application...")
    print(f"  URL: http://{host}:{port}")
    print("  Press Ctrl+C to stop\n")
    
    try:
        # Run streamlit
        subprocess.run([
            'streamlit', 'run', 'app.py',
            '--server.port', str(port),
            '--server.address', host,
            '--theme.base', 'light',
            '--browser.serverAddress', host
        ])
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user")
    except Exception as e:
        print(f"\nError running application: {e}")
        sys.exit(1)

def run_tests():
    """Run unit tests"""
    print("Running unit tests...")
    
    import unittest
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

def generate_report():
    """Generate a system report"""
    print("Generating system report...")
    
    import platform
    import pandas as pd
    from datetime import datetime
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'system': {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'processor': platform.processor()
        },
        'project': {
            'data_file': 'data/dummy_data.csv',
            'model_file': 'models/insulin_predictor.pkl',
            'requirements': 'requirements.txt'
        }
    }
    
    # Check file sizes
    for filepath in ['data/dummy_data.csv', 'models/insulin_predictor.pkl']:
        if Path(filepath).exists():
            size_mb = Path(filepath).stat().st_size / (1024 * 1024)
            report['project'][f'{filepath}_size_mb'] = round(size_mb, 2)
    
    # Save report
    report_file = f"reports/system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    import json
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Report saved to {report_file}")
    return report_file

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='DiabeDose - Insulin Dose Predictor')
    parser.add_argument('command', choices=['run', 'test', 'setup', 'report', 'all'],
                       help='Command to execute')
    parser.add_argument('--port', type=int, default=8501,
                       help='Port for web application (default: 8501)')
    parser.add_argument('--host', default='localhost',
                       help='Host for web application (default: localhost)')
    
    args = parser.parse_args()
    
    # Check dependencies first
    missing = check_dependencies()
    if missing:
        print("Missing dependencies:")
        for package in missing:
            print(f"  ✗ {package}")
        print("\nInstall missing packages with:")
        print("  pip install " + " ".join(missing))
        sys.exit(1)
    
    if args.command == 'setup':
        setup_environment()
    
    elif args.command == 'test':
        return run_tests()
    
    elif args.command == 'report':
        generate_report()
    
    elif args.command == 'run':
        run_web_app(args.port, args.host)
    
    elif args.command == 'all':
        print("Running complete setup and starting application...")
        setup_environment()
        run_web_app(args.port, args.host)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())