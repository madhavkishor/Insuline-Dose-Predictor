"""
Complete Machine Learning model training for insulin dose prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')
import json
import os
from datetime import datetime

class InsulinDosePredictor:
    """
    Complete ML pipeline for insulin dose prediction
    """
    
    def __init__(self, model_name='RandomForest'):
        """
        Initialize the predictor
        
        Args:
            model_name: Type of model to use ('RandomForest', 'GradientBoosting', 
                       'LinearRegression', 'Ridge', 'Lasso')
        """
        self.model_name = model_name
        self.model = None
        self.preprocessor = None
        self.feature_columns = None
        self.target_column = 'recommended_dose'
        self.training_metrics = {}
        
    def prepare_data(self, df, test_size=0.2, random_state=42):
        """
        Prepare and split data for training
        
        Args:
            df: DataFrame containing the data
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Define features to use (only those that exist in dataframe)
        all_possible_features = [
            'glucose', 'hba1c', 'weight', 'age', 'diabetes_type',
            'carbs', 'bmi', 'diabetes_duration', 'insulin_sensitivity',
            'carb_ratio', 'previous_dose'
        ]
        
        self.feature_columns = [f for f in all_possible_features if f in df.columns]
        
        if len(self.feature_columns) < 3:
            # Use minimum required features
            self.feature_columns = ['glucose', 'hba1c', 'weight', 'age', 'diabetes_type', 'carbs']
            # Add only those that exist
            self.feature_columns = [f for f in self.feature_columns if f in df.columns]
        
        print(f"Using features: {self.feature_columns}")
        
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Target mean - Train: {y_train.mean():.2f}, Test: {y_test.mean():.2f}")
        
        return X_train, X_test, y_train, y_test
    
    def create_preprocessing_pipeline(self, X_train):
        """
        Create preprocessing pipeline based on data types
        
        Args:
            X_train: Training features
            
        Returns:
            Preprocessor pipeline
        """
        # Identify feature types
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"Numeric features: {numeric_features}")
        print(f"Categorical features: {categorical_features}")
        
        # Create transformers
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return self.preprocessor
    
    def create_model(self):
        """
        Create the ML model based on model_name
        
        Returns:
            ML model instance
        """
        if self.model_name == 'RandomForest':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_name == 'GradientBoosting':
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif self.model_name == 'LinearRegression':
            model = LinearRegression()
        elif self.model_name == 'Ridge':
            model = Ridge(alpha=1.0)
        elif self.model_name == 'Lasso':
            model = Lasso(alpha=0.1)
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
        
        return model
    
    def train(self, X_train, X_test, y_train, y_test, use_grid_search=False):
        """
        Train the model
        
        Args:
            X_train, X_test, y_train, y_test: Train/test split
            use_grid_search: Whether to use grid search for hyperparameter tuning
            
        Returns:
            Trained model pipeline
        """
        # Create preprocessing pipeline
        preprocessor = self.create_preprocessing_pipeline(X_train)
        
        # Create model
        model = self.create_model()
        
        # Create full pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        if use_grid_search and self.model_name in ['RandomForest', 'GradientBoosting']:
            print(f"\n🔍 Performing Grid Search for {self.model_name}...")
            pipeline = self.grid_search_tuning(pipeline, X_train, y_train)
        else:
            # Standard training
            print(f"\n🏋️ Training {self.model_name} model...")
            pipeline.fit(X_train, y_train)
        
        # Store the model
        self.model = pipeline
        
        # Evaluate on test set
        self.evaluate_model(X_test, y_test)
        
        # Calculate feature importance if available
        if hasattr(model, 'feature_importances_'):
            self.calculate_feature_importance(X_train)
        
        return pipeline
    
    def grid_search_tuning(self, pipeline, X_train, y_train):
        """
        Perform hyperparameter tuning using grid search
        
        Args:
            pipeline: Initial pipeline
            X_train, y_train: Training data
            
        Returns:
            Best estimator from grid search
        """
        if self.model_name == 'RandomForest':
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [5, 10, 15, None],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            }
        elif self.model_name == 'GradientBoosting':
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7]
            }
        else:
            print(f"Grid search not available for {self.model_name}")
            pipeline.fit(X_train, y_train)
            return pipeline
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\n✅ Best Parameters: {grid_search.best_params_}")
        print(f"Best CV Score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model on test set
        
        Args:
            X_test, y_test: Test data
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred),
            'Max_Error': np.max(np.abs(y_test - y_pred)),
            'Mean_Error': np.mean(y_test - y_pred),
            'Std_Error': np.std(y_test - y_pred),
            'Median_Absolute_Error': np.median(np.abs(y_test - y_pred)),
            'Mean_Percentage_Error': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        # Store metrics
        self.training_metrics = metrics
        
        # Print results
        print("\n📊 Model Evaluation on Test Set:")
        print("="*50)
        print(f"MAE: {metrics['MAE']:.3f} units")
        print(f"RMSE: {metrics['RMSE']:.3f} units")
        print(f"R² Score: {metrics['R2']:.3f}")
        print(f"Mean Percentage Error: {metrics['Mean_Percentage_Error']:.1f}%")
        print(f"Max Error: {metrics['Max_Error']:.2f} units")
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=5, scoring='r2')
        print(f"CV R²: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        # Clinical relevance
        self.clinical_evaluation(y_test, y_pred)
        
        return metrics
    
    def clinical_evaluation(self, y_test, y_pred):
        """
        Evaluate clinical relevance of predictions
        
        Args:
            y_test: Actual doses
            y_pred: Predicted doses
        """
        errors = np.abs(y_test - y_pred)
        
        print("\n🏥 Clinical Relevance Analysis:")
        print("="*50)
        
        thresholds = [1, 2, 3, 5]
        for threshold in thresholds:
            within_threshold = np.mean(errors <= threshold)
            print(f"Predictions within ±{threshold} units: {within_threshold:.1%}")
        
        # Dangerous predictions
        dangerous_over = np.sum(y_pred - y_test > 5)
        dangerous_under = np.sum(y_test - y_pred > 5)
        total_dangerous = dangerous_over + dangerous_under
        
        print(f"\nPotentially dangerous predictions (>5 units error): {total_dangerous/len(y_test):.1%}")
        print(f"  Over-prediction >5 units: {dangerous_over}")
        print(f"  Under-prediction >5 units: {dangerous_under}")
    
    def calculate_feature_importance(self, X_train):
        """
        Calculate and display feature importance
        
        Args:
            X_train: Training features
        """
        try:
            # Get feature names after preprocessing
            if hasattr(self.model.named_steps['preprocessor'], 'transformers_'):
                # Extract feature names from preprocessor
                feature_names = []
                
                for name, transformer, columns in self.model.named_steps['preprocessor'].transformers_:
                    if name == 'cat' and hasattr(transformer.named_steps['onehot'], 'get_feature_names_out'):
                        cat_features = transformer.named_steps['onehot'].get_feature_names_out(columns)
                        feature_names.extend(cat_features)
                    elif name == 'num':
                        feature_names.extend(columns)
                
                # Get feature importances
                if hasattr(self.model.named_steps['model'], 'feature_importances_'):
                    importances = self.model.named_steps['model'].feature_importances_
                    
                    # Create importance dataframe
                    importance_df = pd.DataFrame({
                        'Feature': feature_names[:len(importances)],
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    print("\n🔝 Feature Importance:")
                    print("="*50)
                    print(importance_df.head(10).to_string(index=False))
                    
                    # Plot feature importance
                    self.plot_feature_importance(importance_df)
                    
        except Exception as e:
            print(f"\n⚠️ Could not calculate feature importance: {str(e)}")
    
    def plot_feature_importance(self, importance_df, top_n=15):
        """
        Plot feature importance
        
        Args:
            importance_df: DataFrame with feature importances
            top_n: Number of top features to plot
        """
        plt.figure(figsize=(10, 6))
        
        # Plot top N features
        top_features = importance_df.head(top_n)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        plt.barh(top_features['Feature'], top_features['Importance'], color=colors)
        
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('models/feature_importance.png', dpi=100, bbox_inches='tight')
        print("✅ Feature importance plot saved to models/feature_importance.png")
        
        plt.show()
    
    def predict(self, patient_features):
        """
        Predict insulin dose for new patient
        
        Args:
            patient_features: Dictionary or DataFrame of patient features
            
        Returns:
            Predicted insulin dose
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Convert to dataframe if needed
        if isinstance(patient_features, dict):
            patient_df = pd.DataFrame([patient_features])
        else:
            patient_df = patient_features
        
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in patient_df.columns:
                patient_df[col] = 0  # Default value
        
        # Predict
        prediction = self.model.predict(patient_df[self.feature_columns])
        
        return prediction[0]
    
    def save_model(self, filepath='models/insulin_predictor.pkl'):
        """
        Save trained model to file
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare model data
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'model_name': self.model_name,
            'performance': self.training_metrics,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'preprocessor': self.preprocessor
        }
        
        # Save model
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
        
        # Save performance metrics separately
        self.save_performance_metrics(filepath.replace('.pkl', '_metrics.json'))
    
    def save_performance_metrics(self, filepath):
        """
        Save performance metrics to JSON file
        
        Args:
            filepath: Path to save metrics
        """
        metrics_data = {
            'model_name': self.model_name,
            'metrics': self.training_metrics,
            'feature_columns': self.feature_columns,
            'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"✅ Performance metrics saved to {filepath}")
    
    def load_model(self, filepath='models/insulin_predictor.pkl'):
        """
        Load trained model from file
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.model_name = model_data.get('model_name', 'RandomForest')
        self.training_metrics = model_data.get('performance', {})
        self.preprocessor = model_data.get('preprocessor')
        
        print(f"✅ Model loaded from {filepath}")
        print(f"Model: {self.model_name}")
        print(f"Features: {len(self.feature_columns)}")
        print(f"Training Date: {model_data.get('training_date', 'Unknown')}")
        
        return self

def train_complete_pipeline():
    """
    Complete training pipeline from data generation to model saving
    """
    print("="*60)
    print("DIABEDOSE - COMPLETE MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Step 1: Generate or load data
    print("\n📊 Step 1: Loading/Generating Data...")
    
    data_file = 'data/dummy_data.csv'
    
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        print(f"✅ Loaded data from {data_file} ({len(df)} records)")
    else:
        print("⚠️ Data file not found. Generating sample data...")
        from utils.data_generator import DiabetesDataGenerator
        generator = DiabetesDataGenerator()
        df = generator.generate_patient_data(1000)
        generator.save_to_csv(df, data_file)
        print(f"✅ Generated sample data ({len(df)} records)")
    
    # Display data info
    print(f"\nData Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Target variable: 'recommended_dose'")
    
    # Step 2: Initialize predictor
    print("\n🤖 Step 2: Initializing Predictor...")
    predictor = InsulinDosePredictor(model_name='RandomForest')
    
    # Step 3: Prepare data
    print("\n🔧 Step 3: Preparing Data...")
    X_train, X_test, y_train, y_test = predictor.prepare_data(df)
    
    # Step 4: Train model
    print("\n🏋️ Step 4: Training Model...")
    model = predictor.train(X_train, X_test, y_train, y_test, use_grid_search=True)
    
    # Step 5: Make example predictions
    print("\n🧪 Step 5: Making Example Predictions...")
    
    example_patients = [
        {
            'glucose': 150,
            'hba1c': 6.8,
            'weight': 65,
            'age': 35,
            'diabetes_type': 2,
            'carbs': 45,
            'bmi': 24,
            'diabetes_duration': 3
        },
        {
            'glucose': 220,
            'hba1c': 8.5,
            'weight': 85,
            'age': 50,
            'diabetes_type': 1,
            'carbs': 60,
            'bmi': 28,
            'diabetes_duration': 10
        },
        {
            'glucose': 180,
            'hba1c': 7.2,
            'weight': 70,
            'age': 45,
            'diabetes_type': 2,
            'carbs': 30,
            'bmi': 25,
            'diabetes_duration': 5
        }
    ]
    
    for i, patient in enumerate(example_patients, 1):
        try:
            prediction = predictor.predict(patient)
            print(f"\nPatient {i}:")
            print(f"  Glucose: {patient['glucose']} mg/dL")
            print(f"  HbA1c: {patient['hba1c']}%")
            print(f"  Weight: {patient['weight']} kg")
            print(f"  Diabetes Type: {patient['diabetes_type']}")
            print(f"  Predicted Insulin Dose: {prediction:.1f} units")
        except Exception as e:
            print(f"\n⚠️ Error predicting for patient {i}: {str(e)}")
    
    # Step 6: Save model
    print("\n💾 Step 6: Saving Model...")
    predictor.save_model()
    
    # Step 7: Generate training report
    print("\n📄 Step 7: Generating Training Report...")
    generate_training_report(predictor, df, X_test, y_test)
    
    print("\n" + "="*60)
    print("✅ MODEL TRAINING COMPLETE!")
    print("="*60)
    
    return predictor

def generate_training_report(predictor, df, X_test, y_test):
    """
    Generate comprehensive training report
    
    Args:
        predictor: Trained InsulinDosePredictor instance
        df: Full dataset
        X_test, y_test: Test data
    """
    report = f"""
DIABEDOSE MODEL TRAINING REPORT
{"="*60}

Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model Type: {predictor.model_name}

DATASET INFORMATION:
{"-"*40}
Total Samples: {len(df)}
Training Samples: {len(df) - len(X_test)}
Test Samples: {len(X_test)}
Features Used: {len(predictor.feature_columns)}
Feature List: {', '.join(predictor.feature_columns)}

MODEL PERFORMANCE:
{"-"*40}
R² Score: {predictor.training_metrics.get('R2', 0):.3f}
MAE: {predictor.training_metrics.get('MAE', 0):.3f} units
RMSE: {predictor.training_metrics.get('RMSE', 0):.3f} units
Mean Percentage Error: {predictor.training_metrics.get('Mean_Percentage_Error', 0):.1f}%

CLINICAL RELEVANCE:
{"-"*40}
Based on test set predictions:

"""
    
    # Add clinical analysis
    y_pred = predictor.model.predict(X_test)
    errors = np.abs(y_test - y_pred)
    
    thresholds = [1, 2, 3, 5]
    for threshold in thresholds:
        within_threshold = np.mean(errors <= threshold)
        report += f"Predictions within ±{threshold} units: {within_threshold:.1%}\n"
    
    dangerous_over = np.sum(y_pred - y_test > 5)
    dangerous_under = np.sum(y_test - y_pred > 5)
    total_dangerous = dangerous_over + dangerous_under
    
    report += f"\nPotentially dangerous predictions (>5 units error): {total_dangerous/len(y_test):.1%}\n"
    
    report += f"""
RECOMMENDATIONS:
{"-"*40}
1. Model suitable for educational purposes
2. Use as decision support tool only
3. Always verify with healthcare professional
4. Exercise caution with extreme values
5. Monitor model performance regularly

DISCLAIMER:
{"-"*40}
This model is for EDUCATIONAL PURPOSES ONLY.
NEVER use for actual medical treatment.
Always consult healthcare professionals for insulin dosing.

{"="*60}
END OF REPORT
"""
    
    # Save report
    report_file = 'models/training_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"✅ Training report saved to {report_file}")
    
    # Also print to console
    print(report)

def compare_models(df):
    """
    Compare different ML models
    
    Args:
        df: DataFrame containing the data
    """
    print("\n🔍 Comparing Different ML Models...")
    print("="*50)
    
    models_to_try = ['RandomForest', 'GradientBoosting', 'LinearRegression', 'Ridge']
    
    results = {}
    
    for model_name in models_to_try:
        print(f"\nTraining {model_name}...")
        
        predictor = InsulinDosePredictor(model_name=model_name)
        X_train, X_test, y_train, y_test = predictor.prepare_data(df, test_size=0.2)
        model = predictor.train(X_train, X_test, y_train, y_test, use_grid_search=False)
        
        # Store results
        results[model_name] = {
            'model': predictor,
            'metrics': predictor.training_metrics,
            'r2': predictor.training_metrics.get('R2', 0)
        }
        
        print(f"  R²: {predictor.training_metrics.get('R2', 0):.3f}")
        print(f"  MAE: {predictor.training_metrics.get('MAE', 0):.3f}")
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['r2'])
    best_r2 = results[best_model_name]['r2']
    
    print(f"\n🏆 Best Model: {best_model_name} (R²: {best_r2:.3f})")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # R² comparison
    r2_values = [results[m]['r2'] for m in models_to_try]
    axes[0].bar(models_to_try, r2_values, color=['blue', 'green', 'orange', 'red'])
    axes[0].set_title('R² Score Comparison')
    axes[0].set_ylabel('R² Score')
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis='x', rotation=45)
    
    # MAE comparison
    mae_values = [results[m]['metrics']['MAE'] for m in models_to_try]
    axes[1].bar(models_to_try, mae_values, color=['blue', 'green', 'orange', 'red'])
    axes[1].set_title('MAE Comparison')
    axes[1].set_ylabel('MAE (units)')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('models/model_comparison.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Model comparison plot saved to models/model_comparison.png")
    
    return results

if __name__ == "__main__":
    # Set style for plots
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    print("\n" + "="*60)
    print("💉 DIABEDOSE - INSULIN DOSE PREDICTOR MODEL TRAINING")
    print("="*60)
    
    try:
        # Option 1: Run complete training pipeline
        predictor = train_complete_pipeline()
        
        # Option 2: Compare different models (uncomment to run)
        # df = pd.read_csv('data/dummy_data.csv')
        # compare_models(df)
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n✨ Training script completed!")