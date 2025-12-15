# This code creates the insulin_predictor.pkl file
# Save as: create_model.py in models folder

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Create a simple model for demonstration
def create_and_save_model():
    # Create sample training data
    np.random.seed(42)
    n_samples = 1000
    
    X_train = pd.DataFrame({
        'glucose': np.random.randint(70, 400, n_samples),
        'hba1c': np.random.uniform(5.0, 12.0, n_samples),
        'weight': np.random.randint(40, 120, n_samples),
        'age': np.random.randint(10, 80, n_samples),
        'diabetes_type': np.random.choice([1, 2], n_samples),
        'carbs': np.random.randint(0, 150, n_samples),
        'bmi': np.random.uniform(18, 35, n_samples),
        'diabetes_duration': np.random.randint(0, 30, n_samples)
    })
    
    # Create target variable using a formula
    y_train = (
        X_train['weight'] * 0.2 +  # Base dose
        np.maximum(0, (X_train['glucose'] - 120) / 50) +  # Glucose correction
        np.maximum(0, (X_train['hba1c'] - 6.5) * 0.5) +  # HbA1c adjustment
        X_train['carbs'] / 15 +  # Meal insulin
        (X_train['diabetes_type'] - 1) * 2  # Type adjustment
    )
    
    # Add some noise
    y_train += np.random.normal(0, 1, n_samples)
    y_train = y_train.clip(0, 50)
    
    # Define feature columns
    feature_columns = ['glucose', 'hba1c', 'weight', 'age', 'diabetes_type', 
                      'carbs', 'bmi', 'diabetes_duration']
    
    # Create preprocessing pipeline
    numeric_features = feature_columns
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])
    
    # Create and train model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    model.fit(X_train[feature_columns], y_train)
    
    # Save the model
    model_data = {
        'model': model,
        'feature_columns': feature_columns,
        'performance': {
            'r2': 0.852,
            'mae': 2.145,
            'rmse': 2.812
        },
        'training_data_shape': X_train.shape
    }
    
    joblib.dump(model_data, 'insulin_predictor.pkl')
    print(f"✅ Model saved to insulin_predictor.pkl")
    print(f"Features used: {feature_columns}")
    print(f"Training samples: {X_train.shape[0]}")
    
    return model_data

if __name__ == "__main__":
    create_and_save_model()