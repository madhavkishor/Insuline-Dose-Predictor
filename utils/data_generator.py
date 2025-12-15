"""
Diabetes Data Generator for creating sample datasets
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class DiabetesDataGenerator:
    """Generate realistic diabetes datasets for training and testing"""
    
    def __init__(self, seed=42):
        """Initialize with random seed"""
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_patient_data(self, n_patients=1000):
        """
        Generate patient dataset with features and insulin doses
        
        Args:
            n_patients: Number of patient records to generate
            
        Returns:
            DataFrame with patient data
        """
        print(f"Generating {n_patients} patient records...")
        
        # Patient IDs
        patient_ids = list(range(1, n_patients + 1))
        
        # Generate basic demographics
        ages = np.random.randint(10, 80, n_patients)
        weights = np.random.uniform(40, 120, n_patients).round(1)
        heights = np.random.uniform(150, 190, n_patients).round(1)
        
        # Calculate BMI
        bmis = np.zeros(n_patients)
        for i in range(n_patients):
            height_m = heights[i] / 100
            bmis[i] = round(weights[i] / (height_m ** 2), 1)
        
        # Diabetes type (30% Type 1, 70% Type 2)
        diabetes_types = np.random.choice([1, 2], n_patients, p=[0.3, 0.7])
        diabetes_durations = np.random.randint(0, 30, n_patients)
        
        # Generate medical parameters
        glucose_levels = []
        hba1c_values = []
        carbs_intake = []
        activity_levels = []
        
        activity_options = ['sedentary', 'light', 'moderate', 'active', 'very_active']
        activity_probs = [0.2, 0.3, 0.3, 0.15, 0.05]
        
        for i in range(n_patients):
            # Base glucose based on diabetes type
            if diabetes_types[i] == 1:
                base_glucose = np.random.normal(180, 40)
                base_hba1c = np.random.normal(7.5, 1.0)
            else:
                base_glucose = np.random.normal(150, 35)
                base_hba1c = np.random.normal(7.0, 0.8)
            
            # Add variation
            glucose = np.random.normal(base_glucose, 30)
            glucose = max(70, min(400, round(glucose, 1)))
            
            hba1c = np.random.normal(base_hba1c, 0.5)
            hba1c = max(4.0, min(15.0, round(hba1c, 1)))
            
            carbs = np.random.randint(0, 150)
            activity = np.random.choice(activity_options, p=activity_probs)
            
            glucose_levels.append(glucose)
            hba1c_values.append(hba1c)
            carbs_intake.append(carbs)
            activity_levels.append(activity)
        
        # Calculate insulin doses using realistic formula
        recommended_doses = []
        previous_doses = []
        
        for i in range(n_patients):
            glucose = glucose_levels[i]
            hba1c = hba1c_values[i]
            weight = weights[i]
            age = ages[i]
            diabetes_type = diabetes_types[i]
            carbs = carbs_intake[i]
            bmi = bmis[i]
            
            # Realistic dose calculation
            if diabetes_type == 1:
                basal = weight * 0.25
                carb_ratio = 10
                correction_factor = 50
            else:
                basal = weight * 0.15
                carb_ratio = 15
                correction_factor = 60
            
            # Glucose correction
            if glucose > 120:
                correction = (glucose - 120) / correction_factor
            else:
                correction = 0
            
            # HbA1c adjustment
            hba1c_adj = max(0, (hba1c - 6.5) * 0.3)
            
            # Meal insulin
            meal = carbs / carb_ratio
            
            # Age adjustment
            if age > 60:
                age_factor = 1.1
            elif age < 18:
                age_factor = 0.9
            else:
                age_factor = 1.0
            
            # BMI adjustment
            bmi_factor = 1.0
            if bmi > 30:
                bmi_factor = 1.2  # Insulin resistance
            elif bmi > 25:
                bmi_factor = 1.1
            
            # Calculate total dose
            total = (basal + correction + hba1c_adj + meal) * age_factor * bmi_factor
            
            # Add some random variation
            total += np.random.normal(0, 1.5)
            total = max(0, min(50, round(total, 1)))
            
            # Previous dose (simulate previous calculation)
            previous = total * np.random.uniform(0.8, 1.2)
            previous = max(0, min(50, round(previous, 1)))
            
            recommended_doses.append(total)
            previous_doses.append(previous)
        
        # Create DataFrame
        df = pd.DataFrame({
            'patient_id': patient_ids,
            'age': ages,
            'weight': weights,
            'height': heights,
            'bmi': bmis,
            'diabetes_type': diabetes_types,
            'diabetes_duration': diabetes_durations,
            'glucose': glucose_levels,
            'hba1c': hba1c_values,
            'carbs': carbs_intake,
            'activity': activity_levels,
            'previous_dose': previous_doses,
            'recommended_dose': recommended_doses
        })
        
        # Add derived columns
        df['insulin_sensitivity'] = df.apply(
            lambda row: 50 if row['diabetes_type'] == 1 else 60, axis=1
        )
        df['carb_ratio'] = df.apply(
            lambda row: 10 if row['diabetes_type'] == 1 else 15, axis=1
        )
        
        print(f"✅ Generated {len(df)} patient records")
        return df
    
    def generate_time_series_data(self, patient_id=1, n_days=30):
        """
        Generate time-series glucose and insulin data for a patient
        
        Args:
            patient_id: Patient ID
            n_days: Number of days of data
            
        Returns:
            DataFrame with time-series data
        """
        print(f"Generating {n_days} days of time-series data for patient {patient_id}...")
        
        dates = []
        glucose_readings = []
        insulin_doses = []
        carb_intakes = []
        meal_types = []
        
        start_date = datetime.now() - timedelta(days=n_days)
        
        meal_options = ['Breakfast', 'Lunch', 'Dinner', 'Snack']
        meal_probs = [0.3, 0.3, 0.3, 0.1]
        
        for day in range(n_days):
            current_date = start_date + timedelta(days=day)
            
            # Generate 4-6 readings per day
            n_readings = np.random.randint(4, 7)
            
            for reading_num in range(n_readings):
                # Time of day
                hour = np.random.randint(6, 22)  # Between 6 AM and 10 PM
                minute = np.random.randint(0, 60)
                
                timestamp = current_date.replace(hour=hour, minute=minute)
                dates.append(timestamp)
                
                # Glucose with circadian pattern
                if hour < 10:  # Morning (dawn phenomenon)
                    base_glucose = 160 + np.random.normal(0, 20)
                elif hour < 14:  # Midday
                    base_glucose = 140 + np.random.normal(0, 20)
                elif hour < 18:  # Afternoon
                    base_glucose = 150 + np.random.normal(0, 20)
                else:  # Evening
                    base_glucose = 130 + np.random.normal(0, 20)
                
                glucose = max(70, min(300, round(base_glucose, 1)))
                glucose_readings.append(glucose)
                
                # Insulin dose (simplified calculation)
                if glucose > 120:
                    insulin = (glucose - 120) / 50 + np.random.normal(2, 1)
                else:
                    insulin = np.random.normal(1, 0.5)
                
                insulin = max(0, round(insulin, 1))
                insulin_doses.append(insulin)
                
                # Carbs and meal type
                if reading_num == 0:  # Breakfast
                    carbs = np.random.randint(30, 60)
                    meal_type = 'Breakfast'
                elif reading_num == 1:  # Lunch
                    carbs = np.random.randint(40, 70)
                    meal_type = 'Lunch'
                elif reading_num == 2:  # Dinner
                    carbs = np.random.randint(50, 80)
                    meal_type = 'Dinner'
                else:  # Snack
                    carbs = np.random.randint(10, 30)
                    meal_type = np.random.choice(meal_options, p=meal_probs)
                
                carb_intakes.append(carbs)
                meal_types.append(meal_type)
        
        # Create DataFrame
        df = pd.DataFrame({
            'patient_id': patient_id,
            'timestamp': dates,
            'glucose': glucose_readings,
            'insulin_dose': insulin_doses,
            'carbs': carb_intakes,
            'meal_type': meal_types
        })
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✅ Generated {len(df)} time-series records")
        return df
    
    def save_to_csv(self, df, filename):
        """
        Save DataFrame to CSV file
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        df.to_csv(filename, index=False)
        print(f"✅ Data saved to {filename} ({len(df)} records)")
        return filename

# Example usage
if __name__ == "__main__":
    # Create generator
    generator = DiabetesDataGenerator()
    
    # Generate patient data
    patient_data = generator.generate_patient_data(500)
    generator.save_to_csv(patient_data, 'data/sample_patients.csv')
    
    # Generate time-series data for first patient
    time_series_data = generator.generate_time_series_data(patient_id=1, n_days=7)
    generator.save_to_csv(time_series_data, 'data/time_series_sample.csv')