"""
Advanced Insulin Dose Calculator
"""

import numpy as np
from datetime import datetime

class AdvancedInsulinCalculator:
    """Advanced insulin dose calculator with more features"""
    
    def __init__(self):
        # Safety parameters
        self.MAX_DAILY_DOSE = 50  # units
        self.MIN_GLUCOSE = 70     # mg/dL
        self.TARGET_GLUCOSE = 120  # mg/dL
        self.MAX_GLUCOSE = 400    # mg/dL
        
        # Default parameters
        self.BASAL_RATE_TYPE1 = 0.25  # units/kg
        self.BASAL_RATE_TYPE2 = 0.15  # units/kg
        self.CARB_RATIO_TYPE1 = 10    # 1 unit per 10g carbs
        self.CARB_RATIO_TYPE2 = 15    # 1 unit per 15g carbs
        self.CORRECTION_FACTOR = 50   # 1 unit lowers 50 mg/dL
        
    def calculate_risk_score(self, glucose, hba1c, age, weight):
        """Calculate diabetes risk score (0-100)"""
        risk = 0
        
        # Glucose risk (0-40 points)
        if glucose >= 300:
            risk += 40
        elif glucose >= 250:
            risk += 30
        elif glucose >= 180:
            risk += 20
        elif glucose >= 140:
            risk += 10
        elif glucose < 70:
            risk += 30
        elif glucose < 100:
            risk += 5
        
        # HbA1c risk (0-30 points)
        if hba1c >= 10:
            risk += 30
        elif hba1c >= 9:
            risk += 25
        elif hba1c >= 8:
            risk += 20
        elif hba1c >= 7:
            risk += 15
        elif hba1c >= 6.5:
            risk += 10
        
        # Age risk (0-15 points)
        if age >= 60:
            risk += 15
        elif age >= 50:
            risk += 10
        elif age >= 40:
            risk += 5
        
        # Weight risk (BMI estimate, 0-15 points)
        bmi = weight / (1.7 ** 2)  # Assuming average height 1.7m
        if bmi >= 35:
            risk += 15
        elif bmi >= 30:
            risk += 10
        elif bmi >= 25:
            risk += 5
        
        return min(100, risk)
    
    def get_risk_level(self, risk_score):
        """Convert risk score to level"""
        if risk_score >= 60:
            return "High Risk", "danger"
        elif risk_score >= 30:
            return "Medium Risk", "warning"
        else:
            return "Low Risk", "success"
    
    def calculate_comprehensive_dose(self, patient_data):
        """
        Calculate comprehensive insulin dose
        
        Args:
            patient_data (dict): {
                'glucose': current blood glucose (mg/dL),
                'hba1c': HbA1c percentage,
                'weight': weight in kg,
                'age': age in years,
                'diabetes_type': 1 or 2,
                'carbs': carbohydrate intake in grams,
                'activity_level': 'sedentary' to 'very_active',
                'time_since_last_meal': hours since last meal,
                'time_since_last_insulin': hours since last insulin,
                'previous_dose': previous insulin dose
            }
        
        Returns:
            dict: Comprehensive dose calculation
        """
        # Extract values with defaults
        glucose = patient_data.get('glucose', 120)
        hba1c = patient_data.get('hba1c', 7.0)
        weight = patient_data.get('weight', 70)
        age = patient_data.get('age', 45)
        diabetes_type = patient_data.get('diabetes_type', 2)
        carbs = patient_data.get('carbs', 0)
        activity = patient_data.get('activity_level', 'moderate')
        time_since_meal = patient_data.get('time_since_last_meal', 3)
        time_since_insulin = patient_data.get('time_since_last_insulin', 4)
        previous_dose = patient_data.get('previous_dose', 0)
        
        # Calculate risk score
        risk_score = self.calculate_risk_score(glucose, hba1c, age, weight)
        risk_level, risk_color = self.get_risk_level(risk_score)
        
        # Calculate basal dose
        if diabetes_type == 1:
            basal_dose = weight * self.BASAL_RATE_TYPE1
            carb_ratio = self.CARB_RATIO_TYPE1
        else:
            basal_dose = weight * self.BASAL_RATE_TYPE2
            carb_ratio = self.CARB_RATIO_TYPE2
        
        # Calculate correction dose
        if glucose > self.TARGET_GLUCOSE:
            correction_dose = (glucose - self.TARGET_GLUCOSE) / self.CORRECTION_FACTOR
        else:
            correction_dose = 0
        
        # Adjust correction based on HbA1c
        hba1c_factor = 1.0
        if hba1c > 8:
            hba1c_factor = 1.2  # More resistant
        elif hba1c > 7:
            hba1c_factor = 1.1
        elif hba1c < 6:
            hba1c_factor = 0.9  # More sensitive
        
        correction_dose *= hba1c_factor
        
        # Calculate meal dose
        meal_dose = carbs / carb_ratio if carb_ratio > 0 else 0
        
        # Adjust meal dose based on time since last meal
        if time_since_meal < 1:
            meal_dose *= 1.0  # Full dose for recent meal
        elif time_since_meal < 3:
            meal_dose *= 0.7  # Reduced for meal 1-3 hours ago
        else:
            meal_dose *= 0.5  # Further reduced for older meal
        
        # Activity adjustment
        activity_factors = {
            'sedentary': 1.0,
            'light': 0.95,
            'moderate': 0.9,
            'active': 0.85,
            'very_active': 0.8
        }
        activity_factor = activity_factors.get(activity.lower(), 1.0)
        
        # Age adjustment
        if age > 65:
            age_factor = 0.9  # Elderly may be more sensitive
        elif age < 18:
            age_factor = 1.1  # Children may be less sensitive
        else:
            age_factor = 1.0
        
        # Calculate total dose
        total_dose = (basal_dose + correction_dose + meal_dose) * activity_factor * age_factor
        
        # Consider previous dose (avoid stacking)
        if time_since_insulin < 4:
            # Reduce dose if recent insulin
            insulin_on_board = previous_dose * (1 - (time_since_insulin / 4))
            total_dose = max(0, total_dose - insulin_on_board)
        
        # Safety adjustments
        if glucose < self.MIN_GLUCOSE:
            # Reduce dose if low glucose
            total_dose = max(0, total_dose - 2)
        
        # Apply maximum limit
        total_dose = min(total_dose, self.MAX_DAILY_DOSE)
        
        # Ensure non-negative
        total_dose = max(0, total_dose)
        
        # Prepare comprehensive result
        result = {
            'basal_dose': round(basal_dose, 1),
            'correction_dose': round(correction_dose, 1),
            'meal_dose': round(meal_dose, 1),
            'total_dose': round(total_dose, 1),
            'risk_score': round(risk_score, 1),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'carb_ratio': carb_ratio,
            'correction_factor': self.CORRECTION_FACTOR,
            'parameters': {
                'glucose': glucose,
                'hba1c': hba1c,
                'weight': weight,
                'age': age,
                'diabetes_type': diabetes_type,
                'carbs': carbs,
                'activity': activity
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def generate_recommendations(self, result):
        """Generate recommendations based on calculation"""
        recommendations = []
        
        glucose = result['parameters']['glucose']
        hba1c = result['parameters']['hba1c']
        total_dose = result['total_dose']
        risk_level = result['risk_level']
        
        # Glucose recommendations
        if glucose > 250:
            recommendations.append("**Emergency**: Glucose very high (>250 mg/dL). Seek immediate medical attention.")
        elif glucose > 180:
            recommendations.append("**Warning**: Glucose elevated (180-250 mg/dL). Monitor closely and consider lifestyle adjustments.")
        elif glucose < 70:
            recommendations.append("**Emergency**: Glucose low (<70 mg/dL). Treat immediately with fast-acting carbohydrates.")
        elif glucose < 100:
            recommendations.append("**Caution**: Glucose approaching low range. Monitor and have quick sugar available.")
        
        # HbA1c recommendations
        if hba1c > 9:
            recommendations.append(f"**Poor Control**: HbA1c {hba1c}% indicates need for treatment adjustment. Consult healthcare provider.")
        elif hba1c > 8:
            recommendations.append(f"**Suboptimal**: HbA1c {hba1c}% needs improvement. Consider medication or lifestyle changes.")
        elif hba1c > 7:
            recommendations.append(f"**Fair Control**: HbA1c {hba1c}%. Room for improvement to reach target <7%.")
        
        # Dose recommendations
        if total_dose > 40:
            recommendations.append("**High Dose**: Consider splitting dose or consulting healthcare provider.")
        elif total_dose > 30:
            recommendations.append("**Moderate Dose**: Monitor for hypoglycemia, especially if active.")
        
        # Risk level recommendations
        if risk_level == "High Risk":
            recommendations.append("**High Risk Patient**: Requires close monitoring and frequent healthcare provider follow-up.")
        elif risk_level == "Medium Risk":
            recommendations.append("**Medium Risk Patient**: Regular monitoring and lifestyle modifications recommended.")
        
        # General recommendations
        recommendations.extend([
            "**Monitor**: Check glucose 2-4 hours after insulin administration.",
            "**Emergency Preparedness**: Always carry fast-acting carbohydrates.",
            "**Documentation**: Record all insulin doses and glucose readings.",
            "**Medical ID**: Wear identification indicating diabetes status.",
            "**Education**: Ensure family knows hypoglycemia signs and treatment."
        ])
        
        return recommendations

# Singleton instance for easy import
calculator = AdvancedInsulinCalculator()