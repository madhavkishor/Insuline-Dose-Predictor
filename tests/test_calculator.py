"""
Unit tests for insulin calculator
"""

import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.calculator import InsulinCalculator

class TestInsulinCalculator(unittest.TestCase):
    """Test cases for InsulinCalculator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = InsulinCalculator()
        
    def test_calculate_basal_dose_type1(self):
        """Test basal dose calculation for Type 1 diabetes"""
        dose = self.calculator.calculate_basal_dose(70, 1)
        expected = 70 * 0.25  # weight * 0.25
        self.assertAlmostEqual(dose, expected, places=2)
    
    def test_calculate_basal_dose_type2(self):
        """Test basal dose calculation for Type 2 diabetes"""
        dose = self.calculator.calculate_basal_dose(70, 2)
        expected = 70 * 0.15  # weight * 0.15
        self.assertAlmostEqual(dose, expected, places=2)
    
    def test_calculate_correction_dose_normal_glucose(self):
        """Test correction dose for normal glucose"""
        dose = self.calculator.calculate_correction_dose(120, 50)
        self.assertEqual(dose, 0)  # No correction needed
    
    def test_calculate_correction_dose_high_glucose(self):
        """Test correction dose for high glucose"""
        dose = self.calculator.calculate_correction_dose(200, 50)
        expected = (200 - 120) / 50  # (glucose - target) / sensitivity
        self.assertAlmostEqual(dose, expected, places=2)
    
    def test_calculate_meal_dose(self):
        """Test meal dose calculation"""
        dose = self.calculator.calculate_meal_dose(60, 15)
        expected = 60 / 15  # carbs / carb_ratio
        self.assertAlmostEqual(dose, expected, places=2)
    
    def test_calculate_total_dose_basic(self):
        """Test total dose calculation with basic parameters"""
        patient_data = {
            'glucose': 150,
            'hba1c': 7.0,
            'weight': 70,
            'age': 45,
            'diabetes_type': 2,
            'carbs': 45,
            'activity_level': 'moderate'
        }
        
        result = self.calculator.calculate_total_dose(patient_data)
        
        # Check that all required keys are present
        required_keys = ['basal_dose', 'correction_dose', 'meal_dose', 'total_dose']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check that doses are non-negative
        self.assertGreaterEqual(result['total_dose'], 0)
        
        # Check that total dose is within safety limits
        self.assertLessEqual(result['total_dose'], self.calculator.MAX_DAILY_DOSE)
    
    def test_calculate_total_dose_low_glucose(self):
        """Test total dose calculation with low glucose"""
        patient_data = {
            'glucose': 60,  # Low glucose
            'hba1c': 7.0,
            'weight': 70,
            'age': 45,
            'diabetes_type': 2,
            'carbs': 45,
            'activity_level': 'moderate'
        }
        
        result = self.calculator.calculate_total_dose(patient_data)
        
        # With low glucose, dose should be reduced
        self.assertGreaterEqual(result['total_dose'], 0)
    
    def test_calculate_total_dose_very_high_glucose(self):
        """Test total dose calculation with very high glucose"""
        patient_data = {
            'glucose': 350,  # Very high glucose
            'hba1c': 9.0,
            'weight': 70,
            'age': 45,
            'diabetes_type': 1,
            'carbs': 60,
            'activity_level': 'sedentary'
        }
        
        result = self.calculator.calculate_total_dose(patient_data)
        
        # Should have significant correction dose
        self.assertGreater(result['correction_dose'], 0)
        self.assertLessEqual(result['total_dose'], self.calculator.MAX_DAILY_DOSE)
    
    def test_safety_limits(self):
        """Test that safety limits are enforced"""
        patient_data = {
            'glucose': 500,  # Extremely high
            'hba1c': 12.0,   # Extremely high
            'weight': 150,   # High weight
            'age': 30,
            'diabetes_type': 1,
            'carbs': 200,    # High carbs
            'activity_level': 'sedentary'
        }
        
        result = self.calculator.calculate_total_dose(patient_data)
        
        # Total dose should not exceed maximum
        self.assertLessEqual(result['total_dose'], self.calculator.MAX_DAILY_DOSE)
        
        # Dose should not be negative
        self.assertGreaterEqual(result['total_dose'], 0)
    
    def test_generate_recommendations_normal(self):
        """Test recommendation generation for normal parameters"""
        recommendations = self.calculator.generate_recommendations(120, 6.0, 10)
        
        # With normal parameters, should have positive recommendation
        self.assertGreater(len(recommendations), 0)
        self.assertIn("Good Control", recommendations[0])
    
    def test_generate_recommendations_high_glucose(self):
        """Test recommendation generation for high glucose"""
        recommendations = self.calculator.generate_recommendations(280, 7.0, 15)
        
        # Should have warning for high glucose
        self.assertGreater(len(recommendations), 0)
        self.assertIn("High Glucose Alert", recommendations[0])
    
    def test_calculate_risk_level(self):
        """Test risk level calculation"""
        # Test low risk
        risk_level, color = self.calculator.calculate_risk_level(120, 6.0)
        self.assertEqual(risk_level, "Low Risk")
        
        # Test medium risk
        risk_level, color = self.calculator.calculate_risk_level(170, 7.5)
        self.assertEqual(risk_level, "Medium Risk")
        
        # Test high risk
        risk_level, color = self.calculator.calculate_risk_level(300, 9.0)
        self.assertEqual(risk_level, "High Risk")
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Zero weight
        patient_data = {
            'glucose': 120,
            'hba1c': 7.0,
            'weight': 0,
            'age': 45,
            'diabetes_type': 2,
            'carbs': 0,
            'activity_level': 'moderate'
        }
        
        result = self.calculator.calculate_total_dose(patient_data)
        self.assertGreaterEqual(result['total_dose'], 0)
        
        # Zero carbs
        patient_data['weight'] = 70
        result = self.calculator.calculate_total_dose(patient_data)
        self.assertEqual(result['meal_dose'], 0)

if __name__ == '__main__':
    unittest.main()