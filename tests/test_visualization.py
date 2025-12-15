"""
Unit tests for visualization module
"""

import unittest
import sys
import os
import plotly.graph_objects as go

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualization import DiabetesVisualizer

class TestDiabetesVisualizer(unittest.TestCase):
    """Test cases for DiabetesVisualizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.visualizer = DiabetesVisualizer()
    
    def test_create_glucose_gauge(self):
        """Test glucose gauge creation"""
        fig = self.visualizer.create_glucose_gauge(120, "Test Patient")
        
        # Should return a Plotly figure
        self.assertIsInstance(fig, go.Figure)
        
        # Should have correct layout properties
        self.assertIn('height', fig.layout.to_plotly_json())
    
    def test_create_hba1c_gauge(self):
        """Test HbA1c gauge creation"""
        fig = self.visualizer.create_hba1c_gauge(6.5)
        
        # Should return a Plotly figure
        self.assertIsInstance(fig, go.Figure)
        
        # Should have correct layout properties
        self.assertIn('height', fig.layout.to_plotly_json())
    
    def test_create_dose_breakdown_chart(self):
        """Test dose breakdown chart creation"""
        breakdown_data = {
            'Basal': 10,
            'Correction': 3,
            'Meal': 5
        }
        
        fig = self.visualizer.create_dose_breakdown_chart(breakdown_data)
        
        # Should return a Plotly figure
        self.assertIsInstance(fig, go.Figure)
        
        # Should have correct layout properties
        self.assertIn('title', fig.layout.to_plotly_json())
    
    def test_color_scheme(self):
        """Test color scheme initialization"""
        # Should have all required colors
        required_colors = ['low_risk', 'medium_risk', 'high_risk', 'normal', 'warning', 'success']
        
        for color in required_colors:
            self.assertIn(color, self.visualizer.color_scheme)
            self.assertIsInstance(self.visualizer.color_scheme[color], str)
            self.assertTrue(self.visualizer.color_scheme[color].startswith('#'))
    
    def test_create_glucose_trend_chart(self):
        """Test glucose trend chart creation"""
        # Create sample time series data
        time_series_data = {
            'datetime': ['2024-01-01 08:00', '2024-01-01 12:00', '2024-01-01 18:00'],
            'glucose': [120, 145, 130]
        }
        
        import pandas as pd
        df = pd.DataFrame(time_series_data)
        
        fig = self.visualizer.create_glucose_trend_chart(df)
        
        # Should return a Plotly figure
        self.assertIsInstance(fig, go.Figure)
        
        # Should have correct axis labels
        layout = fig.layout.to_plotly_json()
        self.assertIn('xaxis', layout)
        self.assertIn('yaxis', layout)
    
    def test_create_patient_summary_card(self):
        """Test patient summary card creation"""
        patient_data = {
            'glucose': 150,
            'hba1c': 7.2,
            'bmi': 25.5,
            'age': 45,
            'diabetes_duration': 5
        }
        
        fig = self.visualizer.create_patient_summary_card(patient_data)
        
        # Should return a Plotly figure
        self.assertIsInstance(fig, go.Figure)
        
        # Should have subplots
        self.assertGreater(len(fig.data), 0)
    
    def test_create_correlation_heatmap(self):
        """Test correlation heatmap creation"""
        import pandas as pd
        import numpy as np
        
        # Create sample data
        data = {
            'glucose': np.random.normal(150, 30, 100),
            'hba1c': np.random.normal(7.0, 1.0, 100),
            'weight': np.random.normal(70, 10, 100),
            'dose': np.random.normal(12, 4, 100)
        }
        
        df = pd.DataFrame(data)
        
        fig = self.visualizer.create_correlation_heatmap(df)
        
        # Should return a Plotly figure
        self.assertIsInstance(fig, go.Figure)
        
        # Should have heatmap data
        self.assertEqual(len(fig.data), 1)
    
    def test_create_comparison_chart(self):
        """Test comparison chart creation"""
        import pandas as pd
        import numpy as np
        
        # Create sample data
        data = {
            'glucose': np.random.normal(150, 30, 50),
            'hba1c': np.random.normal(7.0, 1.0, 50),
            'weight': np.random.normal(70, 10, 50),
            'diabetes_type': np.random.choice([1, 2], 50),
            'age': np.random.randint(20, 70, 50)
        }
        
        df = pd.DataFrame(data)
        
        fig = self.visualizer.create_comparison_chart(df, 'glucose', 'hba1c')
        
        # Should return a Plotly figure
        self.assertIsInstance(fig, go.Figure)
        
        # Should be a scatter plot
        self.assertEqual(fig.data[0].type, 'scatter')

if __name__ == '__main__':
    unittest.main()