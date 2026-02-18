
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class AppConfig:
    APP_NAME: str = "DiabeDose"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Safety limits
    MAX_DAILY_DOSE: float = 50.0  # units
    MIN_GLUCOSE: float = 70.0     # mg/dL
    MAX_GLUCOSE: float = 400.0    # mg/dL
    TARGET_GLUCOSE: float = 120.0 # mg/dL
    
    # Clinical thresholds
    HBA1C_NORMAL: float = 5.7
    HBA1C_PREDIABETES: float = 6.4
    HBA1C_DIABETES: float = 6.5
    
    GLUCOSE_NORMAL_MAX: float = 140.0
    GLUCOSE_ELEVATED_MAX: float = 180.0
    GLUCOSE_HIGH_MAX: float = 250.0
    
    # File paths
    DATA_PATH: str = "data/dummy_data.csv"
    MODEL_PATH: str = "models/insulin_predictor.pkl"
    EVAL_RESULTS_PATH: str = "models/evaluation_results.json"
    
    # Calculation parameters
    BASAL_RATE_TYPE1: float = 0.25  # units per kg
    BASAL_RATE_TYPE2: float = 0.15  # units per kg
    CARB_RATIO_TYPE1: float = 10.0  # 1 unit per 10g carbs
    CARB_RATIO_TYPE2: float = 15.0  # 1 unit per 15g carbs
    CORRECTION_FACTOR: float = 50.0  # 1 unit lowers 50 mg/dL
    
    # Activity factors
    ACTIVITY_FACTORS: Dict[str, float] = None
    
    # Risk categories
    RISK_CATEGORIES: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default values"""
        self.ACTIVITY_FACTORS = {
            'sedentary': 0.9,
            'light': 1.0,
            'moderate': 1.1,
            'active': 1.2,
            'very_active': 1.3
        }
        
        self.RISK_CATEGORIES = {
            'low': {
                'color': '#28A745',
                'icon': '🟢',
                'message': 'Well controlled'
            },
            'medium': {
                'color': '#FFC107',
                'icon': '🟡',
                'message': 'Monitor closely'
            },
            'high': {
                'color': '#DC3545',
                'icon': '🔴',
                'message': 'Requires immediate attention'
            }
        }
    
    @property
    def glucose_status(self, glucose: float) -> Dict[str, Any]:
        """Determine glucose status"""
        if glucose < self.MIN_GLUCOSE:
            return {
                'status': 'Low',
                'color': '#FF6B6B',
                'action': 'Immediate sugar needed',
                'risk': 'high'
            }
        elif glucose <= self.GLUCOSE_NORMAL_MAX:
            return {
                'status': 'Normal',
                'color': '#51CF66',
                'action': 'Good control',
                'risk': 'low'
            }
        elif glucose <= self.GLUCOSE_ELEVATED_MAX:
            return {
                'status': 'Elevated',
                'color': '#FFD93D',
                'action': 'Monitor',
                'risk': 'medium'
            }
        elif glucose <= self.GLUCOSE_HIGH_MAX:
            return {
                'status': 'High',
                'color': '#FF6B6B',
                'action': 'Correction needed',
                'risk': 'high'
            }
        else:
            return {
                'status': 'Very High',
                'color': '#DC3545',
                'action': 'Medical attention needed',
                'risk': 'high'
            }
    
    @property
    def hba1c_status(self, hba1c: float) -> Dict[str, Any]:
        """Determine HbA1c status"""
        if hba1c < self.HBA1C_NORMAL:
            return {
                'status': 'Normal',
                'color': '#51CF66',
                'risk': 'low'
            }
        elif hba1c < self.HBA1C_DIABETES:
            return {
                'status': 'Pre-diabetes',
                'color': '#FFD93D',
                'risk': 'medium'
            }
        else:
            if hba1c < 7.0:
                level = 'Controlled'
            elif hba1c < 8.0:
                level = 'Suboptimal'
            elif hba1c < 9.0:
                level = 'Poor'
            else:
                level = 'Very Poor'
            
            return {
                'status': f'Diabetes ({level})',
                'color': '#FF6B6B' if hba1c > 8.0 else '#FFA500',
                'risk': 'high' if hba1c > 8.0 else 'medium'
            }

# Global configuration instances
config = AppConfig()

def load_config_from_env():
    """Load configuration from environment variables"""
    import os
    
    config.DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Update from environment if available
    if 'MAX_DOSE_LIMIT' in os.environ:
        config.MAX_DAILY_DOSE = float(os.getenv('MAX_DOSE_LIMIT'))
    
    if 'TARGET_GLUCOSE' in os.environ:
        config.TARGET_GLUCOSE = float(os.getenv('TARGET_GLUCOSE'))
    
    if 'MODEL_PATH' in os.environ:
        config.MODEL_PATH = os.getenv('MODEL_PATH')
    
    if 'DATA_PATH' in os.environ:
        config.DATA_PATH = os.getenv('DATA_PATH')
    
    return config

# Initialize configuration
load_config_from_env()
