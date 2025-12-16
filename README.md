# **💉 DiabeDose - Intelligent Insulin Dose Predictor**

A comprehensive, interactive diabetes management application that calculates personalized insulin doses using multiple patient parameters. Built with Streamlit for real-time visualizations and medical-grade calculations (for educational purposes).

## **📊 Project Overview**
DiabeDose helps diabetes patients and healthcare professionals calculate insulin doses by considering **8+ critical factors**:
- Current Blood Glucose
- Carbohydrate Intake
- Weight & BMI
- Age
- Diabetes Type (Type 1/Type 2)
- Activity Level
- HbA1c Levels
- Individual Correction Factors

## **✨ Features**
- **🧠 Smart Dose Calculation**: Dynamic algorithm adjusting for all patient parameters
- **📈 Real-time Visualizations**: Interactive charts showing glucose trends and insulin history
- **⚠️ Risk Assessment Engine**: Multi-factor scoring with color-coded alerts
- **🎯 Personalized Targets**: Glucose targets adjust based on HbA1c levels
- **📊 CSV Integration**: Works with real patient data (500+ records supported)
- **📱 Responsive Design**: Works on desktop and mobile devices

## **🛠️ Tech Stack**
- **Frontend**: Streamlit, Custom CSS
- **Backend**: Python 3.8+
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib
- **Version Control**: Git, GitHub

## **🚀 Quick Start**

### **Prerequisites**
- Python 3.8 or higher
- Git

### **Installation**

1. **Clone the repository**
```bash
git clone https://github.com/madhavkishor/Insuline-Dose-Predictor.git
cd Insuline-Dose-Predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Open your browser**
Navigate to `http://localhost:8501`

### **Using Sample Data**
The app includes a sample dataset:
- **Location**: `data/dummy_data.csv`
- **Records**: 500+ simulated patient entries
- **Format**: CSV with glucose, insulin, and time data

## **📁 Project Structure**
```
Insuline-Dose-Predictor/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This documentation
│
├── data/
│   └── dummy_data.csv       # Sample patient data (500+ records)
│
├── assets/
│   ├── screenshots/         # Application screenshots
│   └── diagrams/            # Architecture diagrams
│
└── tests/
    └── test_calculations.py  # Unit tests for dose calculations
```

## **🧮 How Calculations Work**

### **Insulin Dose Formula**
```
Total Dose = Correction Dose + Meal Dose + Basal Dose

Correction Dose = (Current Glucose - Target Glucose) ÷ Correction Factor
Meal Dose = Carbohydrates ÷ Carb Ratio
Basal Dose = Weight × Age Factor × Diabetes Factor × Activity Factor × HbA1c Factor
```

### **Factor Adjustments**
| Factor | Impact | Example Values |
|--------|--------|----------------|
| **Age** | Older = Less insulin | 78y: ×0.65, 40y: ×1.0 |
| **Diabetes Type** | Type 1 needs more | Type 1: ×1.2, Type 2: ×1.0 |
| **Activity** | Active = Less insulin | Sedentary: ×1.3, Very Active: ×0.7 |
| **HbA1c** | Higher = More insulin | 10.4%: ×1.17, 7.0%: ×1.0 |

### **Target Glucose by HbA1c**
| HbA1c Range | Target Glucose | Category |
|-------------|----------------|----------|
| < 7.0% | 100 mg/dL | Good Control |
| 7.0-9.0% | 130 mg/dL | Needs Improvement |
| > 9.0% | 150 mg/dL | Poor Control |

## **🎮 Usage Guide**

### **1. Setting Patient Parameters**
Use the sidebar to input:
- **Current Glucose**: 70-500 mg/dL
- **HbA1c**: 4.0-15.0% (affects basal insulin)
- **Carbs**: 0-200g per meal
- **Weight, Age, Diabetes Type, Activity Level**
- **Correction Settings**: Customize sensitivity

### **2. Understanding Results**
- **Green Dose ( <20 units)**: Normal range
- **Orange Dose (20-30 units)**: Monitor closely
- **Red Dose ( >30 units)**: Review with doctor

### **3. Risk Assessment Colors**
- **🟢 Low Risk**: All parameters in safe range
- **🟡 Medium Risk**: 1-2 concerning factors
- **🔴 High Risk**: Multiple risk factors present

## **🧪 Testing**
Run unit tests for calculation accuracy:
```bash
python -m pytest tests/test_calculations.py -v
```

### **Test Cases Include:**
- Dose calculation for different ages
- Type 1 vs Type 2 differences
- Activity level adjustments
- HbA1c impact on targets

## **📈 Performance Metrics**
- **Load Time**: < 2 seconds for 500 records
- **Calculation Speed**: < 100ms per dose calculation
- **Memory Usage**: < 200MB typical

## **🌐 Deployment**

### **Local Development**
```bash
# Development mode with hot reload
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### **Streamlit Cloud Deployment**
1. Push to GitHub
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Select `app.py` as main file
4. Set Python version to 3.8+

### **Docker Deployment**
```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## **🤝 Contributing**
We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### **Code Style**
- Follow PEP 8 guidelines
- Add docstrings for functions
- Include tests for new features
- Update documentation accordingly

## **⚠️ Medical Disclaimer**
**IMPORTANT**: This application is for **EDUCATIONAL AND DEMONSTRATION PURPOSES ONLY**.

- ❌ **NOT** for actual medical treatment
- ❌ **NOT** a substitute for professional medical advice
- ❌ **DO NOT** adjust insulin doses without doctor supervision
- ✅ **ALWAYS** consult with healthcare professionals

The calculations are estimates based on simplified medical formulas. Individual responses to insulin vary significantly.

## **📚 Educational Resources**
- [American Diabetes Association](https://www.diabetes.org/)
- [Insulin Calculation Guidelines](https://www.ncbi.nlm.nih.gov/books/NBK279156/)
- [Diabetes Self-Management Education](https://www.cdc.gov/diabetes/dsmes-toolkit/index.html)

## **👥 Authors**
- **Madhav Kishor** - Initial development

## **📄 License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## **🙏 Acknowledgments**
- Diabetes healthcare professionals for formula guidance
- Streamlit team for the amazing framework
- Open-source community for invaluable tools
- Test users for feedback and suggestions

## **📞 Support**
- **Issues**: [GitHub Issues](https://github.com/madhavkishor/Insuline-Dose-Predictor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/madhavkishor/Insuline-Dose-Predictor/discussions)
- **Email**: project.diabedose@example.com

---

<div align="center">
  
**Made with ❤️ for better diabetes management**

[![GitHub stars](https://img.shields.io/github/stars/madhavkishor/Insuline-Dose-Predictor?style=social)](https://github.com/madhavkishor/Insuline-Dose-Predictor/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/madhavkishor/Insuline-Dose-Predictor?style=social)](https://github.com/madhavkishor/Insuline-Dose-Predictor/network)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## **🔗 Quick Links**
- [View Source Code](app.py)
- [View Sample Data](data/dummy_data.csv)
- [Run Tests](tests/test_calculations.py)
- [Report Bug](https://github.com/madhavkishor/Insuline-Dose-Predictor/issues/new?template=bug_report.md)
- [Request Feature](https://github.com/madhavkishor/Insuline-Dose-Predictor/issues/new?template=feature_request.md)

---

*Last Updated: December 2025 | Version: 2.1*
