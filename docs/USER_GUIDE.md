
---

### **23. **docs/USER_GUIDE.md** (User Guide)
```markdown
# DiabeDose User Guide

## Introduction
DiabeDose is an educational tool for predicting insulin doses based on diabetes parameters. This guide will help you understand and use the application effectively.

## Getting Started

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/example/diabedose.git
cd diabedose

# Install dependencies
pip install -r requirements.txt

# Generate sample data
python -c "from utils.data_generator import DiabetesDataGenerator; \
           g = DiabetesDataGenerator(); \
           df = g.generate_patient_data(1000); \
           g.save_to_csv(df, 'data/dummy_data.csv')"

# Train the model
python models/model_training.py

# Run the application
streamlit run app.py