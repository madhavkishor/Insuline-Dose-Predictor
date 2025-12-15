import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="DiabeDose - Insulin Dose Predictor",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS FOR BETTER STYLING
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .risk-high {
        background-color: #FEE2E2;
        border-left: 5px solid #DC2626;
        padding: 1rem;
        border-radius: 5px;
    }
    .risk-medium {
        background-color: #FEF3C7;
        border-left: 5px solid #D97706;
        padding: 1rem;
        border-radius: 5px;
    }
    .risk-low {
        background-color: #D1FAE5;
        border-left: 5px solid #059669;
        padding: 1rem;
        border-radius: 5px;
    }
    .chart-container {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .insulin-high {
        background: linear-gradient(135deg, #DC2626 0%, #991B1B 100%) !important;
    }
    .insulin-medium {
        background: linear-gradient(135deg, #D97706 0%, #92400E 100%) !important;
    }
    .insulin-low {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    }
    .hba1c-poor {
        color: #DC2626;
        font-weight: bold;
    }
    .hba1c-fair {
        color: #D97706;
        font-weight: bold;
    }
    .hba1c-good {
        color: #059669;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD DATA FUNCTION
# ============================================
@st.cache_data
def load_patient_data():
    """Load patient data from CSV file"""
    try:
        csv_path = os.path.join("data", "dummy_data.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return df, True
        else:
            return None, False
    except Exception as e:
        st.sidebar.error(f"Error loading data: {e}")
        return None, False

# ============================================
# HEADER SECTION
# ============================================
st.markdown('<h1 class="main-header">💉 DiabeDose - Insulin Dose Predictor</h1>', unsafe_allow_html=True)
st.markdown("---")

# ============================================
# SIDEBAR - PATIENT INPUTS
# ============================================
with st.sidebar:
    st.header("🩺 Patient Parameters")
    
    # Load data
    df, data_loaded = load_patient_data()
    
    # Get default values from CSV if available
    if df is not None and data_loaded:
        # Try to get glucose from CSV
        glucose_cols = [col for col in df.columns if any(x in col.lower() for x in ['glucose', 'sugar', 'bg'])]
        if glucose_cols:
            current_glucose_default = int(df[glucose_cols[0]].iloc[-1])
        else:
            current_glucose_default = 250
        
        # Try to get HbA1c from CSV
        hba1c_cols = [col for col in df.columns if any(x in col.lower() for x in ['hba1c', 'a1c', 'hba'])]
        if hba1c_cols:
            hba1c_default = float(df[hba1c_cols[0]].dropna().iloc[-1])
        else:
            hba1c_default = 10.4
            
        st.success(f"✅ Loaded {len(df)} records")
    else:
        current_glucose_default = 250
        hba1c_default = 10.4
        st.warning("📁 Using default values")
    
    # HbA1c Input (NOW DYNAMIC)
    hba1c_value = st.slider(
        "**HbA1c (%)**",
        min_value=4.0,
        max_value=15.0,
        value=hba1c_default,
        step=0.1,
        help="Glycated hemoglobin - indicates 3-month average glucose",
        key="hba1c_input"
    )
    
    # Determine HbA1c status and color
    if hba1c_value >= 9.0:
        hba1c_status = "Poor Control"
        hba1c_delta_color = "inverse"
        hba1c_css_class = "hba1c-poor"
    elif hba1c_value >= 7.0:
        hba1c_status = "Needs Improvement"
        hba1c_delta_color = "off"
        hba1c_css_class = "hba1c-fair"
    else:
        hba1c_status = "Good Control"
        hba1c_delta_color = "normal"
        hba1c_css_class = "hba1c-good"
    
    # Blood Glucose Input
    current_glucose = st.number_input(
        "**Current Blood Glucose (mg/dL)**",
        min_value=50,
        max_value=500,
        value=current_glucose_default,
        step=10,
        help="Enter current blood glucose reading",
        key="glucose_input"
    )
    
    # Other Inputs
    carbs = st.number_input(
        "**Carbohydrate Intake (grams)**",
        min_value=0,
        max_value=200,
        value=50,
        step=5,
        help="Carbs in upcoming meal",
        key="carbs_input"
    )
    
    weight = st.number_input(
        "**Weight (kg)**",
        min_value=30,
        max_value=200,
        value=90,
        step=1,
        key="weight_input"
    )
    
    age = st.number_input(
        "**Age (years)**",
        min_value=1,
        max_value=120,
        value=78,
        step=1,
        key="age_input"
    )
    
    diabetes_type = st.selectbox(
        "**Diabetes Type**",
        ["Type 1", "Type 2"],
        index=1,
        key="diabetes_type_input"
    )
    
    activity_level = st.selectbox(
        "**Activity Level**",
        ["Sedentary", "Light", "Moderate", "Active", "Very Active"],
        index=2,
        key="activity_input"
    )
    
    st.markdown("---")
    st.header("⚙️ Insulin Settings")
    
    correction_factor = st.slider(
        "**Correction Factor**",
        min_value=10,
        max_value=100,
        value=50,
        help="1 unit lowers glucose by X mg/dL",
        key="correction_input"
    )
    
    carb_ratio = st.slider(
        "**Carb Ratio**",
        min_value=5,
        max_value=15,
        value=8,
        help="1 unit covers X grams of carbs",
        key="carb_ratio_input"
    )
    
    # Display current factors
    st.markdown("---")
    st.markdown("**📊 Current Factors:**")
    st.write(f"• HbA1c: <span class='{hba1c_css_class}'>{hba1c_value}%</span>", unsafe_allow_html=True)
    st.write(f"• Age: {age} years")
    st.write(f"• Type: {diabetes_type}")
    st.write(f"• Activity: {activity_level}")

# ============================================
# INSULIN DOSE CALCULATION - DYNAMIC FORMULA
# ============================================
def calculate_insulin_dose(glucose, carbs, weight, age, diabetes_type, activity_level, correction_factor, carb_ratio, hba1c):
    """Calculate insulin dose using all patient parameters"""
    
    # Target glucose (adjust based on HbA1c - higher HbA1c = lower target)
    if hba1c > 9.0:
        target_glucose = 150  # More conservative target for poor control
    elif hba1c > 7.0:
        target_glucose = 130
    else:
        target_glucose = 100  # Standard target for good control
    
    # 1. Correction Dose
    correction_dose = max(0, (glucose - target_glucose) / correction_factor)
    
    # 2. Meal Dose
    meal_dose = carbs / carb_ratio
    
    # 3. Basal Dose with all modifiers
    
    # Age factor: Older patients need less insulin
    if age < 40:
        age_factor = 1.0
    elif age < 60:
        age_factor = 0.85
    elif age < 75:
        age_factor = 0.75
    else:
        age_factor = 0.65
    
    # Diabetes type factor
    diabetes_factor = 1.2 if diabetes_type == "Type 1" else 1.0
    
    # Activity level factors
    activity_factors = {
        "Sedentary": 1.3,      # Less active = more insulin resistance
        "Light": 1.15,
        "Moderate": 1.0,
        "Active": 0.85,        # More active = better insulin sensitivity
        "Very Active": 0.7
    }
    activity_factor = activity_factors[activity_level]
    
    # HbA1c factor: Higher HbA1c indicates more insulin resistance
    hba1c_factor = 1.0 + (hba1c - 7.0) * 0.05  # 5% increase per point above 7.0
    hba1c_factor = max(1.0, min(1.5, hba1c_factor))  # Cap between 1.0 and 1.5
    
    # Weight-based basal calculation
    base_basal_per_kg = 0.5  # Units per kg per day
    weight_basal = base_basal_per_kg * weight
    
    # Apply all factors to basal dose
    basal_dose = weight_basal * age_factor * diabetes_factor * activity_factor * hba1c_factor
    
    # Total dose
    total_dose = correction_dose + meal_dose + basal_dose
    
    # Round values
    total_dose = round(total_dose, 1)
    correction_dose = round(correction_dose, 1)
    meal_dose = round(meal_dose, 1)
    basal_dose = round(basal_dose, 1)
    
    return {
        "total": total_dose,
        "correction": correction_dose,
        "meal": meal_dose,
        "basal": basal_dose,
        "age_factor": age_factor,
        "diabetes_factor": diabetes_factor,
        "activity_factor": activity_factor,
        "hba1c_factor": round(hba1c_factor, 2),
        "weight_basal": round(weight_basal, 1),
        "target_glucose": target_glucose
    }

# Calculate dose
dose_data = calculate_insulin_dose(
    current_glucose, carbs, weight, age, diabetes_type, 
    activity_level, correction_factor, carb_ratio, hba1c_value
)

# ============================================
# MAIN DASHBOARD - TOP METRICS
# ============================================
st.markdown("## 📊 Current Status")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Blood Glucose", f"{current_glucose} mg/dL", 
             delta="High" if current_glucose > 180 else "Normal",
             delta_color="inverse" if current_glucose > 180 else "normal")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("HbA1c", f"{hba1c_value}%", 
             delta=hba1c_status, 
             delta_color=hba1c_delta_color)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    bmi = weight / ((1.75) ** 2)
    bmi_category = "Obese" if bmi >= 30 else "Overweight" if bmi >= 25 else "Normal"
    st.metric("BMI", f"{bmi:.1f}", delta=bmi_category)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Age", f"{age} years", delta=f"{diabetes_type}")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ============================================
# INSULIN DOSE DISPLAY - DYNAMIC COLOR
# ============================================
st.markdown("## 🧪 Recommended Insulin Dose")

# Determine dose color based on total
total_dose = dose_data["total"]
if total_dose > 30:
    dose_class = "insulin-high"
    dose_warning = "⚠️ High Dose - Review with doctor"
elif total_dose > 20:
    dose_class = "insulin-medium"
    dose_warning = "📋 Monitor closely"
else:
    dose_class = "insulin-low"
    dose_warning = "✅ Within normal range"

# Display dose with dynamic styling
st.markdown(f"""
<div class="{dose_class}" style="
            padding: 2rem; 
            border-radius: 15px; 
            color: white;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.2);">
    <h1 style="font-size: 4rem; margin: 0; font-weight: 800;">{total_dose:.1f} units</h1>
    <p style="font-size: 1.2rem; opacity: 0.9;">{dose_warning}</p>
    <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;">
        📍 Factors: HbA1c({hba1c_value}%) | Age({age}) | Type({diabetes_type}) | Activity({activity_level})
    </div>
</div>
""", unsafe_allow_html=True)

# Show detailed breakdown
with st.expander("📝 Show Detailed Calculation & Factors", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Dose Components:**")
        st.write(f"• **Target Glucose:** {dose_data['target_glucose']} mg/dL (based on HbA1c)")
        st.write(f"• **Correction Dose:** {dose_data['correction']:.1f} units")
        st.write(f"• **Meal Dose:** {dose_data['meal']:.1f} units")
        st.write(f"• **Basal Dose:** {dose_data['basal']:.1f} units")
        st.write(f"• **Total:** **{dose_data['total']:.1f} units**")
        
        st.markdown("**🎯 Quick Adjustments:**")
        st.write(f"• Higher HbA1c → Increases basal insulin")
        st.write(f"• Increase age → Decreases basal insulin")
        st.write(f"• Change to Type 1 → Increases basal insulin")
        st.write(f"• More activity → Decreases basal insulin")
    
    with col2:
        st.markdown("**⚙️ Applied Factors:**")
        st.write(f"• **HbA1c Factor:** {dose_data['hba1c_factor']:.2f}x")
        st.write(f"• **Age Factor:** {dose_data['age_factor']:.2f}x")
        st.write(f"• **Diabetes Type:** {dose_data['diabetes_factor']:.1f}x")
        st.write(f"• **Activity Level:** {dose_data['activity_factor']:.1f}x")
        st.write(f"• **Weight Base:** {dose_data['weight_basal']:.1f} units")
        
        st.markdown("**📈 Impact Summary:**")
        hba1c_impact = f"Increases basal by {int((dose_data['hba1c_factor']-1)*100)}%" if dose_data['hba1c_factor'] > 1 else "Normal impact"
        age_impact = f"Reduces basal by {int((1-dose_data['age_factor'])*100)}%" if dose_data['age_factor'] < 1 else "No age impact"
        type_impact = "Increases basal by 20%" if dose_data['diabetes_factor'] > 1 else "Normal type impact"
        activity_impact = f"Adjusts basal by {int((dose_data['activity_factor']-1)*100)}%"
        
        st.write(f"• HbA1c {hba1c_value}%: {hba1c_impact}")
        st.write(f"• Age {age}y: {age_impact}")
        st.write(f"• {diabetes_type}: {type_impact}")
        st.write(f"• {activity_level}: {activity_impact}")
    
    st.markdown("---")
    st.markdown("**🧮 Calculation Details:**")
    st.write(f"""
    **Total Dose Formula:**
    
    `Correction + Meal + (Weight × Age Factor × Diabetes Factor × Activity Factor × HbA1c Factor)`
    
    **Step-by-step:**
    1. **Correction:** ({current_glucose} - {dose_data['target_glucose']}) ÷ {correction_factor} = **{dose_data['correction']:.1f} units**
    2. **Meal:** {carbs} g ÷ {carb_ratio} = **{dose_data['meal']:.1f} units**
    3. **Basal:** {weight} kg × 0.5 × {dose_data['age_factor']:.2f} (age) × {dose_data['diabetes_factor']:.1f} (type) × {dose_data['activity_factor']:.1f} (activity) × {dose_data['hba1c_factor']:.2f} (HbA1c) = **{dose_data['basal']:.1f} units**
    
    **Final:** {dose_data['correction']:.1f} + {dose_data['meal']:.1f} + {dose_data['basal']:.1f} = **{dose_data['total']:.1f} units**
    """)

st.markdown("---")

# ============================================
# VISUALIZATIONS SECTION
# ============================================
st.markdown("## 📈 Visualizations")

# Create 3 columns for the visualization boxes
viz_col1, viz_col2, viz_col3 = st.columns(3)

with viz_col1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 📊 Blood Glucose Trend")
    
    # Create or load glucose data
    if df is not None and len(df) > 0:
        glucose_cols = [col for col in df.columns if any(x in col.lower() for x in ['glucose', 'sugar', 'bg'])]
        if glucose_cols:
            glucose_data = df[glucose_cols[0]].tail(20).values
        else:
            # Generate sample data
            glucose_data = current_glucose + np.random.normal(0, 30, 20)
            glucose_data = np.clip(glucose_data, 50, 400)
    else:
        # Generate sample data based on current glucose
        glucose_data = current_glucose + np.random.normal(0, 30, 20)
        glucose_data = np.clip(glucose_data, 50, 400)
    
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(glucose_data, marker='o', color='red', linewidth=2)
    ax1.axhline(y=dose_data['target_glucose'], color='green', linestyle='--', label=f'Target ({dose_data["target_glucose"]})')
    ax1.axhline(y=70, color='blue', linestyle='--', label='Min (70)')
    ax1.fill_between(range(len(glucose_data)), 70, dose_data['target_glucose'], alpha=0.1, color='green')
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Glucose (mg/dL)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)
    plt.close(fig1)
    st.markdown('</div>', unsafe_allow_html=True)

with viz_col2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 💉 Insulin History")
    
    # Create insulin history with current dose as last value
    insulin_history = np.random.uniform(15, 25, size=13)
    insulin_history = np.append(insulin_history, total_dose)  # Add current calculated dose
    
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    bars = ax2.bar(range(len(insulin_history)), insulin_history, color='orange', alpha=0.7)
    bars[-1].set_color('red')  # Highlight current dose
    ax2.axhline(y=np.mean(insulin_history), color='purple', linestyle='--', 
                label=f'Avg: {np.mean(insulin_history):.1f}')
    ax2.axhline(y=total_dose, color='red', linestyle='-', alpha=0.3, linewidth=2)
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Insulin (units)")
    ax2.set_xticks(range(0, 14, 2))
    ax2.set_xticklabels([f"Day {i-13}" if i < 13 else "Today" for i in range(0, 14, 2)])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close(fig2)
    st.markdown('</div>', unsafe_allow_html=True)

with viz_col3:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 🎯 Factor Impact Analysis")
    
    # Create bar chart showing impact of each factor
    factors = ['Weight', 'Age', 'Diabetes', 'Activity', 'HbA1c']
    
    # Calculate individual impacts
    base_basal = 0.5 * weight
    age_impact = base_basal * dose_data['age_factor']
    type_impact = age_impact * dose_data['diabetes_factor']
    activity_impact = type_impact * dose_data['activity_factor']
    hba1c_impact = activity_impact * dose_data['hba1c_factor']
    
    impacts = [base_basal, age_impact, type_impact, activity_impact, hba1c_impact]
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    bars = ax3.bar(factors, impacts, color=colors, alpha=0.7)
    ax3.set_ylabel("Basal Insulin (units)")
    ax3.set_title("How Each Factor Affects Basal Insulin")
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, impact in zip(bars, impacts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{impact:.1f}', ha='center', va='bottom', fontweight='bold')
    
    st.pyplot(fig3)
    plt.close(fig3)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# PATIENT METRICS SECTION
# ============================================
st.markdown("---")
st.markdown("## 📋 Patient Metrics")

metrics_col1, metrics_col2 = st.columns(2)

with metrics_col1:
    st.markdown("### Health Metrics")
    st.write(f"- **Blood Glucose:** {current_glucose} mg/dL")
    st.write(f"- **HbA1c:** <span class='{hba1c_css_class}'>{hba1c_value}%</span> ({hba1c_status})", unsafe_allow_html=True)
    st.write(f"- **Weight:** {weight} kg")
    st.write(f"- **BMI:** {bmi:.1f} ({bmi_category})")
    st.write(f"- **Age:** {age} years")
    st.write(f"- **Diabetes Type:** {diabetes_type}")

with metrics_col2:
    st.markdown("### Lifestyle & Settings")
    st.write(f"- **Activity Level:** {activity_level}")
    st.write(f"- **Carb Intake:** {carbs} grams per meal")
    st.write(f"- **Correction Factor:** 1 unit per {correction_factor} mg/dL")
    st.write(f"- **Carb Ratio:** 1 unit per {carb_ratio} g carbs")
    st.write(f"- **HbA1c Impact:** {int((dose_data['hba1c_factor']-1)*100)}% basal adjustment")
    st.write(f"- **Age Impact:** {int((1-dose_data['age_factor'])*100)}% basal reduction")

# ============================================
# RISK ASSESSMENT SECTION
# ============================================
st.markdown("---")
st.markdown("## ⚠️ Risk Assessment")

# Calculate risk based on multiple factors
risk_score = 0

# Glucose risk
if current_glucose > 300:
    risk_score += 3
    glucose_risk = "Critical"
elif current_glucose > 250:
    risk_score += 2
    glucose_risk = "High"
elif current_glucose > 180:
    risk_score += 1
    glucose_risk = "Moderate"
else:
    glucose_risk = "Low"

# HbA1c risk
if hba1c_value > 9.0:
    risk_score += 2
    hba1c_risk = "Poor"
elif hba1c_value > 7.0:
    risk_score += 1
    hba1c_risk = "Fair"
else:
    hba1c_risk = "Good"

# Age risk
if age > 75:
    risk_score += 1
    age_risk = "Elevated (elderly)"
else:
    age_risk = "Normal"

# Dose risk
if total_dose > 40:
    risk_score += 2
    dose_risk = "High"
elif total_dose > 30:
    risk_score += 1
    dose_risk = "Moderate"
else:
    dose_risk = "Low"

# Overall risk assessment
if risk_score >= 5:
    risk_class = "risk-high"
    st.markdown('<div class="risk-high">', unsafe_allow_html=True)
    st.markdown("### 🔴 HIGH RISK")
    st.write(f"**Multiple high-risk factors detected:**")
    st.write(f"• Glucose: {current_glucose} mg/dL ({glucose_risk})")
    st.write(f"• HbA1c: {hba1c_value}% ({hba1c_risk})")
    st.write(f"• Age: {age} years ({age_risk})")
    st.write(f"• Insulin Dose: {total_dose} units ({dose_risk})")
    st.write("**Immediate Actions:**")
    st.write("1. Contact healthcare provider immediately")
    st.write("2. Monitor glucose every 2 hours")
    st.write("3. Review all insulin calculations")
    st.write("4. Check for symptoms of hyperglycemia")
    st.markdown('</div>', unsafe_allow_html=True)
    
elif risk_score >= 3:
    risk_class = "risk-medium"
    st.markdown('<div class="risk-medium">', unsafe_allow_html=True)
    st.markdown("### 🟡 MEDIUM RISK")
    st.write(f"**Some concerning factors present:**")
    st.write(f"• Glucose: {current_glucose} mg/dL ({glucose_risk})")
    st.write(f"• HbA1c: {hba1c_value}% ({hba1c_risk})")
    st.write(f"• Age: {age} years ({age_risk})")
    st.write(f"• Insulin Dose: {total_dose} units ({dose_risk})")
    st.write("**Recommended Actions:**")
    st.write("1. Monitor closely for 24 hours")
    st.write("2. Adjust insulin as calculated")
    st.write("3. Follow up with doctor in 3 days")
    st.write("4. Review diet and activity")
    st.markdown('</div>', unsafe_allow_html=True)
    
else:
    risk_class = "risk-low"
    st.markdown('<div class="risk-low">', unsafe_allow_html=True)
    st.markdown("### 🟢 LOW RISK")
    st.write(f"**Risk factors within acceptable range:**")
    st.write(f"• Glucose: {current_glucose} mg/dL ({glucose_risk})")
    st.write(f"• HbA1c: {hba1c_value}% ({hba1c_risk})")
    st.write(f"• Age: {age} years ({age_risk})")
    st.write(f"• Insulin Dose: {total_dose} units ({dose_risk})")
    st.write("**Continue current management plan**")
    st.write("1. Maintain regular monitoring")
    st.write("2. Continue healthy lifestyle")
    st.write("3. Regular follow-ups as scheduled")
    st.markdown('</div>', unsafe_allow_html=True)

# Show risk score
st.caption(f"Risk Score: {risk_score}/8 (lower is better)")

# ============================================
# DISCLAIMER
# ============================================
st.markdown("---")
st.warning("""
**⚠️ IMPORTANT DISCLAIMER:** 
This tool is for EDUCATIONAL and DEMONSTRATION purposes only. It is NOT for actual medical treatment. 
Always consult with a qualified healthcare professional for medical advice. 
Do not adjust insulin doses without medical supervision.
Insulin calculations are estimates and individual responses may vary.
""")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>DiabeDose v2.1</strong> | Insulin Dose Predictor | Last Updated: 2024</p>
    <p>For Educational Use Only | Always consult with your healthcare provider for medical decisions</p>
</div>
""", unsafe_allow_html=True)