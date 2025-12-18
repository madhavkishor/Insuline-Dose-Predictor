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
# ENHANCED CSS FOR PROFESSIONAL LOOK
# ============================================
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Metric cards with hover effects */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    /* Interactive buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Risk styling */
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
    
    /* Chart containers */
    .chart-container {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border: 1px solid #E5E7EB;
    }
    
    /* Insulin dose colors */
    .insulin-high {
        background: linear-gradient(135deg, #DC2626 0%, #991B1B 100%) !important;
    }
    .insulin-medium {
        background: linear-gradient(135deg, #D97706 0%, #92400E 100%) !important;
    }
    .insulin-low {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    }
    
    /* HbA1c colors */
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
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-good { background-color: #10B981; }
    .status-warning { background-color: #F59E0B; }
    .status-danger { background-color: #EF4444; }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-header { font-size: 1.8rem; }
        .metric-card { padding: 1rem; }
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
# INSULIN DOSE CALCULATION FUNCTION
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
    
    # THEME SELECTOR
    st.markdown("---")
    st.markdown("### 🎨 Theme Settings")
    
    theme = st.selectbox(
        "Color Theme",
        ["Default (Medical Blue)", "Dark Mode", "High Contrast", "Colorblind Friendly"],
        index=0
    )
    
    st.markdown("---")
    st.header("📋 Patient Data")
    
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
    st.write(f"• Correction: 1 unit per {correction_factor} mg/dL")
    st.write(f"• Carb Ratio: 1 unit per {carb_ratio} g")

# ============================================
# CALCULATE INSULIN DOSE
# ============================================
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

total_dose = dose_data["total"]

# Determine dose color based on total
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

# ============================================
# ADVANCED ANALYTICS DASHBOARD
# ============================================
st.markdown("---")
st.markdown("## 🔍 Advanced Analytics")

analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs(["📈 Trends", "🎯 Targets", "📋 Statistics"])

with analytics_tab1:
    if df is not None and len(df) > 0:
        # Find numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_x = st.selectbox("X-axis", numeric_cols, index=0, key="x_axis_select")
            with col2:
                selected_y = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1), key="y_axis_select")
            
            # Scatter plot with trendline
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(df[selected_x], df[selected_y], 
                                c=df[selected_x] if len(numeric_cols) > 2 else 'blue',
                                alpha=0.6, s=50, cmap='viridis')
            
            # Add trendline
            try:
                valid_data = df[[selected_x, selected_y]].dropna()
                if len(valid_data) > 1:
                    z = np.polyfit(valid_data[selected_x], valid_data[selected_y], 1)
                    p = np.poly1d(z)
                    ax.plot(valid_data[selected_x], p(valid_data[selected_x]), "r--", alpha=0.8, label=f'Trendline')
            except:
                pass
            
            ax.set_xlabel(selected_x)
            ax.set_ylabel(selected_y)
            ax.set_title(f'{selected_y} vs {selected_x}')
            ax.grid(True, alpha=0.3)
            
            if len(numeric_cols) > 2:
                plt.colorbar(scatter, ax=ax, label=selected_x)
            
            st.pyplot(fig)
            plt.close(fig)
            
            # Calculate correlation
            try:
                correlation = df[selected_x].corr(df[selected_y])
                st.metric("Correlation Coefficient", f"{correlation:.3f}", 
                         delta="Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.3 else "Weak")
            except:
                st.info("Could not calculate correlation")
        else:
            st.info("Need at least 2 numeric columns for trend analysis")
    else:
        st.info("Load CSV data with numeric columns for advanced analytics")

with analytics_tab2:
    st.subheader("Goal Setting & Progress Tracking")
    
    col1, col2 = st.columns(2)
    
    with col1:
        glucose_target = st.slider("Glucose Target (mg/dL)", 80, 150, 130, key="glucose_target")
        hba1c_target = st.slider("HbA1c Target (%)", 5.0, 8.0, 7.0, key="hba1c_target")
        weight_target = st.slider("Weight Target (kg)", 50, 120, 80, key="weight_target")
    
    with col2:
        # Calculate progress
        glucose_progress = min(100, max(0, (1 - (current_glucose - 70) / (glucose_target - 70)) * 100))
        hba1c_progress = min(100, max(0, (1 - (hba1c_value - 5) / (hba1c_target - 5)) * 100))
        weight_progress = min(100, max(0, (1 - (weight - 50) / (weight_target - 50)) * 100))
        
        st.write("**Progress toward targets:**")
        st.progress(int(glucose_progress/100), text=f"Glucose: {glucose_progress:.0f}%")
        st.progress(int(hba1c_progress/100), text=f"HbA1c: {hba1c_progress:.0f}%")
        st.progress(int(weight_progress/100), text=f"Weight: {weight_progress:.0f}%")
        
        # Overall progress
        avg_progress = (glucose_progress + hba1c_progress + weight_progress) / 3
        st.metric("Overall Progress", f"{avg_progress:.1f}%")

with analytics_tab3:
    if df is not None and len(df) > 0:
        st.subheader("Statistical Summary")
        
        # Select column to analyze
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_stat_col = st.selectbox("Select column for statistics", numeric_cols, key="stat_col_select")
            
            if selected_stat_col in df.columns:
                col_data = df[selected_stat_col].dropna()
                
                if len(col_data) > 0:
                    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                    
                    with stats_col1:
                        st.metric("Mean", f"{col_data.mean():.2f}")
                    with stats_col2:
                        st.metric("Median", f"{col_data.median():.2f}")
                    with stats_col3:
                        st.metric("Std Dev", f"{col_data.std():.2f}")
                    with stats_col4:
                        st.metric("Count", f"{len(col_data)}")
                    
                    # Distribution plot
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Histogram
                    ax1.hist(col_data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
                    ax1.axvline(x=col_data.mean(), color='red', linestyle='--', label=f'Mean: {col_data.mean():.2f}')
                    ax1.axvline(x=col_data.median(), color='green', linestyle='--', label=f'Median: {col_data.median():.2f}')
                    ax1.set_xlabel(selected_stat_col)
                    ax1.set_ylabel('Frequency')
                    ax1.set_title(f'Distribution of {selected_stat_col}')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Box plot
                    ax2.boxplot(col_data, vert=False)
                    ax2.set_xlabel(selected_stat_col)
                    ax2.set_title(f'Box Plot of {selected_stat_col}')
                    ax2.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    plt.close(fig)

# ============================================
# PREDICTIVE FEATURES
# ============================================
st.markdown("---")
st.markdown("## 🔮 Predictive Insights")

pred_col1, pred_col2 = st.columns(2)

with pred_col1:
    st.subheader("Future Glucose Prediction")
    
    # Simple linear prediction
    if df is not None and any('glucose' in col.lower() for col in df.columns):
        glucose_cols = [col for col in df.columns if 'glucose' in col.lower()]
        if glucose_cols:
            glucose_col = glucose_cols[0]
            last_values = df[glucose_col].tail(10).values
            
            if len(last_values) >= 5:
                # Simple moving average prediction
                future_hours = st.slider("Predict next (hours)", 1, 12, 4, key="future_hours")
                
                # Calculate trend
                x = np.arange(len(last_values))
                y = last_values
                z = np.polyfit(x, y, 1)
                
                # Predict future values
                future_x = np.arange(len(last_values), len(last_values) + future_hours)
                predicted = z[0] * future_x + z[1]
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(x, y, 'bo-', label='Historical', linewidth=2)
                ax.plot(future_x, predicted, 'r--', label='Predicted', linewidth=2)
                ax.fill_between(future_x, predicted - 20, predicted + 20, alpha=0.2, color='red')
                ax.axhline(y=180, color='green', linestyle=':', label='Target (180)')
                ax.axhline(y=70, color='blue', linestyle=':', label='Min (70)')
                ax.set_xlabel('Hours')
                ax.set_ylabel('Glucose (mg/dL)')
                ax.set_title(f'Glucose Prediction for Next {future_hours} Hours')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.close(fig)
                
                # Show prediction details
                final_prediction = predicted[-1]
                st.write(f"**Predicted glucose in {future_hours} hours:** {final_prediction:.1f} mg/dL")
                
                if final_prediction > 250:
                    st.error("⚠️ Warning: Predicted hyperglycemia")
                elif final_prediction < 70:
                    st.error("⚠️ Warning: Predicted hypoglycemia")
                elif final_prediction > 180:
                    st.warning("⚠️ Predicted elevated glucose")
                else:
                    st.success("✅ Predicted within target range")
            else:
                st.info("Need more glucose data for prediction")

with pred_col2:
    st.subheader("Optimal Dose Simulation")
    
    # Create simulation
    sim_glucose = st.slider("Simulate Glucose (mg/dL)", 70, 300, current_glucose, key="sim_glucose")
    sim_carbs = st.slider("Simulate Carbs (g)", 0, 200, carbs, key="sim_carbs")
    
    # Calculate simulated dose
    sim_dose_data = calculate_insulin_dose(
        sim_glucose, sim_carbs, weight, age, diabetes_type, 
        activity_level, correction_factor, carb_ratio, hba1c_value
    )
    
    # Display comparison
    st.write("**Comparison:**")
    
    comparison_data = {
        "Scenario": ["Current", "Simulated"],
        "Glucose": [current_glucose, sim_glucose],
        "Carbs": [carbs, sim_carbs],
        "Total Dose": [total_dose, sim_dose_data['total']]
    }
    
    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True)
    
    # Show difference
    dose_diff = sim_dose_data['total'] - total_dose
    if abs(dose_diff) > 0.1:
        st.metric("Dose Difference", f"{dose_diff:+.1f} units", 
                 delta="Increase" if dose_diff > 0 else "Decrease")
    
    # Optimization suggestion
    if sim_glucose > current_glucose and dose_diff > 0:
        st.info("💡 **Insight:** Higher glucose requires more insulin")
    elif sim_glucose < current_glucose and dose_diff < 0:
        st.info("💡 **Insight:** Lower glucose requires less insulin")

# ============================================
# VISUALIZATIONS SECTION
# ============================================
st.markdown("---")
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
# REPORT GENERATION FEATURE
# ============================================
st.markdown("---")
st.markdown("## 📄 Generate Patient Report")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💾 Save Current Session", type="primary", use_container_width=True):
        # Save current parameters to session state
        st.session_state['last_save'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state['saved_dose'] = total_dose
        st.session_state['saved_glucose'] = current_glucose
        st.session_state['saved_hba1c'] = hba1c_value
        st.success(f"Session saved at {st.session_state['last_save']}")

with col2:
    if st.button("📈 Export Visualizations", use_container_width=True):
        # Create a comprehensive figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Chart 1: Glucose trend
        if df is not None and len(df) > 0:
            glucose_cols = [col for col in df.columns if any(x in col.lower() for x in ['glucose', 'sugar', 'bg'])]
            if glucose_cols:
                ax1.plot(df[glucose_cols[0]].tail(20), color='red')
        ax1.set_title('Blood Glucose Trend')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('mg/dL')
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Factor impact
        factors = ['Weight', 'Age', 'Type', 'Activity', 'HbA1c']
        impacts = [dose_data['weight_basal'], 
                  dose_data['weight_basal'] * dose_data['age_factor'],
                  dose_data['weight_basal'] * dose_data['age_factor'] * dose_data['diabetes_factor'],
                  dose_data['weight_basal'] * dose_data['age_factor'] * dose_data['diabetes_factor'] * dose_data['activity_factor'],
                  dose_data['basal']]
        ax2.bar(factors, impacts, color=['blue', 'orange', 'green', 'red', 'purple'])
        ax2.set_title('Factor Impact on Basal Insulin')
        ax2.set_ylabel('Units')
        ax2.tick_params(axis='x', rotation=45)
        
        # Chart 3: Risk distribution
        risk_labels = ['Glucose', 'HbA1c', 'Age', 'Dose']
        risk_values = [3 if current_glucose > 300 else 2 if current_glucose > 250 else 1 if current_glucose > 180 else 0,
                      2 if hba1c_value > 9.0 else 1 if hba1c_value > 7.0 else 0,
                      1 if age > 75 else 0,
                      2 if total_dose > 40 else 1 if total_dose > 30 else 0]
        ax3.bar(risk_labels, risk_values, color=['red', 'orange', 'yellow', 'green'])
        ax3.set_title('Risk Factor Distribution')
        ax3.set_ylabel('Risk Score')
        ax3.set_ylim(0, 3)
        
        # Chart 4: Target ranges
        categories = ['Hypoglycemia', 'Target Range', 'Elevated', 'Hyperglycemia']
        ranges = ['<70', '70-180', '180-250', '>250']
        colors = ['blue', 'green', 'orange', 'red']
        ax4.barh(categories, [1, 1, 1, 1], color=colors)
        ax4.set_xlim(0, 1)
        ax4.set_title('Glucose Classification')
        for i, (cat, rng) in enumerate(zip(categories, ranges)):
            ax4.text(0.5, i, rng, va='center', ha='center', fontweight='bold', color='white')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig('patient_report_charts.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a text report
        report_text = f"""
        DIABEDOSE PATIENT REPORT
        Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        PATIENT INFORMATION:
        - Blood Glucose: {current_glucose} mg/dL
        - HbA1c: {hba1c_value}%
        - Weight: {weight} kg
        - Age: {age} years
        - Diabetes Type: {diabetes_type}
        - Activity Level: {activity_level}
        
        INSULIN CALCULATION:
        - Correction Dose: {dose_data['correction']} units
        - Meal Dose: {dose_data['meal']} units
        - Basal Dose: {dose_data['basal']} units
        - TOTAL DOSE: {total_dose} units
        
        RISK ASSESSMENT:
        - Glucose Risk: {'Critical' if current_glucose > 300 else 'High' if current_glucose > 250 else 'Moderate' if current_glucose > 180 else 'Low'}
        - HbA1c Risk: {'Poor' if hba1c_value > 9.0 else 'Fair' if hba1c_value > 7.0 else 'Good'}
        - Overall Risk: {'High' if risk_score >= 5 else 'Medium' if risk_score >= 3 else 'Low'}
        
        RECOMMENDATIONS:
        {dose_warning}
        """
        
        with open('patient_report.txt', 'w') as f:
            f.write(report_text)
        
        st.success("✅ Report exported! Files saved: patient_report_charts.png, patient_report.txt")

with col3:
    if st.button("🔄 Compare with Previous", use_container_width=True):
        if 'saved_dose' in st.session_state:
            dose_change = total_dose - st.session_state['saved_dose']
            glucose_change = current_glucose - st.session_state['saved_glucose']
            hba1c_change = hba1c_value - st.session_state['saved_hba1c']
            
            st.info(f"**Comparison with {st.session_state['last_save']}:**")
            st.write(f"• Dose: {st.session_state['saved_dose']:.1f} → {total_dose:.1f} units ({dose_change:+.1f})")
            st.write(f"• Glucose: {st.session_state['saved_glucose']} → {current_glucose} mg/dL ({glucose_change:+d})")
            st.write(f"• HbA1c: {st.session_state['saved_hba1c']} → {hba1c_value}% ({hba1c_change:+.1f})")
        else:
            st.warning("No previous session saved. Click 'Save Current Session' first.")

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
# HELP & DOCUMENTATION SECTION
# ============================================
st.markdown("---")
with st.expander("📚 Help & Instructions", expanded=False):
    st.markdown("""
    ## 📖 DiabeDose User Guide
    
    ### 🎯 Purpose
    DiabeDose is an educational tool for estimating insulin doses based on multiple patient parameters.
    
    ### 📊 How to Use
    
    1. **Patient Parameters (Sidebar)**
       - Adjust all sliders/number inputs to match patient data
       - HbA1c: Latest glycated hemoglobin percentage
       - Blood Glucose: Current reading in mg/dL
       - Other parameters: Age, weight, diabetes type, activity level
    
    2. **Insulin Settings**
       - Correction Factor: How much 1 unit lowers glucose
       - Carb Ratio: How many carbs 1 unit covers
    
    3. **Results & Visualizations**
       - Top cards show key metrics
       - Insulin dose updates in real-time
       - Charts show trends and factor impacts
    
    4. **Risk Assessment**
       - Automatic risk calculation based on all factors
       - Color-coded recommendations
    
    ### 🧮 Calculation Formulas
    
    **Total Insulin Dose = Correction + Meal + Basal**
    
    - **Correction Dose** = (Current Glucose - Target) ÷ Correction Factor
    - **Meal Dose** = Carbs ÷ Carb Ratio
    - **Basal Dose** = Weight × 0.5 × Age Factor × Diabetes Factor × Activity Factor × HbA1c Factor
    
    ### ⚠️ Important Notes
    
    - This is for **EDUCATIONAL PURPOSES ONLY**
    - Always consult healthcare professionals
    - Individual responses to insulin vary
    - Monitor blood glucose regularly
    
    ### 🐛 Troubleshooting
    
    - **No CSV data loaded?** Ensure `data/dummy_data.csv` exists
    - **Charts not updating?** Check sidebar inputs are changed
    - **Calculation seems off?** Verify all parameters are correct
    
    ### 📞 Support
    For educational institutions: contact@diabedose.edu
    """)

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
    <p><strong>DiabeDose v3.0</strong> | Insulin Dose Predictor | Major Project Edition | 2024</p>
    <p>For Educational Use Only | Always consult with your healthcare provider for medical decisions</p>
</div>
""", unsafe_allow_html=True)
