"""
Visualization utilities for diabetes data
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class DiabetesVisualizer:
    """Create visualizations for diabetes data"""
    
    def __init__(self):
        """Initialize color schemes"""
        self.colors = {
            'low_risk': '#28A745',
            'medium_risk': '#FFC107',
            'high_risk': '#DC3545',
            'normal': '#17A2B8',
            'warning': '#FF6B6B',
            'success': '#51CF66',
            'type1': '#FF6B6B',
            'type2': '#36B9CC'
        }
    
    def create_glucose_gauge(self, glucose_value, patient_name=""):
        """
        Create glucose level gauge chart
        
        Args:
            glucose_value: Current glucose value
            patient_name: Optional patient name
            
        Returns:
            Plotly figure
        """
        # Determine color and status
        if glucose_value < 70:
            color = self.colors['high_risk']
            status = "Low"
            risk = "High"
        elif glucose_value <= 140:
            color = self.colors['success']
            status = "Normal"
            risk = "Low"
        elif glucose_value <= 180:
            color = self.colors['warning']
            status = "Elevated"
            risk = "Medium"
        else:
            color = self.colors['high_risk']
            status = "High"
            risk = "High"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=glucose_value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={
                'text': f"Blood Glucose<br>{status} ({risk} Risk)",
                'font': {'size': 18}
            },
            gauge={
                'axis': {
                    'range': [50, 400],
                    'tickwidth': 1,
                    'tickcolor': "darkblue"
                },
                'bar': {'color': color, 'thickness': 0.75},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [50, 70], 'color': '#FFE5E5'},
                    {'range': [70, 140], 'color': '#E5FFE5'},
                    {'range': [140, 180], 'color': '#FFF9E5'},
                    {'range': [180, 400], 'color': '#FFE5E5'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 180
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_hba1c_gauge(self, hba1c_value):
        """
        Create HbA1c gauge chart
        
        Args:
            hba1c_value: Current HbA1c value
            
        Returns:
            Plotly figure
        """
        # Determine status
        if hba1c_value < 5.7:
            color = self.colors['success']
            status = "Normal"
        elif hba1c_value < 6.5:
            color = self.colors['warning']
            status = "Pre-diabetes"
        else:
            if hba1c_value < 7.0:
                level = "Controlled"
                color = '#FFA500'
            elif hba1c_value < 8.0:
                level = "Suboptimal"
                color = '#FF6B6B'
            elif hba1c_value < 9.0:
                level = "Poor"
                color = '#DC3545'
            else:
                level = "Very Poor"
                color = '#8B0000'
            status = f"Diabetes ({level})"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=hba1c_value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"HbA1c<br>{status}", 'font': {'size': 18}},
            gauge={
                'axis': {'range': [4, 15]},
                'bar': {'color': color},
                'steps': [
                    {'range': [4, 5.7], 'color': '#E5FFE5'},
                    {'range': [5.7, 6.5], 'color': '#FFF9E5'},
                    {'range': [6.5, 7.0], 'color': '#FFE5CC'},
                    {'range': [7.0, 8.0], 'color': '#FFCCCC'},
                    {'range': [8.0, 9.0], 'color': '#FF9999'},
                    {'range': [9.0, 15], 'color': '#FF6666'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 6.5
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_dose_breakdown_chart(self, breakdown_data):
        """
        Create pie chart for insulin dose breakdown
        
        Args:
            breakdown_data: Dictionary with component names and values
            
        Returns:
            Plotly figure
        """
        labels = list(breakdown_data.keys())
        values = list(breakdown_data.values())
        
        # Calculate percentages
        total = sum(values)
        percentages = [f"{v/total*100:.1f}%" for v in values]
        hover_text = [f"{label}: {value} units ({percent})" 
                     for label, value, percent in zip(labels, values, percentages)]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo='label+value',
            textposition='outside',
            hoverinfo='text',
            hovertext=hover_text,
            marker=dict(
                colors=px.colors.qualitative.Set3,
                line=dict(color='white', width=2)
            )
        )])
        
        fig.update_layout(
            title={
                'text': f"Insulin Dose Breakdown<br>Total: {total:.1f} units",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            height=400,
            showlegend=False,
            annotations=[dict(
                text=f"{total:.1f} units",
                x=0.5, y=0.5, font_size=20, showarrow=False
            )]
        )
        
        return fig
    
    def create_glucose_trend_chart(self, time_series_data):
        """
        Create glucose trend line chart
        
        Args:
            time_series_data: DataFrame with timestamp and glucose columns
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add glucose line
        fig.add_trace(go.Scatter(
            x=time_series_data['timestamp'],
            y=time_series_data['glucose'],
            mode='lines+markers',
            name='Glucose',
            line=dict(color='red', width=2),
            marker=dict(size=8, color='red'),
            hovertemplate='<b>Time:</b> %{x}<br><b>Glucose:</b> %{y} mg/dL<extra></extra>'
        ))
        
        # Add target range
        fig.add_hrect(
            y0=70, y1=140,
            fillcolor="green", opacity=0.1,
            line_width=0,
            annotation_text="Target Range",
            annotation_position="bottom right"
        )
        
        fig.add_hrect(
            y0=140, y1=180,
            fillcolor="yellow", opacity=0.1,
            line_width=0,
            annotation_text="Elevated",
            annotation_position="bottom right"
        )
        
        fig.update_layout(
            title="Glucose Trend Over Time",
            xaxis_title="Time",
            yaxis_title="Glucose (mg/dL)",
            height=400,
            hovermode="x unified",
            showlegend=True,
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray'
            )
        )
        
        return fig
    
    def create_patient_summary_dashboard(self, patient_data):
        """
        Create comprehensive patient summary dashboard
        
        Args:
            patient_data: Dictionary with patient metrics
            
        Returns:
            Plotly figure with subplots
        """
        fig = make_subplots(
            rows=2, cols=3,
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                [{"type": "bar", "colspan": 3}, None, None]
            ],
            subplot_titles=("Glucose Level", "HbA1c", "Risk Score", "Key Metrics"),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Glucose indicator
        glucose = patient_data.get('glucose', 120)
        glucose_color = 'green' if glucose <= 140 else 'orange' if glucose <= 180 else 'red'
        
        fig.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=glucose,
                title={"text": "Glucose<br>(mg/dL)"},
                number={'font': {'size': 30}},
                gauge={
                    'shape': "bullet",
                    'axis': {'range': [50, 400]},
                    'bar': {'color': glucose_color},
                    'steps': [
                        {'range': [50, 140], 'color': "lightgreen"},
                        {'range': [140, 180], 'color': "lightyellow"},
                        {'range': [180, 400], 'color': "lightpink"}
                    ]
                }
            ),
            row=1, col=1
        )
        
        # HbA1c indicator
        hba1c = patient_data.get('hba1c', 7.0)
        hba1c_color = 'green' if hba1c < 6.5 else 'orange' if hba1c < 8 else 'red'
        
        fig.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=hba1c,
                title={"text": "HbA1c<br>(%)"},
                number={'font': {'size': 30}},
                gauge={
                    'shape': "bullet",
                    'axis': {'range': [4, 15]},
                    'bar': {'color': hba1c_color},
                    'steps': [
                        {'range': [4, 6.5], 'color': "lightgreen"},
                        {'range': [6.5, 8], 'color': "lightyellow"},
                        {'range': [8, 15], 'color': "lightpink"}
                    ]
                }
            ),
            row=1, col=2
        )
        
        # Risk score indicator
        risk_score = patient_data.get('risk_score', 30)
        risk_color = 'green' if risk_score < 30 else 'orange' if risk_score < 60 else 'red'
        
        fig.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=risk_score,
                title={"text": "Risk<br>Score"},
                number={'font': {'size': 30}},
                gauge={
                    'shape': "bullet",
                    'axis': {'range': [0, 100]},
                    'bar': {'color': risk_color},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 60], 'color': "lightyellow"},
                        {'range': [60, 100], 'color': "lightpink"}
                    ]
                }
            ),
            row=1, col=3
        )
        
        # Key metrics bar chart
        metrics_data = {
            'Weight (kg)': patient_data.get('weight', 70),
            'Age (years)': patient_data.get('age', 45),
            'BMI': patient_data.get('bmi', 25),
            'Diabetes Duration': patient_data.get('diabetes_duration', 5)
        }
        
        fig.add_trace(
            go.Bar(
                x=list(metrics_data.keys()),
                y=list(metrics_data.values()),
                marker_color=['blue', 'green', 'orange', 'purple'],
                text=[str(v) for v in metrics_data.values()],
                textposition='auto'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            paper_bgcolor='white',
            plot_bgcolor='white',
            title_text="Patient Summary Dashboard",
            title_font_size=20
        )
        
        fig.update_yaxes(title_text="Value", row=2, col=1)
        
        return fig
    
    def create_comparison_chart(self, df, x_col, y_col, color_col=None):
        """
        Create scatter plot comparing two features
        
        Args:
            df: DataFrame with data
            x_col: Column for x-axis
            y_col: Column for y-axis
            color_col: Column for color coding (optional)
            
        Returns:
            Plotly figure
        """
        if color_col:
            fig = px.scatter(
                df, x=x_col, y=y_col,
                color=color_col,
                size='weight' if 'weight' in df.columns else None,
                hover_data=df.columns.tolist(),
                title=f"{x_col} vs {y_col}",
                labels={color_col: color_col.replace('_', ' ').title()}
            )
        else:
            fig = px.scatter(
                df, x=x_col, y=y_col,
                title=f"{x_col} vs {y_col}",
                hover_data=df.columns.tolist()
            )
        
        fig.update_layout(
            height=500,
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            hovermode='closest'
        )
        
        return fig
    
    def create_risk_matrix(self, risk_data):
        """
        Create risk matrix visualization
        
        Args:
            risk_data: List of dictionaries with risk metrics
            
        Returns:
            Plotly figure
        """
        df = pd.DataFrame(risk_data)
        
        fig = go.Figure(data=go.Scatter(
            x=df['glucose'],
            y=df['hba1c'],
            mode='markers',
            marker=dict(
                size=df['risk_score']/2,
                color=df['risk_score'],
                colorscale='RdYlGn_r',  # Red to Green (reversed)
                showscale=True,
                colorbar=dict(title="Risk Score")
            ),
            text=df.apply(
                lambda row: f"Glucose: {row['glucose']}<br>HbA1c: {row['hba1c']}<br>Risk: {row['risk_score']:.1f}",
                axis=1
            ),
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Add risk zones
        fig.add_shape(
            type="rect",
            x0=180, x1=400, y0=8, y1=15,
            fillcolor="red", opacity=0.2,
            line=dict(color="red", width=2)
        )
        
        fig.add_shape(
            type="rect",
            x0=140, x1=180, y0=6.5, y1=8,
            fillcolor="orange", opacity=0.2,
            line=dict(color="orange", width=2)
        )
        
        fig.add_shape(
            type="rect",
            x0=70, x1=140, y0=4, y1=6.5,
            fillcolor="green", opacity=0.2,
            line=dict(color="green", width=2)
        )
        
        fig.update_layout(
            title="Diabetes Risk Matrix",
            xaxis_title="Blood Glucose (mg/dL)",
            yaxis_title="HbA1c (%)",
            height=500,
            showlegend=False,
            xaxis=dict(range=[60, 420]),
            yaxis=dict(range=[3.5, 16])
        )
        
        return fig

# Singleton instance
visualizer = DiabetesVisualizer()