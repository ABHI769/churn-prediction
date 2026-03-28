from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for models
rf_model = None
xgb_model = None
scaler = None
X = None  # Add X as global variable

def train_models():
    """Train and cache the models"""
    global rf_model, xgb_model, scaler, X
    
    # Create sample training data (in production, load your actual trained models)
    np.random.seed(42)
    n_samples = 5000
    
    # Generate sample data with realistic patterns
    data = {
        'gender': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'Partner': np.random.choice([0, 1], n_samples, p=[0.48, 0.52]),
        'Dependents': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'tenure': np.random.randint(1, 73, n_samples),
        'PhoneService': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'InternetService_DSL': np.random.choice([0, 1], n_samples, p=[0.35, 0.65]),
        'InternetService_Fiber optic': np.random.choice([0, 1], n_samples, p=[0.45, 0.55]),
        'InternetService_No': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'OnlineSecurity': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'OnlineBackup': np.random.choice([0, 1], n_samples, p=[0.35, 0.65]),
        'DeviceProtection': np.random.choice([0, 1], n_samples, p=[0.35, 0.65]),
        'TechSupport': np.random.choice([0, 1], n_samples, p=[0.25, 0.75]),
        'StreamingTV': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'StreamingMovies': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'Contract_Month-to-month': np.random.choice([0, 1], n_samples, p=[0.55, 0.45]),
        'Contract_One year': np.random.choice([0, 1], n_samples, p=[0.25, 0.75]),
        'Contract_Two year': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'PaperlessBilling': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'PaymentMethod_Bank transfer (automatic)': np.random.choice([0, 1], n_samples, p=[0.23, 0.77]),
        'PaymentMethod_Credit card (automatic)': np.random.choice([0, 1], n_samples, p=[0.22, 0.78]),
        'PaymentMethod_Electronic check': np.random.choice([0, 1], n_samples, p=[0.35, 0.65]),
        'PaymentMethod_Mailed check': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'MonthlyCharges': np.random.uniform(500, 15000, n_samples),  # Indian Rupees
        'TotalCharges': np.random.uniform(500, 200000, n_samples),  # Indian Rupees
        'Churn': np.random.choice([0, 1], n_samples, p=[0.73, 0.27])
    }
    
    df = pd.DataFrame(data)
    
    # Scale numerical features
    scaler = MinMaxScaler()
    df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])
    
    # Train models
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    rf_model.fit(X, y)
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss'
    )
    xgb_model.fit(X, y)
    
    print("Models trained successfully!")

def get_churn_suggestions(customer_data, prediction_prob):
    """Generate personalized suggestions based on customer data and prediction"""
    suggestions = []
    
    # High churn risk suggestions
    if prediction_prob > 0.7:
        suggestions.append({
            "priority": "Critical",
            "title": "Immediate Intervention Required",
            "description": "This customer is at very high risk of churning. Immediate action is recommended.",
            "icon": "solar:shield-warning-linear",
            "colorClass": "text-red-500"
        })
        
        if customer_data.get('Contract_Month-to-month', 0) == 1:
            suggestions.append({
                "priority": "High",
                "title": "Restructure Agreement",
                "description": "High volatility detected. Deploy immediate 15% discount offer to lock in an annual commitment.",
                "icon": "solar:calendar-minimalistic-linear",
                "colorClass": "text-red-500"
            })
        
        if customer_data.get('tenure', 0) < 12:
            suggestions.append({
                "priority": "High",
                "title": "Executive Outreach",
                "description": "Account is failing to find value. Schedule an expedited CS call to audit onboarding process.",
                "icon": "solar:phone-calling-linear",
                "colorClass": "text-amber-500"
            })
    
    elif prediction_prob > 0.4:
        suggestions.append({
            "priority": "Medium",
            "title": "Feedback Loop Initiation",
            "description": "Moderate risk indicators present. Trigger automated email sequence to gauge NPS and gather feedback.",
            "icon": "solar:chat-round-line-linear",
            "colorClass": "text-amber-500"
        })
    
    # Service-specific suggestions
    if customer_data.get('InternetService_Fiber optic', 0) == 1 and customer_data.get('TechSupport', 0) == 0:
        suggestions.append({
            "priority": "Medium",
            "title": "Support Tier Upgrade",
            "description": "Fiber user lacks dedicated support. Offer 3 free months of premium support to reduce technical churn.",
            "icon": "solar:settings-minimalistic-linear",
            "colorClass": "text-zinc-700"
        })
    
    if customer_data.get('OnlineSecurity', 0) == 0 and customer_data.get('InternetService_No', 0) == 0:
        suggestions.append({
            "priority": "Low",
            "title": "Security Package Promotion",
            "description": "Customer lacks online security. Promote security package benefits with limited-time discount.",
            "icon": "solar:shield-check-linear",
            "colorClass": "text-zinc-700"
        })
    
    if customer_data.get('PaymentMethod_Electronic check', 0) == 1:
        suggestions.append({
            "priority": "Low",
            "title": "Automate Billing",
            "description": "Manual payment detected. Incentivize credit card auto-pay with a one-time structural account credit.",
            "icon": "solar:wallet-linear",
            "colorClass": "text-zinc-700"
        })
    
    # Service bundle suggestions
    services_count = sum([
        customer_data.get('OnlineSecurity', 0),
        customer_data.get('OnlineBackup', 0), 
        customer_data.get('DeviceProtection', 0),
        customer_data.get('TechSupport', 0),
        customer_data.get('StreamingTV', 0),
        customer_data.get('StreamingMovies', 0)
    ])
    
    if services_count < 3 and customer_data.get('InternetService_No', 0) == 0:
        suggestions.append({
            "priority": "Low",
            "title": "Bundle Services Offer",
            "description": f"Customer has {services_count} additional services. Create personalized bundle with discounted pricing.",
            "icon": "solar:box-minimalistic-linear",
            "colorClass": "text-zinc-700"
        })
    
    if not suggestions:
        suggestions.append({
            "priority": "Healthy",
            "title": "Maintain Relationship",
            "description": "Account metrics are optimal. Continue standard engagement rhythm and monitor for expansion signals.",
            "icon": "solar:star-linear",
            "colorClass": "text-emerald-500"
        })
    
    return suggestions[:3]  # Return top 3 suggestions

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        data = request.json
        
        # Process input data
        input_data = {
            'gender': 1 if data.get('gender') == 'female' else 0,
            'SeniorCitizen': 1 if data.get('senior') else 0,
            'Partner': 1 if data.get('partner') else 0,
            'Dependents': 1 if data.get('dependents') else 0,
            'tenure': int(data.get('tenure', 12)),
            'PhoneService': 1 if data.get('phone') else 0,
            'MultipleLines': 1 if data.get('multipleLines') == 'yes' else 0,
            'InternetService_DSL': 1 if data.get('internet') == 'dsl' else 0,
            'InternetService_Fiber optic': 1 if data.get('internet') == 'fiber' else 0,
            'InternetService_No': 1 if data.get('internet') == 'no' else 0,
            'OnlineSecurity': 1 if data.get('onlineSecurity') else 0,
            'OnlineBackup': 1 if data.get('onlineBackup') else 0,
            'DeviceProtection': 1 if data.get('deviceProtection') else 0,
            'TechSupport': 1 if data.get('techSupport') else 0,
            'StreamingTV': 1 if data.get('streamingTV') else 0,
            'StreamingMovies': 1 if data.get('streamingMovies') else 0,
            'Contract_Month-to-month': 1 if data.get('contract') == 'monthly' else 0,
            'Contract_One year': 1 if data.get('contract') == 'yearly' else 0,
            'Contract_Two year': 1 if data.get('contract') == 'twoyear' else 0,
            'PaperlessBilling': 1 if data.get('paperless') else 0,
            'PaymentMethod_Bank transfer (automatic)': 1 if data.get('payment') == 'bank' else 0,
            'PaymentMethod_Credit card (automatic)': 1 if data.get('payment') == 'credit' else 0,
            'PaymentMethod_Electronic check': 1 if data.get('payment') == 'electronic' else 0,
            'PaymentMethod_Mailed check': 1 if data.get('payment') == 'mailed' else 0,
            'MonthlyCharges': float(data.get('monthlyCharges', 3000)),
            'TotalCharges': float(data.get('totalCharges', 30000))
        }
        
        # Convert to DataFrame and scale
        input_df = pd.DataFrame([input_data])
        print("Input data columns:", list(input_df.columns))
        print("Training data columns:", list(X.columns))
        
        # Check for missing columns
        missing_cols = set(X.columns) - set(input_df.columns)
        extra_cols = set(input_df.columns) - set(X.columns)
        
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
        if extra_cols:
            print(f"Extra columns: {extra_cols}")
        
        # Reorder columns to match training data
        input_df = input_df[X.columns]
        
        input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(input_df[['tenure', 'MonthlyCharges', 'TotalCharges']])
        
        # Make predictions
        rf_pred_prob = float(rf_model.predict_proba(input_df)[0][1])
        xgb_pred_prob = float(xgb_model.predict_proba(input_df)[0][1])
        
        # Average prediction
        avg_pred_prob = (rf_pred_prob + xgb_pred_prob) / 2
        
        # Generate suggestions
        suggestions = get_churn_suggestions(input_data, avg_pred_prob)
        
        # Determine risk level
        percentage = avg_pred_prob * 100
        if percentage > 70:
            risk_level = "Critical Risk"
            risk_class = "critical"
        elif percentage > 40:
            risk_level = "Elevated Risk"
            risk_class = "elevated"
        else:
            risk_level = "Healthy Account"
            risk_class = "healthy"
        
        response = {
            'success': True,
            'risk_percentage': round(percentage, 1),
            'risk_level': risk_level,
            'risk_class': risk_class,
            'rf_probability': round(rf_pred_prob * 100, 1),
            'xgb_probability': round(xgb_pred_prob * 100, 1),
            'suggestions': suggestions,
            'customer_data': input_data
        }
        
        return jsonify(response)
        
    except Exception as e:
        error_response = jsonify({
            'success': False,
            'error': str(e)
        })
        return error_response, 500

@app.route('/analytics')
def analytics():
    """Provide analytics data"""
    analytics_data = {
        'total_customers': 7051,
        'churn_rate': 26.5,
        'avg_monthly_revenue': 5180,  # Indian Rupees
        'customer_lifetime': 32,
        'churn_by_service': {
            'Fiber Optic': 41.8,
            'DSL Internet': 18.7,
            'Phone Only': 15.2,
            'No Internet': 7.4
        },
        'churn_by_contract': {
            'Month-to-month': 42.7,
            'One year': 11.3,
            'Two year': 2.8
        },
        'insights': [
            "Fiber optic accounts show elevated drop-off; review infrastructure stability or comparative pricing.",
            "Transitioning from monthly to long-term contracts improves retention probability by 4x.",
            "Active tech support add-ons correlate strongly with 15% lower baseline churn.",
            "Manual electronic checks see 22% higher involuntary churn rates than auto-pay CCs."
        ]
    }
    
    return jsonify(analytics_data)

@app.route('/recommendations')
def recommendations():
    """Provide retention strategies"""
    strategies = [
        {
            'title': 'Contract Optimization',
            'impact': 'High',
            'description': 'Migrating month-to-month users to structured agreements stabilizes revenue immediately.',
            'actions': ['Offer 10-15% discount for 1-year terms', 'Free premium tier for 2-year lock-in'],
            'estimated_impact': '+₹2.5L/mo retained',
            'cost': None
        },
        {
            'title': 'Service Bundling',
            'impact': 'Medium',
            'description': 'Multi-service subscribers exhibit higher switching costs and lower churn propensity.',
            'actions': ['Target internet-only users with IPTV trials', 'Create holistic Security Suite bundles'],
            'estimated_impact': '+₹1.8L/mo revenue',
            'cost': None
        },
        {
            'title': 'Support Enhancement',
            'impact': 'High',
            'description': 'Technical friction is a leading involuntary churn indicator for Fiber network customers.',
            'actions': ['Proactive outreach in first 90 days', 'Include basic tech support in all tiers'],
            'estimated_impact': None,
            'cost': '₹1.2L/mo investment'
        },
        {
            'title': 'Auto-Pay Adoption',
            'impact': 'Quick Win',
            'description': 'Frictionless payments prevent involuntary churn originating from missed manual invoices.',
            'actions': ['₹500 one-time credit for auto-pay setup', 'Simplify credit card update workflows'],
            'estimated_impact': '+₹75K/mo retained',
            'cost': None
        }
    ]
    
    return jsonify({'strategies': strategies})

if __name__ == '__main__':
    # Train models on startup
    print("Training models...")
    train_models()
    
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Move HTML file to templates directory
    import shutil
    if os.path.exists('churn_prediction_ui.html'):
        shutil.copy('churn_prediction_ui.html', 'templates/index.html')
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
