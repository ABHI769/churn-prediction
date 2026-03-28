import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffcccc;
        border: 2px solid #ff4444;
    }
    .low-risk {
        background-color: #ccffcc;
        border: 2px solid #44ff44;
    }
    .suggestion-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🔮 Customer Churn Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
This application helps predict customer churn and provides personalized recommendations to improve customer retention.
""")

# Train or load models
@st.cache_resource
def train_models():
    """Train and cache the models"""
    # Create sample training data (in production, this would load your actual trained models)
    np.random.seed(42)
    n_samples = 5000
    
    # Generate sample data
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
        'MonthlyCharges': np.random.uniform(20, 120, n_samples),
        'TotalCharges': np.random.uniform(20, 8000, n_samples),
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
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X, y)
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=6,
        learning_rate=0.1,
        objective='binary:logistic'
    )
    xgb_model.fit(X, y)
    
    return rf_model, xgb_model, scaler

# Load models
rf_model, xgb_model, scaler = train_models()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page:", ["📊 Prediction", "📈 Analytics", "💡 Recommendations"])

def get_churn_suggestions(customer_data, prediction_prob):
    """Generate personalized suggestions based on customer data and prediction"""
    suggestions = []
    
    # High churn risk suggestions
    if prediction_prob > 0.7:
        suggestions.append({
            "priority": "🔴 High Priority",
            "title": "Immediate Intervention Required",
            "description": "This customer is at very high risk of churning. Immediate action is recommended."
        })
        
        if customer_data['Contract_Month-to-month'] == 1:
            suggestions.append({
                "priority": "🔴 High Priority",
                "title": "Offer Long-term Contract Incentives",
                "description": "Customer has month-to-month contract. Offer discounts for 1-year or 2-year contracts with additional benefits."
            })
        
        if customer_data['tenure'] < 12:
            suggestions.append({
                "priority": "🔴 High Priority", 
                "title": "New Customer Onboarding Support",
                "description": "Customer is relatively new. Provide dedicated onboarding support and early-bird retention offers."
            })
    
    elif prediction_prob > 0.4:
        suggestions.append({
            "priority": "🟡 Medium Priority",
            "title": "Proactive Customer Engagement",
            "description": "Customer shows moderate churn risk. Schedule check-in calls and offer personalized solutions."
        })
    
    # Service-specific suggestions
    if customer_data['InternetService_Fiber optic'] == 1 and customer_data['TechSupport'] == 0:
        suggestions.append({
            "priority": "🟡 Medium Priority",
            "title": "Add Tech Support Services",
            "description": "Customer has Fiber optic but no tech support. Offer free tech support trial to improve service experience."
        })
    
    if customer_data['OnlineSecurity'] == 0 and customer_data['InternetService_No'] == 0:
        suggestions.append({
            "priority": "🟢 Low Priority",
            "title": "Security Package Promotion",
            "description": "Customer lacks online security. Promote security package benefits with limited-time discount."
        })
    
    if customer_data['PaymentMethod_Electronic check'] == 1:
        suggestions.append({
            "priority": "🟢 Low Priority",
            "title": "Optimize Payment Method",
            "description": "Customer uses electronic check. Offer incentives for switching to automatic payment methods."
        })
    
    # Service bundle suggestions
    services_count = sum([
        customer_data['OnlineSecurity'], customer_data['OnlineBackup'], 
        customer_data['DeviceProtection'], customer_data['TechSupport'],
        customer_data['StreamingTV'], customer_data['StreamingMovies']
    ])
    
    if services_count < 3 and customer_data['InternetService_No'] == 0:
        suggestions.append({
            "priority": "🟢 Low Priority",
            "title": "Bundle Services Offer",
            "description": f"Customer has {services_count} additional services. Create personalized bundle with discounted pricing."
        })
    
    return suggestions

if page == "📊 Prediction":
    st.header("Customer Churn Prediction")
    
    # Create two columns for input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Information")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.checkbox("Senior Citizen")
        partner = st.checkbox("Has Partner")
        dependents = st.checkbox("Has Dependents")
        tenure = st.slider("Tenure (months)", 1, 72, 12)
        
        st.subheader("Services")
        phone_service = st.checkbox("Phone Service")
        if phone_service:
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
        else:
            multiple_lines = "No"
            
        internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
        
    with col2:
        st.subheader("Additional Services")
        if internet_service != "No":
            online_security = st.checkbox("Online Security")
            online_backup = st.checkbox("Online Backup")
            device_protection = st.checkbox("Device Protection")
            tech_support = st.checkbox("Tech Support")
            streaming_tv = st.checkbox("Streaming TV")
            streaming_movies = st.checkbox("Streaming Movies")
        else:
            online_security = online_backup = device_protection = False
            tech_support = streaming_tv = streaming_movies = False
        
        st.subheader("Billing & Payment")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.checkbox("Paperless Billing")
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", 
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        
        monthly_charges = st.number_input("Monthly Charges ($)", 20.0, 200.0, 50.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)
    
    # Predict button
    if st.button("🔮 Predict Churn Risk", type="primary"):
        # Process input data
        input_data = {
            'gender': 1 if gender == "Female" else 0,
            'SeniorCitizen': 1 if senior_citizen else 0,
            'Partner': 1 if partner else 0,
            'Dependents': 1 if dependents else 0,
            'tenure': tenure,
            'PhoneService': 1 if phone_service else 0,
            'MultipleLines': 1 if multiple_lines == "Yes" else 0,
            'InternetService_DSL': 1 if internet_service == "DSL" else 0,
            'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
            'InternetService_No': 1 if internet_service == "No" else 0,
            'OnlineSecurity': 1 if online_security else 0,
            'OnlineBackup': 1 if online_backup else 0,
            'DeviceProtection': 1 if device_protection else 0,
            'TechSupport': 1 if tech_support else 0,
            'StreamingTV': 1 if streaming_tv else 0,
            'StreamingMovies': 1 if streaming_movies else 0,
            'Contract_Month-to-month': 1 if contract == "Month-to-month" else 0,
            'Contract_One year': 1 if contract == "One year" else 0,
            'Contract_Two year': 1 if contract == "Two year" else 0,
            'PaperlessBilling': 1 if paperless_billing else 0,
            'PaymentMethod_Bank transfer (automatic)': 1 if payment_method == "Bank transfer (automatic)" else 0,
            'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
            'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
            'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # Convert to DataFrame and scale
        input_df = pd.DataFrame([input_data])
        input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(input_df[['tenure', 'MonthlyCharges', 'TotalCharges']])
        
        # Make predictions
        rf_pred_prob = rf_model.predict_proba(input_df)[0][1]
        xgb_pred_prob = xgb_model.predict_proba(input_df)[0][1]
        
        # Average prediction
        avg_pred_prob = (rf_pred_prob + xgb_pred_prob) / 2
        
        # Display results
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        # Risk level display
        if avg_pred_prob > 0.7:
            risk_class = "high-risk"
            risk_emoji = "🔴"
            risk_text = "High Risk"
        elif avg_pred_prob > 0.4:
            risk_class = "medium-risk"
            risk_emoji = "🟡"
            risk_text = "Medium Risk"
        else:
            risk_class = "low-risk"
            risk_emoji = "🟢"
            risk_text = "Low Risk"
        
        # Display prediction
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f'<div class="prediction-box {risk_class}"><h3>{risk_emoji} {risk_text}</h3><h2>{avg_pred_prob:.1%}</h2><p>Churn Probability</p></div>', unsafe_allow_html=True)
        
        with col2:
            st.metric("Random Forest", f"{rf_pred_prob:.1%}")
            st.metric("XGBoost", f"{xgb_pred_prob:.1%}")
        
        with col3:
            if avg_pred_prob > 0.5:
                st.markdown("### ⚠️ Action Required")
                st.markdown("Customer needs immediate attention to prevent churn.")
            else:
                st.markdown("### ✅ Stable Customer")
                st.markdown("Customer relationship appears healthy.")
        
        # Generate and display suggestions
        suggestions = get_churn_suggestions(input_data, avg_pred_prob)
        
        if suggestions:
            st.markdown("---")
            st.subheader("💡 Personalized Recommendations")
            
            for suggestion in suggestions:
                st.markdown(f'''
                <div class="suggestion-box">
                    <h4>{suggestion["priority"]} - {suggestion["title"]}</h4>
                    <p>{suggestion["description"]}</p>
                </div>
                ''', unsafe_allow_html=True)
        
        # Customer profile summary
        st.markdown("---")
        st.subheader("📋 Customer Profile Summary")
        
        profile_data = {
            "Customer Type": "Senior" if senior_citizen else "Regular",
            "Family Status": "Has Family" if partner or dependents else "Individual",
            "Contract Type": contract,
            "Payment Method": payment_method,
            "Monthly Revenue": f"${monthly_charges:.2f}",
            "Customer Lifetime": f"{tenure} months",
            "Services Count": sum([online_security, online_backup, device_protection, 
                                 tech_support, streaming_tv, streaming_movies]) if internet_service != "No" else 0
        }
        
        col1, col2 = st.columns(2)
        with col1:
            for key, value in list(profile_data.items())[:4]:
                st.metric(key, value)
        
        with col2:
            for key, value in list(profile_data.items())[4:]:
                st.metric(key, value)

elif page == "📈 Analytics":
    st.header("Churn Analytics Dashboard")
    
    # Sample analytics data
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", "7,051", "+2.3%")
    
    with col2:
        st.metric("Churn Rate", "26.5%", "-1.2%")
    
    with col3:
        st.metric("Avg Monthly Revenue", "$64.76", "+$3.21")
    
    with col4:
        st.metric("Customer Lifetime", "32 months", "+2 months")
    
    # Churn by service type
    st.subheader("Churn Rate by Service Type")
    
    service_data = {
        'Service': ['Phone Only', 'DSL Internet', 'Fiber Optic', 'No Internet'],
        'Churn Rate': [15.2, 18.7, 41.8, 7.4],
        'Customers': [1526, 2421, 3096, 8]
    }
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bar chart for churn rate
    bars = ax1.bar(service_data['Service'], service_data['Churn Rate'], 
                   color=['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c'])
    ax1.set_xlabel('Service Type')
    ax1.set_ylabel('Churn Rate (%)', color='black')
    ax1.set_title('Churn Rate by Service Type')
    
    # Add percentage labels on bars
    for bar, rate in zip(bars, service_data['Churn Rate']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{rate}%', ha='center', va='bottom')
    
    # Add customer count as text
    for i, (service, customers) in enumerate(zip(service_data['Service'], service_data['Customers'])):
        ax1.text(i, 2, f'n={customers}', ha='center', va='bottom', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Churn by contract type
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn by Contract Type")
        contract_data = {
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'Churn Rate': [42.7, 11.3, 2.8],
            'Customers': [3875, 1473, 1683]
        }
        
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#ff7f0e', '#1f77b4', '#2ca02c']
        bars = ax.bar(contract_data['Contract'], contract_data['Churn Rate'], color=colors)
        
        for bar, rate in zip(bars, contract_data['Churn Rate']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{rate}%', ha='center', va='bottom')
        
        ax.set_ylabel('Churn Rate (%)')
        ax.set_title('Churn Rate by Contract Type')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Revenue Impact")
        revenue_data = {
            'Category': ['Retained Customers', 'Churned Customers'],
            'Monthly Revenue': [450000, 120000]
        }
        
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#2ca02c', '#d62728']
        ax.pie(revenue_data['Monthly Revenue'], labels=revenue_data['Category'], 
               colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Monthly Revenue Distribution')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Key insights
    st.subheader("🔍 Key Insights")
    
    insights = [
        "📈 Fiber optic customers have the highest churn rate (41.8%)",
        "📊 Month-to-month contracts show 42.7% churn vs 2.8% for 2-year contracts",
        "💰 Customers with tech support are 15% less likely to churn",
        "🎯 Senior citizens show 23% lower churn rate",
        "💳 Electronic check payments correlate with higher churn"
    ]
    
    for insight in insights:
        st.markdown(f"• {insight}")

elif page == "💡 Recommendations":
    st.header("Customer Retention Strategy Guide")
    
    st.subheader("🎯 High-Impact Retention Strategies")
    
    strategies = {
        "Contract Optimization": {
            "description": "Long-term contracts significantly reduce churn",
            "actions": [
                "Offer 10-15% discount for 1-year contracts",
                "Provide free premium services for 2-year commitments",
                "Create loyalty rewards for contract renewals"
            ],
            "impact": "High",
            "difficulty": "Medium"
        },
        "Service Bundling": {
            "description": "Customers with more services have lower churn rates",
            "actions": [
                "Create personalized service bundles",
                "Offer bundle discounts (20-30% off)",
                "Provide free trials of additional services"
            ],
            "impact": "High",
            "difficulty": "Low"
        },
        "Payment Method Optimization": {
            "description": "Automatic payments reduce churn by 25%",
            "actions": [
                "Incentivize automatic payment methods",
                "Offer $5 monthly discount for auto-pay",
                "Simplify payment method switching process"
            ],
            "impact": "Medium",
            "difficulty": "Low"
        },
        "Customer Support Enhancement": {
            "description": "Tech support customers churn 15% less",
            "actions": [
                "Include basic tech support in all internet packages",
                "Offer 24/7 premium support for high-value customers",
                "Proactive support for new customers (first 90 days)"
            ],
            "impact": "High",
            "difficulty": "Medium"
        },
        "New Customer Onboarding": {
            "description": "First 90 days are critical for retention",
            "actions": [
                "Dedicated onboarding specialist for new customers",
                "Welcome package with service tutorials",
                "30-day check-in calls and satisfaction surveys"
            ],
            "impact": "High",
            "difficulty": "Medium"
        }
    }
    
    for strategy_name, strategy_info in strategies.items():
        with st.expander(f"📋 {strategy_name}"):
            st.markdown(f"**Description:** {strategy_info['description']}")
            st.markdown("**Recommended Actions:**")
            for action in strategy_info['actions']:
                st.markdown(f"• {action}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Impact:** {strategy_info['impact']}")
            with col2:
                st.markdown(f"**Difficulty:** {strategy_info['difficulty']}")
    
    # Implementation timeline
    st.subheader("📅 Implementation Timeline")
    
    timeline_data = {
        "Phase 1 (0-30 days)": [
            "Launch automatic payment incentives",
            "Create service bundle packages",
            "Start new customer onboarding program"
        ],
        "Phase 2 (30-90 days)": [
            "Implement contract optimization offers",
            "Enhance customer support protocols",
            "Deploy proactive retention campaigns"
        ],
        "Phase 3 (90-180 days)": [
            "Analyze early results and optimize",
            "Scale successful initiatives",
            "Develop advanced predictive models"
        ]
    }
    
    for phase, actions in timeline_data.items():
        with st.expander(phase):
            for action in actions:
                st.markdown(f"• {action}")
    
    # Success metrics
    st.subheader("📊 Success Metrics to Track")
    
    metrics = {
        "Primary Metrics": [
            "Overall churn rate reduction (target: 15% decrease)",
            "Customer lifetime value increase",
            "Revenue retention rate"
        ],
        "Secondary Metrics": [
            "Contract renewal rates",
            "Bundle adoption rates",
            "Customer satisfaction scores",
            "Support ticket resolution time"
        ],
        "Leading Indicators": [
            "Service adoption rates",
            "Payment method changes",
            "Customer engagement levels"
        ]
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Primary Metrics**")
        for metric in metrics["Primary Metrics"]:
            st.markdown(f"• {metric}")
    
    with col2:
        st.markdown("**Secondary Metrics**")
        for metric in metrics["Secondary Metrics"]:
            st.markdown(f"• {metric}")
    
    with col3:
        st.markdown("**Leading Indicators**")
        for metric in metrics["Leading Indicators"]:
            st.markdown(f"• {metric}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔮 Customer Churn Prediction Dashboard | Built with Streamlit</p>
    <p>For best results, use with real customer data and trained models</p>
</div>
""", unsafe_allow_html=True)
