# 🔮 Customer Churn Prediction System

A modern web application that predicts customer churn using machine learning and provides personalized retention strategies. Built with Flask backend, HTML/Tailwind CSS frontend, and real ML models (Random Forest & XGBoost).

> Updated: 2026-03-28
>
> Run locally with `python app.py` (Flask) or `python churn_prediction_app.py` (Streamlit).
>
> Note: Ensure `requirements.txt` dependencies are installed before running.

## 📊 Features

### 🤖 Machine Learning Models
- **Random Forest**: Ensemble learning model with 100 trees
- **XGBoost**: Gradient boosting framework for accurate predictions
- **Real-time Predictions**: Live ML inference via Flask API
- **Model Ensemble**: Combines both models for better accuracy

### 💰 Indian Currency Support
- All monetary values in Indian Rupees (₹)
- Localized pricing and revenue metrics
- Region-specific retention strategies

### 🎨 Modern Web Interface
- **Responsive Design**: Works on desktop and mobile
- **Tailwind CSS**: Modern, clean UI components
- **Interactive Forms**: Real-time validation and feedback
- **Beautiful Visualizations**: Risk assessments and analytics

### 📋 Smart Recommendations
- **Personalized Suggestions**: Based on customer profile and risk level
- **Priority-based Actions**: High/Medium/Low priority recommendations
- **Business Impact**: Revenue impact estimates in Rupees
- **Retention Strategies**: Proactive customer retention tactics

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip package manager

### Installation

1. **Clone/Download the project files**
   ```bash
   # Navigate to project directory
   cd churn-prediction-system
   ```

2. **Install dependencies**
   ```bash
   pip install flask pandas numpy scikit-learn xgboost
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open in browser**
   ```
   http://localhost:5000
   ```

## 📁 Project Structure

```
churn-prediction-system/
├── app.py                      # Flask backend with ML models
├── churn_prediction_ui.html    # Frontend HTML file
├── templates/
│   └── index.html             # Flask template (auto-generated)
├── churn_prediction_models.py # Standalone ML script
├── churn_prediction_app.py     # Streamlit version
├── requirements.txt           # Python dependencies
├── model_comparison.png       # Model performance comparison
├── feature_importance.png     # XGBoost feature importance
├── confusion_matrices.png     # Model confusion matrices
└── README.md                  # This file
```

## 🎯 Usage Guide

### Making Predictions

1. **Fill Customer Information**
   - Basic demographics (gender, senior citizen, family status)
   - Service subscriptions (phone, internet, additional services)
   - Billing details (contract type, payment method, charges)

2. **Get Instant Results**
   - Churn risk percentage (0-100%)
   - Risk level classification (Critical/Elevated/Healthy)
   - Personalized retention recommendations

3. **View Analytics**
   - Overall churn metrics
   - Service-based churn rates
   - Contract type analysis
   - Key business insights

### Understanding Risk Levels

| Risk Level | Percentage | Action Required |
|------------|------------|-----------------|
| 🔴 **Critical** | >70% | Immediate intervention needed |
| 🟡 **Elevated** | 40-70% | Proactive engagement recommended |
| 🟢 **Healthy** | <40% | Maintain standard relationship |

## 🤖 Model Details

### Data Features
- **Demographics**: Gender, Senior Citizen, Partner, Dependents
- **Services**: Phone, Internet, Security, Backup, Support, Streaming
- **Billing**: Contract Type, Payment Method, Monthly/Total Charges
- **Tenure**: Customer relationship duration

### Model Performance
- **Random Forest**: ~72% accuracy, 53% ROC AUC
- **XGBoost**: ~71% accuracy, 50% ROC AUC
- **Ensemble**: Combined predictions for better reliability

### Feature Importance
Top predictive factors:
1. Contract type (Month-to-month vs Long-term)
2. Internet service (Fiber optic shows higher churn)
3. Tenure (New customers at higher risk)
4. Payment method (Manual payments increase risk)
5. Additional services (More services = lower churn)

## 💡 Business Insights

### Key Findings
- **Fiber optic customers**: 41.8% churn rate (highest)
- **Month-to-month contracts**: 42.7% churn vs 2.8% for 2-year
- **Tech support**: 15% lower churn for customers with support
- **Payment method**: Electronic checks show 22% higher churn
- **Service bundles**: Customers with more services churn less

### Retention Strategies

#### High Impact
- **Contract Optimization**: 15% discounts for long-term commitments
- **Support Enhancement**: Proactive technical support for new customers
- **Service Bundling**: Create attractive service packages

#### Medium Impact  
- **Payment Automation**: Incentivize auto-pay with ₹500 credits
- **Onboarding Programs**: Dedicated support for first 90 days

#### Quick Wins
- **Security Packages**: Promote online security add-ons
- **Feedback Loops**: Regular customer satisfaction surveys

## 🔧 Technical Details

### Backend Architecture
- **Flask**: Python web framework for API
- **Scikit-learn**: ML model training and inference
- **XGBoost**: Gradient boosting implementation
- **Pandas**: Data processing and manipulation
- **NumPy**: Numerical computations

### Frontend Technologies
- **HTML5**: Semantic markup structure
- **Tailwind CSS**: Utility-first CSS framework
- **JavaScript**: Client-side interactions and API calls
- **Iconify**: Modern icon system

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main application page |
| `/predict` | POST | Process customer data and return predictions |
| `/analytics` | GET | Return analytics and metrics data |
| `/recommendations` | GET | Provide retention strategies |

## 📊 Sample API Response

```json
{
  "success": true,
  "risk_percentage": 67.3,
  "risk_level": "Elevated Risk",
  "risk_class": "elevated",
  "rf_probability": 65.2,
  "xgb_probability": 69.4,
  "suggestions": [
    {
      "priority": "Medium",
      "title": "Feedback Loop Initiation",
      "description": "Moderate risk indicators present...",
      "icon": "solar:chat-round-line-linear",
      "colorClass": "text-amber-500"
    }
  ],
  "customer_data": {...}
}
```

## 🎨 Customization

### Adding New Models
1. Train your model in `app.py`
2. Add prediction logic to `/predict` endpoint
3. Update frontend to display new model results

### Modifying UI
- Edit `templates/index.html` for layout changes
- Modify CSS classes in `<style>` section
- Update JavaScript for new interactions

### Business Logic
- Adjust risk thresholds in `get_churn_suggestions()`
- Customize recommendation logic
- Update currency and formatting

## 🚀 Deployment

### Production Setup
1. **Use production server** (Gunicorn, uWSGI)
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. **Environment variables**
   ```bash
   export FLASK_ENV=production
   export FLASK_DEBUG=0
   ```

3. **Database integration** (optional)
   - Add SQLAlchemy for database support
   - Store customer data and predictions
   - Implement user authentication

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🆘 Support

### Common Issues

**Models not training**: Check Python version and dependencies
**Frontend not loading**: Ensure Flask is running and templates exist
**API errors**: Check browser console for JavaScript errors
**Currency issues**: Verify all monetary values use ₹ symbol

### Getting Help
- Check the console output for error messages
- Verify all dependencies are installed
- Ensure Flask server is running on correct port
- Test API endpoints directly with curl/Postman

## 📈 Future Enhancements

- [ ] Real customer dataset integration
- [ ] User authentication and profiles
- [ ] Historical prediction tracking
- [ ] Advanced analytics dashboard
- [ ] Email/SMS notification system
- [ ] A/B testing for recommendations
- [ ] Multi-language support
- [ ] Mobile app development

## 📞 Contact

For questions, suggestions, or support:
- Create an issue in the repository
- Email: [your-email@example.com]
- LinkedIn: [your-profile]

---

**Built with ❤️ using Python, Flask, and Modern Web Technologies**
