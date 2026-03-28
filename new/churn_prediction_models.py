import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
print("Loading and preprocessing data...")

# Load dataset (assuming the dataset is in the same directory)
try:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Dataset not found. Creating sample dataset for demonstration...")
    # Create a sample dataset based on the Telco Churn dataset structure
    np.random.seed(42)
    n_samples = 7043
    
    # Generate sample data with similar characteristics
    data = {
        'customerID': [f'CUST{i:06d}' for i in range(n_samples)],
        'gender': np.random.choice(['Female', 'Male'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples, p=[0.48, 0.52]),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'tenure': np.random.randint(1, 73, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.4, 0.5, 0.1]),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.35, 0.45, 0.2]),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.35, 0.45, 0.2]),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.35, 0.45, 0.2]),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.25, 0.55, 0.2]),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.4, 0.2]),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.4, 0.2]),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.2]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], 
                                        n_samples, p=[0.35, 0.2, 0.23, 0.22]),
        'MonthlyCharges': np.random.uniform(20, 120, n_samples),
        'TotalCharges': np.random.uniform(20, 8000, n_samples),
        'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])  # Similar churn rate
    }
    
    df = pd.DataFrame(data)
    print("Sample dataset created successfully!")

# Drop customer ID
df.drop('customerID', axis='columns', inplace=True)

# Handle TotalCharges - remove rows with empty values
df = df[df.TotalCharges != ' ']
df.TotalCharges = pd.to_numeric(df.TotalCharges)

# Replace 'No internet service' and 'No phone service' with 'No'
df.replace('No internet service', 'No', inplace=True)
df.replace('No phone service', 'No', inplace=True)

# Convert binary categorical variables to numeric
yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 
                  'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                  'StreamingMovies', 'PaperlessBilling', 'Churn']

for col in yes_no_columns:
    df[col].replace({'Yes': 1, 'No': 0}, inplace=True)

# Convert gender to numeric
df['gender'].replace({'Female': 1, 'Male': 0}, inplace=True)

# One-hot encode remaining categorical variables
df = pd.get_dummies(data=df, columns=['InternetService', 'Contract', 'PaymentMethod'])

# Scale numerical features
cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = MinMaxScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# Split features and target
X = df.drop('Churn', axis='columns')
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Churn distribution - Training: {y_train.value_counts().to_dict()}")
print(f"Churn distribution - Test: {y_test.value_counts().to_dict()}")

# Random Forest Model
print("\n" + "="*50)
print("Training Random Forest Model...")
print("="*50)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2
)

rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_predictions_proba = rf_model.predict_proba(X_test)[:, 1]

# Evaluate Random Forest
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions)
rf_auc = roc_auc_score(y_test, rf_predictions_proba)

print(f"Random Forest Results:")
print(f"Accuracy: {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall: {rf_recall:.4f}")
print(f"F1-Score: {rf_f1:.4f}")
print(f"ROC AUC: {rf_auc:.4f}")

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_predictions))

# XGBoost Model
print("\n" + "="*50)
print("Training XGBoost Model...")
print("="*50)

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

xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
xgb_predictions_proba = xgb_model.predict_proba(X_test)[:, 1]

# Evaluate XGBoost
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
xgb_precision = precision_score(y_test, xgb_predictions)
xgb_recall = recall_score(y_test, xgb_predictions)
xgb_f1 = f1_score(y_test, xgb_predictions)
xgb_auc = roc_auc_score(y_test, xgb_predictions_proba)

print(f"XGBoost Results:")
print(f"Accuracy: {xgb_accuracy:.4f}")
print(f"Precision: {xgb_precision:.4f}")
print(f"Recall: {xgb_recall:.4f}")
print(f"F1-Score: {xgb_f1:.4f}")
print(f"ROC AUC: {xgb_auc:.4f}")

print("\nXGBoost Classification Report:")
print(classification_report(y_test, xgb_predictions))

# Model Comparison
print("\n" + "="*50)
print("Model Comparison Summary")
print("="*50)

comparison_data = {
    'Random Forest': [rf_accuracy, rf_precision, rf_recall, rf_f1, rf_auc],
    'XGBoost': [xgb_accuracy, xgb_precision, xgb_recall, xgb_f1, xgb_auc]
}

comparison_df = pd.DataFrame(
    comparison_data,
    index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
)

print(comparison_df.round(4))

# Determine best model
best_accuracy = max(rf_accuracy, xgb_accuracy)
best_auc = max(rf_auc, xgb_auc)

if rf_accuracy == best_accuracy:
    best_model_accuracy = "Random Forest"
else:
    best_model_accuracy = "XGBoost"

if rf_auc == best_auc:
    best_model_auc = "Random Forest"
else:
    best_model_auc = "XGBoost"

print(f"\nBest model by Accuracy: {best_model_accuracy} ({best_accuracy:.4f})")
print(f"Best model by ROC AUC: {best_model_auc} ({best_auc:.4f})")

# Create visualizations
print("\nCreating visualizations...")

# Set up the plotting style
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Accuracy Comparison
models = ['Random Forest', 'XGBoost']
accuracies = [rf_accuracy, xgb_accuracy]
colors = ['green', 'orange']

axes[0, 0].bar(models, accuracies, color=colors)
axes[0, 0].set_title('Model Accuracy Comparison')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_ylim(0, 1)
for i, v in enumerate(accuracies):
    axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

# 2. ROC AUC Comparison
auc_scores = [rf_auc, xgb_auc]
axes[0, 1].bar(models, auc_scores, color=colors)
axes[0, 1].set_title('ROC AUC Comparison')
axes[0, 1].set_ylabel('ROC AUC')
axes[0, 1].set_ylim(0, 1)
for i, v in enumerate(auc_scores):
    axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

# 3. Precision, Recall, F1 Comparison
metrics_df = pd.DataFrame({
    'Random Forest': [rf_precision, rf_recall, rf_f1],
    'XGBoost': [xgb_precision, xgb_recall, xgb_f1]
}, index=['Precision', 'Recall', 'F1-Score'])

metrics_df.plot(kind='bar', ax=axes[1, 0], color=colors)
axes[1, 0].set_title('Precision, Recall, F1-Score Comparison')
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_ylim(0, 1)
axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1, 0].tick_params(axis='x', rotation=45)

# 4. ROC Curves
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_predictions_proba)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_predictions_proba)

axes[1, 1].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_auc:.3f})', color='green', linewidth=2)
axes[1, 1].plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {xgb_auc:.3f})', color='orange', linewidth=2)
axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5)
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].set_title('ROC Curves Comparison')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature Importance (XGBoost)
plt.figure(figsize=(12, 8))
feature_importance = xgb_model.feature_importances_
feature_names = X_train.columns
indices = np.argsort(feature_importance)[::-1]

# Plot top 15 features
top_features = 15
plt.title(f'Top {top_features} XGBoost Feature Importance')
plt.bar(range(top_features), feature_importance[indices[:top_features]], color='orange')
plt.xticks(range(top_features), [feature_names[i] for i in indices[:top_features]], rotation=45, ha='right')
plt.ylabel('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Random Forest Confusion Matrix
cm_rf = confusion_matrix(y_test, rf_predictions)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['No Churn', 'Churn'], 
            yticklabels=['No Churn', 'Churn'],
            ax=axes[0])
axes[0].set_title('Random Forest Confusion Matrix')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# XGBoost Confusion Matrix
cm_xgb = confusion_matrix(y_test, xgb_predictions)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['No Churn', 'Churn'], 
            yticklabels=['No Churn', 'Churn'],
            ax=axes[1])
axes[1].set_title('XGBoost Confusion Matrix')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*50)
print("Analysis Complete!")
print("="*50)
print("Files saved:")
print("- model_comparison.png: Overall model comparison")
print("- feature_importance.png: XGBoost feature importance")
print("- confusion_matrices.png: Confusion matrices for both models")
print("\nRecommendation:")
if xgb_auc > rf_auc:
    print("XGBoost shows better performance and is recommended for deployment.")
else:
    print("Random Forest shows better performance and is recommended for deployment.")
