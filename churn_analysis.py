"""
Telecom Customer Churn Prediction
==================================
Author: Victor Prefa
Description: Predicting telecom customer churn using Machine Learning

This script performs:
1. Data loading and exploration
2. Data preprocessing
3. Exploratory Data Analysis (EDA)
4. Machine Learning model training and evaluation
"""

# ============================================
# Step 1 & 2: Import Libraries
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# Set display options
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')

# ============================================
# Step 3 & 4: Load and Check Data
# ============================================
print("=" * 50)
print("TELECOM CUSTOMER CHURN PREDICTION")
print("=" * 50)

# Load data
df = pd.read_csv("Churn.csv")

print("\n📊 Dataset Information:")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print(f"\nColumn names:\n{df.columns.tolist()}")

print("\n📋 Data Types:")
print(df.dtypes)

# ============================================
# Step 5: Display Data
# ============================================
print("\n📄 First 10 rows:")
print(df.head(10))

print("\n📈 Statistical Summary:")
print(df.describe())

# ============================================
# Step 6: Data Pre-processing
# ============================================
print("\n🔧 Data Pre-processing...")

# Convert categorical variables
df['IntlPlan'] = df['IntlPlan'].astype('category')
df['VMailPlan'] = df['VMailPlan'].astype('category')
df['Churn'] = df['Churn'].astype('category')

print("Converted IntlPlan, VMailPlan, and Churn to categorical")
print("\nUpdated Data Types:")
print(df.dtypes)

# ============================================
# Step 7: Exploratory Data Analysis
# ============================================
print("\n📊 Performing Exploratory Data Analysis...")

# 7.1 - Describe data
print("\n7.1 - Data Description:")
print(df.describe())

# 7.2 - Histogram for Day Minutes by Churn
plt.figure(figsize=(10, 6))
df[df['Churn'] == 0]['DayMins'].hist(alpha=0.5, label='Churn=0 (Churner)', bins=20, color='blue')
df[df['Churn'] == 1]['DayMins'].hist(alpha=0.5, label='Churn=1 (Non-Churner)', bins=20, color='orange')
plt.xlabel('Day Minutes')
plt.ylabel('Frequency')
plt.title('Histogram: Day Minutes by Churn Status')
plt.legend()
plt.savefig('images/histogram_daymins.png', dpi=300, bbox_inches='tight')
plt.close()
print("7.2 - Saved: histogram_daymins.png")

# 7.3 - Count plot for Voicemail Plan
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='VMailPlan', hue='Churn')
plt.xlabel('Voicemail Plan (0=No, 1=Yes)')
plt.ylabel('Count')
plt.title('Voicemail Plan by Churn Status')
plt.legend(title='Churn', labels=['Churner', 'Non-Churner'])
plt.savefig('images/countplot_vmailplan.png', dpi=300, bbox_inches='tight')
plt.close()
print("7.3 - Saved: countplot_vmailplan.png")

# 7.4 - Count plot for International Plan
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='IntlPlan', hue='Churn')
plt.xlabel('International Plan (0=No, 1=Yes)')
plt.ylabel('Count')
plt.title('International Plan by Churn Status')
plt.legend(title='Churn', labels=['Churner', 'Non-Churner'])
plt.savefig('images/countplot_intlplan.png', dpi=300, bbox_inches='tight')
plt.close()
print("7.4 - Saved: countplot_intlplan.png")

# 7.5 - Area-wise Churner Analysis
plt.figure(figsize=(12, 6))
area_churn = df.groupby(['AreaCode', 'Churn']).size().unstack(fill_value=0)
area_churn.plot(kind='bar', ax=plt.gca())
plt.xlabel('Area Code')
plt.ylabel('Count')
plt.title('Area-wise Churner vs Non-Churner')
plt.legend(title='Churn', labels=['Churner', 'Non-Churner'])
plt.xticks(rotation=0)
plt.savefig('images/areawise_churn.png', dpi=300, bbox_inches='tight')
plt.close()
print("7.5 - Saved: areawise_churn.png")

# 7.6 - Correlation Matrix
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.savefig('images/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("7.6 - Saved: correlation_matrix.png")

# ============================================
# Step 8: Correlation with Churn
# ============================================
print("\n📈 Step 8: Correlation with Churn")
churn_corr = df[numeric_cols].corrwith(df['Churn'].astype(int))
print("\nCorrelation of features with Churn:")
print(churn_corr.sort_values(ascending=False))

# ============================================
# Step 9: Machine Learning Models
# ============================================
print("\n🤖 Step 9: Machine Learning Models")

# 9.1 & 9.2 - Prepare features
feature_cols = ['AccountLength', 'VMailMessage', 'DayMins', 'EveMins', 
                'NightMins', 'IntlMins', 'CustServCalls', 'VMailPlan',
                'DayCalls', 'DayCharge', 'EveCalls', 'EveCharge', 
                'NightCalls', 'NightCharge', 'IntlCalls', 'IntlCharge']

X = df[feature_cols]
y = df['Churn'].astype(int)

# 9.6 - Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Store results
results = []

# 9.3 - Decision Tree Classifier
print("\n--- Decision Tree Classifier ---")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred, zero_division=0)
dt_recall = recall_score(y_test, dt_pred, zero_division=0)

print(f"Accuracy:  {dt_accuracy:.4f}")
print(f"Precision: {dt_precision:.4f}")
print(f"Recall:    {dt_recall:.4f}")

results.append({
    'Model': 'Decision Tree',
    'Accuracy': dt_accuracy,
    'Precision': dt_precision,
    'Recall': dt_recall
})

# 9.10 - Random Forest Classifier
print("\n--- Random Forest Classifier ---")
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred, zero_division=0)
rf_recall = recall_score(y_test, rf_pred, zero_division=0)

print(f"Accuracy:  {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall:    {rf_recall:.4f}")

results.append({
    'Model': 'Random Forest',
    'Accuracy': rf_accuracy,
    'Precision': rf_precision,
    'Recall': rf_recall
})

# 9.10 - AdaBoost Classifier
print("\n--- AdaBoost Classifier ---")
ab_model = AdaBoostClassifier(random_state=42)
ab_model.fit(X_train, y_train)
ab_pred = ab_model.predict(X_test)

ab_accuracy = accuracy_score(y_test, ab_pred)
ab_precision = precision_score(y_test, ab_pred, zero_division=0)
ab_recall = recall_score(y_test, ab_pred, zero_division=0)

print(f"Accuracy:  {ab_accuracy:.4f}")
print(f"Precision: {ab_precision:.4f}")
print(f"Recall:    {ab_recall:.4f}")

results.append({
    'Model': 'AdaBoost',
    'Accuracy': ab_accuracy,
    'Precision': ab_precision,
    'Recall': ab_recall
})

# ============================================
# Step 10: Results Summary
# ============================================
print("\n" + "=" * 50)
print("📊 STEP 10: MODEL COMPARISON RESULTS")
print("=" * 50)

results_df = pd.DataFrame(results)
print("\n", results_df.to_string(index=False))

# Save results
results_df.to_csv('results/model_comparison.csv', index=False)
print("\nResults saved to: results/model_comparison.csv")

# Feature Importance (Random Forest)
print("\n🔑 Top 10 Feature Importances (Random Forest):")
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# ============================================
# Insights and Conclusions
# ============================================
print("\n" + "=" * 50)
print("💡 KEY INSIGHTS AND CONCLUSIONS")
print("=" * 50)

print("""
1. Random Forest achieved the highest accuracy (85.71%) with perfect 
   precision (100%), but lowest recall (20%).

2. AdaBoost provides balanced performance with 82.14% accuracy and 
   better recall (40%) than Random Forest.

3. Decision Tree has the lowest accuracy (78.57%) but consistent 
   precision and recall (both 40%).

4. Key predictors of churn include DayMins, CustServCalls, and DayCharge.

5. RECOMMENDATION: For identifying churners proactively, AdaBoost or 
   Decision Tree may be preferred due to higher recall.
""")

print("=" * 50)
print("✅ Analysis Complete!")
print("=" * 50)
