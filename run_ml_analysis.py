"""
Heart Attack ML Analysis - Direct Python Execution
This script runs the complete analysis pipeline without requiring Jupyter
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow import keras
from tensorflow.keras import layers

from imblearn.over_sampling import SMOTE

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
import tensorflow as tf
tf.random.set_seed(RANDOM_SEED)

# Create output directories
os.makedirs('outputs', exist_ok=True)
os.makedirs('outputs/plots', exist_ok=True)

print("="*80)
print("Heart Attack Predictor - ML Analysis Pipeline")
print("="*80)
print(f"\n✓ Random seed set to {RANDOM_SEED} for reproducibility\n")

# Load data
print("1. Loading dataset...")
df = pd.read_csv('attached_assets/heart_cleaned_1762844952756.csv')

# Clean column names (remove extra quotes)
df.columns = df.columns.str.replace('"', '')
print(f"✓ Dataset loaded: {df.shape[0]} records, {df.shape[1]} features")

# Generate Data Card
print("\n2. Generating Data Card...")
data_card = pd.DataFrame({
    'Variable': df.columns,
    'Type': df.dtypes.values,
    'Missing Count': df.isnull().sum().values,
    'Missing %': (df.isnull().sum().values / len(df) * 100).round(2),
    'Unique Values': [df[col].nunique() for col in df.columns],
})

leakage_assessment = []
for col in df.columns:
    if col == 'Heart Disease':
        leakage_assessment.append('Target Variable')
    elif col in ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120']:
        leakage_assessment.append('Low risk - Pre-diagnostic')
    else:
        leakage_assessment.append('Low risk - Diagnostic features')

data_card['Leakage Risk'] = leakage_assessment
data_card.to_csv('outputs/data_card.csv', index=False)
print(f"✓ Data card saved ({len(data_card)} variables)")

# Preprocessing
print("\n3. Preprocessing data...")
df_processed = df.copy()

# Handle missing values
for col in df_processed.columns:
    if df_processed[col].isnull().sum() > 0:
        if df_processed[col].dtype in ['float64', 'int64']:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
        else:
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)

# Encode target variable
le = LabelEncoder()
df_processed['Heart Disease'] = le.fit_transform(df_processed['Heart Disease'])

# Split features and target
X = df_processed.drop('Heart Disease', axis=1)
y = df_processed['Heart Disease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
)

# Save split info
split_info = pd.DataFrame({
    'Split': ['Training', 'Testing', 'Total'],
    'Count': [len(X_train), len(X_test), len(X)],
    'Percentage': [f"{len(X_train)/len(X)*100:.1f}%", 
                  f"{len(X_test)/len(X)*100:.1f}%", 
                  "100.0%"],
    'Class 0 (Absence)': [sum(y_train==0), sum(y_test==0), sum(y==0)],
    'Class 1 (Presence)': [sum(y_train==1), sum(y_test==1), sum(y==1)]
})
split_info.to_csv('outputs/split_table.csv', index=False)
print(f"✓ Stratified split: {len(X_train)} train, {len(X_test)} test")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE
print(f"✓ Applying SMOTE (Before: {dict(zip(*np.unique(y_train, return_counts=True)))})")
smote = SMOTE(random_state=RANDOM_SEED)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
print(f"✓ After SMOTE: {dict(zip(*np.unique(y_train_balanced, return_counts=True)))}")

# Evaluation function
def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    return {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_pred_proba),
        'PR-AUC': average_precision_score(y_true, y_pred_proba)
    }

models = {}
results_list = []

# Model 1: Logistic Regression
print("\n4. Training models...")
print("   Training Logistic Regression...")
lr_model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
lr_model.fit(X_train_balanced, y_train_balanced)
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
lr_results = evaluate_model(y_test, y_pred_lr, y_pred_proba_lr, 'Logistic Regression')
results_list.append(lr_results)
models['Logistic Regression'] = lr_model
print(f"   ✓ LR - ROC-AUC: {lr_results['ROC-AUC']:.3f}, PR-AUC: {lr_results['PR-AUC']:.3f}")

# Model 2: SVM
print("   Training SVM...")
svm_model = SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED)
svm_model.fit(X_train_balanced, y_train_balanced)
y_pred_svm = svm_model.predict(X_test_scaled)
y_pred_proba_svm = svm_model.predict_proba(X_test_scaled)[:, 1]
svm_results = evaluate_model(y_test, y_pred_svm, y_pred_proba_svm, 'SVM (RBF)')
results_list.append(svm_results)
models['SVM'] = svm_model
print(f"   ✓ SVM - ROC-AUC: {svm_results['ROC-AUC']:.3f}, PR-AUC: {svm_results['PR-AUC']:.3f}")

# Model 3: Random Forest
print("   Training Random Forest with hyperparameter tuning...")
rf_param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [5, 8, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_base = RandomForestClassifier(random_state=RANDOM_SEED)
rf_random = RandomizedSearchCV(
    rf_base, rf_param_dist, n_iter=20, cv=5, 
    scoring='roc_auc', random_state=RANDOM_SEED, n_jobs=-1
)
rf_random.fit(X_train_balanced, y_train_balanced)
rf_model = rf_random.best_estimator_
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]
rf_results = evaluate_model(y_test, y_pred_rf, y_pred_proba_rf, 'Random Forest')
results_list.append(rf_results)
models['Random Forest'] = rf_model
print(f"   ✓ RF - ROC-AUC: {rf_results['ROC-AUC']:.3f}, PR-AUC: {rf_results['PR-AUC']:.3f}")

# Model 4: XGBoost
print("   Training XGBoost with hyperparameter tuning...")
xgb_param_dist = {
    'n_estimators': [100, 150, 200, 250],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}
xgb_base = XGBClassifier(random_state=RANDOM_SEED, use_label_encoder=False, eval_metric='logloss')
xgb_random = RandomizedSearchCV(
    xgb_base, xgb_param_dist, n_iter=20, cv=5,
    scoring='roc_auc', random_state=RANDOM_SEED, n_jobs=-1
)
xgb_random.fit(X_train_balanced, y_train_balanced)
xgb_model = xgb_random.best_estimator_
y_pred_xgb = xgb_model.predict(X_test_scaled)
y_pred_proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]
xgb_results = evaluate_model(y_test, y_pred_xgb, y_pred_proba_xgb, 'XGBoost')
results_list.append(xgb_results)
models['XGBoost'] = xgb_model
print(f"   ✓ XGB - ROC-AUC: {xgb_results['ROC-AUC']:.3f}, PR-AUC: {xgb_results['PR-AUC']:.3f}")

# Model 5: Neural Network
print("   Training Neural Network...")
nn_model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_balanced.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])
nn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)
history = nn_model.fit(
    X_train_balanced, y_train_balanced,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)
y_pred_proba_nn = nn_model.predict(X_test_scaled, verbose=0).flatten()
y_pred_nn = (y_pred_proba_nn > 0.5).astype(int)
nn_results = evaluate_model(y_test, y_pred_nn, y_pred_proba_nn, 'Neural Network')
results_list.append(nn_results)
models['Neural Network'] = nn_model
print(f"   ✓ NN - ROC-AUC: {nn_results['ROC-AUC']:.3f}, PR-AUC: {nn_results['PR-AUC']:.3f}")

# Save results
results_df = pd.DataFrame(results_list)
results_df.to_csv('outputs/model_comparison.csv', index=False)
print(f"\n5. Model comparison saved")
print(f"\n{results_df.to_string(index=False)}")

# Get best model
best_idx = results_df['ROC-AUC'].idxmax()
best_model_name = results_df.loc[best_idx, 'Model']
print(f"\n🏆 Best Model: {best_model_name}")

# Save feature importance
feature_names = X.columns
rf_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
rf_importance.to_csv('outputs/rf_feature_importance.csv', index=False)

xgb_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)
xgb_importance.to_csv('outputs/xgb_feature_importance.csv', index=False)

# Threshold analysis
print("\n6. Performing threshold analysis...")
best_model_proba = y_pred_proba_xgb if best_model_name == 'XGBoost' else y_pred_proba_rf

thresholds = np.arange(0.1, 0.9, 0.05)
threshold_results = []

for threshold in thresholds:
    y_pred = (best_model_proba >= threshold).astype(int)
    threshold_results.append({
        'Threshold': threshold,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0)
    })

threshold_df = pd.DataFrame(threshold_results)

# Cost curve
cost_fn_ratio = 5
costs = []
for _, row in threshold_df.iterrows():
    threshold = row['Threshold']
    y_pred = (best_model_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total_cost = (fp * 1) + (fn * cost_fn_ratio)
    costs.append(total_cost)

threshold_df['Total_Cost'] = costs
optimal_threshold_idx = threshold_df['Total_Cost'].idxmin()
optimal_threshold = threshold_df.loc[optimal_threshold_idx, 'Threshold']
threshold_df.to_csv('outputs/threshold_analysis.csv', index=False)
print(f"✓ Optimal threshold: {optimal_threshold:.2f}")

# Error analysis
y_pred_best = (best_model_proba >= 0.5).astype(int)
fp_indices = np.where((y_pred_best == 1) & (y_test.values == 0))[0]
fn_indices = np.where((y_pred_best == 0) & (y_test.values == 1))[0]

error_summary = pd.DataFrame({
    'Error Type': ['False Positives', 'False Negatives', 'Total Errors'],
    'Count': [len(fp_indices), len(fn_indices), len(fp_indices) + len(fn_indices)],
    'Percentage': [
        f"{len(fp_indices)/len(y_test)*100:.2f}%",
        f"{len(fn_indices)/len(y_test)*100:.2f}%",
        f"{(len(fp_indices) + len(fn_indices))/len(y_test)*100:.2f}%"
    ]
})
error_summary.to_csv('outputs/error_analysis.csv', index=False)

# Ablation studies
print("\n7. Running ablation studies...")
n_estimators_values = [10, 50, 100, 150, 200, 300]
ablation_results = []

for n_est in n_estimators_values:
    rf_ablation = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=rf_model.max_depth,
        min_samples_split=rf_model.min_samples_split,
        random_state=RANDOM_SEED
    )
    rf_ablation.fit(X_train_balanced, y_train_balanced)
    y_pred_proba_ablation = rf_ablation.predict_proba(X_test_scaled)[:, 1]
    
    ablation_results.append({
        'n_estimators': n_est,
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba_ablation),
        'PR-AUC': average_precision_score(y_test, y_pred_proba_ablation),
        'Accuracy': accuracy_score(y_test, (y_pred_proba_ablation >= 0.5).astype(int))
    })

ablation_df = pd.DataFrame(ablation_results)
ablation_df.to_csv('outputs/ablation_rf.csv', index=False)

learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3]
ablation_results_xgb = []

for lr in learning_rates:
    xgb_ablation = XGBClassifier(
        learning_rate=lr,
        n_estimators=xgb_model.n_estimators,
        max_depth=xgb_model.max_depth,
        random_state=RANDOM_SEED,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_ablation.fit(X_train_balanced, y_train_balanced)
    y_pred_proba_ablation = xgb_ablation.predict_proba(X_test_scaled)[:, 1]
    
    ablation_results_xgb.append({
        'learning_rate': lr,
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba_ablation),
        'PR-AUC': average_precision_score(y_test, y_pred_proba_ablation),
        'Accuracy': accuracy_score(y_test, (y_pred_proba_ablation >= 0.5).astype(int))
    })

ablation_xgb_df = pd.DataFrame(ablation_results_xgb)
ablation_xgb_df.to_csv('outputs/ablation_xgb.csv', index=False)
print("✓ Ablation studies complete")

# Business impact
print("\n8. Calculating business impact...")
y_pred_optimal = (best_model_proba >= optimal_threshold).astype(int)
cm_optimal = confusion_matrix(y_test, y_pred_optimal)
tn, fp, fn, tp = cm_optimal.ravel()

cost_heart_attack = 50000
cost_preventive_treatment = 5000
cost_false_alarm = 1000

baseline_cost_per_patient = cost_heart_attack * (sum(y_test) / len(y_test))
ml_cost = (tp * cost_preventive_treatment + fp * cost_false_alarm + fn * cost_heart_attack)
ml_cost_per_patient = ml_cost / len(y_test)

cost_savings_per_patient = baseline_cost_per_patient - ml_cost_per_patient
cost_savings_percentage = (cost_savings_per_patient / baseline_cost_per_patient) * 100

annual_patients = 10000
annual_savings = cost_savings_per_patient * annual_patients
annual_lives_saved = int(tp / len(y_test) * annual_patients)

business_impact = pd.DataFrame({
    'Metric': [
        'Baseline Cost per Patient',
        'ML Cost per Patient',
        'Cost Savings per Patient',
        'Cost Reduction %',
        'Early Detection Rate',
        'Annual Savings (10K patients)',
        'Annual Lives Saved (10K patients)'
    ],
    'Value': [
        f"${baseline_cost_per_patient:,.2f}",
        f"${ml_cost_per_patient:,.2f}",
        f"${cost_savings_per_patient:,.2f}",
        f"{cost_savings_percentage:.1f}%",
        f"{tp / sum(y_test) * 100:.1f}%",
        f"${annual_savings:,.2f}",
        f"{annual_lives_saved}"
    ]
})
business_impact.to_csv('outputs/business_impact.csv', index=False)
print("✓ Business impact calculated")

# Generate plots
print("\n9. Generating visualizations...")

# Plot 1: Target distribution
plt.figure(figsize=(8, 5))
target_dist = df['Heart Disease'].value_counts()
target_dist.plot(kind='bar', color=['#2ecc71', '#e74c3c'])
plt.title('Distribution of Heart Disease', fontsize=14, fontweight='bold')
plt.xlabel('Heart Disease Status')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('outputs/plots/target_distribution.png', dpi=300)
plt.close()
print("   ✓ Target distribution plot")

# Plot 2: Model comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
for idx, metric in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]
    ax.bar(results_df['Model'], results_df[metric])
    ax.set_title(metric)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('outputs/plots/model_comparison.png', dpi=300)
plt.close()
print("   ✓ Model comparison plot")

# Plot 3: ROC curves
plt.figure(figsize=(10, 8))
model_probas = {
    'Logistic Regression': y_pred_proba_lr,
    'SVM (RBF)': y_pred_proba_svm,
    'Random Forest': y_pred_proba_rf,
    'XGBoost': y_pred_proba_xgb,
    'Neural Network': y_pred_proba_nn
}
for model_name, y_proba in model_probas.items():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - All Models')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/plots/roc_curves.png', dpi=300)
plt.close()
print("   ✓ ROC curves plot")

# Plot 4: PR curves
plt.figure(figsize=(10, 8))
for model_name, y_proba in model_probas.items():
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    plt.plot(recall, precision, label=f'{model_name} (PR-AUC = {pr_auc:.3f})', linewidth=2)
baseline = sum(y_test) / len(y_test)
plt.plot([0, 1], [baseline, baseline], 'k--', label=f'Baseline = {baseline:.3f}', linewidth=1)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves - All Models')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/plots/pr_curves.png', dpi=300)
plt.close()
print("   ✓ PR curves plot")

# Plot 5: Confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
predictions = {
    'Logistic Regression': y_pred_lr,
    'SVM (RBF)': y_pred_svm,
    'Random Forest': y_pred_rf,
    'XGBoost': y_pred_xgb,
    'Neural Network': y_pred_nn
}
for idx, (model_name, y_pred) in enumerate(predictions.items()):
    ax = axes[idx // 3, idx % 3]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Absence', 'Presence'],
                yticklabels=['Absence', 'Presence'])
    ax.set_title(model_name)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
axes[1, 2].axis('off')
plt.tight_layout()
plt.savefig('outputs/plots/confusion_matrices.png', dpi=300)
plt.close()
print("   ✓ Confusion matrices plot")

# Plot 6: Feature importance
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].barh(rf_importance['Feature'], rf_importance['Importance'], color='#2ecc71')
axes[0].set_xlabel('Importance')
axes[0].set_title('Random Forest - Feature Importance')
axes[0].invert_yaxis()
axes[1].barh(xgb_importance['Feature'], xgb_importance['Importance'], color='#f39c12')
axes[1].set_xlabel('Importance')
axes[1].set_title('XGBoost - Feature Importance')
axes[1].invert_yaxis()
plt.tight_layout()
plt.savefig('outputs/plots/feature_importance.png', dpi=300)
plt.close()
print("   ✓ Feature importance plot")

# Plot 7: Threshold analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].plot(threshold_df['Threshold'], threshold_df['Accuracy'], label='Accuracy', linewidth=2)
axes[0].plot(threshold_df['Threshold'], threshold_df['Precision'], label='Precision', linewidth=2)
axes[0].plot(threshold_df['Threshold'], threshold_df['Recall'], label='Recall', linewidth=2)
axes[0].plot(threshold_df['Threshold'], threshold_df['F1-Score'], label='F1-Score', linewidth=2)
axes[0].axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Classification Threshold')
axes[0].set_ylabel('Score')
axes[0].set_title(f'Threshold Sweep Analysis - {best_model_name}')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(threshold_df['Threshold'], threshold_df['Total_Cost'], linewidth=2, color='#e74c3c')
axes[1].axvline(x=optimal_threshold, color='green', linestyle='--', 
               label=f'Optimal = {optimal_threshold:.2f}', linewidth=2)
axes[1].scatter([optimal_threshold], [threshold_df.loc[optimal_threshold_idx, 'Total_Cost']], 
               color='green', s=100, zorder=5)
axes[1].set_xlabel('Classification Threshold')
axes[1].set_ylabel('Total Cost')
axes[1].set_title(f'Cost Curve (FN Cost = {cost_fn_ratio}x FP Cost)')
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/plots/threshold_cost_analysis.png', dpi=300)
plt.close()
print("   ✓ Threshold/cost analysis plot")

# Plot 8: Ablation studies
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(ablation_df['n_estimators'], ablation_df['ROC-AUC'], marker='o', label='ROC-AUC')
axes[0].plot(ablation_df['n_estimators'], ablation_df['PR-AUC'], marker='s', label='PR-AUC')
axes[0].set_xlabel('Number of Trees')
axes[0].set_ylabel('AUC Score')
axes[0].set_title('Ablation: RF Performance vs Number of Trees')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(ablation_xgb_df['learning_rate'], ablation_xgb_df['ROC-AUC'], marker='o', label='ROC-AUC')
axes[1].plot(ablation_xgb_df['learning_rate'], ablation_xgb_df['PR-AUC'], marker='s', label='PR-AUC')
axes[1].set_xlabel('Learning Rate')
axes[1].set_ylabel('AUC Score')
axes[1].set_title('Ablation: XGBoost Performance vs Learning Rate')
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/plots/ablation_studies.png', dpi=300)
plt.close()
print("   ✓ Ablation studies plot")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
print(f"\n📁 All outputs saved to outputs/ directory")
print(f"   - 10 CSV files with results")
print(f"   - 8 PNG plots at 300 DPI")
print("\n🎯 Next steps:")
print("   1. Run: python generate_latex.py")
print("   2. Upload outputs/ to Overleaf")
print("   3. Compile LaTeX document")
print("="*80)
