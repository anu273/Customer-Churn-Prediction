import numpy as np
import pandas as pd
from model_prediction import data
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def data_info():
    df = data()
    #Change categorical to numerical value
    df['Gender'] = df['Gender'].map({'Male':0, 'Female':1})

    #One hot-encoding
    df = pd.get_dummies(df, columns=['Geography'], drop_first=True, dtype=int)

    X = df.drop(columns=['Exited'])
    y = df['Exited']
    X_ref, X_curr, y_ref, y_curr = train_test_split(X, y, test_size=0.3, random_state=2025)
    return X_ref, X_curr, y_ref, y_curr


def calculate_psi(expected, actual, bins=10):

    # Define bins using quantiles of the 'expected' (reference) distribution
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    
    # Calculate histogram counts for each distribution based on the same bins
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Convert counts to percentages
    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)
    
    # Replace 0s with a small number to avoid division by zero in log
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

    # Calculate PSI value
    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return psi_value


def calculate_ks(y_true, y_pred_proba):
    # Separate the predicted probabilities for the two actual classes
    class0_scores = y_pred_proba[y_true == 0]
    class1_scores = y_pred_proba[y_true == 1]
    
    # The ks_2samp function finds the max distance between two CDFs
    ks_stat, _ = ks_2samp(class0_scores, class1_scores)
    return ks_stat


def model_drift_monitoring():
    monitoring_results = {}
    X_ref, X_curr, y_ref, y_curr = data_info()

    best_parameters = {'Logistic Regression': LogisticRegression(C=0.01, max_iter=5000, solver='saga'),
 'Random Forest': RandomForestClassifier(bootstrap=False, max_depth=20, min_samples_split=5, n_estimators=300),
 'Gradient Boosting': GradientBoostingClassifier(learning_rate=0.2, max_depth=5, n_estimators=300, subsample=0.9),
 'XGBoost': XGBClassifier(colsample_bytree=0.8, enable_categorical=False, eval_metric='logloss', gamma=0.2, grow_policy=None,
            learning_rate=0.1, max_depth=7, monotone_constraints=None, multi_strategy=None, n_estimators=300,
            n_jobs=None, num_parallel_tree=None, random_state=None),
 'Decision Tree': DecisionTreeClassifier(max_depth=30, min_samples_split=5),
 'SVM': SVC(C=10, gamma='auto', probability=True),
 'KNN': KNeighborsClassifier(metric='manhattan', n_neighbors=3, weights='distance')}
    
    for name, model in best_parameters.items():
        print(f"\nProcessing Model: {name}...")

        model.fit(X_ref, y_ref)
        
        y_ref_pred_proba = model.predict_proba(X_ref)[:, 1]
        y_curr_pred_proba = model.predict_proba(X_curr)[:, 1]
        
        y_ref_pred = model.predict(X_ref)
        y_curr_pred = model.predict(X_curr)

        prediction_psi = calculate_psi(y_ref_pred_proba, y_curr_pred_proba)
        
        monitoring_results[name] = {
            "Refernece KS": calculate_ks(y_ref, y_ref_pred_proba),
            "Current KS": calculate_ks(y_curr, y_curr_pred_proba),
            "Prediction Drift (PSI)": prediction_psi
        }
    return monitoring_results

def csi_feature_drift():
    features_to_monitor = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'BalanceSalaryRatio', 'TenureByAge','CreditScoreGivenAge']
    csi_results = {}
    X_ref, X_curr, _, _ = data_info()
    for feature in features_to_monitor:
        csi_value = calculate_psi(X_ref[feature], X_curr[feature])
        
        drift_level = "Stable"
        if csi_value >= 0.25:
            drift_level = "Major Shift"
        elif csi_value >= 0.1:
            drift_level = "Moderate Shift"
            
        print(f"  - CSI for '{feature}': {csi_value:.4f} ({drift_level})")
        csi_results[feature] = csi_value
    return csi_results