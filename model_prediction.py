# models.py
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE


# Load saved files
scaler = joblib.load("scaler.pkl")
metrics_data = joblib.load("metrics_data.pkl")
roc_data = joblib.load("roc_data.pkl")

num_cols = ["CreditScore","Age","Balance","EstimatedSalary",
            "BalanceSalaryRatio","TenureByAge","CreditScoreGivenAge"]
 
# Global variables to cache trained models
rf_model = None
gb_model = None
lgbm_model = None

def train_models():
    global rf_model, gb_model, lgbm_model

    # If models are already trained, just return them
    if rf_model is not None and gb_model is not None and lgbm_model is not None and scaler is not None:
        return rf_model, gb_model, lgbm_model

    # Otherwise, train models
    df = pd.read_csv("churn.csv", index_col=False)
    df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)   

    # Encode categorical
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df = pd.get_dummies(df, columns=['Geography'], drop_first=True, dtype=int)

    # Add new features
    df['BalanceSalaryRatio'] = df['Balance'] / df['EstimatedSalary']
    df['TenureByAge'] = df['Tenure'] / df['Age']
    df['CreditScoreGivenAge'] = df['CreditScore'] / df['Age']

    # Split
    X = df.drop(columns=['Exited'])
    y = df['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2025)

    # Scale
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Balance dataset
    smote = SMOTE(random_state=2025)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Train models
    rf_model = RandomForestClassifier(bootstrap=False, max_depth=20, min_samples_split=5, n_estimators=300)
    gb_model = GradientBoostingClassifier(learning_rate=0.2, max_depth=5, n_estimators=300, subsample=0.9)
    lgbm_model = LGBMClassifier(colsample_bytree=0.7, learning_rate=0.05, max_depth=10, n_estimators=300, num_leaves=100, subsample=0.8)

    rf_model.fit(X_train_res, y_train_res)
    gb_model.fit(X_train_res, y_train_res)
    lgbm_model.fit(X_train_res, y_train_res)

    return rf_model, gb_model, lgbm_model


def predict_all_models(input_dict):

    # Convert dict to DataFrame
    df = pd.DataFrame([input_dict])
    rf_model, gb_model, lgbm_model = train_models()

    
    #Change categorical to numerical value
    df['Gender'] = df['Gender'].map({'Male':0, 'Female':1})


    # Add new features
    df['BalanceSalaryRatio'] = df['Balance'] / df['EstimatedSalary']
    df['TenureByAge'] = df['Tenure'] / df['Age']
    df['CreditScoreGivenAge'] = df['CreditScore'] / df['Age']


    # Scale numeric columns using the already fitted scaler
    df[num_cols] = scaler.transform(df[num_cols])

    results = {}

    # Random Forest
    rf_pred = rf_model.predict(df)[0]
    rf_prob = rf_model.predict_proba(df)[0][1]
    results['Random Forest'] = {'prediction': int(rf_pred), 'probability': float(rf_prob)}

    # Gradient Boosting
    gb_pred = gb_model.predict(df)[0]
    gb_prob = gb_model.predict_proba(df)[0][1]
    results['Gradient Boosting'] = {'prediction': int(gb_pred), 'probability': float(gb_prob)}

    # LightGBM
    lgbm_pred = lgbm_model.predict(df)[0]
    lgbm_prob = lgbm_model.predict_proba(df)[0][1]
    results['LightGBM'] = {'prediction': int(lgbm_pred), 'probability': float(lgbm_prob)}

    return results


def data():
    df = pd.read_csv("churn.csv", index_col=False)
    df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)   

    # Add new features
    df['BalanceSalaryRatio'] = df['Balance'] / df['EstimatedSalary']
    df['TenureByAge'] = df['Tenure'] / df['Age']
    df['CreditScoreGivenAge'] = df['CreditScore'] / df['Age']
    return df



def model_metrics():
    metrics = metrics_data
    roc = roc_data
    return metrics, roc 

