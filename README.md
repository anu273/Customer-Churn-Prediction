
# Customer Churn Prediction

A machine learning project that predicts customer churn for banks using multiple ML models with an interactive Streamlit dashboard for real-time predictions and visualizations.


## 🎯 Project Overview

This project predicts whether a bank customer will churn based on their demographic and account information. It implements 7 different machine learning models and provides an interactive web application for real-time predictions.

**Key Highlights:**
- 85% accuracy with Random Forest model
- Interactive Streamlit dashboard
- 7 ML models comparison
- SMOTE for handling imbalanced data
- Real-time predictions with visualizations

## 📊 Live Demo

🌐 **Try the app:** [Customer Churn Prediction Dashboard](https://customer-churn-prediction-f3clxr5umrxqxh3egbwt6p.streamlit.app/)

## 📁 Dataset

The dataset contains 10,000 bank customer records with the following features:

| Feature | Description |
|---------|-------------|
| `CreditScore` | Customer's credit score |
| `Geography` | Customer's location (France, Spain, Germany) |
| `Gender` | Customer's gender |
| `Age` | Customer's age |
| `Tenure` | Years as a bank client |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products used |
| `HasCrCard` | Credit card ownership (0/1) |
| `IsActiveMember` | Active membership status (0/1) |
| `EstimatedSalary` | Customer's estimated salary |
| `Exited` | Churn status (1 = churned, 0 = retained) |

**Class Distribution:**
- Churned Customers: 20.4%
- Retained Customers: 79.6%

## 🔍 Exploratory Data Analysis

The project includes comprehensive EDA covering:
- Churn distribution analysis
- Feature correlation heatmaps
- Churn patterns by geography, gender, and account activity
- Age and balance distribution analysis
- Product usage vs churn relationship

## ⚙️ Feature Engineering

Advanced feature engineering techniques applied:

```python
# New derived features
BalanceSalaryRatio = Balance / EstimatedSalary
TenureByAge = Tenure / Age
CreditScoreGivenAge = CreditScore / Age
```

**Preprocessing Steps:**
- One-hot encoding for `Geography`
- Label encoding for `Gender`
- Feature scaling using StandardScaler
- SMOTE for handling class imbalance

## 🤖 Machine Learning Models

Seven models were trained and evaluated:

| Model | Accuracy | Key Technique |
|-------|----------|---------------|
| **Random Forest** | **85%** | Ensemble learning |
| Gradient Boosting | 85% | Boosting |
| LightGBM | 84% | Gradient boosting framework |
| Logistic Regression | 70% | Linear classifier |
| SVM | 81% | Support vector machine |
| Decision Tree | 77% | Tree-based model |
| KNN | 75% | Instance-based learning |

**Model Optimization:**
- Hyperparameter tuning with RandomizedSearchCV
- Cross-validation for robust evaluation
- SMOTE applied during training
- Performance metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
joblib==1.3.2
plotly==5.17.0
scikit-learn==1.3.0
lightgbm==4.1.0
imbalanced-learn==0.11.0
matplotlib==3.7.2
seaborn==0.12.2
```

## 💻 Usage

### Run the Streamlit App

```bash
streamlit run Churn_app.py
```

The application will open in your browser at `http://localhost:8501`

### App Features

**1. Prediction Interface**
- Input customer details via intuitive form
- Get instant churn predictions from all 7 models
- View prediction probabilities
- Compare model outputs side-by-side

**2. Data Explorer**
- Browse the complete dataset
- Apply filters and search
- View statistical summaries

**3. Visualizations**
- Interactive EDA plots
- Feature importance charts
- Correlation heatmaps
- Churn distribution graphs

**4. Model Performance**
- Detailed metrics comparison
- Confusion matrices for each model
- ROC curves and AUC scores
- Feature importance rankings

## 📊 Results

### Model Performance Summary

- ROC-AUC Curve of all Models
<img width="691" height="545" alt="Image" src="https://github.com/user-attachments/assets/0b2cfee5-5f4d-4248-8234-136af0f36f86" />



- Best Model Performance
```
Best Model: Random Forest
├── Accuracy:    85%
├── Precision:   64%
├── Recall:      68%
├── F1-Score:    62%
└── ROC-AUC:     0.85
```

### Key Insights

- Geography plays a significant role in churn prediction
- Customers with 3-4 products are less likely to churn
- Age and balance are strong predictors
- Active membership significantly reduces churn probability
- Credit card ownership shows moderate correlation with retention

## 📂 Project Structure

```
customer-churn-prediction/
├── Churn_app.py                # Streamlit application
├── churn.csv                   # Dataset
├── models/                     # Trained model files
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── lightgbm.pkl
│   └── ...
├── notebooks/                  # Jupyter notebooks for EDA
│   └── EDA_and_Modeling.ipynb
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
└── images/                     # Screenshots and plots
    ├── dashboard.png
    └── results.png
```

## 🛠️ Technologies Used

- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, LightGBM
- **Data Balancing:** imbalanced-learn (SMOTE)
- **Model Persistence:** Joblib
