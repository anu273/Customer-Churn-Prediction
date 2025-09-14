import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from model_prediction import predict_all_models, data, model_metrics


# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to right bottom, #1a2634, #203A43);
            color: white;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #45a049;
            transform: translateY(-2px);
        }
        .stNumberInput > div > div > input {
            border-radius: 5px;
        }
        .stSelectbox > div > div {
            border-radius: 5px;
        }
        div.block-container {
            padding-top: 2rem;
        }
        h1, h2, h3 {
            color: #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)


# --- Main Panel Display ---
st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h1 style='color: #4CAF50; font-size: 2.5rem; margin-bottom: 1rem;'>
            🏦 Bank Customer Churn Prediction and Analysis
        </h1>
        <h3 style='color: #E0E0E0; font-weight: normal; margin-bottom: 2rem;'>
            Welcome to the Churn Prediction Dashboard
        </h3>
        <p style='color: #CCCCCC; font-size: 1.1rem; line-height: 1.6;'>
            Unlock customer insights with powerful <span style='color: #4CAF50; font-weight: bold;'>churn prediction, 
            data analysis, and visualization</span> tools — powered by 
            <span style='color: #4CAF50; font-weight: bold;'>Logistic Regression, Random Forest, and XGBoost</span> models.
        </p>
        <p style='color: #CCCCCC; font-size: 1.1rem; margin-top: 1rem;'>
            Explore trends, analyze customer behavior, and predict churn risk for new profiles with ease.
        </p>
    </div>
    """, unsafe_allow_html=True)
st.markdown("""
    <hr style='border: none; height: 2px; background: linear-gradient(to right, transparent, #4CAF50, transparent);'>
    """, unsafe_allow_html=True)


# --- Tabs Layout ---
tab1, tab2, tab3 = st.tabs([
    "🔮 Predict Customer Churn", 
    "📊 Data Visualizations & Explorer", 
    "🤖 Model Comparison & Performance"
])


# --- Tab 1: User Input & Prediction ---
with tab1:
    st.header("🔮 Predict Customer Churn")
    st.write("Enter Customer Details for Churn Prediction")

    # Input fields for customer features
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, placeholder="Enter Credit Score")
        age = st.number_input("Age", min_value=18, max_value=100, placeholder="Enter age")
        tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3)
        balance = st.number_input("Balance", min_value=0.0, value=50000.0)
        num_products = st.number_input("Number of Products", min_value=1, max_value=5, value=1)
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        has_cr_card = st.selectbox("Has Credit Card", [0, 1])
        is_active_member = st.selectbox("Is Active Member", [0, 1])

    # Prediction button
    if st.button("🚀 Predict Churn Risk"):
        # Build input dictionary
        if geography == "France":
            germany = 0
            spain = 0
        elif geography == "Germany":
            germany = 1
            spain = 0 
        else:  # Spain
            germany = 0
            spain = 1
        
        input_dict = {
            "CreditScore": credit_score,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active_member,
            "EstimatedSalary": estimated_salary,
            "Geography_Germany": germany,
            "Geography_Spain": spain
        }

        # Call the model prediction function
        results = predict_all_models(input_dict)

        # Display results in a single row with updated style and color
        st.subheader("📌 Prediction Results")
        cols = st.columns(len(results))
        for idx, (model_name, output) in enumerate(results.items()):
            pred_label = "Churn" if output["prediction"] == 1 else "Not Churn"
            prob = output["probability"] * 100
            # New color scheme: blue for Not Churn, orange for Churn
            color = "#FF7926" if pred_label == "Churn" else "#42A5F5"
            bg_color = "rgba(66,165,245,0.10)" if pred_label == "Not Churn" else "rgba(255,167,38,0.10)"
            cols[idx].markdown(
            f"""
            <div style='background: {bg_color}; padding: 1.5rem; border-radius: 16px; text-align: center; box-shadow: 0 2px 12px rgba(66,165,245,0.12);'>
            <h3 style='color: {color}; margin-bottom: 1rem; font-size: 1.3rem;'>{model_name}</h3>
            <p style='font-size: 1.2rem; margin-bottom: 0.5rem;'>
            <span style='color: {color}; font-weight: bold;'>{pred_label}</span>
            </p>
            <div style='width: 85%; margin: 0 auto;'>
            </div>
            <p style='font-size: 1.1rem; color: #F5F5F5;'>
            📊 Probability: <span style='color: {color}; font-weight: bold;'>{prob:.2f}%</span>
            </p>
            </div>
            """, unsafe_allow_html=True
            )
    

# --- Tab 2: Data Visualizations & Explorer ---
with tab2:
    df = data()

    st.header("📊 Data Visualizations & Explorer")
    
        
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # --- Column selection for visualization ---
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    
    # --- Visualization 1: Correlation Heatmap ---
    st.subheader("Correlation Heatmap")
    if len(numeric_cols) > 1:
        fig_corr = px.imshow(df[numeric_cols].corr(), 
                                text_auto=True, 
                                aspect="auto", 
                                color_continuous_scale='Viridis')
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")

    # --- Visualization 2: Pie Chart for Exited vs Not Exited ---
    st.subheader("Churn Distribution (Exited vs Not Exited)")

    if 'Exited' in df.columns:
        churn_counts = df['Exited'].value_counts().rename({0: 'Not Exited', 1: 'Exited'})
        fig_pie = px.pie(
            names=churn_counts.index,
            values=churn_counts.values,
            color=churn_counts.index,
            color_discrete_map={'Not Exited': "#356186", 'Exited': "#D2B536"},
            hole=0.4
        )
        fig_pie.update_traces(textinfo='percent+label', pull=[0, 0.1])
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Column 'Exited' not found in the dataset.")


    # --- Visualization 3: Histogram / Distribution ---
    st.subheader("Distribution of Numeric Features")
    cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'BalanceSalaryRatio', 'TenureByAge','CreditScoreGivenAge']
    num_feature = st.selectbox("Select Numeric Feature", cols)
    fig_hist = px.histogram(df, x=num_feature, nbins=30, color_discrete_sequence=['#4CAF50'])
    st.plotly_chart(fig_hist, use_container_width=True)


    # --- Visualization 4: Countplot / Bar Chart for Categorical Features ---
    st.subheader("Bar Chart Analysis")

    # List of categorical columns in your dataset
    cat_cols = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']

    if cat_cols:
        cat_feature = st.selectbox("Select Feature", cat_cols)
        if 'Exited' in df.columns:
            fig_bar = px.histogram(
                df,
                x=cat_feature,
                color='Exited',
                barmode='group',
                color_discrete_map={0: "#42A5F5", 1: "#FF7926"},
                title=f"{cat_feature} vs Exited Count"
            )
        else:
            fig_bar = px.histogram(
                df,
                x=cat_feature,
                color_discrete_sequence=['#4CAF50'],
                title=f"{cat_feature} Distribution"
            )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No categorical columns available for bar chart.")


    # --- Visualization 5: Boxplot for Numeric Feature vs Target ---
    st.subheader("Boxplot for Numeric Feature vs Target")

    # List of numeric features you want to visualize
    num_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary', 
                'BalanceSalaryRatio', 'TenureByAge', 'CreditScoreGivenAge']

    if 'Exited' in df.columns and num_cols:
        num_feature_box = st.selectbox("Select Numeric Feature for Boxplot", num_cols, key='box')
        fig_box = px.box(
            df,
            x='Exited',
            y=num_feature_box,
            color='Exited',
            color_discrete_map={0: "#42A5F5", 1: "#FF7926"},
            title=f"{num_feature_box} vs Exited"
        )
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("Target column 'Exited' or numeric features missing for boxplot.")


# --- Tab 3: Model Comparison & Performance ---
with tab3:
    st.header("🤖 Model Comparison & Performance")

    all_metrics, roc = model_metrics()

    # --- Metrics Table (Full Width, Styled) ---
    metrics_rows = []
    for model_name, m in all_metrics.items():
        metrics_rows.append({
            "Model": model_name,
            "Accuracy": m["Accuracy"],
            "Precision": m["Precision"],
            "Recall": m["Recall"],
            "F1 Score": m["F1-Score"]
        })

    df_metrics = pd.DataFrame(metrics_rows)
    numeric_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
    for col in numeric_cols:
        df_metrics[col] = df_metrics[col].astype(float)

    st.subheader("Model Performance Metrics")
    st.markdown(
        """
        <style>
        .styled-table th, .styled-table td {padding: 0.7em 1em;}
        </style>
        """, unsafe_allow_html=True
    )
    st.dataframe(
        df_metrics.style.format({col: "{:.2f}" for col in numeric_cols})
        .highlight_max(axis=0, subset=numeric_cols, color='#4CAF50')
        .set_properties(**{'text-align': 'center'}, subset=df_metrics.columns),
        use_container_width=True
    )


    # --- Confusion Matrix Section ---
    st.subheader("Confusion Matrix per Model")
    cm = joblib.load("confusion_matrices_data.pkl")

    # Dropdown to select model
    cm_model_select = st.selectbox(
        "Select Model to View Confusion Matrix",
        list(cm.keys())
    )

    # Get the selected matrix
    cm = np.array(cm.get(cm_model_select))

    # Plot using plotly heatmap, centered in the page
    if cm is not None and cm.size > 0:
        center_col1, center_col2, center_col3 = st.columns([1, 2, 1])
        with center_col2:
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale='Blues',
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Predicted 0", "Predicted 1"],
                y=["Actual 0", "Actual 1"]
            )
            fig_cm.update_layout(
                width=500, height=500,
                xaxis=dict(side="top"),
                plot_bgcolor='rgba(245,245,245,0.5)',
                margin=dict(t=80, b=40),  # Increase top margin for title
                title={
                    'text': f"Confusion Matrix for {cm_model_select}",
                    'y':0.95,  # Move title higher
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                }
            )
            st.plotly_chart(fig_cm, use_container_width=False)
    else:
        st.info(f"No confusion matrix available for {cm_model_select}")



    # --- ROC-AUC Curve ---
    st.subheader("ROC-AUC Curve Comparison")

    if roc:
        fig_roc = go.Figure()

        # Add ROC curves for each model
        for model_name, roc_values in roc.items():
            fpr, tpr, auc = roc_values
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                line=dict(shape='spline', width=2),
                name=f"{model_name} (AUC = {auc:.2f})"
            ))

        # Add diagonal reference line
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash', color='gray', width=1.5),
            showlegend=True
        ))

        # Layout settings
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            title="ROC-AUC Curve",
            plot_bgcolor='rgba(245,245,245,0.5)',
            legend=dict(
                orientation="v",
                yanchor="top", y=1,
                xanchor="left", x=1.05,  
                bordercolor='lightgray',
                borderwidth=1
            ),
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray'),
            width=800,
            height=600
        )

        st.plotly_chart(fig_roc, use_container_width=True)

    else:
        st.info("ROC data not available.")



    # --- Feature Importance ---
    feature_importance = joblib.load("feature_importances.pkl")
    st.subheader("Feature Importance per Model")

    # Dropdown to select model
    model_select = st.selectbox(
        "Select Model to View Feature Importance",
        list(feature_importance.keys())
    )

    # Get the Series for the selected model
    feat_series = feature_importance.get(model_select)

    if feat_series is not None and not feat_series.empty:
        # Convert to DataFrame and sort by importance
        feat_df = feat_series.reset_index()
        feat_df.columns = ["Feature", "Importance"]
        feat_df = feat_df.sort_values(by="Importance", ascending=True)

        # Create horizontal bar chart
        fig_feat = px.bar(
            feat_df,
            x="Importance",
            y="Feature",
            orientation='h',
            color="Importance",
            color_continuous_scale='Viridis',
            title=f"Feature Importance for {model_select}"
        )

        st.plotly_chart(fig_feat, use_container_width=True)

    else:
        st.info(f"Feature importance not available for {model_select}")