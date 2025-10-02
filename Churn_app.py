import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from model_prediction import predict_all_models, data, model_metrics
from Model_Drift import model_drift_monitoring, csi_feature_drift


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
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict Customer Churn",
    "📊 Data Visualizations & Explorer",
    "🤖 Model Comparison & Performance",
    "📈 Model Monitoring & Drift Detection"
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


# --- Tab 4: Model Monitoring & Drift Detection ---
with tab4:
    st.header("📈 Model Monitoring & Drift Detection")

    # Load data
    df = data()
    all_metrics, roc = model_metrics()

    
    # --- Model Drift Monitoring ---
    st.subheader("🤖 Model Drift Monitoring")

    # Run model drift monitoring
    if "drift_results" not in st.session_state:
        st.session_state.drift_results = model_drift_monitoring()
    drift_results = st.session_state.drift_results
    

    # Display results in a table
    drift_df = pd.DataFrame.from_dict(drift_results, orient='index')
    drift_df = drift_df.reset_index().rename(columns={'index': 'Model'})

    # Add drift level based on PSI
    def get_drift_level(psi):
        if psi < 0.1:
            return "Low"
        elif psi < 0.25:
            return "Moderate"
        else:
            return "High"

    drift_df['Drift Level'] = drift_df['Prediction Drift (PSI)'].apply(get_drift_level)

    st.dataframe(drift_df.style.format({
        'Refernece KS': '{:.3f}',
        'Current KS': '{:.3f}',
        'Prediction Drift (PSI)': '{:.4f}'
    }).apply(lambda x: ['background-color: lightgreen' if x['Prediction Drift (PSI)'] < 0.1
                       else 'background-color: lightblue' if x['Prediction Drift (PSI)'] < 0.25
                       else 'background-color: lightcoral' for _ in x], axis=1),
             use_container_width=True)

    # --- Feature Drift Detection (CSI) ---
    st.subheader("📊 Feature Drift Detection (CSI)")

    # Run CSI feature drift
    csi_results = csi_feature_drift()

    # Create DataFrame for display
    csi_df = pd.DataFrame({
        'Feature': list(csi_results.keys()),
        'CSI Value': list(csi_results.values())
    })

    # Add drift level
    def get_csi_drift_level(csi):
        if csi < 0.1:
            return "Stable"
        elif csi < 0.25:
            return "Moderate Shift"
        else:
            return "Major Shift"

    csi_df['Drift Level'] = csi_df['CSI Value'].apply(get_csi_drift_level)

    csi_df_styled = csi_df.style.format({'CSI Value': '{:.4f}'})

    csi_df_styled = csi_df_styled.map(
        lambda val: (
            'background-color: lightgreen' if val == "Stable"
            else 'background-color: lightblue' if val == "Moderate Shift"
            else 'background-color: lightcoral'
        ),
        subset=['Drift Level']  # only style Drift Level column
    )

    st.dataframe(csi_df_styled, use_container_width=True)

    # --- Drift Visualizations ---
    st.subheader("📈 Drift Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        # Feature Drift Bar Chart
        fig_feature_drift = px.bar(
            csi_df,
            x='Feature',
            y='CSI Value',
            color='Drift Level',
            color_discrete_map={'Stable': 'green', 'Moderate Shift': 'orange', 'Major Shift': 'red'},
            title='Feature Drift (CSI Values)',
            labels={'CSI Value': 'CSI Value'}
        )
        fig_feature_drift.add_hline(y=0.1, line_dash="dash", line_color="green", annotation_text="Low Threshold")
        fig_feature_drift.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="High Threshold")
        st.plotly_chart(fig_feature_drift, use_container_width=True)

    with col2:
        # Model Prediction Drift Bar Chart
        fig_model_drift = px.bar(
            drift_df,
            x='Model',
            y='Prediction Drift (PSI)',
            color='Drift Level',
            color_discrete_map={'Low': 'green', 'Moderate': 'orange', 'High': 'red'},
            title='Model Prediction Drift (PSI Values)',
            labels={'Prediction Drift (PSI)': 'PSI Value'}
        )
        fig_model_drift.add_hline(y=0.1, line_dash="dash", line_color="green", annotation_text="Low Threshold")
        fig_model_drift.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="High Threshold")
        st.plotly_chart(fig_model_drift, use_container_width=True)

    # KS Statistics Comparison
    st.subheader("KS Statistics Comparison")
    ks_df = drift_df[['Model', 'Refernece KS', 'Current KS']].melt(id_vars='Model', var_name='Dataset', value_name='KS Value')
    fig_ks = px.bar(
        ks_df,
        x='Model',
        y='KS Value',
        color='Dataset',
        barmode='group',
        color_discrete_map={'Refernece KS': '#42A5F5', 'Current KS': '#FF7926'},
        title='KS Statistics: Reference vs Current Data'
    )
    st.plotly_chart(fig_ks, use_container_width=True)

    # --- Drift Detection Summary Dashboard ---
    st.subheader("🎯 Drift Detection Summary")

    # Calculate summary metrics
    total_features = len(csi_df)
    stable_features = len(csi_df[csi_df['CSI Value'] < 0.1])
    moderate_drift_features = len(csi_df[(csi_df['CSI Value'] >= 0.1) & (csi_df['CSI Value'] < 0.25)])
    major_drift_features = len(csi_df[csi_df['CSI Value'] >= 0.25])

    total_models = len(drift_df)
    low_drift_models = len(drift_df[drift_df['Prediction Drift (PSI)'] < 0.1])
    moderate_drift_models = len(drift_df[(drift_df['Prediction Drift (PSI)'] >= 0.1) & (drift_df['Prediction Drift (PSI)'] < 0.25)])
    high_drift_models = len(drift_df[drift_df['Prediction Drift (PSI)'] >= 0.25])

    # Summary cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        stability_score = (stable_features / total_features) * 100
        st.metric("Feature Stability", f"{stability_score:.1f}%",
                 help=f"{stable_features}/{total_features} features are stable")

    with col2:
        model_stability_score = (low_drift_models / total_models) * 100
        st.metric("Model Stability", f"{model_stability_score:.1f}%",
                 help=f"{low_drift_models}/{total_models} models have low drift")

    with col3:
        critical_alerts = major_drift_features + high_drift_models
        st.metric("Critical Alerts", critical_alerts,
                 help="Features with major drift + models with high prediction drift")

    with col4:
        overall_health = (stability_score + model_stability_score) / 2
        health_color = "🟢" if overall_health > 80 else "🟡" if overall_health > 60 else "🔴"
        st.metric("Overall Health", f"{health_color} {overall_health:.1f}%")

    # Quick status indicators
    st.subheader("Quick Status Overview")

    status_col1, status_col2 = st.columns(2)

    with status_col1:
        st.write("**Feature Drift Status:**")
        if major_drift_features > 0:
            st.error(f"🚨 {major_drift_features} features with major drift")
        if moderate_drift_features > 0:
            st.warning(f"⚠️ {moderate_drift_features} features with moderate drift")
        if stable_features > 0:
            st.success(f"✅ {stable_features} features stable")

    with status_col2:
        st.write("**Model Drift Status:**")
        if high_drift_models > 0:
            st.error(f"🚨 {high_drift_models} models with high prediction drift")
        if moderate_drift_models > 0:
            st.warning(f"⚠️ {moderate_drift_models} models with moderate prediction drift")
        if low_drift_models > 0:
            st.success(f"✅ {low_drift_models} models with low prediction drift")

    

    # --- Alerts & Thresholds ---
    st.subheader("🚨 Monitoring Alerts")

    alerts = []

    # Check for high CSI values (feature drift)
    high_csi_features = csi_df[csi_df['CSI Value'] > 0.25]
    if not high_csi_features.empty:
        alerts.append(f"⚠️ Major feature drift detected in: {', '.join(high_csi_features['Feature'].tolist())}")

    # Check for high model prediction drift
    high_drift_models = drift_df[drift_df['Prediction Drift (PSI)'] > 0.25]
    if not high_drift_models.empty:
        alerts.append(f"⚠️ High prediction drift in models: {', '.join(high_drift_models['Model'].tolist())}")

    # Check for low accuracy (simulated threshold)
    for model, metrics in all_metrics.items():
        if metrics['Accuracy'] < 0.8:
            alerts.append(f"⚠️ {model} accuracy below threshold (0.8)")

    if alerts:
        for alert in alerts:
            st.error(alert)
    else:
        st.success("✅ No critical alerts detected")


    # --- Business Metrics & Drift Impact ---
    st.subheader("💼 Business Impact Metrics & Drift Impact")

    # Calculate some business metrics
    churn_rate = df['Exited'].mean() * 100
    avg_balance_churned = df[df['Exited'] == 1]['Balance'].mean()
    avg_balance_retained = df[df['Exited'] == 0]['Balance'].mean()

    # Calculate potential revenue impact from drift
    # Assuming models with high drift might lead to poor predictions
    high_drift_models = drift_df[drift_df['Prediction Drift (PSI)'] > 0.25]
    drift_impact_factor = len(high_drift_models) * 0.1  # 10% impact per high-drift model
    adjusted_churn_rate = churn_rate * (1 + drift_impact_factor)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Overall Churn Rate", f"{churn_rate:.1f}%")

    with col2:
        st.metric("Projected Churn Rate (with drift)", f"{adjusted_churn_rate:.1f}%",
                 delta=f"+{adjusted_churn_rate - churn_rate:.1f}%")

    with col3:
        st.metric("Avg Balance (Churned)", f"${avg_balance_churned:,.0f}")

    with col4:
        st.metric("Avg Balance (Retained)", f"${avg_balance_retained:,.0f}")

    # Revenue impact calculation
    st.subheader("💰 Revenue Impact Analysis")

    # Estimate potential lost revenue due to drift
    total_customers = len(df)
    churned_customers = int(total_customers * churn_rate / 100)
    potential_lost_revenue = churned_customers * avg_balance_churned

    # Impact from drift
    additional_churn_due_to_drift = int(total_customers * (adjusted_churn_rate - churn_rate) / 100)
    drift_lost_revenue = additional_churn_due_to_drift * avg_balance_churned

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Current Annual Lost Revenue", f"${potential_lost_revenue:,.0f}",
                 help="Estimated revenue lost from current churn rate")

    with col2:
        st.metric("Additional Lost Revenue (Drift Impact)", f"${drift_lost_revenue:,.0f}",
                 help="Additional revenue potentially lost due to model/feature drift",
                 delta=f"+{drift_lost_revenue:,.0f}")

    # Churn rate by segments
    st.subheader("Churn Rate by Segments")

    segment_churn = df.groupby('Geography')['Exited'].agg(['count', 'mean']).round(3)
    segment_churn.columns = ['Count', 'Churn Rate']
    segment_churn['Churn Rate %'] = (segment_churn['Churn Rate'] * 100).round(1)

    st.dataframe(segment_churn[['Count', 'Churn Rate %']], use_container_width=True)

    # Drift Impact by Geography (simulated)
    st.subheader("Drift Impact by Geography")

    # Simulate different drift impacts by geography
    geography_drift = pd.DataFrame({
        'Geography': ['France', 'Germany', 'Spain'],
        'Current Churn %': segment_churn['Churn Rate %'].values,
        'Drift Impact %': np.random.uniform(0.5, 2.0, 3).round(1)
    })
    geography_drift['Projected Churn %'] = geography_drift['Current Churn %'] + geography_drift['Drift Impact %']

    st.dataframe(geography_drift.style.format({
        'Current Churn %': '{:.1f}',
        'Drift Impact %': '{:.1f}',
        'Projected Churn %': '{:.1f}'
    }), use_container_width=True)
