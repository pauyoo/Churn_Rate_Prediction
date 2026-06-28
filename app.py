import streamlit as st
import pandas as pd
import numpy as np
import joblib

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Insurance Churn Prediction",
    page_icon="📋",
    layout="centered"
)

# ASSET LOADING (CACHED FOR SPEED)
@st.cache_resource
def load_model_assets():
    """
    Loads both models and the scaler asset. 
    Wrap them in st.cache_resource so they stay in memory and don't reload on every click.
    """
    try:
        # Load your trained model files (Make sure these names match your saved filenames exactly)
        rf_model = joblib.load('random_forest_model.pkl')
        lr_model = joblib.load('logistic_regression_model.pkl') 
        scaler = joblib.load('scaler.pkl')
        return rf_model, lr_model, scaler
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.info("💡 Ensure 'random_forest_model.pkl', 'logistic_regression_model.pkl', and 'scaler.pkl' are in this exact folder.")
        return None, None, None

rf_model, lr_model, scaler = load_model_assets()

# HEADER SECTION
st.title("📋 Insurance Churn Prediction")
st.write("Input a policyholder's details below.")

# SIDEBAR / MODEL SELECTION CONFIGURATION
st.sidebar.header("🛠️ Model Configuration")
# Allow the user to actively select which mathematical model to process the data with
selected_model_name = st.sidebar.selectbox(
    "Choose Classification Engine:",
    ["Random Forest Classifier", "Logistic Regression"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Note on Pipelines:** Regardless of the chosen engine, your inputs are instantly processed "
    "by your saved `StandardScaler` to match training dimensions before calculations begin."
)

# USER INPUT INTERFACE (FORM LAYOUT)
# Using st.form keeps the app from refreshing while the user is mid-typing.
with st.form("input_form"):
    st.subheader("👤 Demographics & Behavioral Profiling")
    
    # Split the inputs into clean columns for visual appeal
    col1, col2, col3 = st.columns(3)
    
    with col1:
        credit_score = st.number_input("Credit Score", min_value=0, max_value=900, value=650, step=1)
        age = st.number_input("Age (Years)", min_value=15, max_value=100, value=35, step=1)
        tenure = st.number_input("Tenure (Years with Firm)", min_value=0, max_value=20, value=3, step=1)

    with col2:
        balance = st.number_input("Account Balance ($)", min_value=0.0, value=25000.0, step=500.0)
        estimated_salary = st.number_input("Estimated Annual Salary ($)", min_value=0.0, value=65000.0, step=500.0)
        num_products = st.number_input("Number of Products Held", min_value=1, max_value=5, value=1, step=1)

    with col3:
        gender = st.selectbox("Gender", ["Male", "Female"])
        has_credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active = st.selectbox("Is Active Member?", ["Yes", "No"])

    st.subheader("🌍 Geographic Region")
    geography = st.selectbox("Select Core Location:", ["France", "Germany", "Spain"])
    
    # Submit button for the form
    submit_button = st.form_submit_button(label="🚀 Analyze Risk Metrics")

# PREPROCESSING & PREDICTION LOGIC
if submit_button:
    if scaler is None or rf_model is None or lr_model is None:
        st.error("Cannot compute prediction. Asset loading failed.")
    else:
        # Map Categorical / Flag Selectbox text answers back into 0 and 1 numerical flags
        gender_encoded = 1 if gender == "Male" else 0
        has_card_encoded = 1 if has_credit_card == "Yes" else 0
        active_encoded = 1 if is_active == "Yes" else 0
        
        # Re-create the One-Hot Encoded flags for your Geography structure columns
        # (If France is selected, both Germany and Spain flags are 0)
        geo_germany = 1 if geography == "Germany" else 0
        geo_spain = 1 if geography == "Spain" else 0

        # Align inputs into a Pandas DataFrame using your EXACT column schema order
        raw_features = pd.DataFrame([{
            'CreditScore': credit_score,
            'Gender': gender_encoded,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_products,
            'HasCrCard': has_card_encoded,
            'IsActiveMember': active_encoded,
            'EstimatedSalary': estimated_salary,
            'Geography_Germany': geo_germany,
            'Geography_Spain': geo_spain
        }])

        # Step 1 of the pipeline: Standardize the raw features using your scaler
        scaled_features = scaler.transform(raw_features)

        # Route the data to the user's selected engine
        if selected_model_name == "Random Forest Classifier":
            active_engine = rf_model
        else:
            active_engine = lr_model

        # pipeline: Execute mathematical evaluation
        prediction = active_engine.predict(scaled_features)[0]
        # Get the probability score of Class 1 (Risk/Churn)
        probability_score = active_engine.predict_proba(scaled_features)[0][1]

        # DISPLAY PERFORMANCE ANALYSIS RESULTS
        
        st.markdown("---")
        st.subheader(f"📊 Assessment Output via *{selected_model_name}*")
        
        # Display results cleanly based on risk state threshold
        if prediction == 1:
            st.error(f"⚠️ **HIGH RISK OF POLICY CANCELLATION / CHURN**")
            st.progress(float(probability_score))
            st.write(f"The structural algorithm calculated a **{probability_score * 100:.2f}%** operational risk probability score.")
        else:
            st.success(f"✅ **Result Status: LOW RISK / RENEWAL EXPECTED**")
            st.progress(float(probability_score))
            st.write(f"The structural algorithm calculated a **{probability_score * 100:.2f}%** Churn prediction.")

        st.caption("ℹ️ *This metric matrix serves as visual assistance software and is optimized for automated data analytics purposes only.*")