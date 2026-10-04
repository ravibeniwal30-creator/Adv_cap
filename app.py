import streamlit as st
import joblib
import numpy as np

# Set page title
st.set_page_config(page_title="Advertising Sales Predictor", layout="centered")

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load('linear.sav')

model = load_model()

# App title and description
st.title("📈 Advertising Sales Predictor")
st.write("This app uses a trained Linear Regression model to predict sales based on advertising budgets.")

# Input fields for advertising channels
st.subheader("Enter Advertising Budgets ($):")

tv_budget = st.number_input("TV Advertising Budget", min_value=0.0, value=120.0, step=1.0)
radio_budget = st.number_input("Radio Advertising Budget", min_value=0.0, value=10.0, step=1.0)
newspaper_budget = st.number_input("Newspaper Advertising Budget", min_value=0.0, value=5.0, step=1.0)

# Predict button
if st.button("Predict Sales"):
    # Make prediction
    features = np.array([[tv_budget, radio_budget, newspaper_budget]])
    prediction = model.predict(features)[0]
    
    # Display result
    st.success(f"🎯 Predicted Sales: **${prediction:.2f}k**")
