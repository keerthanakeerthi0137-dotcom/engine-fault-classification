import streamlit as st
import joblib
import json
import numpy as np

# Load model using joblib (safer than pickle)
model = joblib.load("engine_fault_pipeline.pkl")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

st.title("Engine Fault Classification")

# Example input fields (replace with your real features)
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

if st.button("Predict"):
    X = np.array([[feature1, feature2, feature3]])
    prediction = model.predict(X)[0]
    st.write("Prediction:", class_names[str(prediction)])
