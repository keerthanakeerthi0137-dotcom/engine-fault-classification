import streamlit as st
import pickle
import json

# Load model (pickle)
with open("engine_fault_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

st.title("Engine Fault Classification")

# Example input fields (replace with your actual features)
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")

if st.button("Predict"):
    X = [[feature1, feature2, feature3]]
    prediction = model.predict(X)[0]
    st.write("Prediction:", class_names[str(prediction)])
