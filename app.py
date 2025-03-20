import streamlit as st
import numpy as np
import pickle

# Load the trained model from pickle file
with open("xgbr_model.pkl", "rb") as f:
    xgbr_model = pickle.load(f)

# print(type(res_model))  # Debugging: cek tipe objek
# print(res_model)        # Debugging: cek isi objek

st.title("Porter Delivery Time Prediction")

# Input fields for user to provide the required features
store_primary_category = st.number_input("primary category store", min_value=18, max_value=70, value=30)
order_protocol = st.number_input("order protocol", min_value=1.0, max_value=5.0, value=5.0, step=0.1)
subtotal = st.number_input("subtotal", min_value=1, max_value=70, value=1)
min_item_price = st.number_input("min item price", min_value=1, max_value=5000, value=1)
max_item_price = st.number_input("max item price", min_value=1, max_value=5000, value=1)
total_onshift_partners = st.number_input("total onshift partners", min_value=1.0, max_value=500.0, value=1.0) 
total_busy_partners = st.number_input("total busy partners", min_value=1.0, max_value=500.0, value=1.0)
total_outstanding_orders = st.number_input("total outstanding orders", min_value=1.0, max_value=500.0, value=1.0)

# Predict button
if st.button("Predict"):
    features = np.array([[store_primary_category,order_protocol,subtotal,min_item_price,max_item_price,total_onshift_partners,total_busy_partners,total_outstanding_orders]])
    predicted_time = xgbr_model.predict(features)
    st.write(f"Predicted Delivery Time: {predicted_time[0]:.2f} Minutes")