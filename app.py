# app.py

import streamlit as st
import joblib
import numpy as np

# Load Model and Scaler

model = joblib.load("airline_pipeline.pkl")
scaler = joblib.load("scaler.pkl")

st.title("✈️ Airline Passenger Satisfaction Prediction")

# User Inputs

online_boarding = st.slider("Online Boarding", 0, 5, 3)

customer_type = st.selectbox(
    "Customer Type",
    ["Loyal Customer", "Disloyal Customer"]
)

type_of_travel = st.selectbox(
    "Type of Travel",
    ["Business travel", "Personal Travel"]
)

travel_class = st.selectbox(
    "Class",
    ["Business", "Eco", "Eco Plus"]
)

ease_of_online_booking = st.slider(
    "Ease of Online Booking",
    0,
    5,
    3
)

overall_service_rating = st.slider(
    "Overall Service Rating",
    0.0,
    5.0,
    3.0
)

inflight_entertainment = st.slider(
    "In-flight Entertainment",
    0,
    5,
    3
)

leg_room_service = st.slider(
    "Leg Room Service",
    0,
    5,
    3
)

seat_comfort = st.slider(
    "Seat Comfort",
    0,
    5,
    3
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=30
)

flight_distance = st.number_input(
    "Flight Distance",
    min_value=0,
    value=1000
)

cleanliness = st.slider(
    "Cleanliness",
    0,
    5,
    3
)

# Encoding

customer_type = 1 if customer_type == "Loyal Customer" else 0

type_of_travel = 1 if type_of_travel == "Business travel" else 0

class_mapping = {
    "Business": 0,
    "Eco": 1,
    "Eco Plus": 2
}

travel_class = class_mapping[travel_class]

# Predict Button

if st.button("Predict Satisfaction"):

    data = np.array([[
        online_boarding,
        customer_type,
        type_of_travel,
        travel_class,
        ease_of_online_booking,
        overall_service_rating,
        inflight_entertainment,
        leg_room_service,
        seat_comfort,
        age,
        flight_distance,
        cleanliness
    ]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("😊 Passenger is Satisfied")
    else:
        st.error("😞 Passenger is Neutral or Dissatisfied")