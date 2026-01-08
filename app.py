import streamlit as st
import numpy as np
import pandas as pd
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Airline Passenger Satisfaction",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Airline Passenger Satisfaction Predictor")
st.markdown(
    """
    This application predicts **airline passenger satisfaction** using a 
    machine learning classification model trained on service quality and travel data.
    Adjust the parameters below and get instant predictions.
    """
)

# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

# --------------------------------------------------
# Sidebar – User Inputs
# --------------------------------------------------
st.sidebar.header("🧾 Passenger & Flight Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
customer_type = st.sidebar.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
travel_type = st.sidebar.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
travel_class = st.sidebar.selectbox("Class", ["Eco", "Eco Plus", "Business"])

age = st.sidebar.slider("Age", 5, 85, 35)
flight_distance = st.sidebar.slider("Flight Distance (km)", 100, 5000, 1200)

st.sidebar.subheader("⭐ Service Ratings (0 = Worst, 5 = Best)")

wifi = st.sidebar.slider("Inflight WiFi Service", 0, 5, 3)
online_booking = st.sidebar.slider("Ease of Online Booking", 0, 5, 3)
gate_location = st.sidebar.slider("Gate Location", 0, 5, 3)
food = st.sidebar.slider("Food & Drink", 0, 5, 3)
seat_comfort = st.sidebar.slider("Seat Comfort", 0, 5, 3)
entertainment = st.sidebar.slider("Inflight Entertainment", 0, 5, 3)
onboard_service = st.sidebar.slider("On-board Service", 0, 5, 3)
legroom = st.sidebar.slider("Leg Room Service", 0, 5, 3)
baggage = st.sidebar.slider("Baggage Handling", 0, 5, 3)
checkin = st.sidebar.slider("Check-in Service", 0, 5, 3)

arrival_delay = st.sidebar.slider("Arrival Delay (minutes)", 0, 300, 10)

# --------------------------------------------------
# Prepare Input Data
# --------------------------------------------------
input_data = pd.DataFrame([{
    "Gender": gender,
    "Customer Type": customer_type,
    "Type of Travel": travel_type,
    "Class": travel_class,
    "Age": age,
    "Flight Distance": flight_distance,
    "Inflight wifi service": wifi,
    "Ease of Online booking": online_booking,
    "Gate location": gate_location,
    "Food and drink": food,
    "Seat comfort": seat_comfort,
    "Inflight entertainment": entertainment,
    "On-board service": onboard_service,
    "Leg room service": legroom,
    "Baggage handling": baggage,
    "Checkin service": checkin,
    "Arrival Delay in Minutes": arrival_delay
}])

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Predict Satisfaction"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    confidence = np.max(probability) * 100

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.success(f"✅ Passenger is **SATISFIED**")
    else:
        st.error(f"❌ Passenger is **NEUTRAL or DISSATISFIED**")

    st.metric("Prediction Confidence", f"{confidence:.2f}%")

# --------------------------------------------------
# Model Insights
# --------------------------------------------------
with st.expander("📈 Model Information"):
    st.markdown(
        """
        **Model Used:** Best Performing Classification Model  
        **Evaluation Metrics:**
        - Accuracy
        - Precision
        - Recall
        - F1-Score  

        This model was selected after comparing multiple classifiers 
        such as Logistic Regression, Random Forest, and Gradient Boosting.
        """
    )

st.markdown("---")
st.markdown(
    "<center>🚀 Built for Competition | Machine Learning + Streamlit</center>",
    unsafe_allow_html=True
)
