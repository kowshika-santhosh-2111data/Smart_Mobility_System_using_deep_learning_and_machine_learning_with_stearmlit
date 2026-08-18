import numpy as np
import pandas as pd
import joblib
import cv2
import streamlit as st

from PIL import Image
from tensorflow.keras.models import load_model

from data_preprocessing import clean_text, preprocess_image_data
from model import sign_classes
from road_risk import risk_mapping
from sentiment_model import sentiment_mapping, get_category

# ---------------- Load Models ----------------
@st.cache_resource
def load_models():

    cnn_model = load_model("traffic_signal_model.keras")
    risk_model = joblib.load("road_risk_model.pkl")
    scaler = joblib.load("scaler.pkl")
    sentiment_model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

    return cnn_model, risk_model, scaler, sentiment_model, vectorizer


cnn_model, loaded_risk_model, loaded_scaler, \
loaded_sentiment_model, loaded_vectorizer = load_models()

def smart_mobility_system(image, road_data, social_media_post):

    # Image
    image = preprocess_image_data(np.array([image]))

    # Traffic Sign
    sign_pred = cnn_model.predict(image, verbose=0)
    pred = sign_pred[0]

    class_id = np.argmax(pred)
    confidence_score = round(float(np.max(pred) * 100), 2)

    if confidence_score < 50:
        sign_name = "Uncertain Sign"
    else:
        sign_name = sign_classes[class_id]

    # Road Risk
    risk_input = pd.DataFrame(
        [road_data],
        columns=[
            "traffic_density",
            "road_condition",
            "visibility",
            "accident_history"
        ]
    )

    risk_input_scaled = loaded_scaler.transform(risk_input)
    risk_pred = loaded_risk_model.predict(risk_input_scaled)[0]
    risk_label = risk_mapping.get(int(risk_pred), "Unknown")

    # Sentiment
    cleaned_text = clean_text(social_media_post)

    text_vector = loaded_vectorizer.transform([cleaned_text])

    sent_pred = loaded_sentiment_model.predict(text_vector)[0]

    sent_label = sentiment_mapping.get(int(sent_pred), "Unknown")

    # Category
    category = get_category(cleaned_text)

    return (
        sign_name,
        confidence_score,
        risk_label,
        sent_label,
        category
    )
#-----------Streamlit App----------------
st.title("Smart Mobility System")

uploaded_file = st.file_uploader(
    "Upload a traffic sign image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

st.header("Your Location")
city = st.text_input("Enter your location (leave blank for current location)")

st.subheader("Road Risk Assessment")

traffic_density = st.number_input("Traffic Density", 0, 100)
road_condition = st.number_input("Road Condition", 1, 10)
visibility = st.number_input("Visibility", 0, 100)
accident_history = st.number_input("Accident History", 0, 10)

st.subheader("Complaint Sentiment Analysis")
text = st.text_area("Enter a social media post or complaint")
st.subheader("Run Smart Mobility System")

if st.button("Analyze"):

    if uploaded_file is None:
        st.warning("Please upload a traffic sign image.")

    elif text.strip() == "":
        st.warning("Please enter a complaint.")

    else:

        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((32, 32))
        image = np.array(image)

        road_data = [
            float(traffic_density),
            float(road_condition),
            float(visibility),
            float(accident_history)
        ]

        sign_name, confidence, risk_label, sent_label, category = smart_mobility_system(
            image,
            road_data,
            text
        )

        # Display results here
        st.markdown("## Results")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image")

        with col2:
            st.success(f"🚦 Traffic Sign: {sign_name} ({confidence:.2f}%)")

            if risk_label == "High Risk":
                st.error(f"🚧 Road Risk: {risk_label}")
            elif risk_label == "Medium Risk":
                st.warning(f"🚧 Road Risk: {risk_label}")
            else:
                st.success(f"🚧 Road Risk: {risk_label}")

            if sent_label == "Negative":
                st.warning(f"💬 Sentiment: {sent_label}")
            else:
                st.success(f"💬 Sentiment: {sent_label}")

            st.info(f"Complaint Category: {category}")        