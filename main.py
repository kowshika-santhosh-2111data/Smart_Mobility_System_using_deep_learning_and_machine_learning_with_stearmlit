import numpy as np
import pandas as pd
import os
import cv2
import importlib
import tensorflow as tf
import joblib
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

#------------------------importing modules--------------------
from data_loader import load_images
from data_preprocessing import preprocess_image_data, clean_text, preprocess_tabular_data
from model import build_cnn, build_transfer_model,train_model, sign_classes
from road_risk import train_tabular_model, risk_mapping
from sentiment_model import train_sentiment_model, sentiment_mapping, get_category
from evaluation import evaluate_model

#-----------------------load_model------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "German_Traffic_sign")

train_csv = os.path.join(DATASET_DIR, "Train.csv")
test_csv = os.path.join(DATASET_DIR, "Test.csv")

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

base_path = DATASET_DIR
    
print("Train CSV:", train_df)
print("Exists:", os.path.exists(train_csv))

print("Test CSV:", test_df)
print("Exists:", os.path.exists(test_csv))

#-----------------------load_images------------------------------
print("Loading images...")
x_train, y_train = load_images(train_df, base_path)
x_test, y_test = load_images(test_df,base_path)

print("Images loaded:", len(x_train), len(y_train))
print("Loading test images...")
x_test, y_test = load_images(test_df, base_path)
print("Test images loaded:", len(x_test), len(y_test))

#-----------------------preprocess_image_data------------------------------
x_train = preprocess_image_data(np.array(x_train))
x_test = preprocess_image_data(np.array(x_test))

y_train = np.array(y_train)
y_test = np.array(y_test)


y_train = to_categorical(y_train, num_classes=43)
y_test = to_categorical(y_test, num_classes=43)

#-----------------------road_risk_model------------------------------
print("Training CNN model...")
road_df = pd.DataFrame({
    "traffic_density": np.random.randint(10, 101, 200),
    "road_condition": np.random.randint(1, 11, 200),
    "visibility": np.random.randint(10, 101, 200),
    "accident_history": np.random.randint(0, 11, 200)
})

print("Training road risk model...")
def calculate_risk(row):
    score = 0

    # Traffic Density
    if row["traffic_density"] > 80:
        score += 2
    elif row["traffic_density"] > 50:
        score += 1

    # Road Condition
    if row["road_condition"] > 6:
        score += 2
    elif row["road_condition"] > 3:
        score += 1

    # Visibility
    if row["visibility"] < 30:
        score += 2
    elif row["visibility"] < 60:
        score += 1

    # Accident History
    if row["accident_history"] > 5:
        score += 2
    elif row["accident_history"] > 2:
        score += 1

    # Final Risk Level
    if score >= 6:
        return 2      # High Risk
    elif score >= 3:
        return 1      # Medium Risk
    else:
        return 0      # Low Risk

road_df["RiskLevel"] = road_df.apply(calculate_risk, axis=1)
#-----------------------sentiment_model------------------------------
print("Training sentiment model...")
complaint_data = pd.DataFrame({
    "text": [
        # Negative (0)
        "Road has potholes",
        "Accident happened here",
        "Road is damaged badly",
        "Heavy traffic jam today",
        "Street lights are not working",
        "Too much congestion in this area",
        "Dangerous curve without sign",
        "Frequent accidents happening here",
        "Road construction causing delay",
        "Bad road conditions everywhere",

        # Neutral (1)
        "Traffic is moderate",
        "Road inspection is underway",
        "Vehicles are moving normally",
        "Road maintenance is scheduled",
        "Traffic signal is operating",
        "Road is open",
        "Traffic flow is average",
        "Normal traffic conditions",
        "Weather is cloudy today",
        "Vehicles are moving at regular speed",

        # Positive (2)
        "Traffic is smooth",
        "No issues on road",
        "Road is clean and well maintained",
        "Very safe road to drive",
        "Signals are working properly",
        "No traffic at all, very smooth",
        "Road is excellent and safe",
        "Clear road, no issues",
        "Road is in excellent condition",
        "Driving is comfortable on this road"
    ],

    "sentiment": [
        # Negative (0)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

        # Neutral (1)
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,

        # Positive (2)
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2
    ]
})
sentiment_mapping = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

#----------------------------------------------
from sklearn.model_selection import train_test_split

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=np.argmax(y_train, axis=1)
)

# Debug: Check labels
print("Training labels:", np.unique(np.argmax(y_train, axis=1)))
print("Testing labels:", np.unique(np.argmax(y_test, axis=1)))
print("Number of training classes:", len(np.unique(np.argmax(y_train, axis=1))))
print("Number of testing classes:", len(np.unique(np.argmax(y_test, axis=1))))

# -------------------- Train / Load Transfer Learning Model --------------------
MODEL_PATH = "traffic_signal_model.keras"

if os.path.exists(MODEL_PATH):
    print("Loading existing Transfer Learning model...")
    model = load_model(MODEL_PATH)

else:
    print("Training MobileNetV2 Transfer Learning model...")

    model = build_cnn()

    model, history = train_model(
        model,
        x_train,
        y_train,
        x_test,
        y_test
    )

    model.save(MODEL_PATH)
    print("Model saved successfully.")

# -------------------- Road Risk Assessment --------------------

# Create RiskLevel
road_df["RiskLevel"] = road_df.apply(calculate_risk, axis=1)

if os.path.exists("road_risk_model.pkl") and os.path.exists("scaler.pkl"):
    print("Loading Road Risk model...")
    risk_model = joblib.load("road_risk_model.pkl")
    scaler = joblib.load("scaler.pkl")

    x_test_tab = road_df.drop("RiskLevel", axis=1)
    y_test_tab = road_df["RiskLevel"]

else:
    print("Training Road Risk model...")

    risk_model, scaler, x_test_tab, y_test_tab = train_tabular_model(road_df)

    joblib.dump(risk_model, "road_risk_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("Road Risk model saved successfully.")

#-----------------------Train or Load Sentiment Model------------------------------
if os.path.exists("sentiment_model.pkl") and os.path.exists("vectorizer.pkl"):
    print("Loading Sentiment model...")
    sentiment_model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
else:
    print("Training Sentiment model...")
    sentiment_model, vectorizer = train_sentiment_model(complaint_data)

    joblib.dump(sentiment_model, "sentiment_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
# -------------------- Evaluate Models --------------------

# Traffic Sign Classification
print("\nEvaluating Traffic Sign Classification Model:")
evaluate_model(model, x_test, y_test)

# Road Risk Assessment
print("\nRoad Risk Model Accuracy:")
print("Accuracy:", risk_model.score(x_test_tab, y_test_tab))

# Sentiment Analysis
print("\nSentiment Model:")
print("Evaluation completed during training.")

def smart_mobility_system(image, road_data, social_media_post):

    # -------------------- Image Preprocessing --------------------
    image = cv2.resize(image, (32,32))      # Use the same size used during training
    image = preprocess_image_data(image)

    image = np.expand_dims(image, axis=0)

    # -------------------- Traffic Sign Prediction --------------------
    sign_pred = model.predict(image, verbose=0)

    class_id = np.argmax(sign_pred, axis=1)[0]
    confidence_score = float(np.max(sign_pred) * 100)

    if confidence_score < 80:
        sign_name = "Uncertain Sign"
    else:
        sign_name = sign_classes[class_id]

    # Debug
    print("Raw Prediction:", sign_pred)

    # -------------------- Road Risk Assessment --------------------
    risk_input = pd.DataFrame([road_data], columns=[
        "traffic_density",
        "road_condition",
        "visibility",
        "accident_history"
    ])

    risk_input_scaled = scaler.transform(risk_input)

    risk_pred = risk_model.predict(risk_input_scaled)[0]
    risk_label = risk_mapping.get(risk_pred, "Unknown")

    # -------------------- Sentiment Analysis --------------------
    cleaned_text = clean_text(social_media_post)

    text_vector = vectorizer.transform([cleaned_text])

    sent_pred = sentiment_model.predict(text_vector)[0]
    sent_label = sentiment_mapping.get(sent_pred, "Unknown")

    # -------------------- Complaint Category --------------------
    category = get_category(cleaned_text)

    # -------------------- Final Output --------------------
    result = {
        "Traffic Sign": sign_name,
        "Confidence": round(confidence_score, 2),
        "Road Risk": risk_label,
        "Sentiment": sent_label,
        "Complaint Category": category
    }

    print("\n========== SMART MOBILITY SYSTEM ==========")
    print(f"Traffic Sign       : {result['Traffic Sign']}")
    print(f"Confidence         : {result['Confidence']}%")
    print(f"Road Risk          : {result['Road Risk']}")
    print(f"Sentiment          : {result['Sentiment']}")
    print(f"Complaint Category : {result['Complaint Category']}")

    return result

# -------------------- Test Call --------------------

# Read first test image
first_image = test_df.iloc[0]["Path"]
image_path = os.path.join(base_path, first_image)

print("Testing image:", image_path)
print("Exists:", os.path.exists(image_path))

# Load image
img = cv2.imread(image_path)

if img is None:
    raise ValueError(f"Image not loaded: {image_path}")

# Convert BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize (must match training size)
img = cv2.resize(img, (64, 64))

# Test inputs
test_image = img

test_road_data = [
    70,   # traffic_density
    5,    # road_condition
    45,   # visibility
    3     # accident_history
]

test_social_media_post = (
    "There is a huge pothole on Main Street causing damage to cars!"
)

# Run the integrated system
result = smart_mobility_system(
    test_image,
    test_road_data,
    test_social_media_post
)

print(result)