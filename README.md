# AI-Powered Smart Mobility System Using Deep Learning, Machine Learning and NLP
## Overview
      The Smart Mobility System is an AI-powered web application that combines Computer Vision, Machine Learning and 
      Natural Language Processing to enhance road safety through intelligent traffic sign recognition, road risk prediction,
      and complaint sentiment analysis. It integrates three intelligent modules into a single Streamlit web application.

* Traffic Sign Classification using Convolutional Neural Network (CNN)
* Road Risk Assessment using Random Forest Classifier
* Complaint Sentiment Analysis using TF-IDF and Logistic Regression
Users can upload a traffic sign image, enter road condition details and submit a traffic related complaint.
---
## Technologies Used
### Programming Language
* Python
### Deep Learning                  
* TensorFlow                
* Keras
### Machine Learning                  
* Scikit-learn              
* Random Forest Classifier
* Logistic Regression
### NLP
* TF - IDF Vectorizer
* Text Preprocessing
## Libraries used
* Numpy
* Pandas
* OpenCV
* PIL (Pillow)
* Matplotlib
* Seaborn
* Joblib
  
## Dataset
### Traffic Sign Dataset
* German Traffic Sign Recognition Dataset (GTSRB)
* 43 Traffic Sign Classes
### Road Risk Dataset 
A synthetic dataset generated for training using:
* Traffic Density
* Road Condition
* Visibility
* Accident History
### Sentiment Dataset
Custom traffic complaint dataset containing:
* Positive Complaints
* Neutral Complaints
* Negative Complaints
## Project Structure
```text
Smart_Mobility_System/
│
├── dataset/
│   ├── Train/
│   ├── Test/
│   ├── Train.csv
│   └── Test.csv
│
├── outputs/
│   ├── sample.png
│   ├── sample_images.png
│   ├── Class_distribution.png
│   ├── Pedestrian_crossing_sign.png
│   ├── Stop_sign.png
|   ├── Yield_sign.png
│   ├── no_entry.png
│   └── speed_limit_50kmhr.png
│
├── app.py
├── main.py
├── data_loader.py
├── data_preprocessing.py
├── eda.py
├── model.py
├── road_risk.py
├── sentiment_model.py
├── evaluation.py
│
├── traffic_signal_model.keras
├── road_risk_model.pkl
├── sentiment_model.pkl
├── scaler.pkl
├── vectorizer.pkl
│
├── requirements.txt
├── runtime.txt
└── README.md
```

## Features
* Traffic Sign Recognition from uploaded images
* Road Risk Prediction (Low, Medium, High)
* Complaint Sentiment Analysis (Positive, Neutral, Negative)
* Complaint Category Detection
* Interactive Streamlit Web Application
* Uses Deep Learning, Machine Learning, and NLP in one application

## Modules
## 1.Traffic Sign Classification
* Image Preprocessing
* CNN - based traffic sign recognition
* Supports 43 traffic sign classes
* Displays predicted sign with confidence score
## 2. Road Risk Assessment
Predicts road risk using Traffic Density, Road Condition, Visibility, Accident History
## Output
Low Risk, Medium Risk, High Risk
## 3.Complaint Sentiment Analysis
Processes traffic related complaints by
* Cleaning text
* Tokenization
* Stop-Word removal
* TF-IDF Vectorization
* Logistic Regression Classification
## Outputs
* Positive
* Neutral
* Negative
It also identifies complaint categories such as:
* Road Condition
* Traffic Issue
* Signal Issue
* Accident
* General Complaint
## Installation
```
Clone the repository
git clone https://github.com/yourusername/Smart-Mobility-System.git
```
Move into the project directory
```
cd Smart-Mobility-System
```
Install dependencies
```
pip install -r requirements.txt
```
## Run the application 
```
streamlit run app.py
```
## How to Use
1. Upload a traffic sign image
2. Enter road information
     * Traffic Density
     * Road Condition
     * Visibility
     * Accident History
3. Enter a traffic related complaint
4. Click Analyse
5. View:
     * Predicted Traffic Sign
     * Confidence Score
     * Road Risk Level
     * Sentiment
     * Complaint Category
## Future Enhancements
* Real-time traffic sign detection using a webcam
* GPS-based location integration
* Live traffic and weather data integration
* Cloud deployment
* Mobile application support
* Advanced deep learning models for higher accuracy
## Author
**Kowshika Santhosh**
GitHub: https://github.com/kowshika-santhosh-2111data
## License
This project is developed for educational and academic purposes.
