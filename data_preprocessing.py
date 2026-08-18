import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------------------- Image Data Preprocessing -------------------------------

def preprocess_image_data(x):
    """
    Normalize image pixel values to the range [0, 1].
    """
    x = np.asarray(x, dtype=np.float32)
    x /= 255.0
    return x


# ------------------------------- Tabular Data Preprocessing -------------------------------

def preprocess_tabular_data(df, target_column):
    """
    Fill missing numeric values, separate features and target,
    and standardize numeric features.
    """
    df = df.copy()

    # Fill missing values in numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Features and target
    x = df.drop(columns=[target_column])
    y = df[target_column]

    # Standardize only numeric columns
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    return x_scaled, y, scaler


# ------------------------------- Text Cleaning -------------------------------

def clean_text(text):
    """
    Convert text to lowercase and remove punctuation.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ------------------------------- Tokenization -------------------------------

def tokenize_text(text):
    """
    Split text into tokens.
    """
    return text.split()


# ------------------------------- Stop Word Removal -------------------------------

def remove_stop_words(tokens, stop_words):
    """
    Remove stop words from token list.
    """
    return [word for word in tokens if word not in stop_words]


# ------------------------------- TF-IDF Vectorization -------------------------------

def tfidf_vectorize(corpus):
    """
    Convert text corpus into TF-IDF features.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    return tfidf_matrix, vectorizer