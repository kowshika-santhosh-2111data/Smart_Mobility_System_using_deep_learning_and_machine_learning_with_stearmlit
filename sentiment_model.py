from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_preprocessing import clean_text, tokenize_text, remove_stop_words


def train_sentiment_model(df, text_column=None, target_column="sentiment"):
    """
    Train a sentiment analysis model using TF-IDF and Logistic Regression.
    """

    df = df.copy()

    # Detect text column automatically if not provided
    if text_column is None:
        text_column = df.columns[0]

    # Clean text
    df["CleanedText"] = df[text_column].astype(str).apply(clean_text)

    # Tokenize
    df["Tokens"] = df["CleanedText"].apply(tokenize_text)

    # Stop words
    stop_words = {
        "the", "is", "in", "and", "to", "of",
        "a", "an", "for", "on", "at", "with"
    }

    df["FilteredTokens"] = df["Tokens"].apply(
        lambda tokens: remove_stop_words(tokens, stop_words)
    )

    # Convert back to sentence
    df["ProcessedText"] = df["FilteredTokens"].apply(" ".join)

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)

    X = vectorizer.fit_transform(df["ProcessedText"])
    y = df[target_column]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Logistic Regression
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nSentiment Model Accuracy : {accuracy:.4f}\n")

    print("Classification Report")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    return model, vectorizer


# ---------------- Sentiment Mapping ---------------- #

sentiment_mapping = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


# ---------------- Complaint Category ---------------- #

def get_category(text):
    """
    Classify complaint category from text.
    """

    text = text.lower()

    if any(word in text for word in [
        "pothole",
        "road damage",
        "broken road",
        "crack"
    ]):
        return "Road Condition"

    elif any(word in text for word in [
        "traffic jam",
        "traffic",
        "congestion"
    ]):
        return "Traffic Issue"

    elif any(word in text for word in [
        "signal",
        "traffic light",
        "red light"
    ]):
        return "Signal Issue"

    elif any(word in text for word in [
        "accident",
        "crash",
        "collision"
    ]):
        return "Accident"

    else:
        return "General Complaint"