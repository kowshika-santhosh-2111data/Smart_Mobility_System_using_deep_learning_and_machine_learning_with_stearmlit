from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_preprocessing import preprocess_tabular_data


def train_tabular_model(df, target_column="RiskLevel"):
    """
    Train a Random Forest model on tabular data.
    """

    # Preprocess data
    x, y, scaler = preprocess_tabular_data(df, target_column)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Build model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    # Train
    model.fit(x_train, y_train)

    # Prediction
    y_pred = model.predict(x_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTabular Model Accuracy: {accuracy:.4f}\n")
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    return model, scaler, x_test, y_test


# ---------------- Risk Labels ---------------- #

risk_mapping = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk"
}