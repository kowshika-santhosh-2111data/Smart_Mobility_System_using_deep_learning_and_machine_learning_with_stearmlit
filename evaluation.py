from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import numpy as np


def evaluate_model(model, x_test, y_test):
    """
    Evaluate CNN or Machine Learning model.
    """

    # Predict
    y_pred = model.predict(x_test)

    # CNN/Deep Learning model (predict returns probabilities)
    if len(y_pred.shape) > 1:
        y_pred_labels = np.argmax(y_pred, axis=1)
    else:
        y_pred_labels = y_pred

    # One-hot encoded labels
    if len(y_test.shape) > 1:
        y_test_labels = np.argmax(y_test, axis=1)
    else:
        y_test_labels = y_test

    # Accuracy
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    print(f"\nAccuracy: {accuracy:.4f}")

    # Classification Report
    print("\nClassification Report")
    print(
        classification_report(
            y_test_labels,
            y_pred_labels,
            zero_division=0
        )
    )

    # Confusion Matrix
    print("\nConfusion Matrix")
    print(confusion_matrix(y_test_labels, y_pred_labels))

    return accuracy