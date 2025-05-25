def evaluate_model(model, X_test, y_test):
    """
    Evaluate the performance of the trained model on the test dataset.

    Parameters:
    model: The trained model to evaluate.
    X_test: The features of the test dataset.
    y_test: The true labels of the test dataset.

    Returns:
    metrics: A dictionary containing evaluation metrics such as accuracy, precision, recall, and F1 score.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Compile metrics into a dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return metrics