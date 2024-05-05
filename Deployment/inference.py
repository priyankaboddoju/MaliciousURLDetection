import os
import json
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Deserialize fitted model
def model_fn(model_dir):
    """
    Load the trained RandomForestClassifier model from the model directory.

    Args:
    model_dir (str): Directory where model artifacts are stored.

    Returns:
    RandomForestClassifier: Loaded model object.
    """
    model_path = os.path.join(model_dir, "model.pkl")
    model = joblib.load(model_path)
    return model

# Deserialize input data
def input_fn(request_body, request_content_type):
    """
    Deserialize the input data from the request body.

    Args:
    request_body (str): Raw request body containing JSON-formatted data.
    request_content_type (str): Content type of the request body.

    Returns:
    dict: Deserialized input data.
    """
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError("This model only supports application/json input")

# Preprocess input data
def process_input(input_data, model):
    """
    Preprocess the input data before passing it to the model for prediction.
    Here, we transform the input data using the TF-IDF vectorizer.

    Args:
    input_data (dict): Input data dictionary containing 'url' key with list of URLs.
    model (RandomForestClassifier): Trained model object.

    Returns:
    numpy.ndarray: Processed input data ready for prediction.
    """
    X = input_data['url']
    vectorizer_path = os.path.join("opt/ml/model", "tfidf_vectorizer.pkl")
    vectorizer = joblib.load(vectorizer_path)
    X_vect = vectorizer.transform(X)
    return X_vect

# Perform inference using the model
def predict_fn(input_data, model):
    """
    Perform inference using the trained model on the preprocessed input data.

    Args:
    input_data (dict): Preprocessed input data.
    model (RandomForestClassifier): Trained model object.

    Returns:
    numpy.ndarray: Model predictions.
    """
    # Process the input data if necessary
    processed_data = process_input(input_data, model)
    # Make predictions using the model
    predictions = model.predict(processed_data)
    return predictions

# Serialize the output predictions
def output_fn(prediction, content_type):
    """
    Serialize the model predictions to JSON format.

    Args:
    prediction (numpy.ndarray): Model predictions.
    content_type (str): Expected content type of the response.

    Returns:
    str: JSON-formatted string representing the predictions.
    """
    prediction_str = prediction.tolist()
    response = {"type": prediction_str}
    return json.dumps(response)
