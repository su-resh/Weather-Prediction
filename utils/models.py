import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
import pickle
import os
import warnings
from sklearn.exceptions import DataConversionWarning

# Suppress specific warnings
warnings.filterwarnings('ignore', category=DataConversionWarning)

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model_type):
    """
    train the model based on the the model type
    Model type can be one of: ["Regression", "Multi-Class Classifier", "Binary Classifier"]
    """
    # Convert to numpy arrays if they're DataFrames
    X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.values if isinstance(y_train, pd.DataFrame) else y_train

    scaler = None
    if model_type == "Regression":
        model = LinearRegression()
        # Scaling for regression
        scaler = StandardScaler()
        
    elif model_type == "Multi-Class Classifier":
        model = RandomForestClassifier(n_estimators=20, n_jobs=-1)
        # No Scaling for RandomForestClassifier. No need scaling for tree-based models
    elif model_type == "Binary Classifier":
        model = RandomForestClassifier(n_estimators=20, n_jobs=-1)
        # No Scaling for RandomForestClassifier. No need scaling for tree-based models
    else:
        raise ValueError('"model_type" must be one of ["Regression", "Multi-Class Classifier", "Binary Classifier"]')

    if scaler:
        # fit the scaler to the train set, it will learn the parameters
        scaler.fit(X_train)
        # Transform train and test set
        X_train_scaled = scaler.transform(X_train)
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)
    
    # Store feature names for later use
    if hasattr(X_train, 'columns'): # in case sklearn < 1.0 save features name from X_train column name
        # For sklearn >=1.0 built-in 'feature_names_in_' exists
        model.feature_names_ = list(X_train.columns)
        scaler.feature_names_ = list(X_train.columns)

    return model, scaler

def predict_model(X_test, model, scaler=None):
    """
    Predict using the trained model. Apply scaling if needed.
    """
    # Convert to numpy arrays if they're DataFrames
    X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

    if scaler:
        # Scale the test data using the same scaler fitted on training data
        X_test_scaled = scaler.transform(X_test)
        return model.predict(X_test_scaled)
    else:
        return model.predict(X_test)
    
def evaluate_model(model, scaler, X_train, y_train, X_test, y_test, model_type):
    """
    Evaluate the model based on model type and return performance metrics.
    Model type can be one of: ["Regression", "Multi-Class Classifier", "Binary Classifier"]
    """
    # Get predictions for both train and test sets
    y_pred_train = predict_model(X_train, model, scaler)
    y_pred_test = predict_model(X_test, model, scaler)

    # For consistent format convert predictions to dataframe
    y_pred_train = pd.DataFrame(y_pred_train, columns=y_train.columns)
    y_pred_test = pd.DataFrame(y_pred_test, columns=y_test.columns)

    if model_type == "Regression":
        # Calculate the metrics for Linear Regression
        metrics = {
            'model_type': "Regression",
            'train_rmse' : np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse' : np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_r2' : r2_score(y_train, y_pred_train),
            'test_r2' : r2_score(y_test, y_pred_test),
            'y_test' : y_test,
            'y_pred_test': y_pred_test
        }
        
    elif model_type in ["Multi-Class Classifier", "Binary Classifier"]:
        # Calculate the metrics for Classification
        metrics = {
            'model_type': "Classification",
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'train_confusion_matrix': confusion_matrix(y_train, y_pred_train),
            'test_confusion_matrix': confusion_matrix(y_test, y_pred_test),
            'train_classification_report': classification_report(y_train, y_pred_train, output_dict=True),
            'test_classification_report': classification_report(y_test, y_pred_test, output_dict=True),
            'y_test' : y_test,
            'y_pred_test': y_pred_test
        }
    else:
        raise ValueError(f'Invalid model_type: {model_type}. Must be one of ["Regression", "Multi-Class Classifier", "Binary Classifier"]')

    return metrics

def save_model(model, scaler, model_name):
    """
    Save the trained model and scaler to disk.
    model_name: string without extension
    """
    print(f'saving model: "{model_name}"')
    # Make sure the 'models/' directory exists
    os.makedirs("models", exist_ok=True)

    # save the model
    with open(f'models/{model_name}_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    # Save the scaler if it exists
    if scaler != None:
        with open(f'models/{model_name}_scaler.pkl', 'wb') as file:
            pickle.dump(scaler, file)
    

def load_model(model_name):
    """
    Load the trained model and scaler from disk.
    model_name: string without extension
    """
    # load model
    try:
        with open(f'models/{model_name}_model.pkl', 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        model =  None
    # load scaler
    try:
        with open(f'models/{model_name}_scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
    except:
        scaler = None
    
    return model, scaler
