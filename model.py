from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from typing import Tuple
import pandas as pd

def train(X_train_res: pd.DataFrame, y_train_res: pd.Series) -> SVC:
    """
    Train an SVM classifier on the resampled training data.
    
    Input:
        - X_train_res: pd.DataFrame, resampled training features.
        - y_train_res: pd.Series, resampled training labels.
    
    Output: 
        - SVC, trained SVM classifier.
    """
    # Create the SVM classifier using best hyperparameters from the gridsearch
    model_svc = SVC(probability=True, kernel='rbf', C=25, gamma="scale")
    model_svc.fit(X_train_res, y_train_res)
    return model_svc

def eval(X_train_res: pd.DataFrame, X_val_scaled: pd.DataFrame, model_svc: SVC, y_train_res: pd.Series, y_val: pd.Series):
    """
    Evaluate the SVM classifier using the training and validation data.
    
    Input:
        - X_train_res: pd.DataFrame, resampled training features.
        - X_val_scaled: pd.DataFrame, scaled validation features.
        - model_svc: SVC, trained SVM classifier.
        - y_train_res: pd.Series, resampled training labels.
        - y_val: pd.Series, validation labels.
    
    Output: 
        - Prints the F1 Score and Accuracy for both training and validation datasets.
    """
    # Evaluate the model
    y_pred_train = model_svc.predict(X_train_res)
    y_pred_val = model_svc.predict(X_val_scaled)

    print("Train F1_Score: ", f1_score(y_train_res, y_pred_train, average='micro'))
    print("Val F1_Score: ", f1_score(y_val, y_pred_val, average='micro'))
    print("Train Accuracy: ", accuracy_score(y_train_res, y_pred_train))
    print("Val Accuracy: ", accuracy_score(y_val, y_pred_val))

def train_model(X_train_res: pd.DataFrame, y_train_res: pd.Series, X_val_scaled: pd.DataFrame, y_val: pd.Series):
    """
    Calls the training and evaluation of the SVM model.
    
    Input:
        - X_train_res: pd.DataFrame, resampled training features.
        - y_train_res: pd.Series, resampled training labels.
        - X_val_scaled: pd.DataFrame, scaled validation features.
        - y_val: pd.Series, validation labels.
    
    Output: 
        - None, calls train and eval functions.
    """
    model_svc = train(X_train_res, y_train_res)
    eval(X_train_res, X_val_scaled, model_svc, y_train_res, y_val)