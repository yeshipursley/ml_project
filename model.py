from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC


def train(X_train_res, y_train_res):    
    """
    Input:
        - 
    Output: it returns the data
    """
    # Create the SVM classifier using best hyperparameters from the gridsearch
    model_svc = SVC(probability=True, kernel='rbf', C=25, gamma="scale")
    model_svc.fit(X_train_res, y_train_res)
    return model_svc

def eval(X_train_res, X_val_scaled, model_svc, y_train_res, y_val):
    """
    Input:
        - 
    Output: it returns the data
    """
    # Evaluate the model
    y_pred_train = model_svc.predict(X_train_res)
    y_pred_val = model_svc.predict(X_val_scaled)

    print("Train F1_Score: ", f1_score(y_train_res, y_pred_train, average='micro'))
    print("Val F1_Score: ", f1_score(y_val, y_pred_val, average='micro'))
    print("Train Accuracy: ", accuracy_score(y_train_res, y_pred_train))
    print("Val Accuracy: ", accuracy_score(y_val, y_pred_val))

def train_model(X_train_res, y_train_res, X_val_scaled, y_val):
    
    model_svc =train(X_train_res, y_train_res)
    eval(X_train_res, X_val_scaled, model_svc, y_train_res, y_val)
    

    