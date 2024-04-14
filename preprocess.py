import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def load_data(path: str):
    """
    Input:
        - path: path to the csv file
    Output: it returns the data
    """
    df = pd.read_csv(path)
    return df

# def clean_data(data: pd.DataFrame):
#     """
#     Input:
#         - path: path to the csv file
#     Output: it returns the data
#     """

def scale_data(X_train, X_val):
    """
    Input:
        - 
    Output: it returns the data
    """
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_scaled)
    X_val_scaled = scaler.transform(X_val_scaled)

    return X_train_scaled, X_val_scaled

def split_data(df):
    """
    Input:
        - 
    Output: it returns the data
    """
    X = df.drop(columns=['Class'])
    y = df['Class']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_val, y_train, y_val

def oversample_data(X_train_scaled, y_train):
    """
    Input:
        - 
    Output: it returns the data
    """
    # Apply SMOTE to the training set
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    return X_train_res, y_train_res


def preprocess_data(df):

    X_train, X_val, y_train, y_val = split_data(df)
    X_train_scaled, X_val_scaled =scale_data(X_train, X_val)
    X_train_res, y_train_res = oversample_data(X_train_scaled, y_train)
    
    return X_train_res, y_train_res, X_val_scaled, y_val

