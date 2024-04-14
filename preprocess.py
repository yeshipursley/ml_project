import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from typing import Tuple
from sklearn import preprocessing

def load_data(path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Input:
        - path: str, path to the csv file.
    
    Output: 
        - pd.DataFrame, the loaded data.
    """
    df = pd.read_csv(path)
    return df

def scale_data(X_train: pd.DataFrame, X_val: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale data using MinMaxScaler followed by StandardScaler.
    
    Input:
        - X_train: pd.DataFrame, training data to scale.
        - X_val: pd.DataFrame, validation data to scale.
    
    Output: 
        - Tuple[pd.DataFrame, pd.DataFrame], scaled training and validation data.
    """
    # Apply MinMax scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Apply Standard scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_scaled)
    X_val_scaled = scaler.transform(X_val_scaled)

    return X_train_scaled, X_val_scaled

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataframe into training and validation sets.
    
    Input:
        - df: pd.DataFrame, dataframe to split.
    
    Output: 
        - Tuple containing training features, validation features, training labels, and validation labels.
    """
    le = preprocessing.LabelEncoder()
    df['Class'] = le.fit_transform(df['Class'])
    X = df.drop(columns=['Class'])
    y = df['Class']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_val, y_train, y_val

def oversample_data(X_train_scaled: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to oversample the training data.
    
    Input:
        - X_train_scaled: pd.DataFrame, scaled training data to oversample.
        - y_train: pd.Series, training labels.
    
    Output: 
        - Tuple[pd.DataFrame, pd.Series], oversampled training data and labels.
    """
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    return X_train_res, y_train_res

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Preprocess the data by splitting, scaling, and oversampling.
    
    Input:
        - df: pd.DataFrame, the dataframe to preprocess.
    
    Output: 
        - Tuple containing resampled training features, resampled training labels, scaled validation features, and validation labels.
    """
    X_train, X_val, y_train, y_val = split_data(df)
    X_train_scaled, X_val_scaled = scale_data(X_train, X_val)
    X_train_res, y_train_res = oversample_data(X_train_scaled, y_train)
    
    return X_train_res, y_train_res, X_val_scaled, y_val