from preprocess import preprocess_data, load_data
from model import train_model

def main():
    data_path = 'dry-beans-data.csv'
    df = load_data(data_path)
    X_train_res, y_train_res, X_val_scaled, y_val = preprocess_data(df)
    train_model(X_train_res, y_train_res, X_val_scaled, y_val)

if __name__ == '__main__':
    main()