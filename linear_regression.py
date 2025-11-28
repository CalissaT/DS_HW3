import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

'''
Features: 
Genre_encoded, pub_encoded
Target:
Global sales
'''

#function to encode genre and publisher for linear regression
def encode(df):
    df = df.copy()
    df = df.dropna(subset=["Genre", "Publisher", "Global_Sales"])
    df["Publisher_freq"] = df["Publisher"].map(df["Publisher"].value_counts())
    df_encoded = pd.get_dummies(df, columns=["Genre"], prefix="Genre", drop_first=False)
    genre_cols = [col for col in df_encoded.columns if col.startswith("Genre_")]
    return df_encoded, genre_cols


#get the data, split into training and test set
def get_data(df, genre_cols):
    feature_cols = genre_cols + ["Publisher_freq"]
    train_df, test_df = train_test_split(df, test_size=0.30, random_state=42)
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df["Global_Sales"]
    y_test = test_df["Global_Sales"]

    return X_train, y_train, X_test, y_test, feature_cols

#add the bias column to X
def add_bias(X):
    X = np.asarray(X, dtype=float)
    X_new = np.hstack([np.ones((X.shape[0], 1)), X])
    return X_new

#linear regression on the data
def linear_regression(X_train, y_train):
    XTX = X_train.T @ X_train
    XTX_inv = np.linalg.inv(XTX)
    XT_y = X_train.T @ y_train
    w = XTX_inv @ XT_y
    return w

def predict(X, w):
    return X @ w

def print_error_report(y_true, y_pred, label=""):
    errors = y_true - y_pred

    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))

    print(f"Root Mean Squared Error: {rmse:.4f} million units")
    print(f"Mean Absolute Error: {mae:.4f} million units")
    print(f"Min error:  {np.min(errors):.4f} (best case)")
    print(f"Max error:  {np.max(errors):.4f} (worst case)")
    print(f"25th percentile error: {np.percentile(errors, 25):.4f}")
    print(f"50th percentile error: {np.percentile(errors, 50):.4f} (median)")
    print(f"75th percentile error: {np.percentile(errors, 75):.4f}")
    print("==============================\n")


if __name__ == "__main__":
    df = pd.read_csv("vgsales.csv")
    df_encoded, genre_cols = encode(df)
    X_train, y_train, X_test, y_test, feature_cols = get_data(df_encoded, genre_cols)

    #make into np arrays
    X_train = add_bias(X_train)
    X_test = add_bias(X_test)
    y_train = np.array(y_train, dtype=float)
    y_test = np.array(y_test, dtype=float)


    w = linear_regression(X_train, y_train)
    pred_train = predict(X_train, w)
    pred_test  = predict(X_test, w)

    E_train = np.sum((y_train - pred_train) ** 2)
    E_test = np.sum((y_test - pred_test) ** 2)
    
    print_error_report(y_test, pred_test)