import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def testError(predictions, y):
    E_test = 0
    for i in range(len(predictions)):
        E_test += (predictions[i] - y.iloc[i])**2
    return E_test / len(predictions)

if __name__ == "__main__":
    df = pd.read_csv("vgsales.csv")

    # Features: "Platform", "Year", "Genre", "Publisher", "NA_Sales", 
    # "EU_Sales", "JP_Sales", "Other_Sales"
    features = [ "Platform", "Year", "Genre", "Publisher", "NA_Sales", 
                "EU_Sales", "JP_Sales", "Other_Sales"]
    
    X = df.loc[:, features].copy()

    # Target: Global Sales
    y = df["Global_Sales"]

    categoricalFeatures = ["Platform", "Genre", "Publisher"]
    for col in categoricalFeatures: 
        encoder = LabelEncoder()           # creates an encoder per category column
        X[col] = encoder.fit_transform(X[col])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    error = testError(predictions, y_test)

    print("Average units of error: ", error)
