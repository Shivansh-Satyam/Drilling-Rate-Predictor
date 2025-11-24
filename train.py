import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

#Loading data
filename = 'drillingdata.csv'
print(f"--- Loading data from {filename} ---")

try:
    df = pd.read_csv(filename)
    print("File Loaded Successfully!")
    print(f"Total Rows: {len(df)}")
    print("Columns found:", list(df.columns))

except FileNotFoundError:
    print(f"Error: '{filename}' nahi mili. Make sure csv file same folder mein ho.")
    exit()

#Preparing target and features
target_col = 'penetration_rate_m/min'

#Checking if columns exists or not
if target_col not in df.columns:
    print(f"Error: '{target_col}' naam ka column nahi mila. CSV check karo.")
    exit()

#X - all columns except target
#Y - target column
X = df.drop(columns=[target_col])
y = df[target_col]

#Spliting - 80% training 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Training XGBoost model
print("\n--- Training XGBoost Model ---")

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,     #Number of trees
    learning_rate=0.1,    #Training speed
    max_depth=6,          #Depth of trees
    random_state=42
)

model.fit(X_train, y_train)
print("Model Trained Successfully!")

#Model evaluation
print("\n--- Checking Accuracy ---")
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"RMSE (Average Error): {rmse:.4f} m/min")
print(f"R2 Score (Accuracy):  {r2:.4f} (jitna 1.0 ke paas, utna behtar)")

#Saving models
pkl_filename = "drilling_prediction_model.pkl"
joblib.dump(model, pkl_filename)

print(f"\nDone! Model saved as '{pkl_filename}'")
