import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ==========================================
# STEP 1: LOAD DATA
# ==========================================
filename = 'drillingdata.csv'
print(f"--- Loading data from {filename} ---")

try:
    # Tumhari file standard format mein hai, seedha read kar rahe hain
    df = pd.read_csv(filename)
    print("✅ File Loaded Successfully!")
    print(f"Total Rows: {len(df)}")
    print("Columns found:", list(df.columns))

except FileNotFoundError:
    print(f"❌ Error: '{filename}' nahi mili. Make sure csv file same folder mein ho.")
    exit()

# ==========================================
# STEP 2: PREPARE TARGET & FEATURES
# ==========================================
# Maine file check ki, tumhare target ka exact naam ye hai:
target_col = 'penetration_rate_m/min'

# Check karte hain ki ye column exist karta hai ya nahi
if target_col not in df.columns:
    print(f"❌ Error: '{target_col}' naam ka column nahi mila. CSV check karo.")
    exit()

# X = Saare columns sivaaye target ke
# y = Sirf target column
X = df.drop(columns=[target_col])
y = df[target_col]

# Split: 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# STEP 3: TRAIN XGBOOST MODEL
# ==========================================
print("\n--- Training XGBoost Model ---")

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,     # Number of trees
    learning_rate=0.1,    # Speed
    max_depth=6,          # Depth of trees
    random_state=42
)

model.fit(X_train, y_train)
print("✅ Model Trained Successfully!")

# ==========================================
# STEP 4: EVALUATE MODEL
# ==========================================
print("\n--- Checking Accuracy ---")
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"RMSE (Average Error): {rmse:.4f} m/min")
print(f"R2 Score (Accuracy):  {r2:.4f} (jitna 1.0 ke paas, utna behtar)")

# ==========================================
# STEP 5: SAVE MODEL (PKL FILE)
# ==========================================
pkl_filename = "drilling_prediction_model.pkl"
joblib.dump(model, pkl_filename)

print(f"\n🎉 Kaam Ho Gaya! Model saved as '{pkl_filename}'")
print("Ab tum is .pkl file ko use karke naye inputs par prediction kar sakte ho.")