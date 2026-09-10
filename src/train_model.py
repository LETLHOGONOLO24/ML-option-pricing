import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------

df = pd.read_csv("C:/Users/HLOGIZNBUCKS/Downloads/ML-option-pricing/data/processed/black_scholes_dataset.csv")

print("Dataset loaded successfully.")
print(f"Number of observations: {len(df)}")


# --------------------------------------------------
# 2. Define input variables and target variable
# --------------------------------------------------

X = df[["S", "K", "T", "r", "sigma"]]
y = df["call_price"]


# --------------------------------------------------
# 3. Split the dataset into training and testing sets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nData split completed.")
print(f"Training observations: {len(X_train)}")
print(f"Testing observations: {len(X_test)}")


# --------------------------------------------------
# 4. Create the Random Forest model
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=150,
    max_depth=18,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 5. Train the model
# --------------------------------------------------

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed.")


# --------------------------------------------------
# 6. Make predictions on the test set
# --------------------------------------------------

y_pred = model.predict(X_test)

print("\nPredictions generated successfully.")
print("First 10 predictions:")

for actual, predicted in zip(y_test.iloc[:10], y_pred[:10]):
    print(f"Black-Scholes: {actual:.6f} | ML Prediction: {predicted:.6f}")


# --------------------------------------------------
# 7. Save the trained model
# --------------------------------------------------

joblib.dump(
    model,
    "C:/Users/HLOGIZNBUCKS/Downloads/ML-option-pricing/results/random_forest_model.pkl"
)

print("\nTrained model saved successfully.")
print("Location: results/random_forest_model.pkl")