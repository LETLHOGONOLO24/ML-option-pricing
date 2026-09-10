import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
# 3. Recreate the same train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTest dataset recreated.")
print(f"Testing observations: {len(X_test)}")


# --------------------------------------------------
# 4. Load the trained Random Forest model
# --------------------------------------------------

model = joblib.load(
    "C:/Users/HLOGIZNBUCKS/Downloads/ML-option-pricing/results/random_forest_model.pkl"
)

print("\nTrained Random Forest model loaded successfully.")


# --------------------------------------------------
# 5. Generate predictions for the test dataset
# --------------------------------------------------

y_pred = model.predict(X_test)

print("Predictions generated successfully.")


# --------------------------------------------------
# 6. Calculate evaluation metrics
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


# --------------------------------------------------
# 7. Display evaluation results
# --------------------------------------------------

print("\n----------------------------------------")
print("MODEL EVALUATION RESULTS")
print("----------------------------------------")

print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
print(f"R-squared (R²): {r2:.6f}")

print("----------------------------------------")


# --------------------------------------------------
# 8. Calculate prediction errors
# --------------------------------------------------

errors = y_pred - y_test.values

print("\nPrediction error statistics:")
print(f"Mean error: {np.mean(errors):.6f}")
print(f"Minimum error: {np.min(errors):.6f}")
print(f"Maximum error: {np.max(errors):.6f}")
print(f"Standard deviation of error: {np.std(errors):.6f}")


# --------------------------------------------------
# 9. Save evaluation metrics
# --------------------------------------------------

metrics_df = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R_squared",
        "Mean Error",
        "Minimum Error",
        "Maximum Error",
        "Error Standard Deviation"
    ],
    "Value": [
        mae,
        rmse,
        r2,
        np.mean(errors),
        np.min(errors),
        np.max(errors),
        np.std(errors)
    ]
})

metrics_df.to_csv(
    "C:/Users/HLOGIZNBUCKS/Downloads/ML-option-pricing/results/tables/ml_results.csv",
    index=False
)

print("\nEvaluation metrics saved to:")
print("results/tables/ml_results.csv")


# --------------------------------------------------
# 10. Create predicted vs Black-Scholes price plot
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

# Ideal prediction line: y = x
minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Black-Scholes Price")
plt.ylabel("ML Predicted Price")
plt.title("ML Predicted Prices vs Black-Scholes Prices")

plt.tight_layout()

plt.savefig(
    "C:/Users/HLOGIZNBUCKS/Downloads/ML-option-pricing/results/figures/predicted_vs_black_scholes.png",
    dpi=300
)

plt.close()

print("\nFigure 1 saved to:")
print("results/figures/predicted_vs_black_scholes.png")


# --------------------------------------------------
# 11. Create prediction error plot
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    errors,
    alpha=0.5
)

# Zero-error reference line
plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Black-Scholes Price")
plt.ylabel("Prediction Error")
plt.title("ML Prediction Error")

plt.tight_layout()

plt.savefig(
    "C:/Users/HLOGIZNBUCKS/Downloads/ML-option-pricing/results/figures/ml_prediction_error.png",
    dpi=300
)

plt.close()

print("\nFigure 2 saved to:")
print("results/figures/ml_prediction_error.png")


# --------------------------------------------------
# 12. Save detailed predictions
# --------------------------------------------------

results_df = X_test.copy()

results_df["black_scholes_price"] = y_test.values
results_df["ml_predicted_price"] = y_pred
results_df["prediction_error"] = errors
results_df["absolute_error"] = np.abs(errors)

results_df.to_csv(
    "results/tables/ml_predictions.csv",
    index=False
)

print("\nDetailed predictions saved to:")
print("results/tables/ml_predictions.csv")


# --------------------------------------------------
# 13. Finish
# --------------------------------------------------

print("\nEvaluation completed successfully.")