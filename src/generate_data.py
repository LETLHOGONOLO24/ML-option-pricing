import numpy as np
import pandas as pd

from src.black_scholes import black_scholes_call

# Reproducibility
np.random.seed(42)

# Number of observations
n = 20000

# Generate option parameters
S = np.random.uniform(50, 150, n)
K = np.random.uniform(50, 150, n)
T = np.random.uniform(0.05, 2.0, n)
r = np.random.uniform(0.01, 0.10, n)
sigma = np.random.uniform(0.10, 0.60, n)

# Calculate Black-Scholes call prices
call_price = black_scholes_call(
    S,
    K,
    T,
    r,
    sigma
)


# Create DataFrame
df = pd.DataFrame({
    "S": S,
    "K": K,
    "T": T,
    "r": r,
    "sigma": sigma,
    "call_price": call_price
})


# Display first five observations
print(df.head())


# Display summary statistics
print("\nSummary statistics:")
print(df.describe())


# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())


# Save dataset
df.to_csv(
    "C:/Users/HLOGIZNBUCKS/Downloads/ML-option-pricing/data/processed/black_scholes_dataset.csv",
    index=False
)


print("\nDataset successfully generated.")
print(f"Number of observations: {len(df)}")