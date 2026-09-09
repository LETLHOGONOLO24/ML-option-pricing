import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):

    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    C = (
        S * norm.cdf(d1)
        - K * np.exp(-r * T) * norm.cdf(d2)
    )

    return C

price = black_scholes_call(
    S=100,
    K=100,
    T=1,
    r=0.05,
    sigma=0.20
)

print(price)