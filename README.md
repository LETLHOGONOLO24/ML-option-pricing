# ML-option-pricing

# Numerical Methods and Machine Learning for Option Pricing

## Overview

This repository contains the computational work for an Honours Applied Mathematics
research project titled:

**A Comparative Study of Machine Learning and Numerical Methods for Option Pricing**

The project investigates numerical approaches to option pricing, with particular
focus on the Black--Scholes framework, finite-difference methods, and numerical
methods for high-dimensional derivative pricing. A machine learning model is
then introduced as a computational approximation to the option-pricing function.

The main objective is not to replace the mathematical theory of option pricing
with machine learning, but to investigate how a data-driven model can approximate
option prices generated from a mathematical pricing framework and how its
performance compares with established numerical approaches.

---

## Research Objectives

The main objectives of this project are:

1. To review the mathematical foundations of European option pricing using the
   Black--Scholes model.

2. To investigate the use of finite-difference methods for solving the
   Black--Scholes partial differential equation (PDE).

3. To study the Crank--Nicolson finite-difference method for numerical option
   pricing.

4. To investigate numerical methods for option pricing in high-dimensional
   settings using PCA and ANOVA expansions.

5. To implement a machine learning model that approximates the Black--Scholes
   option-pricing function.

6. To evaluate the accuracy of the machine learning approximation using
   quantitative error measures.

7. To compare the mathematical and computational characteristics of
   Crank--Nicolson, PCA--ANOVA, and Random Forest approaches.

---

## Mathematical Background

The project begins with the Black--Scholes framework for pricing European
options.

Under the standard Black--Scholes assumptions, the option value satisfies the
Black--Scholes PDE

$$
\frac{\partial V}{\partial t}
+
\frac{1}{2}\sigma^2 S^2
\frac{\partial^2 V}{\partial S^2}
+
rS\frac{\partial V}{\partial S}
-rV=0.
$$

For a European call option, the terminal condition is

$$
V(S,T)=\max(S-K,0),
$$

where:

- $S$ is the underlying asset price,
- $K$ is the strike price,
- $T$ is the maturity,
- $r$ is the risk-free interest rate,
- $\sigma$ is the volatility.

The analytical Black--Scholes call price provides a reference value against
which numerical and machine learning approximations can be evaluated.

---

## Literature Review

### 1. Crank--Nicolson Method

The first paper considered in this research is:

> Putri, I. E. W. and Artiono, R.  
> *Numerical Pricing of European Stock Option Based on Black-Scholes Model
> Using Crank-Nicolson Method.*

The paper investigates the numerical pricing of European stock options using
the Crank--Nicolson finite-difference method.

The method discretises the Black--Scholes PDE in both the asset-price and
time dimensions. The temporal derivative is approximated using a central
average between two consecutive time levels, while spatial derivatives are
approximated using central finite differences.

The resulting discretisation produces a tridiagonal system that can be solved
efficiently at each time step.

The paper investigates the effect of grid refinement on numerical accuracy.
The reported results show that increasing the number of spatial and temporal
grid points substantially reduces the numerical error.

For example, the paper reports a relative error of approximately 17.94% for
a grid with $M=N=50$, while refinement to $M=N=600$ reduces the relative error
to approximately 0.072%.

These values are **results reported by the authors of the paper** and are not
results produced by this repository.

---

### 2. PCA--ANOVA for High-Dimensional Derivatives

The second paper considered is:

> Reisinger, C. and Wissmann, R.  
> *Numerical Valuation of Derivatives in High-Dimensional Settings via PDE
> Expansions.*

This work addresses the difficulty of applying grid-based PDE methods to
high-dimensional derivative-pricing problems.

The central problem is the **curse of dimensionality**: conventional grid
methods become increasingly expensive as the number of underlying risk factors
increases.

The authors combine:

- Principal Component Analysis (PCA),
- anchored ANOVA decompositions, and
- PDE-based numerical approximations.

PCA is used to transform the original risk factors into a smaller number of
important principal components. The ANOVA expansion is then used to approximate
the solution using lower-dimensional component interactions.

The approach is particularly useful for high-dimensional derivative-pricing
problems where a direct multidimensional PDE solution would be computationally
expensive.

The paper reports that the proposed PDE expansion can provide accurate
approximations while requiring substantially less computational time than the
corresponding Monte Carlo calculations for the numerical examples considered.

Again, these are **results reported by the authors** and are not results
generated by this repository.

---

## Research Methodology

The research consists of two connected components.

### Component 1: Mathematical and Numerical Analysis

The first component investigates established mathematical and numerical
approaches to option pricing.

The workflow is:

```text
Black--Scholes Model
        |
        v
Black--Scholes PDE
        |
        +----------------------+
        |                      |
        v                      v
Crank--Nicolson          PCA--ANOVA
Finite Differences       PDE Expansion
        |                      |
        v                      v
Numerical Pricing       High-Dimensional
                        Derivative Pricing
