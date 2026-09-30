# Vectorized Stochastic Differential Equation Engine: Geometric Brownian Motion & Ornstein-Uhlenbeck Frameworks

**Author:** Mauro Martínez Barquero  
**Affiliation:** B.S. in Mathematical Engineering & Physics, Universidad CEU Cardenal Herrera  
**Topic:** Quantitative Finance, Stochastic Calculus & High-Performance Numerical Simulation  

---

## Abstract
This repository presents a high-performance, parallelized Python simulation framework for two foundational continuous-time stochastic differential equations (SDEs) used in quantitative finance and physical modeling: **Geometric Brownian Motion (GBM)** and the **Ornstein-Uhlenbeck (OU) Mean-Reverting Process**. 

We derive the exact analytical solution for GBM via Itô’s Lemma, implement the Euler-Maruyama numerical scheme for the OU process, and showcase the performance benefits of SIMD matrix vectorization over interpreted loops.

---

## 1. Geometric Brownian Motion (GBM)

### 1.1 Mathematical Formulation
Geometric Brownian Motion models continuous-time asset dynamics under the assumption that returns are log-normally distributed with independent increments:

$$\mathrm{d}S_t = \mu S_t \mathrm{d}t + \sigma S_t \mathrm{d}W_t$$

Where:
* $S_t$ is the asset price at time $t$.
* $\mu \in \mathbb{R}$ is the expected annual drift rate.
* $\sigma > 0$ is the annualized volatility (diffusion intensity).
* $W_t$ is a 1D Wiener process satisfying $W_t \sim \mathcal{N}(0, t)$.

### 1.2 Exact Derivation via Itô's Lemma
Applying Itô's Lemma to $f(S_t) = \ln(S_t)$:

$$\mathrm{d}f(S_t) = \frac{\partial f}{\partial S_t}\mathrm{d}S_t + \frac{1}{2}\frac{\partial^2 f}{\partial S_t^2}(\mathrm{d}S_t)^2$$

Computing partial derivatives:
$$\frac{\partial f}{\partial S_t} = \frac{1}{S_t}, \quad \frac{\partial^2 f}{\partial S_t^2} = -\frac{1}{S_t^2}$$

Using Itô multiplication rules where $(\mathrm{d}W_t)^2 = \mathrm{d}t$ and $(\mathrm{d}t)^2 = 0$:
$$(\mathrm{d}S_t)^2 = \sigma^2 S_t^2 \mathrm{d}t$$

Substituting back into the Taylor expansion yields:
$$\mathrm{d}\ln(S_t) = \left(\mu - \frac{1}{2}\sigma^2\right)\mathrm{d}t + \sigma \mathrm{d}W_t$$

Integrating from $0$ to $T$ and exponentiating both sides yields the exact analytical solution implemented in our code:

$$S_T = S_0 \exp\left( \left(\mu - \frac{1}{2}\sigma^2\right)T + \sigma W_T \right)$$

The $-\frac{1}{2}\sigma^2$ term represents the **volatility drag (Itô correction)** resulting from non-linear exponential compounding over continuous time.

---

## 2. Ornstein-Uhlenbeck (OU) Mean-Reverting Process

### 2.1 Mathematical Formulation
The OU process models mean-reverting financial quantities, such as interest rates, commodity spreads, or statistical arbitrage pairs:

$$\mathrm{d}X_t = \theta(\mu - X_t)\mathrm{d}t + \sigma \mathrm{d}W_t$$

Where:
* $\theta > 0$ is the mean-reversion speed (restoring force strength).
* $\mu$ is the long-term equilibrium mean target.
* $\sigma$ is the noise/volatility coefficient.

### 2.2 Euler-Maruyama Numerical Discretization
Discretizing over time step $\Delta t$:

$$X_{t+\Delta t} = X_t + \theta(\mu - X_t)\Delta t + \sigma \sqrt{\Delta t} \, Z_t, \quad Z_t \sim \mathcal{N}(0, 1)$$

---

## 3. Computational Architecture & Vectorization

Rather than using slow, interpreted Python loops, this engine utilizes **NumPy matrix vectorization**:
1. Generates a random normal tensor $\mathbf{Z}$ of shape `(steps, paths)` in a single C-level memory allocation.
2. Computes Brownian paths $\mathbf{W}$ using fast row-wise cumulative summation (`np.cumsum(..., axis=0)`).
3. Leverages memory broadcasting (`t[:, np.newaxis]`) to evaluate time vectors across 1,000 parallel paths simultaneously.

---

## 4. How to Run

Execute the engine using `uv`:

```bash
uv run stochastic_engine.py
