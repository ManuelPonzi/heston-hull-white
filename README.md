# Heston-Hull-White

## Introduction
This repository contains a Python implementation of the deterministic approximation of the Heston-Hull & White model proposed by L. A. Grzelak and C. W. Oosterlee in their paper *On the Heston Model with Stochastic Interest Rates*. 
A practical application consisting in the calibration of the model to historical interest rates data and option data is also included. 

## Repository Structure
- **`main.ipynb`**  
  Jupyter notebook containing the calibration of the model to market data and the Monte Carlo simulation of the stock price, volatility and interest rate processes.

- **`classes/heston_hull_white.py`**  
  Contains the implementation of the Heston-Hull-White model.

- **`data/`**  
  Directory for input datasets:
  - `AMD_2025-07-07_options.csv` – Call options written on AMD stock, available as of July 7th, 2025, retrieved from Yahoo Finance.
  - `SOFR.csv` – SOFR daily time series, retrieved from FRED data library, used for calibrating the Hull-White model.

## The Heston-Hull & White Model
The Heston-Hull & White (HHW) model is a sophisticated financial model designed for pricing long-dated contingent claims, particularly those sensitive to both stochastic volatility and stochastic interest rates.
The model combines the Heston (1993) stochastic volatility model with the Hull and White (1990, 1994) interest rate model. This integrated approach allows for a more accurate representation of market dynamics
compared to simpler models like Black-Scholes (1973) or even the standalone Heston model, especially for complex derivatives where the correlation between interest rates and asset volatility plays a significant role.

### Model Dynamics

Let:

- $S_t$ be the asset price,
- $v_t$ the stochastic variance,
- $r_t$ the short-term interest rate, and
- $x_t = \ln S_t$ the log of the asset price.

The HHW model is defined by the following SDEs:

$$
\begin{aligned}
dx_t &= \left( r_t - \frac{1}{2} v_t \right) dt + \sqrt{v_t} \, dW_t^x, \\
dv_t &= \kappa (\bar{v} - v_t) dt + \gamma \sqrt{v_t} \, dW_t^v, \\
dr_t &= \lambda (\theta_t - r_t) dt + \eta \, dW_t^r,
\end{aligned}
$$

with the correlation structure:

$$
dW_t^x \, dW_t^v = \rho_{xv} dt, \quad
dW_t^x \, dW_t^r = \rho_{xr} dt, \quad
dW_t^v \, dW_t^r = \rho_{vr} dt.
$$

### Deterministic Approximation Approach
The deterministic approximation approach consists in:

1. Replacing $\sqrt{v_t}$ with its expected value:

$$
\mathbb{E}[\sqrt{v_t}] \approx \Lambda_t = \sqrt{c(t)(\lambda(t) - 1) + c(t)d + \frac{c(t)d}{2(d + \lambda(t))}},
$$

with:

$$
c(t) = \frac{\gamma^2 (1 - e^{-\kappa t})}{4 \kappa}, \quad
d = \frac{4\kappa \bar{v}}{\gamma^2}, \quad
\lambda(t) = \frac{4\kappa v_0 e^{-\kappa t}}{\gamma^2 (1 - e^{-\kappa t})}.
$$

2. Assuming constant mean-reversion level: $\theta_t = \theta$ (Vasicek model).

3. Setting the correlation $\rho_{vr} = 0$.

### Characteristic Function

Under these assumptions, the characteristic function is:

$$
\phi(u, \tau) = \exp\left[A(u, \tau) + B(u, \tau) x_t + C(u, \tau) r_t + D(u, \tau) v_t\right],
$$

where $\tau = T - t$ and:

$$
\begin{aligned}
B(u, \tau) &= iu, \\
C(u, \tau) &= \frac{iu - 1}{\lambda}(1 - e^{-\lambda \tau}), \\
D(u, \tau) &= \frac{1 - e^{-D_1 \tau}}{\gamma^2 (1 - g e^{-D_1 \tau})} (\kappa - \gamma \rho_{xv} iu - D_1), \\
A(u, \tau) &= \lambda \theta I_1(\tau) + \kappa \bar{v} I_2(\tau) + \frac{1}{2} \eta^2 I_3(\tau) + \eta \rho_{xr} I_4(\tau),
\end{aligned}
$$

with:

$$
D_1 = \sqrt{(\gamma \rho_{xv} iu - \kappa)^2 - \gamma^2 iu (iu - 1)}, \quad
g = \frac{\kappa - \gamma \rho_{xv} iu - D_1}{\kappa - \gamma \rho_{xv} iu + D_1}.
$$

and

$$
\begin{aligned}
I_1(\tau) &= \frac{1}{\lambda}(iu - 1)\left(\tau + \frac{1}{\lambda}(e^{-\lambda \tau} - 1)\right), \\
I_2(\tau) &= \frac{1}{\gamma^2}(\kappa - \gamma \rho_{xv} iu - D_1)\tau - \frac{2}{\gamma^2} \ln\left(\frac{1 - g e^{-D_1 \tau}}{1 - g}\right), \\
I_3(\tau) &= \frac{1}{2\lambda^3}(iu + u^2)\left(3 + e^{-2\lambda \tau} - 4e^{-\lambda \tau} - 2\lambda \tau\right), \\
I_4(\tau) &= -\frac{1}{\lambda}(iu + u^2) \int_0^{\tau} \Lambda_{T - s}(1 - e^{-\lambda s}) \, ds.
\end{aligned}
$$

### COS Method for Pricing

The price of an option $v(x, t)$ is approximated using the COS method:

$$
v(x, t) \approx \sum_{k=0}^{N-1}{}^{'} \text{Re} \left[ \varphi\left(\frac{k\pi}{b-a}\right) e^{i k \pi \frac{x - a}{b - a}} \right] V_k,
$$

where:

- $\phi(u; x) = \varphi(u) e^{iux}$;
- $\sum^{'}$ means the first term is halved;
- $V_k$ are cosine coefficients of the payoff function;
- $N = 200$ is the number of cosine terms;
- $[a, b]=[-8\sqrt{\tau}, 8\sqrt{\tau}]$ is the integration interval.

The cosine coefficients for call and put options are given by

$$
V_k^{\text{call}} = \frac{2K}{b - a} \left[ \chi_k(0, b) - \psi_k(0, b) \right]
$$

$$
V_k^{\text{put}} = \frac{2K}{b - a} \left[ -\chi_k(a, 0) + \psi_k(a, 0) \right]
$$

where $K$ is the strike price and:

$$
\chi_k(c, d) = \frac{1}{1 + \left( \frac{k\pi}{b - a} \right)^2} \left[\cos\left(\frac{k\pi(d - a)}{b - a} \right) e^d -\cos\left(\frac{k\pi(c - a)}{b - a} \right) e^c+ \frac{k\pi}{b - a} \left(\sin\left(\frac{k\pi(d - a)}{b - a} \right) e^d -\sin\left(\frac{k\pi(c - a)}{b - a} \right) e^c\right) \right]
$$

$$
\psi_k(c, d) =
\begin{cases}
\frac{b - a}{k\pi} \left[ \sin\left( \frac{k\pi(d - a)}{b - a} \right) - \sin\left( \frac{k\pi(c - a)}{b - a} \right) \right], & k > 0 \\
d - c, & k = 0
\end{cases}
$$

## References
- Grzelak, Lech Aleksander and Oosterlee, Cornelis W., On the Heston Model with Stochastic Interest Rates (July 30, 2010). SIAM Journal on Financial Mathematics 2, 255–286, 2011, Available at SSRN: https://ssrn.com/abstract=1382902
