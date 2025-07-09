import numpy as np
from collections.abc import Iterable
from scipy.optimize import minimize

class HestonHullWhite():
    """
    Heston Hull-White model implementation for modeling price, volatility and interest rates processes.
    Combines stochastic volatility (Heston) and stochastic interest rate (Hull-White) dynamics.

    Parameters:
    -----------
    r0 : float
        Initial short rate (interest rate).
    q : float
        Continuous dividend yield.
    v0 : float
        Initial variance of the underlying asset.
    vbar : float
        Long-term mean variance.
    kappa : float
        Speed of mean reversion of the variance process.
    rho_xv : float
        Correlation between asset returns and volatility.
    gamma : float
        Volatility of volatility parameter.
    rho_xr : float
        Correlation between asset returns and interest rates.
    lmdb : float
        Speed of mean reversion of the short rate.
    theta : float
        Long-term mean level of the short rate.
    eta : float
        Volatility of the short rate.
    """
    def __init__(self, r0=0.02, q=0, v0=0.01, vbar=0.01, kappa=1, rho_xv=-0.5, gamma=0.1, rho_xr=-0.1, lmdb=0.05, theta=0.03, eta=0.1):
        for name, value in locals().items():
            if name != 'self':
                setattr(self, name, value)   

    def set_params(self, param_dict):
        """
        Update multiple model parameters at once from a dictionary.

        Parameters:
        -----------
        param_dict : dict
            Dictionary with parameter names as keys and parameter values as values.
        """
        for key, value in param_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def A(self, u, tau):
        """
        Calculate coefficient A(u, tau) used in the characteristic function of the model.

        Parameters:
        -----------
        u : complex or array-like
            Complex argument in characteristic function.
        tau : float or array-like
            Time to maturity.

        Returns:
        --------
        complex or array-like
            Value of coefficient A(u, tau).
        """
        v0, vbar, kappa, gamma = self.v0, self.vbar, self.kappa, self.gamma
        rho_xr, rho_xv = self.rho_xr, self. rho_xv
        lmdb, theta, eta = self.lmdb, self.theta, self.eta

        c = lambda t: 1. / (4 * kappa) * gamma**2 * (1. - np.exp(-kappa * t))
        d = 4 * kappa * vbar / gamma**2
        l = lambda t: 4 * kappa * v0 * np.exp(-kappa * t) / (gamma**2 * (1 - np.exp(-kappa * t)))
        Lambda = lambda t: np.sqrt(c(t) * (l(t) - 1.) + c(t) * d + c(t) * d / (2 * (d + l(t))))
        
        # Trapezoidal integration
        integral = np.zeros_like(tau)
        integral_time_step = 0.001
        for i, t in enumerate(tau):
            n_subs = int(t / integral_time_step)
            integral_lb = 1e-4
            integral_ub = t - 1e-4
            s = np.linspace(integral_lb, integral_ub, n_subs)
            ds = s[1] - s[0]
            lambda_vals = Lambda(t - s) * (1 - np.exp(-lmdb * s))
            integral[i] = np.trapezoid(lambda_vals, dx=ds)
        
        D1 = np.sqrt((gamma * rho_xv * 1j * u - kappa)**2 - gamma**2 * 1j * u * (1j * u - 1.))
        g = (kappa - gamma * rho_xv * 1j * u - D1) / (kappa - gamma * rho_xv * 1j * u + D1)
        I1 = 1. / lmdb * (1j * u - 1.) * (tau + 1. / lmdb * (np.exp(-lmdb * tau) - 1.))
        I2 = tau / gamma**2 * (kappa - gamma * rho_xv * 1j * u - D1) - 2. / gamma**2 * np.log((1. - g * np.exp(- D1 * tau)) / (1. - g))
        I3 = 1. / (2 * lmdb**3) * (1j + u)**2 * (3 + np.exp(-2 * lmdb * tau) - 4 * np.exp(-lmdb * tau) - 2 * lmdb * tau)        
        I4 = - 1. / lmdb * (1j * u + u**2) * integral
        return lmdb * theta * I1 + kappa * vbar * I2 + 1. / 2 * eta**2 * I3 + eta * rho_xr * I4

    def C(self, u, tau):
        """
        Calculate coefficient C(u, tau) in the characteristic function.

        Parameters:
        -----------
        u : complex or array-like
            Complex argument.
        tau : float or array-like
            Time to maturity.

        Returns:
        --------
        complex or array-like
            Value of coefficient C(u, tau).
        """
        lmdb = self.lmdb
        return (1j * u - 1) / lmdb * (1 - np.exp(-lmdb * tau))

    def D(self, u, tau):
        """
        Calculate coefficient D(u, tau) in the characteristic function.

        Parameters:
        -----------
        u : complex or array-like
            Complex argument.
        tau : float or array-like
            Time to maturity.

        Returns:
        --------
        complex or array-like
            Value of coefficient D(u, tau).
        """
        gamma, kappa = self.gamma, self.kappa
        rho_xv = self.rho_xv
        D1 = np.sqrt((gamma * rho_xv * 1j * u - kappa)**2 - gamma**2 * 1j * u * (1j * u - 1.))
        g = (kappa - gamma * rho_xv * 1j * u - D1) / (kappa - gamma * rho_xv * 1j * u + D1)
        return (1 - np.exp(-D1 * tau)) / (gamma**2 * (1 - g * np.exp(-D1 * tau))) * (kappa - gamma * rho_xv * 1j * u - D1) 

    def phi(self, u, tau):
        """
        Calculate the characteristic function phi(u, tau) of log-asset prices under the Heston Hull-White model.

        Parameters:
        -----------
        u : complex or array-like
            Complex argument.
        tau : float or array-like
            Time to maturity.

        Returns:
        --------
        complex or array-like
            Value of characteristic function phi(u, tau).
        """
        r0, v0, q = self.r0, self.v0, self.q 
        return np.exp(self.A(u, tau) + self.C(u, tau) * (r0 - q) + self.D(u, tau) * v0) 

    def chi(self, k, a, b, c, d):
        """
        Helper function computing cosine coefficients for Fourier cosine expansion used in pricing.

        Parameters:
        -----------
        k : int
            Term index in cosine expansion.
        a, b, c, d : float
            Integration bounds for the expansion.

        Returns:
        --------
        float
            Value of the chi coefficient.
        """
        return 1. / (1 + (k * np.pi / (b - a))**2) * (np.cos(k * np.pi * (d - a) / (b - a)) * np.exp(d) - np.cos(k * np.pi * (c - a) / (b - a)) * np.exp(c) + k * np.pi / (b - a) * (np.sin(k * np.pi * (d - a) / (b - a)) * np.exp(d) - np.sin(k * np.pi * (c - a) / (b - a)) * np.exp(c)))
    
    def psi(self, k, a, b, c, d):
        """
        Helper function computing sine coefficients for Fourier cosine expansion used in pricing.

        Parameters:
        -----------
        k : int
            Term index in cosine expansion.
        a, b, c, d : float
            Integration bounds for the expansion.

        Returns:
        --------
        float
            Value of the psi coefficient.
        """
        if k == 0:
            return d - c
        else:
            return (b - a) / (k * np.pi) * (np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a) / (b - a))) 

    def V_call(self, k, a, b, K):
        """
        Compute cosine series coefficient for European call option payoff expansion.

        Parameters:
        -----------
        k : int
            Term index.
        a, b : float
            Expansion bounds.
        K : float
            Strike price.

        Returns:
        --------
        float
            Cosine coefficient for call option.
        """
        return 2. * K / (b - a) * (self.chi(k=k, a=a, b=b, c=0, d=b) - self.psi(k=k, a=a, b=b, c=0, d=b))
    
    def V_put(self, k, a, b, K):
        """
        Compute cosine series coefficient for European put option payoff expansion.

        Parameters:
        -----------
        k : int
            Term index.
        a, b : float
            Expansion bounds.
        K : float
            Strike price.

        Returns:
        --------
        float
            Cosine coefficient for put option.
        """
        return 2. * K / (b - a) * (-self.chi(k=k, a=a, b=b, c=a, d=0) + self.psi(k=k, a=a, b=b, c=a, d=0))
    
    def eu_put_price(self, S0, K, tau):
        """
        Calculate the price of a European put option using the COS method and characteristic function.

        Parameters:
        -----------
        S0 : float
            Current underlying asset price.
        K : float
            Strike price.
        tau : float or array-like
            Time to maturity.

        Returns:
        --------
        float or array-like
            Price of the European put option.
        """
        x = np.log(S0 / K)
        a = - 8. * np.sqrt(tau)
        b = 8. * np.sqrt(tau)
        # Set number of cosine terms to be summed
        N = 200 
        call_price = 0
        tau = tau if isinstance(tau, Iterable) else np.array([tau])
        for k in range(N):
            u = k * np.pi / (b - a)
            call_price += 1. / (1 + int(k==0)) * np.real(self.phi(u=u, tau=tau) * np.exp(1j * k * np.pi * (x - a) / (b - a))) * self.V_put(k=k, a=a, b=b, K=K) 
        return call_price
    
    def eu_call_price(self, S0, K, tau):
        """
        Calculate the price of a European call option using put-call parity.

        Parameters:
        -----------
        S0 : float
            Current underlying asset price.
        K : float
            Strike price.
        tau : float or array-like
            Time to maturity.

        Returns:
        --------
        float or array-like
            Price of the European call option.
        """        
        put_price = self.eu_put_price(S0, K, tau)
        call_price = put_price + S0 * np.exp(-self.q * tau) - K * np.exp(-self.r0 * tau)
        return call_price  

    def log_likelihood_vasicek(self, params, r, dt):
        """
        Negative log-likelihood function for Vasicek interest rate model used in calibration.

        Parameters:
        -----------
        params : tuple
            Parameters (lmdb, theta, eta) for Vasicek model.
        r : array-like
            Observed interest rates.
        dt : float
            Time step between observations.

        Returns:
        --------
        float
            Negative log-likelihood value for given parameters.
        """
        lmdb, theta, eta = params        
        r = np.asarray(r)
        rt = r[:-1]
        rt1 = r[1:]
        # Conditional mean and variance of Vasicek process
        m = rt * np.exp(-lmdb * dt) + theta * (1 - np.exp(-lmdb * dt))
        s2 = (eta**2 / (2 * lmdb)) * (1 - np.exp(-2 * lmdb * dt))
        # Negative log-likelihood for minimization
        ll = -0.5 * np.sum(np.log(2 * np.pi * s2) + ((rt1 - m)**2) / s2)
        return -ll  

    def calibrate_vasicek(self, rates, dt, n_trials=1000):
        """
        Calibrate Vasicek model parameters using maximum likelihood estimation.

        Parameters:
        -----------
        rates : array-like
            Observed interest rate time series.
        dt : float
            Time step between observations.
        n_trials : int, optional
            Number of random initializations for optimizer.

        Returns:
        --------
        OptimizeResult
            Result of optimization containing calibrated parameters.
        """
        x0 = (0.1, np.mean(rates), 0.01)
        # Bounds for lambda, theta and eta
        bounds = [(1e-6, 20), (-2, 2), (1e-6, 2)] 
        best_result = None
        best_fun = np.inf
        # Repeat minimization with n_trials randomly generated initial guesses (within imposed boundaries)
        for _ in range(n_trials):
            x0 = [np.random.uniform(low, high) for (low, high) in bounds]
            result = minimize(
                self.log_likelihood_vasicek,
                x0=x0,
                args=(rates, dt),
                method='L-BFGS-B',
                bounds=bounds
            )
            if result.success and result.fun < best_fun:
                best_fun = result.fun
                best_result = result

        if best_result is not None:
            param_keys = ['lmdb', 'theta', 'eta']
            for key, value in zip(param_keys, best_result.x):
                setattr(self, key, value) 
        return best_result

    def objective_function_heston(self, param_values, S0, calib_opt_data, call_opt_id):
        """
        Objective function for Heston model calibration: Mean Squared Error between market and model call prices.

        Parameters:
        -----------
        param_values : array-like
            Model parameters [v0, vbar, rho_xv, kappa, gamma].
        S0 : float
            Underlying asset price.
        calib_opt_data : np.array with shape (n, 4)
            Options to calibrate Heston model. Array columns should be ordered as follows: option price, option time to maturity, option strike, option type.
        call_opt_id: Any
            Parameter indicating how call options are identified in the 4th column of calib_opt_data

        Returns:
        --------
        float
            Mean Squared Error value.
        """
        param_keys = ['v0', 'vbar', 'rho_xv', 'kappa', 'gamma']
        for i in range(len(param_keys)):
            setattr(self, param_keys[i], param_values[i])
        opt_types = calib_opt_data[:, 3]
        call_data = calib_opt_data[opt_types==call_opt_id][:, :3]
        put_data = calib_opt_data[opt_types!=call_opt_id][:, :3]
        model_prices = np.array([])
        mkt_prices = np.array([])
        if call_data.shape[1] > 0:
            mkt_call_prices, call_ttms, call_strikes = call_data[:, 0].astype(float), call_data[:, 1].astype(float), call_data[:, 2].astype(float)
            heston_call_prices = self.eu_call_price(S0, call_strikes, call_ttms)
            model_prices = np.concatenate([model_prices, heston_call_prices])
            mkt_prices = np.concatenate([mkt_prices, mkt_call_prices])
        if put_data.shape[1] > 0:
            mkt_put_prices, put_ttms, put_strikes = put_data[:, 0].astype(float), put_data[:, 1].astype(float), put_data[:, 2].astype(float)
            heston_put_prices = self.eu_put_price(S0, put_strikes, put_ttms)
            model_prices = np.concatenate([model_prices, heston_put_prices])
            mkt_prices = np.concatenate([mkt_prices, mkt_put_prices])
        mse = np.mean((model_prices - mkt_prices)**2)
        return mse 
    
    def calibrate_heston(self, S0, calib_opt_data, call_opt_id='call'):
        """
        Calibrate Heston model parameters by minimizing the MSE with observed call option prices.

        Parameters:
        -----------
        S0 : float
            Underlying asset price.
        calib_opt_data : np.array with shape (n, 4)
            Options to calibrate Heston model. Array columns should be ordered as follows: option price, option time to maturity, option strike, option type.
        call_opt_id: Any
            Parameter indicating how call options are identified in the 4th column of calib_opt_data

        Returns:
        --------
        OptimizeResult
            Result of optimization with calibrated parameters.
        """
        # Set initial guess
        x0 = (self.v0, self.vbar, self.rho_xv, self.kappa, self.gamma)
        # Define bounds
        bounds = ((0.005, 0.95), (0.01, 0.95), (-0.9, -0.1), (0.1, 5), (0.05, 0.9))
        # Set constraint: 8 * kappa * vbar < gamma**2, so that Lambda(t) > 0
        constraints = {'type': 'ineq', 'fun': lambda x: 8 * x[3] * x[1] - x[4]**2}       
        args = (S0, calib_opt_data, call_opt_id)
        # Run optimization
        optimization_result = minimize(self.objective_function_heston, x0=x0, args=args, bounds=bounds, constraints=[constraints], method='SLSQP', options={'maxiter': 50})
        # Save calibrated parameters
        param_keys = ['v0', 'vbar', 'rho_xv', 'kappa', 'gamma']
        param_values = optimization_result.x
        for i in range(len(param_keys)):
            setattr(self, param_keys[i], param_values[i])
        return optimization_result

    def simulate_paths(self, n_paths, dt, T, S0):
        """
        Simulate paths for the underlying asset price (S), variance (v), and interest rate (r)
        under the Heston Hull-White model using Euler discretization.

        Parameters:
        -----------
        n_paths : int
            Number of Monte Carlo simulation paths.
        dt : float
            Time increment for each step.
        T : float
            Total simulation time horizon.
        S0 : float
            Initial underlying asset price.

        Returns:
        --------
        tuple of numpy.ndarray
            Arrays containing simulated paths for (S, v, r), each shape (n_paths, n_steps + 1).
        """
        n_steps = int(T / dt)
        v0, vbar, kappa, gamma = self.v0, self.vbar, self.kappa, self.gamma
        rho_xr, rho_xv = self.rho_xr, self. rho_xv
        lmdb, theta, eta = self.lmdb, self.theta, self.eta
        r0 = self.r0

        X = np.full(shape=(n_paths, n_steps + 1), fill_value=np.log(S0))
        v = np.full(shape=(n_paths, n_steps + 1), fill_value=v0)
        r = np.full(shape=(n_paths, n_steps + 1), fill_value=r0)
        corr_matrix = np.array([[1, rho_xv, rho_xr], [rho_xv, 1, 0], [rho_xr, 0, 1]])
        L = np.linalg.cholesky(corr_matrix)

        for i in range(n_steps):
            Z = np.random.normal(0, 1, size=(3, n_paths))            
            W = L @ Z
            X[:, i+1] = X[:, i] + (r[:, i] - 0.5 * np.maximum(v[:, i], 0)) * dt + np.sqrt(np.maximum(v[:, i], 0) * dt) * W[0]
            v[:, i+1] = v[:, i] + kappa * (vbar - np.maximum(v[:, i], 0)) * dt + gamma * np.sqrt(np.maximum(v[:, i], 0) * dt) * W[1]
            r[:, i+1] = r[:, i] + lmdb * (theta - r[:, i]) * dt + eta * np.sqrt(dt) * W[2]

        S = np.exp(X)

        return S, v, r        
   