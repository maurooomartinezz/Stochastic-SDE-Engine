"""
Stochastic Differential Equation (SDE) Engine
=============================================
Author: Mauro Martínez
Description: A vectorized high-performance Python engine for simulating continuous-time 
             stochastic processes (Geometric Brownian Motion & Ornstein-Uhlenbeck).
"""

from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np


class StochasticEngine:
    """Vectorized numerical simulator for continuous-time stochastic differential equations."""

    def __init__(self, seed: int | None = 42) -> None:
        """
        Initialize the random number generator.
        Setting a seed guarantees that every time you run the script, the 'random'
        dice rolls are identical so your results are 100% reproducible.
        """
        if seed is not None:
            np.random.seed(seed)  # Lock in NumPy's pseudo-random number generator state

    def simulate_gbm(
        self,
        S0: float,
        mu: float,
        sigma: float,
        T: float,
        steps: int,
        paths: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulates Geometric Brownian Motion (GBM) using the exact analytical solution derived from Itô's Lemma:
            dS_t = μ * S_t * dt + σ * S_t * dW_t
            S_t  = S_0 * exp((μ - 0.5 * σ^2) * t + σ * W_t)
        """
        # Step 1: Calculate time increment dt (e.g., 1 year / 252 days = 0.00396 years per step)
        dt = T / steps
        
        # Step 2: Build a 1D timeline array from t = 0 to t = T with (steps + 1) points
        t = np.linspace(0, T, steps + 1)

        # Step 3: Roll standard normal dice Z ~ N(0, 1) for all time steps and paths at once
        # Creates a 2D matrix of shape (steps, paths) -> e.g., 252 rows by 1,000 columns
        Z = np.random.normal(loc=0.0, scale=1.0, size=(steps, paths))

        # Step 4: Scale dice rolls by sqrt(dt) to construct Brownian motion increments (dW = sqrt(dt) * Z)
        dW = np.sqrt(dt) * Z

        # Step 5: Stack a row of zeros at t=0 and take the cumulative sum down rows (axis=0)
        # This converts step-by-step shocks into continuous Brownian trajectories W_t
        W = np.vstack([np.zeros(paths), np.cumsum(dW, axis=0)])

        # Step 6: Reshape 1D time array 't' from (253,) to a 2D column vector (253, 1) using np.newaxis
        # This lets NumPy broadcast time across all 1,000 universe paths simultaneously
        time_grid = t[:, np.newaxis]

        # Step 7: Compute predictable growth (drift) including the -0.5 * sigma^2 volatility drag correction
        drift = (mu - 0.5 * sigma**2) * time_grid

        # Step 8: Compute the random diffusion impact by scaling accumulated noise W by volatility sigma
        diffusion = sigma * W

        # Step 9: Plug combined drift and diffusion into e^x and scale by initial price S0
        S = S0 * np.exp(drift + diffusion)

        # Return the time timeline and the 2D matrix of simulated stock price paths
        return t, S

    def simulate_ou(
        self,
        X0: float,
        theta: float,
        mu: float,
        sigma: float,
        T: float,
        steps: int,
        paths: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulates an Ornstein-Uhlenbeck (OU) Mean-Reverting Process via Euler-Maruyama discretization:
            dX_t = θ * (μ - X_t) * dt + σ * dW_t
        """
        # Step 1: Calculate the size of each discrete time step dt
        dt = T / steps

        # Step 2: Create a 1D time array covering 0 to T
        t = np.linspace(0, T, steps + 1)

        # Step 3: Allocate a blank matrix filled with zeros of shape (steps + 1, paths) to store values
        X = np.zeros((steps + 1, paths))

        # Step 4: Set the initial starting value X0 across all 1,000 paths on row 0
        X[0] = X0

        # Step 5: Pre-draw all random Gaussian noise Z ~ N(0, 1) for the entire simulation upfront
        Z = np.random.normal(loc=0.0, scale=1.0, size=(steps, paths))

        # Step 6: Loop through time step-by-step (Euler-Maruyama loop)
        for i in range(steps):
            # Calculate restoring force (drift): pulls positive if X[i] < mu, negative if X[i] > mu
            drift_step = theta * (mu - X[i]) * dt

            # Calculate random thermal perturbation for this specific time step
            diffusion_step = sigma * np.sqrt(dt) * Z[i]

            # Update tomorrow's value (row i + 1) for all 1,000 paths simultaneously
            X[i + 1] = X[i] + drift_step + diffusion_step

        # Return the timeline array and the simulated spread trajectories
        return t, X


def plot_simulation_results(
    t: np.ndarray,
    paths: np.ndarray,
    title: str,
    ylabel: str,
    mean_reversion_level: float | None = None,
) -> None:
    """Utility function to visualize simulation trajectories and ensemble statistics."""
    # Create a figure canvas with dimensions 12 inches wide by 6 inches tall
    plt.figure(figsize=(12, 6))

    # Plot only the first 50 paths (columns) out of 1,000 with thin lines (lw=0.8) and 50% opacity (alpha=0.5)
    # This prevents the plot from becoming a solid illegible block of color
    plt.plot(t, paths[:, :50], lw=0.8, alpha=0.5)

    # Compute the average value across ALL 1,000 paths for every time step (averaging across columns, axis=1)
    ensemble_mean = np.mean(paths, axis=1)

    # Plot the Monte Carlo mean path as a thick, solid black line
    plt.plot(
        t,
        ensemble_mean,
        color="black",
        linewidth=2.5,
        label=f"Monte Carlo Mean (N={paths.shape[1]:,})",
    )

    # If a mean-reversion level is provided (e.g. for OU process), draw a horizontal red dashed line
    if mean_reversion_level is not None:
        plt.axhline(
            y=mean_reversion_level,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Long-Term Equilibrium (μ = {mean_reversion_level})",
        )

    # Format title, axis labels, grid lines, and legend
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Time Horizon (Years)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)  # Add faint dashed background grid
    plt.legend(loc="upper left", frameon=True) # Display legend box in top-left corner
    plt.tight_layout()                          # Adjust margins so labels don't get clipped
    plt.show()                                  # Display graph window


def show_mauro_out_banner() -> None:
    """Displays the custom dark-mode sign-off signature upon completion."""
    # Create a figure and axis object with a dark GitHub background (#0d1117)
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")  # Match inner plot area color to the outer figure

    # Render main title "MAURO OUT" in large bright blue text centered at coordinates (x=0.5, y=0.6)
    ax.text(
        0.5,
        0.6,
        "MAURO OUT",
        fontsize=46,
        fontweight="bold",
        color="#58a6ff",          # GitHub dark theme accent blue
        ha="center",              # Horizontal alignment centered
        va="center",              # Vertical alignment centered
        family="sans-serif",
    )

    # Render subtitle signature in lighter grey italic text centered at coordinates (x=0.5, y=0.3)
    ax.text(
        0.5,
        0.3,
        "Quantitative Engineering & Stochastic Engine • Mauro Martínez",
        fontsize=13,
        color="#8b949e",          # Muted grey text color
        ha="center",
        va="center",
        style="italic",
    )

    # Turn off graph axes, ticks, and border lines since this is a visual banner rather than a data chart
    ax.axis("off")

    # Fit content neatly inside the canvas and render the banner window
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Instantiate the stochastic simulation engine with seed 42
    sim_engine = StochasticEngine(seed=42)

    # 1. Run Geometric Brownian Motion Simulation
    print("[+] Executing Vectorized Geometric Brownian Motion Simulation...")
    t_gbm, gbm_trajectories = sim_engine.simulate_gbm(
        S0=100.0,   # Initial stock price ($100)
        mu=0.08,    # 8% annual expected drift
        sigma=0.20, # 20% annualized volatility
        T=1.0,      # 1 Year time horizon
        steps=252,  # 252 daily trading steps
        paths=1000  # 1,000 parallel paths
    )
    # Plot GBM results
    plot_simulation_results(
        t_gbm, gbm_trajectories, "Geometric Brownian Motion (Asset Dynamics)", "Price ($)"
    )

    # 2. Run Ornstein-Uhlenbeck Mean-Reversion Simulation
    print("[+] Executing Vectorized Ornstein-Uhlenbeck Mean-Reversion Simulation...")
    t_ou, ou_trajectories = sim_engine.simulate_ou(
        X0=5.0,     # Initial spread value ($5.00)
        theta=3.0,  # Mean-reversion speed (strong restoring force)
        mu=1.5,     # Target equilibrium level ($1.50)
        sigma=0.5,  # Volatility of spread
        T=2.0,      # 2 Year time horizon
        steps=500,  # 500 discrete evaluation steps
        paths=1000  # 1,000 parallel paths
    )
    # Plot OU results with target line at 1.5
    plot_simulation_results(
        t_ou,
        ou_trajectories,
        "Ornstein-Uhlenbeck Process (Mean-Reverting Spread)",
        "Spread Value ($)",
        mean_reversion_level=1.5,
    )

    # 3. Launch Custom Sign-Off Screen
    print("[+] Simulation Complete. Launching Signature Banner...")
    show_mauro_out_banner()