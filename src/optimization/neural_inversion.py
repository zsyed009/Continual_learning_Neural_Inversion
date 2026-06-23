"""
neural_inversion.py
===================
Neural Inversion-Based Real-Time Control Optimization.

Per the paper (Section 2.2.5):
  Instead of updating model weights (θ), the gradient of the cost (CSO)
  is backpropagated to the INPUT layer (x) to iteratively update the
  control parameters until CSO is minimized.

  x* = arg min_x  C(f_θ(x))
  x_new = x_old - η · ∇_x C
  ∇_x C = ∂C/∂f_θ(x) · ∂f_θ(x)/∂x

Implementation specifics (Section 2.4):
  - Adam optimizer, η = 0.002
  - Max 200 iterations with early stopping
  - Actions clipped to [0, 1] at each iteration (physical feasibility)
"""

import torch
import torch.nn as nn
import numpy as np


class NeuralInversionOptimizer:
    """
    Gradient-based optimizer that freezes the trained surrogate model and
    optimizes the control actions (gate openings + pump settings) by
    backpropagating the CSO cost through the model to the input layer.
    """

    def __init__(
        self,
        model,
        precipitation,
        lr=0.002,
        max_iters=200,
        early_stop_patience=20,
        early_stop_tol=1e-4,
        is_bayesian=False,
    ):
        """
        Args:
            model (nn.Module): Trained surrogate (SDNN or BDNN). Will be frozen.
            precipitation (np.ndarray): Fixed rainfall array for the event.
            lr (float): Learning rate for Adam optimizer (paper: η = 0.002).
            max_iters (int): Maximum optimization iterations (paper: 200).
            early_stop_patience (int): Stop if no improvement for N iterations.
            early_stop_tol (float): Minimum improvement threshold.
            is_bayesian (bool): If True, uses deterministic forward (sample=False).
        """
        self.model = model
        self.model.eval()
        # Freeze all model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.precipitation = torch.tensor(precipitation, dtype=torch.float32)
        self.lr = lr
        self.max_iters = max_iters
        self.early_stop_patience = early_stop_patience
        self.early_stop_tol = early_stop_tol
        self.is_bayesian = is_bayesian
        self.num_rain_features = len(precipitation)
        self.num_actions = 10  # 7 gates + 3 pumps

    def optimize(self, init_actions=None):
        """
        Runs the neural inversion optimization loop.

        Per paper: "the control gates were initialized to completely random
        states" and "the optimized input vector was strictly bounded at each
        iteration using value clipping within the [0,1] fractional range."

        Args:
            init_actions (np.ndarray or None): Initial action vector [10].
                If None, randomly initialized in [0, 1].

        Returns:
            best_actions (np.ndarray): Optimized control actions [10].
            best_cso (float): Predicted CSO under the optimized actions.
            history (list): Per-iteration CSO values for convergence tracking.
        """
        # Initialize actions as a learnable parameter
        if init_actions is not None:
            actions = torch.tensor(init_actions, dtype=torch.float32)
        else:
            actions = torch.rand(self.num_actions)

        # Make actions a leaf tensor that requires gradients
        actions = actions.clone().detach().requires_grad_(True)

        # Adam optimizer on the action vector only (paper: η = 0.002)
        optimizer = torch.optim.Adam([actions], lr=self.lr)

        best_cso = float('inf')
        best_actions = actions.data.clone()
        patience_counter = 0
        history = []

        for iteration in range(self.max_iters):
            optimizer.zero_grad()

            # Construct full input: [precipitation | actions]
            # Precipitation is fixed; only actions are optimized
            full_input = torch.cat([self.precipitation, actions]).unsqueeze(0)

            # Forward pass through frozen model
            if self.is_bayesian:
                predicted_cso = self.model(full_input, sample=False).squeeze()
            else:
                predicted_cso = self.model(full_input).squeeze()

            # Cost = predicted CSO volume (minimize this)
            cost = predicted_cso

            # Backpropagate ∂C/∂x through the frozen model to the input actions
            cost.backward()

            # Update actions via Adam
            optimizer.step()

            # Enforce physical constraints: clip actions to [0, 1]
            with torch.no_grad():
                actions.clamp_(0.0, 1.0)

            current_cso = cost.item()
            history.append(current_cso)

            # Track best solution
            if current_cso < best_cso - self.early_stop_tol:
                best_cso = current_cso
                best_actions = actions.data.clone()
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.early_stop_patience:
                break

        return best_actions.numpy(), best_cso, history

    def optimize_multi_start(self, n_starts=100):
        """
        Runs the neural inversion from multiple random initializations.

        Per paper (Section 3.3): "The neural inversion process was executed
        100 independent times per event from these random starting points."

        Args:
            n_starts (int): Number of independent random starts (paper: 100).

        Returns:
            all_actions (list of np.ndarray): Optimized actions per trial.
            all_cso_predicted (list of float): Predicted CSO per trial.
            all_histories (list of list): Convergence histories.
        """
        all_actions = []
        all_cso_predicted = []
        all_histories = []

        for trial in range(n_starts):
            # Random initialization for each trial
            init = np.random.uniform(0.0, 1.0, size=self.num_actions)
            opt_actions, pred_cso, hist = self.optimize(init_actions=init)

            all_actions.append(opt_actions)
            all_cso_predicted.append(pred_cso)
            all_histories.append(hist)

            if (trial + 1) % 25 == 0:
                print(f"    Trial {trial+1}/{n_starts}  "
                      f"Predicted CSO: {pred_cso:.2f}")

        return all_actions, all_cso_predicted, all_histories
