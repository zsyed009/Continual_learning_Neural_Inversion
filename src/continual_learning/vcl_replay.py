"""
vcl_replay.py
=============
Variational Continual Learning (VCL) with Replay Buffer.

Per the paper (Eq. 3):
  L^t_VCL(q_t(θ)) = Σ_n E[log p(y|θ,x)] - KL(q_t(θ) || q_{t-1}(θ))

The VCL framework uses the Bayesian DNN (BDNN) and:
  1. Treats the previous task's posterior as the new prior.
  2. Maximizes the ELBO (expected log-likelihood minus KL divergence).
  3. Augments each training batch with replay buffer samples (27-30%).

Also implements EVCL (Eq. 4) which adds an EWC penalty on top of VCL.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.bayesian_dnn import BDNN, BayesianLinear
from continual_learning.buffer import ReplayBuffer


# ==========================================
# 1. VCL LOSS (ELBO)
# ==========================================

def compute_elbo_loss(model, X, y, num_samples_mc=5, beta=1.0):
    """
    Computes the Evidence Lower Bound (ELBO) loss for the BDNN.

    ELBO = E_q[log p(y|x,θ)] - β * KL(q(θ) || p(θ))

    Args:
        model (BDNN): The Bayesian DNN.
        X (torch.Tensor): Input features.
        y (torch.Tensor): Target CSO volumes.
        num_samples_mc (int): Number of Monte Carlo samples for expectation.
        beta (float): Task-specific scaling for KL divergence term.
        
    Returns:
        loss (torch.Tensor): Negative ELBO (to be minimized).
    """
    # Monte Carlo estimate of expected log-likelihood
    log_likelihood = 0.0
    for _ in range(num_samples_mc):
        preds = model(X, sample=True).squeeze()
        # Gaussian log-likelihood ∝ -MSE
        log_likelihood += -F.mse_loss(preds, y, reduction='sum')
    log_likelihood /= num_samples_mc

    # KL divergence between current posterior and prior
    kl = model.get_kl_divergence()

    # Negative ELBO = -log_likelihood + β * KL
    loss = -log_likelihood + beta * kl
    return loss


# ==========================================
# 2. PRIOR UPDATE (Task Transition)
# ==========================================

def update_prior(model):
    """
    After training on a task, snapshots the current posterior parameters
    (mu, rho) to serve as the prior for the next task.
    
    Per paper: "For each new task, it uses the accumulated approximate 
    posteriors from previous tasks as a prior."
    
    Returns:
        prior_params: dict mapping layer_name -> (mu_copy, rho_copy)
    """
    prior_params = {}
    for name, module in model.named_modules():
        if isinstance(module, BayesianLinear):
            prior_params[name] = {
                'weight_mu': module.weight_mu.data.clone(),
                'weight_rho': module.weight_rho.data.clone(),
                'bias_mu': module.bias_mu.data.clone(),
                'bias_rho': module.bias_rho.data.clone(),
            }
    return prior_params


def compute_kl_from_prior(model, prior_params):
    """
    Computes KL(q_t(θ) || q_{t-1}(θ)) using stored prior parameters,
    instead of the standard unit Gaussian prior.
    
    This is the key VCL mechanism: the posterior from the previous task
    becomes the prior for the current task.
    """
    kl = 0.0
    for name, module in model.named_modules():
        if isinstance(module, BayesianLinear) and name in prior_params:
            prior = prior_params[name]

            # Weight KL
            sigma_q = torch.log1p(torch.exp(module.weight_rho))
            sigma_p = torch.log1p(torch.exp(prior['weight_rho']))
            mu_q = module.weight_mu
            mu_p = prior['weight_mu']

            kl += 0.5 * torch.sum(
                2 * torch.log(sigma_p) - 2 * torch.log(sigma_q)
                + (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (sigma_p ** 2) - 1
            )

            # Bias KL
            sigma_q_b = torch.log1p(torch.exp(module.bias_rho))
            sigma_p_b = torch.log1p(torch.exp(prior['bias_rho']))
            mu_q_b = module.bias_mu
            mu_p_b = prior['bias_mu']

            kl += 0.5 * torch.sum(
                2 * torch.log(sigma_p_b) - 2 * torch.log(sigma_q_b)
                + (sigma_q_b ** 2 + (mu_q_b - mu_p_b) ** 2) / (sigma_p_b ** 2) - 1
            )

    return kl


# ==========================================
# 3. EVCL PENALTY (Eq. 4)
# ==========================================

def compute_evcl_penalty(model, prior_params, fisher_dict, lambda_evcl=1.0):
    """
    Computes the EVCL elastic penalty on top of VCL.
    
    Per paper (Eq. 4):
      Σ_i (λ/2) * F_{t-1}_i * [(μ_t,i - μ_{t-1,i})^2 + (σ^2_t,i - σ^2_{t-1,i})]
      
    Args:
        model (BDNN): Current model.
        prior_params (dict): Snapshot of previous task's posterior.
        fisher_dict (dict): Fisher Information computed on previous task's mu parameters.
        lambda_evcl (float): Elastic penalty weight.
    """
    penalty = 0.0
    for name, module in model.named_modules():
        if isinstance(module, BayesianLinear) and name in prior_params:
            prior = prior_params[name]
            fisher_key = name

            if fisher_key in fisher_dict:
                f_weight = fisher_dict[fisher_key]['weight']
                f_bias   = fisher_dict[fisher_key]['bias']

                # Mean shift penalty
                mu_diff_w = (module.weight_mu - prior['weight_mu']) ** 2
                mu_diff_b = (module.bias_mu - prior['bias_mu']) ** 2

                # Variance shift penalty
                sigma_q_w = torch.log1p(torch.exp(module.weight_rho)) ** 2
                sigma_p_w = torch.log1p(torch.exp(prior['weight_rho'])) ** 2
                sigma_q_b = torch.log1p(torch.exp(module.bias_rho)) ** 2
                sigma_p_b = torch.log1p(torch.exp(prior['bias_rho'])) ** 2

                penalty += (lambda_evcl / 2) * torch.sum(f_weight * (mu_diff_w + (sigma_q_w - sigma_p_w)))
                penalty += (lambda_evcl / 2) * torch.sum(f_bias * (mu_diff_b + (sigma_q_b - sigma_p_b)))

    return penalty


def compute_fisher_bdnn(model, dataloader):
    """
    Computes the empirical Fisher on the mu parameters of a BDNN.
    Used by EVCL: F = ∇_μ L_MSE
    """
    model.eval()
    fisher = {}
    for name, module in model.named_modules():
        if isinstance(module, BayesianLinear):
            fisher[name] = {
                'weight': torch.zeros_like(module.weight_mu),
                'bias':   torch.zeros_like(module.bias_mu),
            }

    for X_batch, y_batch in dataloader:
        model.zero_grad()
        output = model(X_batch, sample=False).squeeze()
        loss = F.mse_loss(output, y_batch)
        loss.backward()

        for name, module in model.named_modules():
            if isinstance(module, BayesianLinear):
                if module.weight_mu.grad is not None:
                    fisher[name]['weight'] += module.weight_mu.grad.data ** 2
                if module.bias_mu.grad is not None:
                    fisher[name]['bias'] += module.bias_mu.grad.data ** 2

    for name in fisher:
        fisher[name]['weight'] /= len(dataloader)
        fisher[name]['bias']   /= len(dataloader)

    return fisher


# ==========================================
# 4. VCL+REPLAY TRAINING LOOP
# ==========================================

def train_vcl_with_replay(
    model, 
    dataloader, 
    replay_buffer, 
    prior_params=None,
    beta=1.0, 
    replay_ratio=0.3,
    epochs=50, 
    lr=1e-3,
    fisher_dict=None,
    lambda_evcl=0.0,
):
    """
    Trains the BDNN on a single task using the VCL objective with replay.
    Optionally adds the EVCL penalty if fisher_dict and lambda_evcl > 0.

    Args:
        model (BDNN): The Bayesian DNN.
        dataloader: DataLoader for the current task.
        replay_buffer (ReplayBuffer): Episodic memory buffer.
        prior_params (dict or None): Previous task's posterior (None for baseline).
        beta (float): Task-specific KL scaling coefficient.
        replay_ratio (float): Fraction of batch drawn from replay (paper: 0.27-0.30).
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        fisher_dict (dict or None): Fisher for EVCL penalty.
        lambda_evcl (float): EVCL penalty weight (0 = pure VCL).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()

            # --- Mix replay samples into the batch (paper: 27-30%) ---
            if len(replay_buffer) > 0:
                replay_size = max(1, int(len(X_batch) * replay_ratio))
                X_replay, y_replay = replay_buffer.sample(replay_size)

                if X_replay is not None:
                    X_batch = torch.cat([X_batch, X_replay], dim=0)
                    y_batch = torch.cat([y_batch, y_replay], dim=0)

            # --- Compute VCL ELBO ---
            # Monte Carlo expected log-likelihood
            log_likelihood = 0.0
            mc_samples = 5
            for _ in range(mc_samples):
                preds = model(X_batch, sample=True).squeeze()
                log_likelihood += -F.mse_loss(preds, y_batch, reduction='sum')
            log_likelihood /= mc_samples

            # KL divergence against prior
            if prior_params is not None:
                kl = compute_kl_from_prior(model, prior_params)
            else:
                kl = model.get_kl_divergence()

            loss = -log_likelihood + beta * kl

            # --- Add EVCL elastic penalty if configured ---
            if lambda_evcl > 0 and fisher_dict is not None and prior_params is not None:
                loss += compute_evcl_penalty(model, prior_params, fisher_dict, lambda_evcl)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}  ELBO Loss: {epoch_loss / len(dataloader):.4f}")
