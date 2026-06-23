"""
ewc.py
======
Elastic Weight Consolidation (EWC) for Continual Learning.

Per the paper (Eq. 1 & 2):
  L_new = L_Dt+1(θ) + λ * Σ_i F_i * (θ_i - θ'_i)^2

  where F_i is the Fisher Information Matrix (diagonal approximation),
  θ'_i are the optimal parameters from the previous task,
  and λ is a task-specific regularization strength.
"""

import copy
import torch
import torch.nn as nn


class EWC:
    """
    Elastic Weight Consolidation regularizer.
    Computes the Fisher Information Matrix after each task and penalizes
    deviations from important parameters during subsequent training.
    """
    def __init__(self, model, lambda_ewc=1.0):
        """
        Args:
            model (nn.Module): The SDNN model to regularize.
            lambda_ewc (float): Regularization strength (λ in the paper).
        """
        self.model = model
        self.lambda_ewc = lambda_ewc

        # Stores accumulated Fisher information and optimal params from past tasks
        self.fisher_dict = {}
        self.optpar_dict = {}

    def compute_fisher(self, dataloader):
        """
        Computes the empirical Fisher Information Matrix (diagonal approximation)
        after finishing training on a task.

        Per paper: F_i = E[(∂ log P(y|x;θ) / ∂θ_i)^2]
        Approximated as: F ≈ ∇_θ L_MSE (empirical Fisher proxy)
        
        Args:
            dataloader: DataLoader with the task's training data.
        """
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}

        for X_batch, y_batch in dataloader:
            self.model.zero_grad()
            output = self.model(X_batch)
            # MSE loss as proxy for log-likelihood gradient
            loss = nn.functional.mse_loss(output.squeeze(), y_batch)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2

        # Average over dataset
        for n in fisher:
            fisher[n] /= len(dataloader)

        # Store Fisher and optimal parameters
        for n, p in self.model.named_parameters():
            self.fisher_dict[n] = fisher[n].clone()
            self.optpar_dict[n] = p.data.clone()

    def penalty(self):
        """
        Computes the EWC penalty term to add to the current task's loss.
        
        Returns:
            ewc_loss (torch.Tensor): λ * Σ_i F_i * (θ_i - θ'_i)^2
        """
        ewc_loss = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher_dict:
                ewc_loss += (self.fisher_dict[n] * (p - self.optpar_dict[n]) ** 2).sum()
        return self.lambda_ewc * ewc_loss


def train_with_ewc(model, optimizer, dataloader, ewc_module=None, 
                    replay_buffer=None, replay_ratio=0.3, epochs=50):
    """
    Standard training loop with optional EWC regularization and replay mixing.
    
    Args:
        model (nn.Module): The SDNN model.
        optimizer: PyTorch optimizer.
        dataloader: DataLoader for the current task.
        ewc_module (EWC or None): If provided, adds the EWC penalty to the loss.
        replay_buffer: ReplayBuffer or None. If provided, mixes replay samples
                       into each batch (paper: 27-30% of batch from buffer).
        replay_ratio (float): Fraction of batch drawn from replay buffer.
        epochs (int): Number of training epochs.
    """
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()

            # Mix replay samples into the batch (paper: 27-30%)
            if replay_buffer is not None and len(replay_buffer) > 0:
                replay_size = max(1, int(len(X_batch) * replay_ratio))
                X_replay, y_replay = replay_buffer.sample(replay_size)
                if X_replay is not None:
                    X_batch = torch.cat([X_batch, X_replay], dim=0)
                    y_batch = torch.cat([y_batch, y_replay], dim=0)

            # Standard MSE loss on new data
            preds = model(X_batch).squeeze()
            loss = nn.functional.mse_loss(preds, y_batch)

            # Add EWC penalty if we have consolidated knowledge
            if ewc_module is not None and len(ewc_module.fisher_dict) > 0:
                loss += ewc_module.penalty()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}  Loss: {epoch_loss / len(dataloader):.4f}")
