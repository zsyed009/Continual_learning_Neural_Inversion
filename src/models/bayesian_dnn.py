import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianLinear(nn.Module):
    """
    A single Bayesian Linear layer using the reparameterization trick.
    Instead of fixed weights, it learns the mean (mu) and variance (rho) of a Gaussian distribution.
    """
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5.0, -4.0))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5.0, -4.0))

    def forward(self, x, sample=True):
        """
        Forward pass. 
        If sample=True, we sample from the distribution for training/uncertainty.
        If sample=False, we just use the mean (mu) for deterministic prediction.
        """
        if sample:
            # Reparameterization trick: W = mu + sigma * epsilon
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            weight_epsilon = torch.randn_like(weight_sigma)
            weight = self.weight_mu + weight_sigma * weight_epsilon
            
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_epsilon = torch.randn_like(bias_sigma)
            bias = self.bias_mu + bias_sigma * bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)
        
    def kl_divergence(self, prior_mu=0.0, prior_sigma=1.0):
        """
        Computes the KL divergence between the current approximate posterior (q) and a Gaussian prior (p).
        This is a core component of the Evidence Lower Bound (ELBO) in VCL.
        """
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        # KL(q || p) for weights
        kl_weight = 0.5 * torch.sum(
            2 * math.log(prior_sigma) - 2 * torch.log(weight_sigma) 
            + (weight_sigma ** 2 + (self.weight_mu - prior_mu) ** 2) / (prior_sigma ** 2) - 1
        )
        
        # KL(q || p) for biases
        kl_bias = 0.5 * torch.sum(
            2 * math.log(prior_sigma) - 2 * torch.log(bias_sigma) 
            + (bias_sigma ** 2 + (self.bias_mu - prior_mu) ** 2) / (prior_sigma ** 2) - 1
        )
        
        return kl_weight + kl_bias


# ==========================================
# 3. BDNN (Bayesian Deep Neural Network)
# ==========================================
class BDNN(nn.Module):
    """
    Bayesian Deep Neural Network (Model 3 in the paper).
    Architecture matches SDNN: 3 hidden layers, 256 neurons each, 
    but uses BayesianLinear layers to treat weights and biases as probability distributions.
    This serves as the core architecture for VCL and EVCL.
    """
    def __init__(self, input_dim):
        super(BDNN, self).__init__()
        
        self.fc1 = BayesianLinear(input_dim, 256)
        self.fc2 = BayesianLinear(256, 256)
        self.fc3 = BayesianLinear(256, 256)
        self.fc4 = BayesianLinear(256, 1)

    def forward(self, x, sample=True):
        x = F.relu(self.fc1(x, sample))
        x = F.relu(self.fc2(x, sample))
        x = F.relu(self.fc3(x, sample))
        x = self.fc4(x, sample)
        return x
        
    def get_kl_divergence(self, prior_mu=0.0, prior_sigma=1.0):
        """
        Sums the KL divergence across all Bayesian layers.
        Used when computing the VCL ELBO loss.
        """
        kl = 0.0
        kl += self.fc1.kl_divergence(prior_mu, prior_sigma)
        kl += self.fc2.kl_divergence(prior_mu, prior_sigma)
        kl += self.fc3.kl_divergence(prior_mu, prior_sigma)
        kl += self.fc4.kl_divergence(prior_mu, prior_sigma)
        return kl
