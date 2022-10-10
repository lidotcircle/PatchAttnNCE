import torch


def gaussian_reparameterization(mu, logvar):
    mu_shape = mu.shape
    mu = mu.view(mu.size(0), -1)
    logvar = logvar.view(logvar.size(0), -1)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    ans = mu + eps * std
    return ans.view(mu_shape)

def univariate_gaussian_KLD(mu, logvar):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))