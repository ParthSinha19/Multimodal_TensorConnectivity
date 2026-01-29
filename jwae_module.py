import torch
import torch.nn as nn
import torch.nn.functional as F

class JWAE(nn.Module):
    def __init__(self, input_dims, latent_dim):
        """
        Joint Wasserstein Autoencoder for Multimodal Alignment.
        
        Args:
            input_dims (dict): e.g., {'image': 4096, 'text': 300}
            latent_dim (int): The shared latent space dimension (z).
        """

        super(JWAE, self).__init__()

        # 1. Modality-Specific Encoders (f_v, f_t)
        # These map raw high-dimensional data to the latent space

        self.encoders = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(dim, 512),
                nn.ReLU(),
                nn.Linear(512, latent_dim)
            )for m, dim in input_dims.items()
        })

        #2. Modality-Specific Decoders
        self.decoders = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(latent_dim, 512),
                nn.ReLU(),
                nn.Linear(512, dim)
            ) for m , dim in input_dims.items()
        })

    def forward(self, x_dict):
        """
        Forward pass producing latent embeddings for each modality.
        """

        latents = {}
        reconstructions = {}

        for m, x in x_dict.items():
            z = self.encoders[m](x)
            latents[m] = z
            reconstructions[m] = self.decoders[m](z)

        return latents, reconstructions

    def compute_loss(self, x_dict, latents, reconstructions, lambda_wae=10.0):
        """
        Implements the jWAE objective: Reconstruction + Joint Wasserstein Regularization.
        Ensures latent embeddings lie on a smooth, continuous manifold[cite: 60].
        """
        recon_loss = 0

        for m in x_dict.keys():
            recon_loss += F.mse_loss(reconstructions[m], x_dict[m])

        # MMD (Maximum Mean Discrepancy) as a proxy for the Wasserstein distance
        # This enforces the shared Gaussian prior

        mmd_loss = 0
        for m, z in latents.items():
            prior_z = torch.randn_like(z) # Standard Gaussian Prior
            mmd_loss += self._mmd_kernel(z, prior_z)

        return recon_loss + (lambda_wae * mmd_loss)

    def _mmd_kernel(self, z, prior_z, sigma=2.0):
        """Helper to compute MMD with an RBF kernel."""
        # Simple RBF kernel implementation for geometric regularization 
        def rbf_kernel(x, y):
            dist = torch.cdist(x,y).pow(2)
            return torch.exp(-dist / (2 * sigma**2)).mean()
        return rbf_kernel(z, z) + rbf_kernel(prior_z, prior_z) - 2 * rbf_kernel(z, prior_z)
        