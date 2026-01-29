import torch
import torch.nn as nn
import torch.nn.functional as F

class LowRankMultimodalFusion(nn.Module):
    def __init__(self, input_dims, output_dim, rank):
        """
        Implementation of Low-Rank Multimodal Fusion (LMF).
        
        Args:
            input_dims (list of int): List containing dimensions of each modality (e.g., [128, 128] for image/text).
            output_dim (int): The final dimension for the prediction/classification task.
            rank (int): The number of rank-specific factors (r).
        """

        super(LowRankMultimodalFusion, self).__init__()
        
        self.output_dim = output_dim
        self.input_dims = input_dims
        self.rank = rank

        # We define modality-specific projection matrices W_m in R^(r x (d + 1))
        self.factors = nn.ParameterList([
          nn.Parameter(torch.Tensor(rank, dim+1, output_dim))  
          for dim in input_dims
        ])

        self.fusion_weights = nn.Parameter(torch.Tensor(1, rank))

        # Initializing Weights
        for factor in self.factors:
            nn.init.xavier_normal_(factor)
        nn.init.xavier_normal_(self.fusion_weights)

    def forward(self, modalities):
        """
        Forward propagation using the Low-Rank Tensor Factorization.
        
        Args:
            modalities (list of torch.Tensor): List of tensors from jWAE encoders.
        """
        batch_size = modalities[0].shape[0]
        # 1. Bias Augmentation: Append 1 to each latent vector
        # Resulting shape: (batch_size, dim + 1)

        augmented_modalities = []
        for m in modalities:
            ones = torch.ones(batch_size, 1).to(m.device)
            augmented_modalities.append(torch.cat([m, ones], dim=1))

        # 2. Parallel Rank-Specific Projection
        # We project each modality into the rank-space.
        # Efficiently compute the element-wise (Hadamard) product across modalities.
        fusion_res = None
        for i , m in enumerate(augmented_modalities):
            # (batch_size, dim + 1) @ (dim + 1, rank * output_dim) -> (batch_size, rank, output_dim)
            # We reshape the factor to allow batch multiplication across the rank
            weight = self.factors[i].view(self.input_dims[i] + 1, -1)
            projected = torch.matmul(m,weight).reshape(batch_size, self.rank, self.output_dim)

            if fusion_res == None:
                fusion_res = projected
            else:
                fusion_res = fusion_res * projected
        # 3. Final Summation across the Rank dimension
        # h = sum_{i=1}^{r} (W_weight * fused_factors)
        # Result shape: (batch_size, output_dim)
        final_fusion = torch.sum(fusion_res * self.fusion_weights.unsqueeze(-1) , dim = 1)

        return final_fusion