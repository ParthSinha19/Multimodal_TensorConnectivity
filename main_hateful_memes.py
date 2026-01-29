import torch
import torch.nn as nn
import torch.optim as optim
import os
import kagglehub

# Assuming these are saved as separate local .py files
from lmf_module import LowRankMultimodalFusion
from jwae_module import JWAE
from hateful_memes_loader import get_dataloaders

# --- 1. Configuration ---
USE_REAL_DATA = True
raw_path = kagglehub.dataset_download("parthplc/facebook-hateful-meme-dataset")
REAL_PATH = {
    'jsonl': os.path.join(raw_path, 'data', 'train.jsonl'), 
    'img_dir': os.path.join(raw_path, 'data') 
}
BATCH_SIZE = 32
LATENT_DIM = 128  # Shared Gaussian manifold dimension 
RANK = 8          # Rank factors for interaction approximation 
EPOCHS = 10
LEARNING_RATE = 1e-4

# Set up Device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 2. Setup Data ---
train_loader, input_dims = get_dataloaders(
    real_path=REAL_PATH if USE_REAL_DATA else None, 
    batch_size=BATCH_SIZE
)

# --- 3. Model Architecture ---
class HatefulMemesPipeline(nn.Module):
    def __init__(self, input_dims, latent_dim, rank):
        super(HatefulMemesPipeline, self).__init__()
        # jWAE for geometric conditioning and linearization [cite: 18, 59]
        self.jwae = JWAE(input_dims=input_dims, latent_dim=latent_dim)
        
        # LMF for additive interaction modeling in low-rank space [cite: 43, 51]
        # input_dims here refers to the latent dimensions produced by jWAE
        self.lmf = LowRankMultimodalFusion(
            input_dims=[latent_dim, latent_dim], 
            output_dim=1, 
            rank=rank
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x_dict):
        # Step A: Map raw inputs to shared latent vectors z_v, z_t 
        latents, recons = self.jwae(x_dict)
        
        # Step B: Perform low-rank fusion on the conditioned latents [cite: 70]
        # We pass a list of vectors: [z_image, z_text]
        fusion_out = self.lmf([latents['image'], latents['text']])
        
        return self.sigmoid(fusion_out), recons, latents

# Initialize Pipeline
model = HatefulMemesPipeline(input_dims, LATENT_DIM, RANK).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion_cls = nn.BCELoss()

# --- 4. Training Loop ---
print(f"Starting Training: Rank={RANK}, Latent={LATENT_DIM}")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        # A. Filter and Move Data to Device
        # jWAE expects only feature modalities, not the 'label' 
        x_dict = {
            'image': batch['image'].to(device),
            'text': batch['text'].to(device)
        }
        labels = batch['label'].to(device).unsqueeze(1)
        
        # B. Forward Pass
        preds, recons, latents = model(x_dict)
        
        # C. Hybrid Loss Calculation
        # 1. Classification Accuracy (Task specific)
        loss_cls = criterion_cls(preds, labels)
        
        # 2. Geometric Regularization (jWAE Alignment + Prior) [cite: 13, 84]
        # Weighted higher initially to ensure linearization [cite: 85]
        loss_jwae = model.jwae.compute_loss(x_dict, latents, recons)
        
        # Combined Loss: Balancing task accuracy and geometric faithfulness [cite: 97]
        loss = loss_cls + (0.05 * loss_jwae)
        
        # D. Backpropagation
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # E. Metrics Tracking
        predicted_labels = (preds > 0.5).float()
        correct += (predicted_labels == labels).sum().item()
        total += labels.size(0)
        
    avg_loss = total_loss / len(train_loader)
    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f} - Accuracy: {acc:.2f}%")

print("\n--- Training Complete ---")

# --- 5. Interpretability Snapshot ---
# Validating the hypothesis that rank factors correspond to semantic interactions [cite: 24]
print("\nMechanistic Transparency Analysis:")
print("Global Fusion Weights (Significance of each Rank Factor):")
# Higher weights indicate the specific 'interaction block' is more influential [cite: 55, 56]
print(model.lmf.fusion_weights.data.cpu().numpy())