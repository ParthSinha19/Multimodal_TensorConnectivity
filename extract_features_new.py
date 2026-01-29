import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from transformers import BertTokenizer, BertModel
from PIL import Image
import json
import os
import tqdm
import kagglehub

# --- Configuration ---
# Reduced Batch Size to allow multiprocessing without crashing /dev/shm
BATCH_SIZE = 8 
NUM_WORKERS = 2  # Uses parallel loading (much faster than 0)
SAVE_FILE = "hateful_memes_features.pt"

class RawHatefulDataset(Dataset):
    def __init__(self, jsonl_path, img_dir):
        self.samples = []
        if not os.path.exists(jsonl_path):
             raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.img_dir = img_dir
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_name = sample['img'].split('/')[-1] if '/' in sample['img'] else sample['img']
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(image)
        except Exception:
            img_tensor = torch.zeros(3, 224, 224)

        return {
            'text': sample['text'],
            'image': img_tensor,
            'label': torch.tensor(sample['label']).float()
        }

def fast_extract():
    # 1. FORCE CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("❌ STOP! GPU not found. Do not run this on CPU.")
    
    device = torch.device('cuda')
    print(f"Acceleration: Using {torch.cuda.get_device_name(0)}")

    # 2. Locate Data
    raw_path = kagglehub.dataset_download("parthplc/facebook-hateful-meme-dataset")
    jsonl_path = os.path.join(raw_path, 'data', 'train.jsonl')
    img_dir = os.path.join(raw_path, 'data', 'img') 

    # 3. Load Models
    print("Loading Models to GPU...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased').to(device).eval()
    
    resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()
    resnet = resnet.to(device).eval()

    # 4. Loader with Workers
    dataset = RawHatefulDataset(jsonl_path, img_dir)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True # Faster transfer to GPU
    )

    extracted_data = []
    print(f"--- Starting GPU Extraction (Batch: {BATCH_SIZE}) ---")
    
    with torch.no_grad():
        for batch in tqdm.tqdm(loader):
            # Text
            text_inputs = tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            text_emb = bert(**text_inputs).last_hidden_state[:, 0, :] 

            # Image
            img_inputs = batch['image'].to(device)
            img_emb = resnet(img_inputs)

            # Save
            curr_batch_size = img_emb.size(0)
            for i in range(curr_batch_size):
                extracted_data.append({
                    'text': text_emb[i].cpu().clone(),
                    'image': img_emb[i].cpu().clone(),
                    'label': batch['label'][i].cpu()
                })

    torch.save(extracted_data, SAVE_FILE)
    print(f"DONE! Saved features to {SAVE_FILE}")

if __name__ == "__main__":
    fast_extract()