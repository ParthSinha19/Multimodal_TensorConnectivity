import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from transformers import BertTokenizer, BertModel

class HatefulMemesDataset(Dataset):
    def __init__(self, jsonl_path, img_dir):
        """
        Real Hateful Memes Data Loader.
        Prepares raw image and text data for geometric conditioning via jWAE[cite: 59, 63].
        """
        # 1. Load labels and metadata [cite: 63]
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"JSONL file not found at: {jsonl_path}")
            
        with open(jsonl_path, 'r') as f:
            self.samples = [json.loads(line) for line in f]
        self.img_dir = img_dir
        
        # 2. Extractors: Turn raw data into high-dimensional vectors 
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased').eval()
        
        # ResNet50 for image features (2048-dimensional vector) 
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = torch.nn.Identity() # Remove classifier layer 
        self.resnet.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Text -> BERT Vector (768 dimensions) 
        inputs = self.tokenizer(sample['text'], return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.bert(**inputs)
            # Use CLS token as the static vector representation for jWAE conditioning [cite: 18, 63]
            text_emb = outputs.last_hidden_state[:, 0, :].squeeze()

        # Image -> ResNet Vector (2048 dimensions) 
        # Handling the image path structure from the Kaggle download
        img_name = sample['img'].split('/')[-1] if '/' in sample['img'] else sample['img']
        img_path = os.path.join(self.img_dir, 'img', img_name)
        
        if not os.path.exists(img_path):
            # Fallback check for different folder structures
            img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            img_emb = self.resnet(img_tensor).squeeze()

        # Return aligned modalities ready for joint training 
        return {
            'text': text_emb, 
            'image': img_emb, 
            'label': torch.tensor(sample['label']).float()
        }

def get_dataloaders(real_path=None, batch_size=32):
    """
    Factory function to initialize the dataset and dimensions for the main pipeline.
    Ensures a valid dataset object is always returned to the DataLoader.
    """
    if real_path and os.path.exists(real_path['jsonl']):
        # Initialize the real Hateful Memes Dataset [cite: 91, 94]
        dataset = HatefulMemesDataset(real_path['jsonl'], real_path['img_dir'])
        print(f"Loaded REAL Hateful Memes Dataset from {real_path['jsonl']}")
    else:
        # Fallback to Mock dataset if real path is missing or invalid [cite: 16, 88]
        dataset = MockHatefulMemesDataset(size=100)
        print("Real path invalid or not provided. Loading MOCK dataset for testing.")
        
    # Dimensions remain consistent for BERT (768) and ResNet (2048) [cite: 10, 82]
    dims = {'image': 2048, 'text': 768}
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dims