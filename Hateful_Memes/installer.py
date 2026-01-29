import kagglehub

# Download latest version
path = kagglehub.dataset_download("parthplc/facebook-hateful-meme-dataset")

print("Path to dataset files:", path)