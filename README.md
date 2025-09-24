# An exploration of Multimodal data fusion: From MKL to Low Rank Tensors
The repository shall document an independent research where I explore methods for multimodal data fusion, wherein, my aim would be to achieve tensor connectivity across multiple heterogenous data sources.
#
# Motivation 
The project was motivated by the idea of achieving a quantifiable approach, perhaps a learnable constant, or a variable equation to be able to quantize data from multiple resources and be able to use it to train modern machine learning algorithms 
The goal is to able to mitigate performance of these kernels through incorporation of data, that is not defined by the boundaries of media
#
# Methodology
### 1. Theoretical Explanation of MKL 
Concept: For now, I can provide a principled method to linearly combine multiple base kernels, where each kernel can correspond to different data modality and feature representation. The goal is to learn the optimal combination of these kernels to improve classification performances
Optimization: To solve the constrained optimization problem inherent in MKL, Lagrangian expansion was explored as a technique to de-constrain the equation
### 2. Practical Implementation: Low-Rank Multimodal Fusion (LMF)
To apply these concepts in a modern deep learning context, a Low-Rank Multimodal Fusion (LMF) model was built from scratch in PyTorch.
Data Modalities: The model was designed to fuse three simulated data types:
Visual: 128-dimensional feature vector.
Audio: 64-dimensional feature vector.
Text: 300-dimensional feature vector.
Architecture: The LMF model projects each modality's feature vector into a lower-dimensional space defined by a specific 'rank'. These projections are then combined using an element-wise multiplication (Hadamard product) to create a single, fused representation that captures inter-modality interactions efficiently.
## Key Findings
The MKL experiments, visualized using PCA, demonstrated that a combined kernel can create a more suitable vector space for classification, improving the separability of data points
The from-scratch implementation of the LMF model was successful. It was trained on a sample regression task, and the training loss showed a consistent decrease over epochs, validating that the model was implemented correctly and is capable of learning from multimodal inputs.
#
# Future Work
This study serves as a foundational step. The ultimate goal is to build upon these findings to develop a novel method for quantitatively combining tensors. The work is heavily inspired by the ideas presented in the paper "Interpretable Tensor Fusion" by Varshneya et al.
# 
# How to Run
The primary implementation is contained within the Low_Rank_Multimodal_Fusion.ipynb Jupyter Notebook. It includes the model definition, testing functions, and a sample training loop. The notebook can be run in Google Colab or any standard Jupyter environment with PyTorch installed.
