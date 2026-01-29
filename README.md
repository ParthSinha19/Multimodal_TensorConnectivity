# Robustness and Geometry in Low-Rank Multimodal Fusion

This repository establishes a theoretically grounded framework for advanced multimodal machine learning, integrating **Joint Wasserstein Autoencoders (jWAE)** with **Low-Rank Multimodal Fusion (LMF)**. The project addresses the instability of fusing heterogeneous data—such as image and text—by ensuring geometric alignment and spectral robustness.

---

## Research Motivation

Traditional multimodal systems often suffer from geometrically misaligned latent spaces, leading to brittle models vulnerable to distribution shifts and adversarial attacks. Furthermore, high-dimensional fusion often incurs a "curse of dimensionality" through colossal over-parametrization.

This project hypothesizes that **low-rank regularization** serves as a spectral procedure to prevent high-frequency noise while improving adversarial robustness and uncertainty quantification.

---

## Methodology

The architecture is derived from a synthesis of **Multiple Kernel Learning (MKL)** and **Tensor Factorization**.

### 1. Geometric Conditioning (jWAE)

A **Joint Wasserstein Autoencoder** acts as a critical geometric conditioner. By enforcing a shared Gaussian prior on latent embeddings, it ensures semantic continuity and "linearizes" the latent manifold. This alignment is a prerequisite for efficient tensor approximation.

### 2. Low-Rank Multimodal Fusion (LMF)

To avoid the exponential computational cost of full tensor products (), the system employs **Low-Rank Decomposition**.

**The Rank Function ():** Acts as a bottleneck forcing the model to learn only salient interactions.


**Fusion Equation:** The fusion is calculated as a summation of element-wise (Hadamard) products across the rank dimension:


### 3. Faithfulness vs. Performance

The architecture prioritizes **faithfulness**—the capacity to rationalize the model's reasoning process—over the raw accuracy of "black-box" Transformers. By isolating explicit rank factors, the model provides mechanistic transparency into how specific modalities change the context of others.

---

## Repository Structure

**`lmf_module.py`**: Implementation of the Low-Rank Multimodal Fusion layer from scratch, including bias augmentation () and rank-specific projection matrices .


**`jwae_module.py`**: The geometric conditioning engine utilizing encoders () to map raw inputs to a smooth latent manifold.


**`hateful_memes_loader.py`**: A specialized data pipeline for the **Facebook Hateful Memes** dataset, utilizing pre-trained **ResNet-50** (Image) and **BERT** (Text) feature extractors to provide 2048 and 768-dimensional vectors respectively.
**`main_hateful_memes.py`**: The primary end-to-end training script implementing hybrid loss: Classification (BCE) + Geometric Regularization (MMD).

---

## Implementation & Data

The framework is validated using datasets where modalities must interact to resolve ambiguity:

**CMU-MOSI**: Sentiment analysis requiring alignment of text, visual, and acoustic features.


**MUSTARD**: Sarcasm detection where text meaning is modified by tone or context.


**Hateful Memes**: Challenging binary classification where benign text (e.g., "Love the way you smell") becomes hateful only when paired with specific visual context (e.g., a skunk).

---
How to Run

To execute the research pipeline and replicate the results on the **Hateful Memes** dataset, follow these steps:

### 1. Environment Setup

Clone the repository and install the necessary dependencies using the provided `requirements.txt`.

```bash
pip install -r requirements.txt

```

### 2. Dataset Acquisition

The framework is designed to work with the **Facebook Hateful Memes** dataset. You can download it directly into your environment using `kagglehub`:

```python
import kagglehub
# This captures the local path to the versioned dataset
raw_path = kagglehub.dataset_download("parthplc/facebook-hateful-meme-dataset")

```

### 3. Configuration

In `main_hateful_memes.py`, update the `REAL_PATH` dictionary to point to your downloaded data:

**`jsonl`**: Path to `train.jsonl`.


**`img_dir`**: Path to the parent folder of the `img/` directory.



### 4. Execution

Run the main training script. This script integrates the **jWAE** geometric conditioner with the **LMF** fusion layer.

```bash
python main_hateful_memes.py

```

**Training Loop**: The script calculates a hybrid loss combining **Binary Cross Entropy** (classification) and **Joint Wasserstein Regularization** (geometry).


**Output**: Upon completion, the script provides a "Mechanistic Transparency" snapshot, showing the global weights of each **Rank Factor**.



---

## Research Continuity: Future Directions

The current implementation focuses on **static vector fusion**, transforming a complex data landscape into a "smooth bowl shape" via jWAE. To advance the research, we can now look toward:

**Temporal Integration**: Extending the **LMF**'s additive interactions to handle sequential effects in video and audio tokens.


**Explainability Expansion**: Proving that specific rank factors correspond to distinct semantic interactions (e.g., Rank 1 for sentiment, Rank 2 for context) to further verify the mathematical basis for explainability.


**Adversarial Probing**: Testing the model's resistance to high-frequency noise and non-robust features using the spectral regularization properties of low-rank factors.

---
## References

* Gurevych et al., "Joint Wasserstein Autoencoders for Aligning Multimodal Embeddings," 2019.


* Liu et al., "Efficient Low-rank Multimodal Fusion with Modality-Specific Factors," 2018.


* Varshneya et al., "Interpretable Tensor Fusion," 2024.
---
