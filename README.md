# Multi-Modal Product Classifier (Text + Image)

A multi-modal deep learning model that classifies fashion products by combining product images and text titles.

## Overview

This project uses the [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) (~44,000 products) to predict a product's `subCategory` using both its image and its `productDisplayName` text.

## Approach

- **Image Encoder:** ResNet18 (pretrained, frozen)
- **Text Encoder:** DistilBERT (pretrained, frozen)
- **Fusion:** Feature concatenation → MLP classifier
- **Target label:** `subCategory` (29 classes after removing rare categories with <75 samples)
- **Class imbalance:** handled with balanced class weights in the loss function

Three models were trained and compared:
1. Image-only
2. Text-only
3. Fusion (Image + Text)

## Results

| Model | Macro-F1 |
|---|---:|
| Image-only | see notebook |
| Text-only | see notebook |
| Fusion | see notebook |

Macro-F1 was used as the main metric due to significant class imbalance in the dataset.

## Tech Stack

- PyTorch
- Hugging Face Transformers (DistilBERT)
- torchvision (ResNet18)
- scikit-learn (metrics, splitting, class weights)

## Files

- `build-a-multi-modal-product-classifier-w6.ipynb` — full notebook (EDA, preprocessing, model training, evaluation)

## How to Run

1. Download the dataset from Kaggle and update the file paths in the notebook.
2. Install dependencies: `torch`, `torchvision`, `transformers`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`.
3. Run the notebook cells in order.
