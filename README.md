# Multimodal Sentiment Analysis with Transformer Model

This project demonstrates a multimodal approach to sentiment analysis using **RoBERTa** for text encoding, **DenseNet** for image encoding, and a **Transformer** model for feature fusion. The model classifies tweets into sentiment categories: **Certainly Fake**, **Probably Fake**, **Probably Real**, and **Certainly Real**.

Key metrics tracked include **accuracy**, **F1-score**, and **loss** over training epochs, with the best-performing models saved.

---

## Table of Contents
- [Setup and Imports](#setup-and-imports)
- [Dataset Loading and Preprocessing](#dataset-loading-and-preprocessing)
- [Text and Image Preprocessing](#text-and-image-preprocessing)
  - [Text Preprocessing](#text-preprocessing)
  - [Image Preprocessing](#image-preprocessing)
- [Model Architecture](#model-architecture)
  - [Text Encoder (RoBERTa)](#1-text-encoder-roberta)
  - [Image Encoder (DenseNet-121)](#2-image-encoder-densenet-121)
  - [Multimodal Transformer Model](#3-multimodal-transformer-model)
  - [Custom Collate Function](#4-custom-collate-function)
  - [Visual Comprehension of Model](#5-visual-comprehension-of-model)
- [Training Loop](#training-loop)
- [Model Performance Analysis](#model-performance-analysis)
  - [Training History](#training-history-accuracy-f1-score-and-loss-over-epochs)
  - [Confusion Matrix and Classification Report](#confusion-matrix-and-classification-report)
  - [ROC-AUC Curves](#roc-auc-curves)
- [Requirements](#requirements)
- [Cloning the Repository](#cloning-the-repository)
- [Running And Training The Model](#running-and-training-the-model)
- [Output Files](#output-files)
  - [highest_acc/](#highest_acc)
  - [highest_f1/](#highest_f1)
  - [Training_history.csv](#training_historycsv)

---

## Setup and Imports

First, install the necessary libraries and set up the environment for GPU debugging.

```bash
!pip install torch-geometric
%env CUDA_LAUNCH_BLOCKING=1
```

## Import essential libraries:
```bash
import os
import shutil
import torch
import torch.nn as nn
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
from torchvision import models, transforms
from torch_geometric.loader import DataLoader
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
```
# Dataset Loading and Preprocessing
Load the dataset and remove errors and missing values.
```bash

data = pd.read_csv("/kaggle/input/tweet-data-game-on/Updated_Tweet_Data.csv")
data = data[data['Content'] != "Error: Tweet could not be scraped"]
data = data.dropna(subset=['Image_Name']).reset_index(drop=True)
```
# Plot label distribution to visualize class balance.
```bash
label_counts = data['Label'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Pie Chart Distribution of Labels")
plt.show()
print(data.Label.value_counts())
```
![image](https://github.com/user-attachments/assets/705bbfd9-22b9-4f47-9c9f-6d7ee63c4b82)

# Text and Image Preprocessing
## Text Preprocessing
We use the RoBERTa tokenizer with padding and truncation.

```bash
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
def preprocess_text(text):
    encoding = tokenizer(text, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0)
```
## Image Preprocessing
Images are resized, normalized, and augmented for better model generalization.
```bash
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
def preprocess_image(image_name):
    if pd.isna(image_name) or not isinstance(image_name, str):
        return torch.zeros(3, 224, 224)
    image_path = os.path.join("/kaggle/input/tweet-data-game-on/Tweet_Images/", image_name)
    try:
        image = Image.open(image_path).convert("RGB")
        return image_transform(image)
    except FileNotFoundError:
        print(f"Warning: Image file '{image_path}' not found. Using placeholder.")
        return torch.zeros(3, 224, 224)
```
# Model Architecture

## 1). Text Encoder (RoBERTa)
The text encoder uses **RoBERTa** with a dropout layer. Layers are initially frozen for faster training.

```python
class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        for param in self.roberta.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return self.dropout(outputs.pooler_output)

```
## 2). Image Encoder (DenseNet-121)
The image encoder is built on DenseNet-121 to extract image features.
```python
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        densenet = models.densenet121(pretrained=True)
        self.densenet = nn.Sequential(*list(densenet.features.children()))
        for param in self.densenet.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, images):
        features = self.densenet(images)
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        return self.dropout(features.view(features.size(0), -1))
```
## 3). Multimodal Transformer Model
A Transformer model fuses text and image features for sentiment classification.
```python
class MultimodalTransformerModel(nn.Module):
    def __init__(self, text_dim=768, image_dim=1024, hidden_dim=512, output_dim=4):
        super(MultimodalTransformerModel, self).__init__()
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text_features, image_features):
        batch_size = text_features.size(0)
        text_features = self.text_projection(text_features)
        image_features = self.image_projection(image_features)
        combined_features = torch.cat([text_features.unsqueeze(1), image_features.unsqueeze(1)], dim=1)
        fused_features = self.transformer_encoder(combined_features).mean(dim=1)
        return self.classifier(fused_features)
```
## 4). Custom Collate Function
In this model, a custom collate function is used for handling multimodal inputs (text and images) during batching. The function is designed to ensure that the text and image data are properly paired together and padded when necessary.
```python
from torch.utils.data import DataLoader

def collate_fn(batch):
    text_data = [item['text'] for item in batch]
    image_data = [item['image'] for item in batch]
    text_labels = [item['label'] for item in batch]
    
    # Padding text data and converting to tensors
    text_input_ids = tokenizer(text_data, padding=True, truncation=True, return_tensors="pt")
    text_attention_mask = text_input_ids['attention_mask']
    text_input_ids = text_input_ids['input_ids']

    # Converting image data to tensors
    images = torch.stack(image_data)

    return text_input_ids, text_attention_mask, images, torch.tensor(text_labels)
```
## 5). Visual Comprehension of Model
To help visualize how the model works and understand its architecture, a video demonstration has been provided. It showcases the fusion of text and image features, and how they are processed through the model for sentiment classification.

https://github.com/user-attachments/assets/4ad9cf85-e0c4-4b31-886f-6f6bf13e05e0


# Training Loop
The training loop saves models with the highest accuracy and F1-score across epochs.

```python
best_accuracy = 0.0
best_f1_score = 0.0
for epoch in range(100):  # Adjust epochs as needed
    # Training and validation code
    
    # Check for new best accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        # Save model with highest accuracy
    
    # Check for new best F1 score
    if f1 > best_f1_score:
        best_f1_score = f1
        # Save model with highest F1 score
```
# Model Performance Analysis
## This section includes:
### 1. Training History: Accuracy, F1 score, and loss over epochs.
![image](https://github.com/user-attachments/assets/0aa31c74-cfc5-4ea2-92a6-030f500b5665)
![image](https://github.com/user-attachments/assets/4545714c-02b5-4509-8cf9-f849070f4766)
![image](https://github.com/user-attachments/assets/6d67545d-bb13-48a2-959f-56202ec82100)

### 2. Confusion Matrix and Classification Report: For detailed performance analysis.
![image](https://github.com/user-attachments/assets/4d212836-dd97-48a7-9528-bdd3dcde0b77)

### 3. ROC-AUC Curves: To evaluate performance across all classes.
![image](https://github.com/user-attachments/assets/8ed2bd8a-3722-4531-9ba4-52a7a9089a97)
![image](https://github.com/user-attachments/assets/344b62bb-002e-4bbe-8670-748fb13be03f)


# Requirements
**1).Python 3.7 or later**

**2). PyTorch**

**3). Transformers (pip install transformers)**

**4). Torchvision (pip install torchvision)**

**5). Torch Geometric (pip install torch-geometric)**

**6). PIL (Python Imaging Library)**

**7). scikit-learn**

**8). Matplotlib**

**9). Seaborn**

**10). Numpy**

**11). Pandas**

# Install all dependencies with a single command:
```python
pip install -r requirements.txt
```
# Cloning the Repository
Clone this repository and navigate into the project directory:
```python
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
```
# Running And Training The Model
To begin training and evaluating the model, run the main training script:
```python
python train.py
```
# Output files
Upon completion of training, the following outputs will be generated:

## highest_acc/: 
Stores model checkpoints with the highest accuracy achieved during training.
## highest_f1/: 
Contains the best-performing model based on the F1-score metric.
## Training_history.csv: 
Logs key metrics such as accuracy, F1-score, and loss for each epoch. This file provides a comprehensive view of the model's performance progression.
