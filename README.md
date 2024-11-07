# Multimodal Sentiment Analysis with Transformer Model

This project demonstrates a multimodal approach to sentiment analysis using **RoBERTa** for text encoding, **DenseNet** for image encoding, and a **Transformer** model for feature fusion. The model classifies tweets into sentiment categories: **Certainly Fake**, **Probably Fake**, **Probably Real**, and **Certainly Real**.

Key metrics tracked include **accuracy**, **F1-score**, and **loss** over training epochs, with the best-performing models saved.

---

## Table of Contents
- [Setup and Imports](#setup-and-imports)
- [Dataset Loading and Preprocessing](#dataset-loading-and-preprocessing)
- [Text and Image Preprocessing](#text-and-image-preprocessing)
- [Model Architecture](#model-architecture)
- [Training Loop](#training-loop)
- [Model Performance Analysis](#model-performance-analysis)

---

## Setup and Imports

First, install the necessary libraries and set up the environment for GPU debugging.

```bash
!pip install torch-geometric
%env CUDA_LAUNCH_BLOCKING=1
```

# Import essential libraries:
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
# Text and Image Preprocessing
Text Preprocessing
We use the RoBERTa tokenizer with padding and truncation.

```bash
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
def preprocess_text(text):
    encoding = tokenizer(text, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0)
```
Image Preprocessing
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
## Model Architecture

### 1. Text Encoder (RoBERTa)
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
Training Loop
The training loop saves models with the highest accuracy and F1-score across epochs.

python
Copy code
best_accuracy = 0.0
best_f1_score = 0.0
for epoch in range(100):  # Adjust epochs as needed
    # Training and validation code here...
    
    # Check for new best accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        # Save model with highest accuracy
    
    # Check for new best F1 score
    if f1 > best_f1_score:
        best_f1_score = f1
        # Save model with highest F1 score
Model Performance Analysis
This section includes:

Training History: Accuracy, F1 score, and loss over epochs.
Confusion Matrix and Classification Report: For detailed performance analysis.
ROC-AUC Curves: To evaluate performance across all classes.
