Multimodal Sentiment Classification
This repository contains a Jupyter notebook, Multimodal Sentiment Classification with Transformer Model, demonstrating a multimodal sentiment analysis approach using RoBERTa for text encoding, DenseNet for image encoding, and a Transformer-based architecture for feature fusion. The goal is to classify tweets with both text and image components into sentiment categories: Certainly Fake, Probably Fake, Probably Real, and Certainly Real.

Model Architecture
The project employs the following architecture:

Text Encoder: RoBERTa, pre-trained on large language corpora, encodes tweet text. We use dropout for regularization.
Image Encoder: DenseNet-121, pre-trained on ImageNet, encodes tweet images. This module is also equipped with dropout layers.
Multimodal Transformer: Combines text and image features using a Transformer encoder, followed by classification using a fully connected layer.
Key Features
Custom Dataloader with Dynamic Padding: To handle varied text lengths and image processing, we implemented a custom collate function.
Training and Evaluation: Tracks key metrics (accuracy, F1 score, and loss) across epochs, saving the best models based on accuracy and F1 score.
Model Persistence: Best-performing models are saved for later use and reproducibility.
Performance Metrics
Accuracy and F1 Score: Recorded across epochs, with detailed analysis and visualization for insights into class-wise performance.
Confusion Matrix and ROC-AUC Curves: Presented for each class, highlighting model strengths and potential areas for improvement.
