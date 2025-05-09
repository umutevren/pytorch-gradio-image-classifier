# PyTorch-Gradio Image Classifier & Clustering System

A modern deep learning solution for image classification and clustering, built with PyTorch and Gradio. This project implements an image classification and clustering system using deep learning techniques. It includes various components for image processing, model training, inference, and visualization.

## Project Structure

The project consists of several key components:

### Core Components
- `model.py`: Implements the neural network architecture using EfficientNet-B0 as the base model
- `train.py`: Handles model training with PyTorch, including training and validation loops
- `inference.py`: Provides functionality for model inference
- `data_model.py`: Contains dataset handling and data loading utilities

### Image Processing
- `image_utils.py`: Utility functions for image processing
- `smartcrop.py`: Implements smart cropping functionality
- `image_processing.ipynb`: Jupyter notebook for image processing experiments

### Clustering and Embeddings
- `clustering.py`: Implements clustering algorithms for image analysis
- `get_embeddings.py`: Generates embeddings from images
- `image_clustering.ipynb`: Notebook for clustering analysis
- `image_embedding.ipynb`: Notebook for embedding generation and analysis

### Data Management
- `dataset_splitter.py`: Utility for splitting datasets
- `dataset_split.ipynb`: Notebook for dataset preparation
- `scrapers.py`: Web scraping utilities for data collection

### Visualization and Interface
- `visualization.py`: Tools for visualizing results
- `gradio_infer.py`: Gradio-based web interface for model inference

## Methodology

### Model Architecture
- Uses EfficientNet-B0 as the base model
- Implements transfer learning with pre-trained weights
- Custom classifier head for specific classification tasks

### Training Approach
- Implements PyTorch-based training pipeline
- Uses Adam optimizer with CrossEntropyLoss
- Includes validation during training
- Supports GPU acceleration when available

### Image Processing Pipeline
1. Image resizing and normalization
2. Smart cropping for better feature extraction
3. Data augmentation for improved generalization

### Clustering and Embedding
- Generates embeddings using the trained model
- Implements clustering algorithms for image grouping
- Provides visualization tools for analysis

## Requirements

The project requires the following main dependencies:
- PyTorch
- torchvision
- timm
- numpy
- gradio (for web interface)
- tqdm (for progress bars)

## Notebooks

The project includes several Jupyter notebooks for interactive analysis:
- `image_processing.ipynb`: Image processing experiments
- `image_clustering.ipynb`: Clustering analysis
- `image_embedding.ipynb`: Embedding generation and analysis
- `dataset_split.ipynb`: Dataset preparation 