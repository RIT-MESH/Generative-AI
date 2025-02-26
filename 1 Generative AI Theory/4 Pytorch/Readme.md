# PyTorch Custom Neural Network and Dataset

## Overview
This repository contains implementations of a custom neural network and a custom dataset in PyTorch. The purpose of this project is to demonstrate how to define and use a simple feedforward neural network and dataset loader for machine learning tasks.

## Table of Contents
1. [Installation](#installation)
2. [Creating a Custom Neural Network](#creating-a-custom-neural-network)
3. [Creating a Custom Dataset](#creating-a-custom-dataset)
4. [Training the Model](#training-the-model)
5. [Evaluating the Model](#evaluating-the-model)
6. [Usage Example](#usage-example)
7. [PyTorch Overview and Core Features](#pytorch-overview-and-core-features)
8. [PyTorch vs TensorFlow](#pytorch-vs-tensorflow)
9. [Core PyTorch Modules](#core-pytorch-modules)
10. [PyTorch Domain Libraries](#pytorch-domain-libraries)
11. [PyTorch Ecosystem](#pytorch-ecosystem)
12. [Who Uses PyTorch](#who-uses-pytorch)
13. [Contributing](#contributing)
14. [License](#license)

---

## Installation
To use this project, you need to install **PyTorch**. You can install it using pip:

```bash
pip install torch torchvision torchaudio
```

---

## Creating a Custom Neural Network

A simple feedforward neural network in PyTorch can be implemented using `torch.nn.Module`. Below is an example:

```python
# Import necessary modules
import torch
import torch.nn as nn

# Define a custom neural network
class MySimpleNN(nn.Module):
    def __init__(self, input_features):  # Constructor to initialize the model
        super().__init__()
        
        # Define the layers of the neural network
        self.linear1 = nn.Linear(input_features, 3)  # First fully connected layer
        self.relu = nn.ReLU()  # ReLU activation function
        self.linear2 = nn.Linear(3, 1)  # Second fully connected layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function for binary classification

    # Define the forward pass
    def forward(self, x):
        out = self.linear1(x)  # Pass input through the first layer
        out = self.relu(out)  # Apply ReLU activation
        out = self.linear2(out)  # Pass through the second layer
        out = self.sigmoid(out)  # Apply Sigmoid activation
        return out  # Return final output
```

### Explanation
- `nn.Linear()` defines fully connected layers.
- `nn.ReLU()` applies a non-linearity to the model.
- `nn.Sigmoid()` maps the output to a probability between 0 and 1.

---

## Creating a Custom Dataset
A custom dataset can be created by extending `torch.utils.data.Dataset`:

```python
from torch.utils.data import Dataset

# Define a custom dataset
class CustomDataSet(Dataset):
    def __init__(self, features, labels):
        self.features = features  # Store input features
        self.labels = labels  # Store corresponding labels

    def __len__(self):
        return len(self.features)  # Return the total number of samples

    def __getitem__(self, index):
        return self.features[index], self.labels[index]  # Return the sample and label
```

### Explanation
- `__len__` returns the dataset size.
- `__getitem__` retrieves a specific sample from the dataset.

---

## PyTorch Overview and Core Features

PyTorch is an **open-source deep learning library** developed by Meta AI. It evolved from the original Torch framework written in Lua to a Python-based framework.

### Core Features:
- **Tensor Computation**: Supports multi-dimensional tensors.
- **GPU Acceleration**: Optimized for GPU computations.
- **Dynamic Computation Graph**: Defined at runtime.
- **Automatic Differentiation**: Supports automatic gradient calculation.
- **Distributed Training**: Enables training across multiple devices.
- **Interoperability**: Works with NumPy and other libraries.

---

## PyTorch vs TensorFlow
| Feature | PyTorch | TensorFlow |
|---------|---------|------------|
| Programming Language | Python-based | Python with C++ core |
| Ease of Use | More intuitive | Steeper learning curve |
| Deployment | Catching up | More widely used in production |
| Community & Ecosystem | Strong | Larger |
| Preferred Use Case | Research | Production |
| Model Zoo | torchvision | TensorFlow Hub |

---

## Core PyTorch Modules
- **torch**: Core module for tensor operations.
- **torch.autograd**: Automatic differentiation.
- **torch.nn**: Defines and constructs neural networks.
- **torch.optim**: Implements optimization algorithms.
- **torch.utils.data**: Utilities for data loading.
- **torch.cuda**: CUDA support for GPU computations.

---

## PyTorch Domain Libraries
- **torchvision**: Computer vision tools.
- **torchtext**: Text processing tools.
- **torchaudio**: Audio processing tools.
- **pytorch_lightning**: High-level API for deep learning.

---

## PyTorch Ecosystem
- **Huggingface Transformers**: NLP models.
- **FastAI**: High-level API for deep learning.
- **PyTorch Geometric**: Graph-based learning tools.
- **TorchMetrics**: Model evaluation metrics.
- **Optuna**: Hyperparameter optimization.

---

## Who Uses PyTorch
- **Meta**: Developers of PyTorch.
- **Microsoft**: AI projects.
- **Tesla**: AI applications.
- **OpenAI**: GPT and reinforcement learning models.
- **Uber**: AI research.
- **Walmart**: Demand forecasting and recommendation systems.

---

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.
```

