#!/usr/bin/env python3
"""
  Improved CNN Model for Silent Cities - Addressing Performance Issues

  References:
  -----------
  [1] LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 
  436-444.
  [2] Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network 
  training 
      by reducing internal covariate shift. International Conference on Machine 
  Learning, 448-456.
  [3] Srivastava, N., et al. (2014). Dropout: A simple way to prevent neural networks 
  from 
      overfitting. Journal of Machine Learning Research, 15(1), 1929-1958.
  [4] Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep 
  learning 
      library. Advances in Neural Information Processing Systems, 32, 8024-8035.
""" 
"""
Improved CNN Model for Silent Cities - Addressing Performance Issues
"""

import torch  # PyTorch deep learning framework
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Activation functions and utilities

class ImprovedSpectrogramCNN(nn.Module):
    """Simplified CNN with better regularization for loudness prediction."""
    
    def __init__(self, input_channels=4, n_mels=64, dropout=0.5):
        super(ImprovedSpectrogramCNN, self).__init__()
        
        # SOLUTION 1: Simpler architecture (fewer parameters)
        # First convolutional layer: 4 input channels -> 32 output channels
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, padding=2)
        # Second convolutional layer: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        # Third convolutional layer: 64 -> 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # SOLUTION 2: Better normalization
        # Batch normalization for each conv layer to stabilize training
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # SOLUTION 3: More aggressive pooling
        # Max pooling to reduce spatial dimensions
        self.pool = nn.MaxPool2d(2, 2)
        # Adaptive pooling to ensure fixed output size regardless of input dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Smaller than before
        
        # SOLUTION 4: Much simpler FC layers
        # Fully connected layers for final regression
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # Drastically reduced parameters
        self.fc2 = nn.Linear(256, 64)  # Hidden layer
        self.fc3 = nn.Linear(64, 1)  # Output layer for loudness prediction
        
        # SOLUTION 5: Heavier dropout
        # Dropout layers for regularization to prevent overfitting
        self.dropout = nn.Dropout(dropout)  # Standard dropout
        self.dropout_heavy = nn.Dropout(0.7)  # Heavy dropout for critical layers
        
    def forward(self, x):
        # Feature extraction with heavy regularization
        # First conv block: Conv -> BatchNorm -> ReLU -> Dropout -> Pool
        x = self.dropout(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(x)  # Reduce spatial dimensions by factor of 2
        
        # Second conv block: Conv -> BatchNorm -> ReLU -> Dropout -> Pool
        x = self.dropout(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(x)  # Further spatial reduction
        
        # Third conv block: Conv -> BatchNorm -> ReLU -> Dropout -> Pool
        x = self.dropout(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(x)  # Final spatial reduction
        
        # Global pooling to fixed size
        x = self.adaptive_pool(x)  # Ensure consistent tensor size
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        
        # Simple classification head
        # First FC layer with heavy dropout
        x = self.dropout_heavy(F.relu(self.fc1(x)))
        # Second FC layer with standard dropout
        x = self.dropout(F.relu(self.fc2(x)))
        # Output layer - no activation for regression task
        x = self.fc3(x)  # No activation for regression
        
        return x

class LoudnessRegressionCNN(nn.Module):
    """Alternative: Domain-specific architecture for acoustic features."""
    
    def __init__(self, input_channels=4):
        super(LoudnessRegressionCNN, self).__init__()
        
        # SOLUTION 6: Frequency-aware convolutions
        # Separate processing for different frequency ranges to capture domain-specific patterns
        self.low_freq = nn.Conv2d(input_channels, 16, (1, 5))   # Temporal patterns (horizontal)
        self.mid_freq = nn.Conv2d(input_channels, 16, (3, 3))   # Local patterns (square)
        self.high_freq = nn.Conv2d(input_channels, 16, (5, 1))  # Frequency patterns (vertical)
        
        # Combine features from different frequency processing paths
        self.combine = nn.Conv2d(48, 64, (3, 3), padding=1)  # 48 = 16*3 channels
        self.pool = nn.AdaptiveAvgPool2d((8, 8))  # Fixed output size
        
        # Simple regressor - sequential fully connected layers
        self.regressor = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),  # First layer: flatten to 128 features
            nn.ReLU(),  # ReLU activation
            nn.Dropout(0.6),  # Heavy dropout for regularization
            nn.Linear(128, 32),  # Second layer: reduce to 32 features
            nn.ReLU(),  # ReLU activation
            nn.Dropout(0.5),  # Standard dropout
            nn.Linear(32, 1)  # Output layer: single loudness value
        )
    
    def forward(self, x):
        # Multi-scale feature extraction - process different frequency patterns
        low = F.relu(self.low_freq(x))  # Extract temporal patterns
        mid = F.relu(self.mid_freq(x))  # Extract local spatial patterns
        high = F.relu(self.high_freq(x))  # Extract frequency-specific patterns
        
        # Adjust dimensions and combine features from different scales
        # Interpolate to match middle frequency branch dimensions
        low = F.interpolate(low, size=mid.shape[2:], mode='bilinear')
        high = F.interpolate(high, size=mid.shape[2:], mode='bilinear')
        
        # Concatenate all feature maps along channel dimension
        combined = torch.cat([low, mid, high], dim=1)  # Combine 16+16+16=48 channels
        # Further feature processing and dimensionality reduction
        x = F.relu(self.combine(combined))  # Reduce to 64 channels
        x = self.pool(x)  # Pool to fixed 8x8 spatial size
        
        # Flatten and pass through regression head
        x = x.view(x.size(0), -1)  # Flatten to vector
        return self.regressor(x)  # Predict loudness value

# SOLUTION 7: Better training configuration
def get_improved_training_config():
    """Returns optimized training configuration."""
    return {
        'batch_size': 32,           # Larger batches for more stable gradients
        'lr': 5e-4,                # Lower learning rate to prevent overshooting
        'weight_decay': 1e-3,      # Stronger L2 regularization to prevent overfitting
        'epochs': 200,             # More epochs for thorough training
        'patience': 25,            # More patience before early stopping
        'scheduler': 'ReduceLROnPlateau',  # Adaptive learning rate scheduler
        'factor': 0.5,             # LR reduction factor when plateau detected
        'min_lr': 1e-6,           # Minimum learning rate threshold
    }

# SOLUTION 8: Better loss function
class HuberLossWithRegularization(nn.Module):
    """Custom loss with additional regularization terms."""
    
    def __init__(self, delta=1.0, reg_weight=0.01):
        super().__init__()
        # Huber loss is more robust to outliers than MSE
        self.huber = nn.HuberLoss(delta=delta)  # Delta controls transition point
        self.reg_weight = reg_weight  # Weight for regularization term
    
    def forward(self, predictions, targets, model):
        # Main regression loss - Huber loss for robustness
        main_loss = self.huber(predictions, targets)
        
        # L1 regularization on model weights to encourage sparsity
        l1_reg = sum(p.abs().sum() for p in model.parameters())
        
        # Combined loss: main loss + weighted regularization
        return main_loss + self.reg_weight * l1_reg