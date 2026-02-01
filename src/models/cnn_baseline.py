"""
Minimal CNN Baseline for Malaria Classification

Architecture:
- 3 convolutional blocks
- <1M parameters
- Standard for binary classification

Purpose: Baseline comparison for ViT failure-awareness claims
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBaseline(nn.Module):
    """
    Minimal CNN for binary malaria classification.
    
    Architecture:
        Block 1: Conv(3→32, k=3) → ReLU → MaxPool(2×2)
        Block 2: Conv(32→64, k=3) → ReLU → MaxPool(2×2)
        Block 3: Conv(64→128, k=3) → ReLU → MaxPool(2×2)
        Block 4: Conv(128→256, k=3) → ReLU → AdaptiveAvgPool(4×4)
        Flatten → Dropout(0.3) → FC(4096→256) → ReLU → FC(256→2)
    
    Total params: ~1.1M
    """
    
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        
        # Convolutional blocks
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Flattened size: 256 * 4 * 4 = 4096
        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) input images
        
        Returns:
            logits: (B, num_classes) unnormalized predictions
        """
        # Block 1
        x = self.pool(F.relu(self.conv1(x)))  # (B, 32, 112, 112)
        
        # Block 2
        x = self.pool(F.relu(self.conv2(x)))  # (B, 64, 56, 56)
        
        # Block 3
        x = self.pool(F.relu(self.conv3(x)))  # (B, 128, 28, 28)
        
        # Block 4 + adaptive pooling
        x = F.relu(self.conv4(x))  # (B, 256, 28, 28)
        x = self.adaptive_pool(x)  # (B, 256, 4, 4)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (B, 4096)
        
        # Fully connected
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def get_features(self, x):
        """Extract features before final classification (for uncertainty analysis)"""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return x


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = CNNBaseline()
    print(f"CNN Baseline")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 224, 224)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test feature extraction
    features = model.get_features(x)
    print(f"Feature shape: {features.shape}")
