"""
Model architectures for fairness-aware deepfake detection.

This module provides model creation utilities using EfficientNet-B0
as the backbone architecture, with support for transfer learning
from ImageNet pre-trained weights.
"""

import torch
import torch.nn as nn
import timm


class DeepfakeDetector(nn.Module):
    """
    Deepfake detection model based on EfficientNet-B0.
    
    Uses transfer learning from ImageNet pre-trained weights,
    with a randomly initialized 2-class classification head.
    """
    
    def __init__(self, model_name='efficientnet_b0', num_classes=2, pretrained=True):
        """
        Args:
            model_name (str): Name of timm model architecture
            num_classes (int): Number of output classes (2 for binary classification)
            pretrained (bool): Whether to use ImageNet pre-trained weights
        """
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        self.model_name = model_name
        self.num_classes = num_classes
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, 224, 224)
        
        Returns:
            torch.Tensor: Logits of shape (B, num_classes)
        """
        return self.model(x)
    
    def get_num_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(model_name='efficientnet_b0', num_classes=2, pretrained=True, device=None):
    """
    Factory function to create a deepfake detection model.
    
    Args:
        model_name (str): Architecture name (default: 'efficientnet_b0')
        num_classes (int): Number of output classes (default: 2)
        pretrained (bool): Use ImageNet pre-trained weights (default: True)
        device (torch.device, optional): Device to move model to
    
    Returns:
        DeepfakeDetector: Initialized model
    """
    model = DeepfakeDetector(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained
    )
    
    if device is not None:
        model = model.to(device)
    
    print(f"Created {model_name} model")
    print(f"Parameters: {model.get_num_parameters():,}")
    print(f"Pre-trained: {pretrained}")
    
    return model


def load_model(checkpoint_path, model_name='efficientnet_b0', num_classes=2, device=None):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to model checkpoint (.pth file)
        model_name (str): Architecture name
        num_classes (int): Number of output classes
        device (torch.device, optional): Device to load model on
    
    Returns:
        DeepfakeDetector: Loaded model in evaluation mode
    """
    # Create model architecture
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,  # Don't download weights, load from checkpoint
        device=device
    )
    
    # Load weights
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint with metadata")
        if 'test_acc' in checkpoint:
            print(f"Test accuracy: {checkpoint['test_acc']:.2f}%")
    else:
        model.load_state_dict(checkpoint)
        print(f"Loaded model weights")
    
    model.eval()
    print(f"Model loaded from: {checkpoint_path}")
    
    return model


def save_model(model, save_path, optimizer=None, epoch=None, metrics=None):
    """
    Save model checkpoint with optional metadata.
    
    Args:
        model (nn.Module): Model to save
        save_path (str): Path to save checkpoint
        optimizer (torch.optim.Optimizer, optional): Optimizer state to save
        epoch (int, optional): Training epoch number
        metrics (dict, optional): Performance metrics to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if metrics is not None:
        checkpoint.update(metrics)
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to: {save_path}")


if __name__ == '__main__':
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(device=device)
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    
    print(f"\nTest forward pass:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")
