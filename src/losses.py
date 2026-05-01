"""
Loss functions for fairness-aware deepfake detection.

This module implements the proposed variance-based fairness-aware loss
function that augments cross-entropy with a demographic regularization term.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FairDeepfakeLoss(nn.Module):
    """
    Fairness-aware loss function for deepfake detection.
    
    Combines standard cross-entropy classification loss with a variance-based
    regularization term that penalizes prediction inconsistencies across
    demographic groups.
    
    The total loss is:
        L_fair = L_CE + lambda * R_var
    
    Where:
        L_CE is the standard cross-entropy loss
        R_var is the variance of mean predictions across demographic groups
        lambda controls the strength of the fairness constraint
    """
    
    def __init__(self, lambda_fair=0.5):
        """
        Args:
            lambda_fair (float): Weight for the fairness regularization term.
                Higher values enforce stronger fairness constraints.
                - 0.0: Standard training (no fairness)
                - 0.5: Moderate fairness regularization
                - 2.0: Strong fairness regularization
        """
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda_fair = lambda_fair
    
    def forward(self, logits, labels, sensitive_attrs):
        """
        Compute the fairness-aware loss.
        
        Args:
            logits (torch.Tensor): Model output of shape (B, num_classes)
            labels (torch.Tensor): Ground truth labels of shape (B,)
            sensitive_attrs (torch.Tensor): Demographic attributes of shape (B,)
        
        Returns:
            tuple: (total_loss, classification_loss, fairness_loss)
        """
        # Standard classification loss
        classification_loss = self.ce_loss(logits, labels)
        
        # Fairness regularization
        fairness_loss = self._compute_fairness_loss(logits, labels, sensitive_attrs)
        
        # Total loss
        total_loss = classification_loss + self.lambda_fair * fairness_loss
        
        return total_loss, classification_loss, fairness_loss
    
    def _compute_fairness_loss(self, logits, labels, sensitive_attrs):
        """
        Compute variance of mean predictions across demographic groups.
        
        Mathematically:
            R_var = (1/|A|) * sum_{a in A} (mu_a - mu_global)^2
        
        Where:
            mu_a is the mean prediction probability for group a
            mu_global is the global mean across all groups
            A is the set of demographic groups
        
        Args:
            logits (torch.Tensor): Model logits
            labels (torch.Tensor): Ground truth labels (unused but kept for API consistency)
            sensitive_attrs (torch.Tensor): Demographic group indices
        
        Returns:
            torch.Tensor: Scalar variance penalty
        """
        # Get probability of "fake" class (class 1)
        probs = F.softmax(logits, dim=1)[:, 1]
        
        # Get unique demographic groups in this batch
        unique_groups = torch.unique(sensitive_attrs)
        
        # Need at least 2 groups to compute variance
        if len(unique_groups) < 2:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Compute mean prediction for each group
        group_means = []
        for group in unique_groups:
            mask = (sensitive_attrs == group)
            if mask.sum() > 0:
                group_mean = probs[mask].mean()
                group_means.append(group_mean)
        
        # Need at least 2 groups with samples
        if len(group_means) < 2:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Compute variance across groups
        group_means_tensor = torch.stack(group_means)
        variance = torch.var(group_means_tensor, unbiased=False)
        
        return variance


class StandardLoss(nn.Module):
    """
    Standard cross-entropy loss wrapper for compatibility.
    
    Used as baseline (lambda = 0) to compare against fairness-aware approaches.
    """
    
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, logits, labels, sensitive_attrs=None):
        """
        Compute standard cross-entropy loss.
        
        Args:
            logits (torch.Tensor): Model output
            labels (torch.Tensor): Ground truth labels
            sensitive_attrs: Ignored (for API compatibility)
        
        Returns:
            tuple: (total_loss, classification_loss, fairness_loss=0)
        """
        loss = self.ce_loss(logits, labels)
        zero = torch.tensor(0.0, device=logits.device)
        return loss, loss, zero


def create_loss(lambda_fair=0.0):
    """
    Factory function to create appropriate loss based on lambda value.
    
    Args:
        lambda_fair (float): Fairness regularization weight
            - 0.0: Returns StandardLoss
            - > 0.0: Returns FairDeepfakeLoss
    
    Returns:
        nn.Module: Configured loss function
    """
    if lambda_fair == 0.0:
        print("Using standard cross-entropy loss (no fairness regularization)")
        return StandardLoss()
    else:
        print(f"Using fairness-aware loss with lambda = {lambda_fair}")
        return FairDeepfakeLoss(lambda_fair=lambda_fair)


if __name__ == '__main__':
    # Example usage and testing
    print("Testing FairDeepfakeLoss...")
    
    # Create dummy data
    batch_size = 16
    num_classes = 2
    num_groups = 5
    
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    labels = torch.randint(0, num_classes, (batch_size,))
    sensitive_attrs = torch.randint(0, num_groups, (batch_size,))
    
    # Test with different lambda values
    for lambda_val in [0.0, 0.5, 2.0]:
        print(f"\nLambda = {lambda_val}")
        loss_fn = create_loss(lambda_fair=lambda_val)
        
        if lambda_val == 0.0:
            total, cls, fair = loss_fn(logits, labels)
        else:
            total, cls, fair = loss_fn(logits, labels, sensitive_attrs)
        
        print(f"  Total Loss: {total.item():.4f}")
        print(f"  Classification Loss: {cls.item():.4f}")
        print(f"  Fairness Loss: {fair.item():.4f}")
    
    print("\nAll tests passed!")
