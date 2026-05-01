"""
Explainability analysis for fairness-aware deepfake detection.

This module implements Integrated Gradients-based attribution analysis
to verify that fairness-aware models achieve equitable predictions
through consistent reasoning patterns across demographic groups.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from captum.attr import IntegratedGradients

from dataset import get_default_transform
from models import load_model


SKIN_TONES = ['light', 'medium-light', 'medium', 'medium-dark', 'dark']


def generate_attribution(model, image_tensor, target_class, device, n_steps=50):
    """
    Generate Integrated Gradients attribution map for an image.
    
    Args:
        model (nn.Module): Trained model
        image_tensor (torch.Tensor): Preprocessed image of shape (1, 3, H, W)
        target_class (int): Target class for attribution (0=real, 1=fake)
        device (torch.device): Device for computation
        n_steps (int): Number of integration steps (higher = more accurate)
    
    Returns:
        np.ndarray: Aggregated attribution map of shape (H, W)
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    # Initialize Integrated Gradients
    ig = IntegratedGradients(model)
    
    # Use black image as baseline
    baseline = torch.zeros_like(image_tensor)
    
    # Compute attributions
    attributions = ig.attribute(
        image_tensor,
        baselines=baseline,
        target=target_class,
        n_steps=n_steps
    )
    
    # Aggregate across RGB channels
    attribution_map = attributions.squeeze().cpu().detach().numpy()
    attribution_map = np.abs(attribution_map).sum(axis=0)
    
    return attribution_map


def visualize_attribution(image_path, attribution_map, title="", save_path=None):
    """
    Create visualization showing original image and attribution heatmap.
    
    Args:
        image_path (str): Path to original image
        attribution_map (np.ndarray): Attribution map from generate_attribution()
        title (str): Title for the visualization
        save_path (str, optional): Path to save figure
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Load original image
    original = Image.open(image_path).convert('RGB')
    original = original.resize((224, 224))
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original image
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attribution heatmap
    im = axes[1].imshow(attribution_map, cmap='hot')
    axes[1].set_title('Attribution Map')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def analyze_demographic_groups(model, test_csv, save_dir, device=None):
    """
    Generate attribution maps across demographic groups for analysis.
    
    Selects representative real and fake samples from each skin tone group
    and generates side-by-side attribution visualizations to enable
    qualitative auditing of demographic fairness in model reasoning.
    
    Args:
        model (nn.Module): Trained model to analyze
        test_csv (str): Path to test annotations CSV
        save_dir (str): Directory to save visualizations
        device (torch.device, optional): Device for computation
    
    Returns:
        list: Paths to generated visualization files
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load annotations
    df = pd.read_csv(test_csv)
    transform = get_default_transform()
    
    print(f"Generating attribution maps for {len(SKIN_TONES)} skin tone groups...")
    
    saved_paths = []
    
    for skin_tone in ['light', 'medium', 'dark']:
        for label_text, label_idx in [('real', 0), ('fake', 1)]:
            # Get representative sample
            mask = (df['skin_tone'] == skin_tone) & (df['label'] == label_idx)
            samples = df[mask]
            
            if len(samples) == 0:
                print(f"  No samples for {skin_tone} {label_text}, skipping")
                continue
            
            sample = samples.iloc[0]
            image_path = sample['image_path']
            
            # Verify image exists
            if not os.path.exists(image_path):
                print(f"  Image not found: {image_path}")
                continue
            
            # Preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)
            
            # Generate attribution
            print(f"  Processing {skin_tone} {label_text}...")
            attribution = generate_attribution(
                model, image_tensor, label_idx, device
            )
            
            # Create visualization
            title = f"{skin_tone.title()} Skin - {label_text.title()} Image"
            save_path = os.path.join(save_dir, f'xai_{skin_tone}_{label_text}.png')
            
            fig = visualize_attribution(image_path, attribution, title, save_path)
            plt.close(fig)
            
            saved_paths.append(save_path)
    
    print(f"\nGenerated {len(saved_paths)} attribution maps")
    return saved_paths


def create_comparison_grid(attribution_paths, save_path):
    """
    Create a grid visualization combining all demographic group attributions.
    
    Args:
        attribution_paths (list): List of paths to individual attribution images
        save_path (str): Path to save the combined grid
    """
    # Organize by skin tone and label
    real_paths = sorted([p for p in attribution_paths if 'real' in p])
    fake_paths = sorted([p for p in attribution_paths if 'fake' in p])
    
    n_groups = max(len(real_paths), len(fake_paths))
    
    if n_groups == 0:
        print("No attribution maps to combine")
        return
    
    fig, axes = plt.subplots(2, n_groups, figsize=(5 * n_groups, 8))
    
    if n_groups == 1:
        axes = axes.reshape(2, 1)
    
    for i, path in enumerate(real_paths[:n_groups]):
        img = Image.open(path)
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        axes[0, i].set_title(os.path.basename(path).replace('xai_', '').replace('.png', ''))
    
    for i, path in enumerate(fake_paths[:n_groups]):
        img = Image.open(path)
        axes[1, i].imshow(img)
        axes[1, i].axis('off')
        axes[1, i].set_title(os.path.basename(path).replace('xai_', '').replace('.png', ''))
    
    fig.suptitle('Attribution Maps Across Demographic Groups', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Comparison grid saved: {save_path}")


def main():
    """Main XAI analysis script."""
    parser = argparse.ArgumentParser(description='Generate XAI attribution maps')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--test_csv', type=str, default='./data/test_annotations.csv',
                        help='Path to test annotations CSV')
    parser.add_argument('--save_dir', type=str, default='./results/xai',
                        help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.model}")
    model = load_model(args.model, device=device)
    
    # Generate attributions
    print(f"\nGenerating attribution maps...")
    attribution_paths = analyze_demographic_groups(
        model=model,
        test_csv=args.test_csv,
        save_dir=args.save_dir,
        device=device
    )
    
    # Create comparison grid
    if attribution_paths:
        grid_path = os.path.join(args.save_dir, 'attribution_grid.png')
        create_comparison_grid(attribution_paths, grid_path)
    
    print(f"\nXAI analysis complete!")
    print(f"Results saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
