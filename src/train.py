"""
Training script for fairness-aware deepfake detection.

This module provides the complete training pipeline including:
- Training and evaluation loops
- Multi-experiment configuration support
- Comprehensive logging and metrics tracking
- Model checkpointing
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DemographicDeepfakeDataset, get_default_transform
from models import create_model, save_model
from losses import create_loss
from metrics import FairnessMetrics


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, criterion, optimizer, device, use_fairness=True):
    """
    Train for one epoch.
    
    Args:
        model: Neural network model
        loader: Training DataLoader
        criterion: Loss function (StandardLoss or FairDeepfakeLoss)
        optimizer: Optimizer
        device: Device to train on
        use_fairness: Whether to use demographic info in loss
    
    Returns:
        dict: Training metrics for this epoch
    """
    model.train()
    running_total_loss = 0.0
    running_class_loss = 0.0
    running_fair_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        if use_fairness:
            skin_tones = batch['skin_tone'].to(device)
            total_loss, class_loss, fair_loss = criterion(outputs, labels, skin_tones)
        else:
            total_loss, class_loss, fair_loss = criterion(outputs, labels)
        
        total_loss.backward()
        optimizer.step()
        
        # Track metrics
        running_total_loss += total_loss.item()
        running_class_loss += class_loss.item()
        running_fair_loss += fair_loss.item()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'fair': f'{fair_loss.item():.4f}',
            'acc': f'{100.*correct/total:.1f}%'
        })
    
    return {
        'total_loss': running_total_loss / len(loader),
        'class_loss': running_class_loss / len(loader),
        'fair_loss': running_fair_loss / len(loader),
        'accuracy': 100. * correct / total
    }


def evaluate(model, loader, device):
    """
    Evaluate model on test set with demographic tracking.
    
    Args:
        model: Trained model
        loader: Test DataLoader
        device: Device to evaluate on
    
    Returns:
        dict: Evaluation results including predictions, labels, and demographics
    """
    model.eval()
    
    all_labels = []
    all_predictions = []
    all_skin_tones = []
    all_genders = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_skin_tones.extend(batch['skin_tone_text'])
            all_genders.extend(batch['gender_text'])
    
    accuracy = 100. * sum(
        l == p for l, p in zip(all_labels, all_predictions)
    ) / len(all_labels)
    
    return {
        'accuracy': accuracy,
        'y_true': np.array(all_labels),
        'y_pred': np.array(all_predictions),
        'skin_tones': np.array(all_skin_tones),
        'genders': np.array(all_genders)
    }


def run_experiment(config):
    """
    Run a single experiment configuration.
    
    Args:
        config (dict): Experiment configuration with keys:
            - name: Experiment identifier
            - train_csv: Path to training annotations
            - test_csv: Path to test annotations
            - lambda_fair: Fairness regularization weight
            - epochs: Number of training epochs
            - batch_size: Batch size
            - learning_rate: Learning rate
            - save_path: Where to save the model
    
    Returns:
        dict: Experiment results including model and evaluation metrics
    """
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {config['name']}")
    print(f"{'='*70}")
    print(f"Lambda: {config['lambda_fair']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create datasets
    transform = get_default_transform()
    train_dataset = DemographicDeepfakeDataset(config['train_csv'], transform=transform)
    test_dataset = DemographicDeepfakeDataset(config['test_csv'], transform=transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=2
    )
    
    # Create model
    model = create_model(device=device)
    
    # Setup loss and optimizer
    criterion = create_loss(lambda_fair=config['lambda_fair'])
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    use_fairness = config['lambda_fair'] > 0.0
    
    # Training loop
    print(f"\nStarting training...")
    history = []
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, use_fairness
        )
        
        history.append({
            'epoch': epoch + 1,
            **train_metrics
        })
        
        print(f"Train Loss: {train_metrics['total_loss']:.4f} | "
              f"Train Acc: {train_metrics['accuracy']:.2f}%")
    
    # Evaluation
    print(f"\nEvaluating on test set...")
    eval_results = evaluate(model, test_loader, device)
    
    # Compute fairness metrics
    fairness_metrics = FairnessMetrics()
    
    # Skin tone analysis
    skin_results = fairness_metrics.comprehensive_evaluation(
        eval_results['y_true'],
        eval_results['y_pred'],
        eval_results['skin_tones'],
        group_name='Skin Tone'
    )
    fairness_metrics.print_evaluation(skin_results)
    
    # Save model
    if config.get('save_path'):
        save_model(
            model,
            config['save_path'],
            optimizer=optimizer,
            epoch=config['epochs'],
            metrics={
                'test_acc': eval_results['accuracy'],
                'tpr_disparity': skin_results['tpr_disparity']['disparity'],
                'fpr_disparity': skin_results['fpr_disparity']['disparity'],
                'history': history
            }
        )
    
    return {
        'config': config,
        'model': model,
        'history': history,
        'eval_results': eval_results,
        'fairness_results': skin_results
    }


def run_all_experiments(data_dir, models_dir):
    """
    Run all four experiments described in the paper.
    
    Args:
        data_dir (str): Directory containing dataset CSVs
        models_dir (str): Directory to save trained models
    
    Returns:
        list: Results from all experiments
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # Define experiments
    experiments = [
        {
            'name': 'E1: Balanced Baseline',
            'train_csv': os.path.join(data_dir, 'train_annotations.csv'),
            'test_csv': os.path.join(data_dir, 'test_annotations.csv'),
            'lambda_fair': 0.0,
            'epochs': 10,
            'batch_size': 16,
            'learning_rate': 1e-4,
            'save_path': os.path.join(models_dir, 'baseline_balanced.pth')
        },
        {
            'name': 'E2: Biased Baseline',
            'train_csv': os.path.join(data_dir, 'biased_train_annotations.csv'),
            'test_csv': os.path.join(data_dir, 'test_annotations.csv'),
            'lambda_fair': 0.0,
            'epochs': 10,
            'batch_size': 16,
            'learning_rate': 1e-4,
            'save_path': os.path.join(models_dir, 'baseline_biased.pth')
        },
        {
            'name': 'E3: Moderate Fairness',
            'train_csv': os.path.join(data_dir, 'biased_train_annotations.csv'),
            'test_csv': os.path.join(data_dir, 'test_annotations.csv'),
            'lambda_fair': 0.5,
            'epochs': 10,
            'batch_size': 16,
            'learning_rate': 1e-4,
            'save_path': os.path.join(models_dir, 'fair_moderate.pth')
        },
        {
            'name': 'E4: Strong Fairness',
            'train_csv': os.path.join(data_dir, 'biased_train_annotations.csv'),
            'test_csv': os.path.join(data_dir, 'test_annotations.csv'),
            'lambda_fair': 2.0,
            'epochs': 10,
            'batch_size': 16,
            'learning_rate': 1e-4,
            'save_path': os.path.join(models_dir, 'fair_strong.pth')
        }
    ]
    
    # Run all experiments
    all_results = []
    for exp_config in experiments:
        results = run_experiment(exp_config)
        all_results.append(results)
    
    # Print summary table
    print("\n" + "="*70)
    print("FINAL COMPARISON ACROSS ALL EXPERIMENTS")
    print("="*70)
    
    summary = []
    for results in all_results:
        summary.append({
            'Experiment': results['config']['name'],
            'Lambda': results['config']['lambda_fair'],
            'Accuracy': f"{results['eval_results']['accuracy']:.2f}%",
            'TPR Disparity': f"{results['fairness_results']['tpr_disparity']['disparity']:.3f}",
            'FPR Disparity': f"{results['fairness_results']['fpr_disparity']['disparity']:.3f}"
        })
    
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    print("="*70)
    
    return all_results


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Train fairness-aware deepfake detection models'
    )
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing dataset CSVs')
    parser.add_argument('--models_dir', type=str, default='./models',
                        help='Directory to save trained models')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    print("Fairness-Aware Deepfake Detection")
    print("=" * 70)
    print(f"Data directory: {args.data_dir}")
    print(f"Models directory: {args.models_dir}")
    print(f"Seed: {args.seed}")
    print("=" * 70)
    
    # Run all experiments
    results = run_all_experiments(args.data_dir, args.models_dir)
    
    print("\nAll experiments completed successfully!")


if __name__ == '__main__':
    main()
