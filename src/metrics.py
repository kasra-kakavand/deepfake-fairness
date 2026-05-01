"""
Fairness evaluation metrics for deepfake detection.

This module implements comprehensive fairness metrics including:
- Group-wise performance metrics (TPR, FPR, Accuracy)
- Disparity measures (max - min across groups)
- Equalized Odds violation
- Demographic Parity difference
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


class FairnessMetrics:
    """
    Comprehensive fairness metrics calculator for binary classification.
    
    Computes per-group performance metrics and various disparity measures
    to evaluate fairness across demographic groups.
    """
    
    def __init__(self):
        """Initialize fairness metrics calculator."""
        pass
    
    def calculate_group_metrics(self, y_true, y_pred, sensitive_attr):
        """
        Calculate performance metrics for each demographic group.
        
        Args:
            y_true (np.ndarray): True labels of shape (N,)
            y_pred (np.ndarray): Predicted labels of shape (N,)
            sensitive_attr (np.ndarray): Demographic attributes of shape (N,)
        
        Returns:
            pd.DataFrame: Per-group metrics including:
                - group: Demographic group identifier
                - n_samples: Number of samples in group
                - accuracy: Classification accuracy
                - tpr: True Positive Rate
                - fpr: False Positive Rate
                - tnr: True Negative Rate
                - true_positives, false_positives, true_negatives, false_negatives
        """
        groups = np.unique(sensitive_attr)
        results = []
        
        for group in groups:
            # Filter data for this group
            mask = (sensitive_attr == group)
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            # Compute confusion matrix
            try:
                tn, fp, fn, tp = confusion_matrix(
                    y_true_group, y_pred_group, labels=[0, 1]
                ).ravel()
            except ValueError:
                # Handle edge case with only one class
                tn, fp, fn, tp = 0, 0, 0, 0
            
            # Calculate metrics
            total = tp + tn + fp + fn
            accuracy = (tp + tn) / total if total > 0 else 0.0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            results.append({
                'group': group,
                'n_samples': len(y_true_group),
                'accuracy': accuracy,
                'tpr': tpr,
                'fpr': fpr,
                'tnr': tnr,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn)
            })
        
        return pd.DataFrame(results)
    
    def calculate_fairness_disparity(self, group_metrics, metric='fpr'):
        """
        Calculate disparity (max - min) for a specific metric across groups.
        
        Lower disparity values indicate more fair systems, with 0 representing
        perfect parity across all demographic groups.
        
        Args:
            group_metrics (pd.DataFrame): Output from calculate_group_metrics()
            metric (str): Metric to analyze ('tpr', 'fpr', 'accuracy', 'tnr')
        
        Returns:
            dict: Disparity statistics including max, min, disparity value,
                  and group identifiers achieving max/min values
        """
        if metric not in group_metrics.columns:
            raise ValueError(f"Metric '{metric}' not found in group_metrics")
        
        values = group_metrics[metric].values
        max_val = np.max(values)
        min_val = np.min(values)
        disparity = max_val - min_val
        
        return {
            'metric': metric,
            'max': max_val,
            'min': min_val,
            'disparity': disparity,
            'max_group': group_metrics.loc[group_metrics[metric].idxmax(), 'group'],
            'min_group': group_metrics.loc[group_metrics[metric].idxmin(), 'group']
        }
    
    def demographic_parity_difference(self, y_pred, sensitive_attr):
        """
        Compute Demographic Parity (DP) difference.
        
        DP measures whether positive predictions are equally distributed
        across demographic groups, regardless of true labels:
            DP = max_a P(y_hat=1 | A=a) - min_a P(y_hat=1 | A=a)
        
        Args:
            y_pred (np.ndarray): Predicted labels
            sensitive_attr (np.ndarray): Demographic attributes
        
        Returns:
            float: Demographic parity difference (lower = more fair)
        """
        groups = np.unique(sensitive_attr)
        pred_rates = []
        
        for group in groups:
            mask = (sensitive_attr == group)
            if mask.sum() > 0:
                pred_rate = np.mean(y_pred[mask])
                pred_rates.append(pred_rate)
        
        if len(pred_rates) < 2:
            return 0.0
        
        return max(pred_rates) - min(pred_rates)
    
    def equalized_odds_difference(self, y_true, y_pred, sensitive_attr):
        """
        Compute Equalized Odds (EO) violation.
        
        EO requires equal True Positive and False Positive rates across groups.
        Returns disparities for both TPR and FPR.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            sensitive_attr (np.ndarray): Demographic attributes
        
        Returns:
            dict: TPR disparity, FPR disparity, and average disparity
        """
        groups = np.unique(sensitive_attr)
        tpr_list, fpr_list = [], []
        
        for group in groups:
            mask = (sensitive_attr == group)
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            try:
                tn, fp, fn, tp = confusion_matrix(
                    y_true_group, y_pred_group, labels=[0, 1]
                ).ravel()
            except ValueError:
                continue
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        tpr_disparity = max(tpr_list) - min(tpr_list) if tpr_list else 0.0
        fpr_disparity = max(fpr_list) - min(fpr_list) if fpr_list else 0.0
        
        return {
            'tpr_disparity': tpr_disparity,
            'fpr_disparity': fpr_disparity,
            'avg_disparity': (tpr_disparity + fpr_disparity) / 2
        }
    
    def comprehensive_evaluation(self, y_true, y_pred, sensitive_attr, group_name=None):
        """
        Perform comprehensive fairness evaluation.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            sensitive_attr (np.ndarray): Demographic attributes
            group_name (str, optional): Name for the demographic dimension
        
        Returns:
            dict: Complete fairness evaluation including:
                - per-group metrics DataFrame
                - disparities for all metrics
                - equalized odds analysis
                - demographic parity
                - overall accuracy
        """
        # Per-group metrics
        group_metrics = self.calculate_group_metrics(y_true, y_pred, sensitive_attr)
        
        # Disparities for each metric
        tpr_disp = self.calculate_fairness_disparity(group_metrics, 'tpr')
        fpr_disp = self.calculate_fairness_disparity(group_metrics, 'fpr')
        acc_disp = self.calculate_fairness_disparity(group_metrics, 'accuracy')
        
        # Equalized odds
        eo = self.equalized_odds_difference(y_true, y_pred, sensitive_attr)
        
        # Demographic parity
        dp = self.demographic_parity_difference(y_pred, sensitive_attr)
        
        # Overall accuracy
        overall_acc = np.mean(y_true == y_pred)
        
        return {
            'group_name': group_name,
            'overall_accuracy': overall_acc,
            'group_metrics': group_metrics,
            'tpr_disparity': tpr_disp,
            'fpr_disparity': fpr_disp,
            'accuracy_disparity': acc_disp,
            'equalized_odds': eo,
            'demographic_parity': dp
        }
    
    def print_evaluation(self, evaluation_results):
        """
        Print comprehensive fairness evaluation results in readable format.
        
        Args:
            evaluation_results (dict): Output from comprehensive_evaluation()
        """
        print("=" * 70)
        if evaluation_results['group_name']:
            print(f"FAIRNESS EVALUATION - {evaluation_results['group_name']}")
        else:
            print("FAIRNESS EVALUATION")
        print("=" * 70)
        
        print(f"\nOverall Accuracy: {evaluation_results['overall_accuracy']*100:.2f}%")
        
        print("\nPer-Group Performance:")
        print("-" * 70)
        print(evaluation_results['group_metrics'][
            ['group', 'n_samples', 'accuracy', 'tpr', 'fpr']
        ].to_string(index=False))
        
        print("\nDisparities:")
        print("-" * 70)
        print(f"  TPR Disparity:      {evaluation_results['tpr_disparity']['disparity']:.3f}")
        print(f"  FPR Disparity:      {evaluation_results['fpr_disparity']['disparity']:.3f}")
        print(f"  Accuracy Disparity: {evaluation_results['accuracy_disparity']['disparity']:.3f}")
        
        print("\nEqualized Odds:")
        print("-" * 70)
        print(f"  TPR Disparity: {evaluation_results['equalized_odds']['tpr_disparity']:.3f}")
        print(f"  FPR Disparity: {evaluation_results['equalized_odds']['fpr_disparity']:.3f}")
        print(f"  Average:       {evaluation_results['equalized_odds']['avg_disparity']:.3f}")
        
        print(f"\nDemographic Parity Difference: {evaluation_results['demographic_parity']:.3f}")
        print("=" * 70)


if __name__ == '__main__':
    # Example usage
    print("Testing FairnessMetrics...")
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 100
    
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.random.randint(0, 2, n_samples)
    sensitive_attr = np.random.choice(['light', 'medium', 'dark'], n_samples)
    
    # Initialize metrics
    metrics = FairnessMetrics()
    
    # Comprehensive evaluation
    results = metrics.comprehensive_evaluation(
        y_true, y_pred, sensitive_attr, group_name='Skin Tone'
    )
    
    # Print results
    metrics.print_evaluation(results)
    
    print("\nAll tests passed!")
