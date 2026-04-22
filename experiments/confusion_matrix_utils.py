"""Utility functions for generating stratified confusion matrices across racial groups."""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm


def _extract_logits(model_output):
    """Return a logits tensor from either raw tensor or model output container."""
    if isinstance(model_output, torch.Tensor):
        return model_output
    if hasattr(model_output, "logits"):
        return model_output.logits
    if isinstance(model_output, dict) and "logits" in model_output:
        return model_output["logits"]
    raise TypeError(
        f"Unsupported model output type for logits extraction: {type(model_output)}"
    )


def _forward_logits(model, images):
    """Run a forward pass and normalize outputs to logits tensor.

    Supports torchvision-style calls (`model(images)`) and Hugging Face ViT-style
    calls (`model(pixel_values=images)`).
    """
    try:
        outputs = model(images)
    except TypeError:
        outputs = model(pixel_values=images)
    return _extract_logits(outputs)


def get_predictions_and_labels(model, dataloader, device):
    """
    Generate predictions and labels from a model on a dataloader.
    
    Returns:
        Tuple of (predictions, ground_truth_labels)
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, leave=False, desc="Getting predictions"):
            images = batch["pixel_values"]
            labels = batch["labels"]
            
            if labels.ndim == 1:
                labels = labels.unsqueeze(1)
            labels = labels.float()

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = _forward_logits(model, images)
            if logits.ndim > 1 and logits.shape[-1] > 1:
                preds = torch.argmax(logits, dim=-1, keepdim=True).float()
            else:
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()

            all_preds.extend(preds.cpu().numpy().flatten().astype(np.int64).tolist())
            all_labels.extend(labels.cpu().numpy().flatten().astype(np.int64).tolist())

    return np.array(all_preds), np.array(all_labels)


def compute_confusion_matrices(
    model, 
    eval_groups: Dict[str, object],
    dataloader_dict: Dict[str, object],
    device,
    race_names: List[str]
) -> Dict[str, Dict]:
    """
    Compute confusion matrices for each race group.
    
    Args:
        model: The trained model
        eval_groups: Dict mapping group names to datasets
        dataloader_dict: Dict mapping group names to dataloaders
        device: torch device
        race_names: List of race names for reference
    
    Returns:
        Dict with confusion matrices and metrics for each race
    """
    confusion_matrices = {}
    
    for group_name, loader in dataloader_dict.items():
        preds, labels = get_predictions_and_labels(model, loader, device)
        
        # Compute confusion matrix
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        
        # Extract metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        confusion_matrices[group_name] = {
            "confusion_matrix": cm.tolist(),
            "matrix_labels": ["Female (0)", "Male (1)"],
            "metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "specificity": float(specificity),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
            "n_samples": int(len(labels)),
        }
    
    return confusion_matrices


def create_comparison_summary(
    confusion_matrices: Dict[str, Dict],
    race_names: List[str]
) -> Dict:
    """
    Create a summary comparing metrics across racial groups.
    
    Returns:
        Dict with comparison statistics
    """
    # Extract per-race metrics (skip aggregate groups like "ood::all")
    race_metrics = {}
    
    for group_name, data in confusion_matrices.items():
        # Include per-race groups and skip aggregate group.
        if "ood::all" in group_name:
            continue
        if group_name.startswith("in_dist::") or group_name.startswith("ood::"):
            # Extract race name from group
            if "::" in group_name:
                race_name = group_name.split("::", 1)[1]
            else:
                race_name = group_name
            
            if race_name not in race_metrics:
                race_metrics[race_name] = data["metrics"]
    
    # Calculate disparities
    accuracies = [m["accuracy"] for m in race_metrics.values() if m is not None]
    f1_scores = [m["f1_score"] for m in race_metrics.values() if m is not None]
    
    comparison = {
        "per_race_metrics": race_metrics,
        "accuracy_statistics": {
            "mean": float(np.mean(accuracies)) if accuracies else 0,
            "std": float(np.std(accuracies)) if accuracies else 0,
            "min": float(np.min(accuracies)) if accuracies else 0,
            "max": float(np.max(accuracies)) if accuracies else 0,
            "gap": float(np.max(accuracies) - np.min(accuracies)) if accuracies else 0,
        },
        "f1_score_statistics": {
            "mean": float(np.mean(f1_scores)) if f1_scores else 0,
            "std": float(np.std(f1_scores)) if f1_scores else 0,
            "min": float(np.min(f1_scores)) if f1_scores else 0,
            "max": float(np.max(f1_scores)) if f1_scores else 0,
            "gap": float(np.max(f1_scores) - np.min(f1_scores)) if f1_scores else 0,
        }
    }
    
    return comparison


def save_confusion_matrices(confusion_matrices: Dict[str, Dict], output_dir: str):
    """Save confusion matrices to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    json_path = os.path.join(output_dir, "confusion_matrices.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(confusion_matrices, f, indent=2)
    
    return json_path


def visualize_confusion_matrices(
    confusion_matrices: Dict[str, Dict],
    output_dir: str,
    figsize: Tuple[int, int] = (6, 5)
):
    """
    Generate visualizations of confusion matrices for each race.
    
    Creates both individual matrices and a multi-panel comparison figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create individual confusion matrix plots
    for group_name, data in confusion_matrices.items():
        cm = np.array(data["confusion_matrix"])
        
        # Skip aggregate groups for individual plots
        if "ood::all" in group_name:
            continue
        
        fig, ax = plt.subplots(figsize=figsize)
        
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Female (0)", "Male (1)"]
        )
        disp.plot(ax=ax, cmap="Blues")
        
        # Add metrics as text
        metrics = data["metrics"]
        title = f"{group_name}\nAccuracy: {metrics['accuracy']:.3f} | F1: {metrics['f1_score']:.3f}"
        ax.set_title(title, fontsize=12, fontweight="bold")
        
        plt.tight_layout()
        
        # Save with sanitized filename
        safe_name = group_name.replace("::", "_").replace(" ", "_")
        plt.savefig(os.path.join(output_dir, f"cm_{safe_name}.png"), dpi=100, bbox_inches="tight")
        plt.close()
    
    # Create a comparison figure with key metrics
    race_groups = {
        name: data for name, data in confusion_matrices.items()
        if "ood::all" not in name
    }
    
    if race_groups:
        n_groups = len(race_groups)
        fig, axes = plt.subplots(
            (n_groups + 2) // 3, 3,
            figsize=(15, 5 * ((n_groups + 2) // 3))
        )
        
        if n_groups == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (group_name, data) in enumerate(race_groups.items()):
            cm = np.array(data["confusion_matrix"])
            
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["Female", "Male"]
            )
            disp.plot(ax=axes[idx], cmap="Blues")
            
            metrics = data["metrics"]
            title = f"{group_name}\nAcc: {metrics['accuracy']:.3f} | F1: {metrics['f1_score']:.3f} | n={data['n_samples']}"
            axes[idx].set_title(title, fontsize=10)
        
        # Hide unused subplots
        for idx in range(len(race_groups), len(axes)):
            axes[idx].axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrices_comparison.png"), dpi=100, bbox_inches="tight")
        plt.close()


def create_metrics_comparison_table(
    confusion_matrices: Dict[str, Dict],
    output_dir: str
):
    """
    Create a text summary comparing metrics across races.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    summary_lines = ["=" * 100]
    summary_lines.append("CONFUSION MATRIX ANALYSIS - METRICS BY RACE")
    summary_lines.append("=" * 100)
    summary_lines.append("")
    
    # Create table header
    header = f"{'Group Name':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'n_samples':<12}"
    summary_lines.append(header)
    summary_lines.append("-" * 100)
    
    # Add per-group metrics
    for group_name, data in sorted(confusion_matrices.items()):
        if "ood::all" not in group_name:
            metrics = data["metrics"]
            line = (
                f"{group_name:<30} "
                f"{metrics['accuracy']:<12.4f} "
                f"{metrics['precision']:<12.4f} "
                f"{metrics['recall']:<12.4f} "
                f"{metrics['f1_score']:<12.4f} "
                f"{data['n_samples']:<12d}"
            )
            summary_lines.append(line)
    
    summary_lines.append("-" * 100)
    summary_lines.append("")
    
    # Add statistics
    comparison = create_comparison_summary(confusion_matrices, [])
    stats = comparison["accuracy_statistics"]
    
    summary_lines.append("ACCURACY STATISTICS:")
    summary_lines.append(f"  Mean:  {stats['mean']:.4f}")
    summary_lines.append(f"  Std:   {stats['std']:.4f}")
    summary_lines.append(f"  Min:   {stats['min']:.4f}")
    summary_lines.append(f"  Max:   {stats['max']:.4f}")
    summary_lines.append(f"  Gap:   {stats['gap']:.4f}")
    summary_lines.append("")
    
    summary_text = "\n".join(summary_lines)
    
    # Save to file
    txt_path = os.path.join(output_dir, "confusion_matrices_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    
    print(summary_text)
    
    return txt_path
