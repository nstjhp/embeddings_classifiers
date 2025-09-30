import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    fbeta_score,
)
from sklearn.utils import compute_sample_weight

# --- Data Loading ---

def load_data(
    path: str,
    *,
    balance: bool = True,
    n_negatives: int = None,
    negative_ratio: float = None,
    random_state: int = None,
    shuffle: bool = True,
    label_col: str = "label",
    drop_cols: tuple = ("label", "h5_index"), # protein is handled separately
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads and optionally downsamples the negative class from a CSV file.

    Args:
        path: Path to the input CSV file.
        balance: If True, downsample the negative class.
        n_negatives: If provided, sample exactly this many negative instances.
        negative_ratio: If provided, sample neg_ratio * n_positives instances.
                        If n_negatives is also given, n_negatives takes precedence.
        random_state: Seed for reproducibility of sampling and shuffling.
        shuffle: If True, shuffle the final dataset.
        label_col: Name of the column containing the labels (0 or 1).
        drop_cols: Columns to drop to create the feature matrix X.

    Returns:
        A tuple containing the feature DataFrame (X), the label Series (y),
        and the protein ID Series.
    """
    print(f"Loading data from {path}...")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {path}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file {path}: {e}")
    
    print(f"Initial dataset shape: {df.shape}")
    
    # Validate required columns exist
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in data. Available columns: {list(df.columns)}")
    
    if "protein" not in df.columns:
        raise ValueError(f"'protein' column not found in data. Available columns: {list(df.columns)}")

    if balance:
        positives = df[df[label_col] == 1]
        negatives = df[df[label_col] == 0]
        n_pos = len(positives)
        
        if n_negatives is not None:
            n_neg_to_sample = n_negatives
        elif negative_ratio is not None:
            n_neg_to_sample = int(negative_ratio * n_pos)
        else:
            n_neg_to_sample = n_pos  # Default to 1:1 ratio

        print(f"Balancing data: {n_pos} positives and sampling {n_neg_to_sample} negatives.")
        
        if n_neg_to_sample > len(negatives):
            print(f"Warning: Requested {n_neg_to_sample} negatives, but only {len(negatives)} are available. Using all negatives.")
            n_neg_to_sample = len(negatives)

        sampled_negatives = negatives.sample(n=n_neg_to_sample, random_state=random_state)
        df = pd.concat([positives, sampled_negatives])
        print(f"Balanced dataset shape: {df.shape}")

    if shuffle:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    y = df[label_col]
    protein_ids = df["protein"]
    
    # Create feature matrix by dropping specified cols and also protein/label
    cols_to_drop = list(drop_cols) + [label_col, "protein"]
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    return X, y, protein_ids


# --- Metrics Calculation ---

def calculate_at_k_metrics(y_true: np.ndarray, y_score: np.ndarray, k_values: list[int]) -> dict:
    """Calculates Precision@K and Recall@K for a list of k values."""
    if y_score is None or len(np.unique(y_score)) < 2:
        # Handle constant-score case (e.g., DummyClassifier)
        prevalence = y_true.mean()
        metrics = {f"P@{k}": prevalence for k in k_values}
        metrics.update({f"R@{k}": k / len(y_true) if len(y_true) > 0 else 0 for k in k_values})
        return metrics

    # Sort by score descending
    desc_score_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_score_indices]

    metrics = {}
    total_positives = np.sum(y_true)

    for k in k_values:
        if k > len(y_true):
            continue
        
        top_k = y_true_sorted[:k]
        
        precision_at_k = np.mean(top_k) if k > 0 else 0.0
        recall_at_k = np.sum(top_k) / total_positives if total_positives > 0 else 0.0
        
        metrics[f"P@{k}"] = precision_at_k
        metrics[f"R@{k}"] = recall_at_k

    metrics["positives_total"] = int(total_positives)
    return metrics

def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculates Expected Calibration Error (ECE) using quantile binning."""
    probs = np.asarray(y_prob)
    y_true = np.asarray(y_true).astype(int)
    # Use quantile binning; handle constant probs
    if np.allclose(np.std(probs), 0.0):
        # single bin
        acc = y_true.mean()
        conf = probs.mean()
        return float(abs(acc - conf))
    # quantile edges
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(probs, qs)
    # ensure strictly increasing edges
    edges = np.unique(edges)
    if len(edges) <= 2:
        # degenerate: fall back to uniform bins
        edges = np.linspace(probs.min(), probs.max() + 1e-12, min(n_bins, len(np.unique(probs))) + 1)
    total = len(probs)
    ece = 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not np.any(mask):
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = probs[mask].mean()
        w = mask.mean()
        ece += abs(bin_acc - bin_conf) * w
    return float(ece)

# --- Thresholding ---

def find_operating_point(y_true: np.ndarray, y_score: np.ndarray, strategy: str) -> float:
    """Finds the optimal threshold on a validation set based on a given strategy."""
    if strategy == 'youden':
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        # Add a small epsilon to handle cases where tpr and fpr are equal
        idx = np.argmax(tpr - fpr + 1e-9)
        return thresholds[idx]

    prec, recall, thresholds = precision_recall_curve(y_true, y_score)
    # The last precision is 1, recall is 0, with no threshold. We remove it.
    prec, recall = prec[:-1], recall[:-1]
    
    # Ensure thresholds, precision and recall arrays are not empty
    if len(thresholds) == 0:
        return 0.5 # Default fallback

    if strategy == 'f05':
        # Calculate F-beta score for each threshold
        # F-beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
        # For beta=0.5, this weights precision more heavily than recall
        beta = 0.5
        beta_squared = beta ** 2
        # Avoid division by zero
        f05_scores = []
        for i, thresh in enumerate(thresholds):
            if prec[i] == 0 and recall[i] == 0:
                f05_scores.append(0)
            else:
                f05 = (1 + beta_squared) * (prec[i] * recall[i]) / ((beta_squared * prec[i]) + recall[i] + 1e-9)
                f05_scores.append(f05)
        
        f05_scores = np.array(f05_scores)
        idx = np.argmax(f05_scores)
        return thresholds[idx]
    
    if strategy.startswith('fpr@'):
        try:
            fpr_target = int(strategy.split('@')[1]) / 100.0
        except (IndexError, ValueError):
            raise ValueError(f"Invalid fpr@ strategy format: {strategy}. Expected format: 'fpr@X' where X is an integer (e.g., 'fpr@1', 'fpr@5')")
        
        fpr, _, thresholds_roc = roc_curve(y_true, y_score)
        # Find the highest threshold that gives FPR <= target
        valid_indices = np.where(fpr <= fpr_target)[0]
        if len(valid_indices) == 0:
            return thresholds_roc[np.argmin(fpr)] # Fallback to lowest FPR
        return thresholds_roc[valid_indices[-1]]
    
    if strategy.startswith('ppv@'):
        try:
            ppv_target = int(strategy.split('@')[1]) / 100.0
        except (IndexError, ValueError):
            raise ValueError(f"Invalid ppv@ strategy format: {strategy}. Expected format: 'ppv@X' where X is an integer (e.g., 'ppv@30', 'ppv@50')")
        
        # Find the lowest threshold that gives precision >= target
        valid_indices = np.where(prec >= ppv_target)[0]
        if len(valid_indices) == 0:
            return thresholds[np.argmax(prec)] # Fallback to best precision
        return thresholds[valid_indices[0]]
        
    raise ValueError(f"Unknown operating point strategy: {strategy}")

# --- Model Tuning ---

HGB_PARAM_GRID = {
    'learning_rate': [0.02, 0.05, 0.1],
    'max_leaf_nodes': [31, 63, 127],
    'min_samples_leaf': [20, 50, 100],
    'l2_regularization': [0.0, 0.5, 1.0],
}

XGB_PARAM_GRID = {
    'learning_rate': [0.03, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'min_child_weight': [1, 5, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_lambda': [0, 1, 5],
    'reg_alpha': [0, 0.1, 0.5],
}
