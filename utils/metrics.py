"""
Metrics calculation utilities for the HotSwapPII.
"""
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from nervaluate import Evaluator
from config.config import GROUND_TRUTH_TAGS

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def calculate_span_iou(
    span1: Dict[str, Any],
    span2: Dict[str, Any]
) -> float:
    """
    Calculate the Intersection over Union (IoU) between two spans.
    
    Args:
        span1: First span with 'start' and 'end' keys
        span2: Second span with 'start' and 'end' keys
        
    Returns:
        IoU value between 0 and 1
    """
    # Get start and end positions
    start1, end1 = span1['start'], span1['end']
    start2, end2 = span2['start'], span2['end']
    
    # Calculate intersection
    start_i = max(start1, start2)
    end_i = min(end1, end2)
    
    if start_i >= end_i:
        # No overlap
        return 0.0
    
    intersection = end_i - start_i
    
    # Calculate union
    span1_length = end1 - start1
    span2_length = end2 - start2
    union = span1_length + span2_length - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0.0
    
    return iou


def find_best_match(
    gt_span: Dict[str, Any],
    pred_spans: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
    require_type_match: bool = True
) -> Tuple[Optional[int], float]:
    """
    Find the best matching prediction for a ground truth span.
    
    Args:
        gt_span: Ground truth span
        pred_spans: List of prediction spans
        iou_threshold: Minimum IoU to consider a match
        require_type_match: Whether entity types must match
        
    Returns:
        Tuple of (best match index or None, best IoU score)
    """
    best_match_idx = None
    best_iou = iou_threshold  # Initialize to threshold
    
    for i, pred_span in enumerate(pred_spans):
        # Skip if entity types don't match and type matching is required
        if require_type_match and gt_span.get('label') != pred_span.get('label'):
            continue
        
        # Calculate IoU
        iou = calculate_span_iou(gt_span, pred_span)
        
        # Check if better than current best
        if iou > best_iou:
            best_iou = iou
            best_match_idx = i
    
    return best_match_idx, best_iou


def calculate_binary_metrics(
    tp: int,
    fp: int,
    fn: int
) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        tp: Number of true positives
        fp: Number of false positives
        fn: Number of false negatives
        
    Returns:
        Dictionary with metrics
    """
    # Calculate precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Calculate recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def calculate_exact_match_metrics(
    ground_truth: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate metrics based on exact span matching.
    
    Args:
        ground_truth: List of ground truth spans
        predictions: List of predicted spans
        
    Returns:
        Dictionary with metrics
    """
    # Create sets for exact matching
    gt_spans = set((span['start'], span['end'], span.get('label', '')) for span in ground_truth)
    pred_spans = set((span['start'], span['end'], span.get('label', '')) for span in predictions)
    
    # Calculate matches
    true_positives = len(gt_spans.intersection(pred_spans))
    false_positives = len(pred_spans) - true_positives
    false_negatives = len(ground_truth) - true_positives
    
    # Calculate metrics
    metrics = calculate_binary_metrics(true_positives, false_positives, false_negatives)
    
    # Add counts
    metrics.update({
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_ground_truth": len(ground_truth),
        "total_predicted": len(predictions)
    })
    
    return metrics


def calculate_overlap_match_metrics(
    ground_truth: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
    require_type_match: bool = True
) -> Dict[str, Any]:
    """
    Calculate metrics based on span overlap matching.
    
    Args:
        ground_truth: List of ground truth spans
        predictions: List of predicted spans
        iou_threshold: Minimum IoU to consider a match
        require_type_match: Whether entity types must match
        
    Returns:
        Dictionary with metrics
    """
    # Track matches
    matched_gt = set()
    matched_pred = set()
    
    # For each ground truth span, find the best matching prediction
    for gt_idx, gt_span in enumerate(ground_truth):
        best_pred_idx, best_iou = find_best_match(
            gt_span,
            predictions,
            iou_threshold,
            require_type_match
        )
        
        if best_pred_idx is not None:
            matched_gt.add(gt_idx)
            matched_pred.add(best_pred_idx)
    
    # Calculate metrics
    true_positives = len(matched_gt)
    false_positives = len(predictions) - len(matched_pred)
    false_negatives = len(ground_truth) - len(matched_gt)
    
    metrics = calculate_binary_metrics(true_positives, false_positives, false_negatives)
    
    # Add counts
    metrics.update({
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_ground_truth": len(ground_truth),
        "total_predicted": len(predictions),
        "matched_gt_indices": list(matched_gt),
        "matched_pred_indices": list(matched_pred)
    })
    
    return metrics


def calculate_entity_type_metrics(
    all_ground_truth: List[List[Dict[str, Any]]],
    all_predictions: List[List[Dict[str, Any]]],
    iou_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Calculate metrics for each entity type.
    
    Args:
        all_ground_truth: List of lists of ground truth spans by document
        all_predictions: List of lists of predicted spans by document
        iou_threshold: IoU threshold for span matching
        
    Returns:
        DataFrame with metrics by entity type
    """
    # Track metrics for each entity type
    entity_metrics = {}
    
    # Process each document
    for doc_gt, doc_pred in zip(all_ground_truth, all_predictions):
        # Track matched spans
        matched_gt = set()
        matched_pred = set()
        
        # For each ground truth span, find the best matching prediction
        for gt_idx, gt_span in enumerate(doc_gt):
            entity_type = gt_span.get('label', '')
            
            # Initialize entity metrics if not already present
            if entity_type not in entity_metrics:
                entity_metrics[entity_type] = {
                    'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0, 'total_pred': 0
                }
            
            # Count this ground truth instance
            entity_metrics[entity_type]['total_gt'] += 1
            
            # Find best matching prediction
            best_pred_idx, best_iou = find_best_match(
                gt_span,
                doc_pred,
                iou_threshold,
                require_type_match=True
            )
            
            # Count true positive or false negative
            if best_pred_idx is not None:
                entity_metrics[entity_type]['tp'] += 1
                matched_gt.add(gt_idx)
                matched_pred.add(best_pred_idx)
            else:
                entity_metrics[entity_type]['fn'] += 1
        
        # Count entity types in predictions and false positives
        for pred_idx, pred_span in enumerate(doc_pred):
            entity_type = pred_span.get('label', '')
            
            # Initialize entity metrics if not already present
            if entity_type not in entity_metrics:
                entity_metrics[entity_type] = {
                    'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0, 'total_pred': 0
                }
            
            # Count this prediction
            entity_metrics[entity_type]['total_pred'] += 1
            
            # Count false positive if not matched
            if pred_idx not in matched_pred:
                entity_metrics[entity_type]['fp'] += 1
    
    # Calculate metrics for each entity type
    entity_results = []
    for entity_type, metrics in entity_metrics.items():
        # Calculate binary metrics
        binary_metrics = calculate_binary_metrics(
            metrics['tp'],
            metrics['fp'],
            metrics['fn']
        )
        
        # Add to results
        entity_results.append({
            'entity_type': entity_type,
            'precision': binary_metrics['precision'],
            'recall': binary_metrics['recall'],
            'f1': binary_metrics['f1'],
            'true_positives': metrics['tp'],
            'false_positives': metrics['fp'],
            'false_negatives': metrics['fn'],
            'total_ground_truth': metrics['total_gt'],
            'total_predicted': metrics['total_pred']
        })
    
    # Convert to DataFrame and sort by ground truth count (most common first)
    df = pd.DataFrame(entity_results)
    if not df.empty:
        df = df.sort_values(by='total_ground_truth', ascending=False)
    
    return df


def calculate_confidence_threshold_curve(
    ground_truth: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    thresholds: List[float],
    iou_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Calculate precision, recall, and F1 for different confidence thresholds.
    
    Args:
        ground_truth: List of ground truth spans
        predictions: List of predicted spans with 'score' key
        thresholds: List of confidence thresholds to evaluate
        iou_threshold: IoU threshold for span matching
        
    Returns:
        DataFrame with metrics for each threshold
    """
    results = []
    
    for threshold in thresholds:
        # Filter predictions by threshold
        filtered_preds = [p for p in predictions if p.get('score', 0) >= threshold]
        
        # Calculate metrics
        metrics = calculate_overlap_match_metrics(
            ground_truth,
            filtered_preds,
            iou_threshold
        )
        
        # Add to results
        results.append({
            'threshold': threshold,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'true_positives': metrics['true_positives'],
            'false_positives': metrics['false_positives'],
            'false_negatives': metrics['false_negatives'],
            'total_predictions': len(filtered_preds)
        })

    return pd.DataFrame(results)

def get_nervaluate_metrics(true, pred):
    evaluator = Evaluator(true, pred, tags=GROUND_TRUTH_TAGS)
    # evaluator = Evaluator(true, pred, tags=VALIDATION_TO_SYSTEM_MAPPING.keys())
    # Returns overall metrics and metrics for each tag
    results, results_per_tag, result_indices, result_indices_by_tag = evaluator.evaluate()

    return results, results_per_tag, result_indices, result_indices_by_tag