"""
Model evaluation functionality for the HotSwapPII.
"""
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import streamlit as st
from presidio_analyzer import RecognizerResult

from config.config import SYSTEM_TO_VALIDATION_MAPPING, VALIDATION_TO_SYSTEM_MAPPING
from core.detector import analyze_text, filter_overlapping_entities
from utils.metrics import (
    calculate_confidence_threshold_curve,
    calculate_entity_type_metrics,
    calculate_overlap_match_metrics,
)

logger = logging.getLogger(__name__)

def process_validation_data(file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Process uploaded validation data in CSV format.
    
    Args:
        file: Uploaded CSV file
        
    Returns:
        Tuple of (processed DataFrame, error message)
        If processing is successful, error message will be None
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Check required columns
        if 'text' not in df.columns:
            return None, "CSV file must contain a 'text' column"
        
        if 'label' not in df.columns:
            return None, "CSV file must contain a 'label' column with annotation data"
            
        # Process the label column
        try:
            # Parse JSON annotation data
            df['annotations'] = df['label'].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x.strip() else 
                         (json.loads(str(x)) if not pd.isna(x) else [])
            )
            
            # Process annotations to standardized format
            df['processed_annotations'] = df.apply(
                lambda row: [
                    {
                        "start": anno["start"], 
                        "end": anno["end"],
                        "label": anno.get("pii_type", "") or 
                                (anno["labels"][0] if "labels" in anno and 
                                 isinstance(anno["labels"], list) and 
                                 len(anno["labels"]) > 0 else "")
                    } 
                    for anno in row['annotations']
                    if isinstance(anno, dict) and "start" in anno and "end" in anno
                ] if isinstance(row['annotations'], list) else [],
                axis=1
            )
            
            return df, None
        except Exception as e:
            logger.error(f"Error processing label column: {str(e)}")
            return None, f"Error processing annotation data: {str(e)}"
            
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        return None, f"Error reading CSV file: {str(e)}"


def analyze_validation_text(
    model_family: str,
    model_path: str,
    text: str,
    threshold: float,
    exclude_overlaps: bool = True,
    overlap_tolerance: int = 1
) -> List[Dict[str, Any]]:
    """
    Analyze text from validation dataset.
    
    Args:
        model_family: The model family (spacy, huggingface, gliner)
        model_path: The specific model path or name
        text: The text to analyze
        threshold: Confidence threshold for detection
        exclude_overlaps: Whether to exclude overlapping entities
        overlap_tolerance: Character tolerance for overlaps
        
    Returns:
        List of spans in the format {'start': int, 'end': int, 'label': str, 'score': float}
    """
    try:
        # Get system entities to detect (all from validation mapping)
        system_entities = list(set(VALIDATION_TO_SYSTEM_MAPPING.values()))
        
        # Analyze the text
        results = analyze_text(
            model_family=model_family,
            model_path=model_path,
            text=text,
            entities=system_entities,
            language="en",
            score_threshold=threshold,
        )
        
        # Filter overlapping entities if needed
        if exclude_overlaps:
            results = filter_overlapping_entities(
                results=results,
                exclude_overlaps=exclude_overlaps,
                overlap_tolerance=overlap_tolerance
            )
        
        # Convert to spans
        spans = []
        for res in results:
            # Map system entity type to validation entity type
            mapped_label = res.entity_type
            for val_type, sys_type in VALIDATION_TO_SYSTEM_MAPPING.items():
                if sys_type == res.entity_type:
                    mapped_label = val_type
                    break
            
            # Create span
            spans.append({
                "start": int(res.start),
                "end": int(res.end),
                "label": mapped_label,
                "original_label": res.entity_type,
                "score": float(res.score)
            })
        
        # Sort by start position
        spans.sort(key=lambda x: x["start"])
        
        return spans
    except Exception as e:
        logger.error(f"Error analyzing validation text: {str(e)}")
        return []


def evaluate_model(
    df: pd.DataFrame,
    model_family: str,
    model_path: str,
    threshold: float,
    exclude_overlaps: bool = True,
    overlap_tolerance: int = 1,
    overlap_threshold: float = 0.5,
    progress_callback: Optional[Callable[[float], None]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """
    Evaluate model on validation dataset.
    
    Args:
        df: DataFrame with validation data
        model_family: The model family (spacy, huggingface, gliner)
        model_path: The specific model path or name
        threshold: Confidence threshold for detection
        exclude_overlaps: Whether to exclude overlapping entities
        overlap_tolerance: Character tolerance for overlaps
        overlap_threshold: IoU threshold for entity matching
        progress_callback: Optional callback for progress updates
        
    Returns:
        Tuple of (results DataFrame, overall metrics, entity-type metrics DataFrame)
    """
    # Initialize result containers
    results = []
    all_ground_truth = []
    all_predictions = []
    
    total_rows = len(df)
    
    # Process each text in the dataset
    for idx, row in df.iterrows():
        # Get text and ground truth
        text = row['text']
        ground_truth = row.get('processed_annotations', [])
        
        # Update progress if callback provided
        if progress_callback:
            progress_callback((idx + 1) / total_rows)
        
        # Skip invalid texts
        if not isinstance(text, str) or not text.strip():
            continue
        
        # Analyze text
        predictions = analyze_validation_text(
            model_family=model_family,
            model_path=model_path,
            text=text,
            threshold=threshold,
            exclude_overlaps=exclude_overlaps,
            overlap_tolerance=overlap_tolerance
        )
        
        # Calculate metrics for this text using the metrics utility
        metrics = calculate_overlap_match_metrics(
            ground_truth=ground_truth,
            predictions=predictions,
            iou_threshold=overlap_threshold,
            require_type_match=True
        )
        
        # Add to results
        results.append({
            "text": text,
            "ground_truth": ground_truth,
            "predictions": predictions,
            **metrics
        })
        
        # Store for overall metrics
        all_ground_truth.append(ground_truth)
        all_predictions.append(predictions)
    
    # Calculate entity-type metrics using the metrics utility
    entity_metrics_df = calculate_entity_type_metrics(
        all_ground_truth=all_ground_truth,
        all_predictions=all_predictions,
        iou_threshold=overlap_threshold
    )
    
    # Calculate overall metrics by aggregating
    overall_metrics = {
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "total_ground_truth": 0,
        "total_predicted": 0
    }
    
    if results:
        # Sum counts
        for key in ['true_positives', 'false_positives', 'false_negatives', 'total_ground_truth', 'total_predicted']:
            overall_metrics[key] = sum(result[key] for result in results)
        
        # Calculate overall metrics
        tp = overall_metrics['true_positives']
        fp = overall_metrics['false_positives']
        fn = overall_metrics['false_negatives']
        
        overall_metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        overall_metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        overall_metrics['f1'] = 2 * (overall_metrics['precision'] * overall_metrics['recall']) / (overall_metrics['precision'] + overall_metrics['recall']) if (overall_metrics['precision'] + overall_metrics['recall']) > 0 else 0
    
    return pd.DataFrame(results), overall_metrics, entity_metrics_df


def analyze_threshold_sensitivity(
    df: pd.DataFrame,
    model_family: str,
    model_path: str,
    thresholds: List[float],
    exclude_overlaps: bool = True,
    overlap_tolerance: int = 1,
    overlap_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Analyze model performance across different confidence thresholds.
    
    Args:
        df: DataFrame with validation data
        model_family: The model family (spacy, huggingface, gliner)
        model_path: The specific model path or name
        thresholds: List of confidence thresholds to evaluate
        exclude_overlaps: Whether to exclude overlapping entities
        overlap_tolerance: Character tolerance for overlaps
        overlap_threshold: IoU threshold for entity matching
        
    Returns:
        DataFrame with performance metrics for each threshold
    """
    # Sample a subset of data for threshold analysis (for performance)
    if len(df) > 10:
        sample_df = df.sample(10, random_state=42)
    else:
        sample_df = df
    
    # Get all ground truth spans
    all_ground_truth = []
    
    # Get all predictions with scores
    all_raw_predictions = []
    
    # Process each text in the sample
    for _, row in sample_df.iterrows():
        # Get text and ground truth
        text = row['text']
        ground_truth = row.get('processed_annotations', [])
        
        # Skip invalid texts
        if not isinstance(text, str) or not text.strip():
            continue
        
        # Add to ground truth list
        all_ground_truth.append(ground_truth)
        
        # Get predictions with a very low threshold to capture all candidates
        predictions = analyze_validation_text(
            model_family=model_family,
            model_path=model_path,
            text=text,
            threshold=0.01,  # Very low threshold to get all candidates
            exclude_overlaps=exclude_overlaps,
            overlap_tolerance=overlap_tolerance
        )
        
        # Add to predictions list
        all_raw_predictions.append(predictions)
    
    # Initialize result container
    threshold_results = []
    
    # Evaluate each threshold
    for threshold in thresholds:
        # Filter predictions for this threshold
        filtered_predictions = []
        
        for preds in all_raw_predictions:
            filtered_preds = [p for p in preds if p['score'] >= threshold]
            filtered_predictions.append(filtered_preds)
        
        # Calculate metrics for this threshold
        tp_sum = fp_sum = fn_sum = 0
        
        for gt, pred in zip(all_ground_truth, filtered_predictions):
            metrics = calculate_overlap_match_metrics(
                ground_truth=gt,
                predictions=pred,
                iou_threshold=overlap_threshold,
                require_type_match=True
            )
            
            tp_sum += metrics['true_positives']
            fp_sum += metrics['false_positives']
            fn_sum += metrics['false_negatives']
        
        # Calculate overall metrics for this threshold
        precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0
        recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Add to results
        threshold_results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp_sum,
            'false_positives': fp_sum,
            'false_negatives': fn_sum
        })
    
    return pd.DataFrame(threshold_results)