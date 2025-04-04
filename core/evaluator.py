"""
Model evaluation functionality for the HotSwapPII.
"""
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import streamlit as st
from presidio_analyzer import RecognizerResult, EntityRecognizer, AnalyzerEngine

from config.config import SYSTEM_TO_VALIDATION_MAPPING, VALIDATION_TO_SYSTEM_MAPPING, MODEL_DETAILS
from core.detector import analyze_text, filter_overlapping_entities, initialize_analyzer_engine
from models.model_factory import get_independent_model
from utils.metrics import (
    calculate_confidence_threshold_curve,
    calculate_entity_type_metrics,
    calculate_overlap_match_metrics,
    get_nervaluate_metrics
)

from config.config import MODEL_ENTITIES_TO_STANDARDIZED_ENTITY_MAPPING

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
    _analyzer: AnalyzerEngine,
    text: str,
    threshold: float,
    entities: Optional[List[str]] = None,
    exclude_overlaps: bool = True,
    overlap_tolerance: int = 1,
    ad_hoc_recognizers: List[EntityRecognizer] = None,
    model_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Analyze text from validation dataset.
    
    Args:
        _analyzer: The analyzer engine to use
        text: The text to analyze
        threshold: Confidence threshold for detection
        entities: List of entities to detect
        exclude_overlaps: Whether to exclude overlapping entities
        overlap_tolerance: Character tolerance for overlaps
        ad_hoc_recognizers: List of ad-hoc recognizers
        model_name: Optional name of the model producing these predictions
        
    Returns:
        List of spans in the format {'start': int, 'end': int, 'label': str, 'score': float, 'model': str}
    """
    try:
        # Get system entities to detect (all from validation mapping)
        if entities is None:
            entities = list(set(VALIDATION_TO_SYSTEM_MAPPING.values()))

        # Analyze the text
        results = analyze_text(
            _analyzer=_analyzer,
            text=text,
            entities=entities,
            language="en",
            score_threshold=threshold,
            ad_hoc_recognizers=ad_hoc_recognizers
        )

        # Convert to spans
        spans = []
        for res in results:
            spans.append({
                "start": int(res.start),
                "end": int(res.end),
                "label": MODEL_ENTITIES_TO_STANDARDIZED_ENTITY_MAPPING.get(res.entity_type, 'UNKNOWN'),
                "original_label": res.entity_type,
                "score": float(res.score),
                "model": model_name or "default"
            })
        
        # Sort by start position
        spans.sort(key=lambda x: x["start"])
        
        return spans
    except Exception as e:
        logger.error(f"Error analyzing validation text: {str(e)}")
        return []


def resolve_overlapping_predictions(
    predictions: List[Dict[str, Any]],
    custom_pipeline: Dict[str, str],
    overlap_tolerance: int = 1
) -> List[Dict[str, Any]]:
    """
    Resolve overlapping predictions from different models based on custom pipeline configuration.
    
    Args:
        predictions: List of predictions from different models
        custom_pipeline: Dictionary mapping entity types to model names
        overlap_tolerance: Character tolerance for overlaps
        
    Returns:
        List of resolved predictions
    """
    if not predictions:
        return []
        
    # Sort predictions by start position
    predictions.sort(key=lambda x: x["start"])
    
    # Group overlapping predictions
    groups = []
    current_group = [predictions[0]]
    
    for pred in predictions[1:]:
        # Check if current prediction overlaps with the last one in current group
        last_pred = current_group[-1]
        if (pred["start"] <= last_pred["end"] + overlap_tolerance and 
            pred["end"] >= last_pred["start"] - overlap_tolerance):
            current_group.append(pred)
        else:
            groups.append(current_group)
            current_group = [pred]
    
    groups.append(current_group)
    
    # Resolve each group
    resolved_predictions = []
    for group in groups:
        if len(group) == 1:
            # No overlap, keep the prediction
            resolved_predictions.append(group[0])
        else:
            # Find the best prediction for this overlap
            best_pred = None
            best_score = -1
            
            for pred in group:
                # Check if this model is responsible for this entity type
                is_responsible = False
                for entity_type, model_name in custom_pipeline.items():
                    if (pred["label"] == entity_type and 
                        pred["model"] == model_name):
                        is_responsible = True
                        break
                
                # If this model is responsible for this entity type
                if is_responsible:
                    if pred["score"] > best_score:
                        best_pred = pred
                        best_score = pred["score"]
            
            # If no responsible model found, use highest score
            if best_pred is None:
                best_pred = max(group, key=lambda x: x["score"])
            
            resolved_predictions.append(best_pred)
    
    return resolved_predictions


def evaluate_model(
    df: pd.DataFrame,
    base_model: str,
    model_family: str,
    model_path: str,
    threshold: float,
    entities: Optional[List[str]] = None,
    exclude_overlaps: bool = True,
    overlap_tolerance: int = 1,
    overlap_threshold: float = 0.5,
    progress_callback: Optional[Callable[[float], None]] = None,
    custom_pipeline: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate model on validation dataset.
    
    Args:
        df: DataFrame with validation data
        base_model: presidio or independent
        model_family: The model family (spacy, huggingface, gliner)
        model_path: The specific model path or name
        threshold: Confidence threshold for detection
        exclude_overlaps: Whether to exclude overlapping entities
        overlap_tolerance: Character tolerance for overlaps
        overlap_threshold: IoU threshold for entity matching
        progress_callback: Optional callback for progress updates
        custom_pipeline: Optional dictionary mapping entity types to model names
        
    Returns:
        Tuple of (results DataFrame, overall metrics, entity-type metrics DataFrame)
    """
    # Initialize result containers
    results = []
    all_ground_truth = []
    all_predictions = []
    
    total_rows = len(df)

    # Initialize models based on whether custom pipeline is being used
    if custom_pipeline:
        # Initialize models for custom pipeline
        models = {}
        for entity_type, model_name in custom_pipeline.items():
            print('DETAILS:', entity_type, model_name)
            # Get model details from MODEL_DETAILS
            model_info = next(
                (info for name, info in MODEL_DETAILS.items() if name == model_name),
                None
            )
            if model_info:
                if model_info["base_model"] == "presidio":
                    analyzer, ad_hoc_recognizers = initialize_analyzer_engine(
                        model_family=model_info["model_family"],
                        model_path=model_info["model_path"]
                    )
                    models[entity_type] = (analyzer, ad_hoc_recognizers, model_name)
                else:
                    model = get_independent_model(
                        model_family=model_info["model_family"],
                        model_path=model_info["model_path"]
                    )
                    models[entity_type] = (model, [], model_name)
    else:
        # Initialize single model for regular evaluation
        if base_model == "presidio":
            analyzer, ad_hoc_recognizers = initialize_analyzer_engine(
                model_family=model_family,
                model_path=model_path
            )
        else:
            analyzer = get_independent_model(
                model_family=model_family,
                model_path=model_path
            )
            ad_hoc_recognizers = []

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

        # Analyze text using analyze_validation_text for both custom and non-custom pipeline
        if custom_pipeline:
            # Process with custom pipeline
            all_predictions_from_models = []
            for entity_type, (model, ad_hoc_recognizers, model_name) in models.items():
                # Get predictions from each model with all entities
                entity_predictions = analyze_validation_text(
                    _analyzer=model,
                    text=text,
                    entities=list(set(VALIDATION_TO_SYSTEM_MAPPING.values())),
                    threshold=threshold,
                    exclude_overlaps=False,  # Don't filter overlaps yet
                    overlap_tolerance=overlap_tolerance,
                    ad_hoc_recognizers=ad_hoc_recognizers,
                    model_name=model_name
                )
                all_predictions_from_models.extend(entity_predictions)

            print(all_predictions_from_models)
            # Resolve overlapping predictions
            predictions = resolve_overlapping_predictions(
                predictions=all_predictions_from_models,
                custom_pipeline=custom_pipeline,
                overlap_tolerance=overlap_tolerance
            )
        else:
            # Use regular model analysis
            predictions = analyze_validation_text(
                _analyzer=analyzer,
                text=text,
                entities=entities,
                threshold=threshold,
                exclude_overlaps=exclude_overlaps,
                overlap_tolerance=overlap_tolerance,
                ad_hoc_recognizers=ad_hoc_recognizers
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

    # Get nervaluate metrics
    nervaluate_overall_metrics, nervaluate_entity_metrics, _, _ = get_nervaluate_metrics(
        all_ground_truth,
        all_predictions
    )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return results_df, metrics, entity_metrics_df, nervaluate_overall_metrics, nervaluate_entity_metrics


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