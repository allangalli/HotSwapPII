"""
Core PII detection logic for the HotSwapPII.
"""
import logging
from typing import Any, Dict, List, Optional, Set, Union

import streamlit as st
from presidio_analyzer import (Pattern, PatternRecognizer,
                               RecognizerResult, AnalyzerEngine, RecognizerRegistry, EntityRecognizer)

from models.model_factory import get_analyzer_engine, get_independent_model
from config.config import MODEL_DETAILS

logger = logging.getLogger(__name__)

# cache?
def initialize_analyzer_engine(
    model_family: str,
    model_path: str,
    deny_list: Optional[List[str]] = None,
    regex_pattern: Optional[str] = None,
    regex_entity_type: Optional[str] = None,
    regex_score: Optional[float] = None,
    regex_context: Optional[List[str]] = None
) -> (AnalyzerEngine, List[EntityRecognizer]):
    """
        Analyze text to detect PII entities.

        Args:
            model_family: The model family (spacy, huggingface, gliner)
            model_path: The specific model path or name
            text: The text to analyze
            entities: List of entities to detect. If None, all supported entities are detected.
            language: The language of the text
            score_threshold: Minimum confidence score for detected entities
            return_decision_process: Whether to return the decision process in the results
            allow_list: List of words to ignore (not considered PII)
            deny_list: List of words to always flag as PII
            regex_pattern: Custom regex pattern for detection
            regex_entity_type: Entity type to assign to regex matches
            regex_score: Confidence score for regex matches
            regex_context: Context words that increase the confidence of regex matches

        Returns:
            A list of RecognizerResult objects
        """

    analyzer = get_analyzer_engine(model_family, model_path)

    create_default_regex_recognizers(analyzer)

    # Set up ad-hoc recognizers if needed
    ad_hoc_recognizers = []

    # Create deny list recognizer if needed
    if deny_list and len(deny_list) > 0:
        deny_list_recognizer = create_deny_list_recognizer(deny_list)
        if deny_list_recognizer:
            ad_hoc_recognizers.append(deny_list_recognizer)

    # Create regex recognizer if needed
    if regex_pattern and regex_entity_type:
        regex_recognizer = create_regex_recognizer(
            regex_pattern,
            regex_entity_type,
            regex_score or 0.6,
            regex_context
        )
        if regex_recognizer:
            ad_hoc_recognizers.append(regex_recognizer)

    return analyzer, ad_hoc_recognizers

# @st.cache_data
def analyze_text(
    _analyzer: AnalyzerEngine,
    text: str,
    entities: Optional[List[str]] = None,
    language: str = "en",
    score_threshold: float = 0.35,
    return_decision_process: bool = False,
    allow_list: Optional[List[str]] = None,
    ad_hoc_recognizers: List[EntityRecognizer] = None,
) -> List[RecognizerResult]:
    """
    Analyze text to detect PII entities.
    
    Args:
        _analyzer:
        text: The text to analyze
        entities: List of entities to detect. If None, all supported entities are detected.
        language: The language of the text
        score_threshold: Minimum confidence score for detected entities
        return_decision_process: Whether to return the decision process in the results
        allow_list: List of words to ignore (not considered PII)
        ad_hoc_recognizers:
        
    Returns:
        A list of RecognizerResult objects
    """
    if not text:
        return []


    # Analyze the text
    results = _analyzer.analyze(
        text=text,
        entities=entities,
        language=language,
        # score_threshold=score_threshold,
        # return_decision_process=return_decision_process,
        # allow_list=allow_list,
        # ad_hoc_recognizers=ad_hoc_recognizers if ad_hoc_recognizers else None,
    )

    return results


def process_with_custom_pipeline(
    text: str,
    custom_pipeline: Dict[str, str],
    settings: Dict[str, Any]
) -> List[RecognizerResult]:
    """
    Process text through a custom pipeline of models, where each model is responsible
    for detecting specific entity types.
    
    Args:
        text: The text to analyze
        custom_pipeline: Dictionary mapping entity types to model selections
        settings: Dictionary of global settings
        
    Returns:
        Combined list of RecognizerResult objects
    """
    if not text or not custom_pipeline:
        return []
    
    logger.info(f"Processing with custom pipeline: {len(custom_pipeline.keys())} entity types")
    
    # Get global settings
    threshold = settings.get("threshold", 0.35)
    allow_list = settings.get("allow_list", [])
    deny_list = settings.get("deny_list", [])
    regex_pattern = settings.get("regex_pattern", "")
    regex_entity_type = settings.get("regex_entity_type", "")
    regex_score = settings.get("regex_score", 0.6)
    regex_context = settings.get("regex_context", [])
    
    # Group entity types by model to avoid redundant processing
    model_to_entities = {}
    for entity_type, model_name in custom_pipeline.items():
        if model_name not in model_to_entities:
            model_to_entities[model_name] = []
        model_to_entities[model_name].append(entity_type)
    
    # Process text through each model
    all_results = []
    
    for model_name, entity_types in model_to_entities.items():
        # Get model details
        model_details = MODEL_DETAILS.get(model_name, {})
        if not model_details:
            logger.warning(f"Model {model_name} not found in MODEL_DETAILS")
            continue
        
        base_model = model_details.get("base_model")
        model_family = model_details.get("model_family")
        model_path = model_details.get("model_path")
        
        if not all([base_model, model_family, model_path]):
            logger.warning(f"Incomplete model details for {model_name}")
            continue
        
        logger.info(f"Processing with model {model_name} for entity types: {entity_types}")
        
        # Initialize analyzer for this model
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
        
        # Process text with this model
        if base_model == "presidio":
            results = analyze_text(
                _analyzer=analyzer,
                text=text,
                entities=entity_types,
                language="en",
                score_threshold=threshold,
                allow_list=allow_list,
                ad_hoc_recognizers=ad_hoc_recognizers,
            )
        else:
            # Handle non-presidio models if needed
            # (This would need to be implemented)
            results = []
            logger.warning(f"Non-Presidio models not yet supported in custom pipeline: {base_model}")
        
        # Keep only the entities that this model is responsible for
        filtered_results = [r for r in results if r.entity_type in entity_types]
        all_results.extend(filtered_results)
    
    # Sort all results by start position
    sorted_results = sorted(all_results, key=lambda x: (x.start, -x.score))
    
    # Handle overlapping entities from different models
    final_results = []
    i = 0
    while i < len(sorted_results):
        current = sorted_results[i]
        
        # Look for overlapping entities
        overlaps = []
        for j in range(i+1, len(sorted_results)):
            if sorted_results[j].start > current.end:
                # No overlap possible with subsequent entities
                break
                
            # Check for overlap
            next_entity = sorted_results[j]
            if (current.start <= next_entity.start < current.end or
                current.start < next_entity.end <= current.end or
                next_entity.start <= current.start < next_entity.end or
                next_entity.start < current.end <= next_entity.end):
                
                overlaps.append(next_entity)
        
        if not overlaps:
            # No overlaps, keep the current entity
            final_results.append(current)
            i += 1
            continue
        
        # For overlapping entities, keep the one with highest confidence
        all_candidates = [current] + overlaps
        best_entity = max(all_candidates, key=lambda x: x.score)
        final_results.append(best_entity)
        
        # Skip all overlapping entities
        i += len(overlaps) + 1
    
    return final_results


def filter_overlapping_entities(
    results: List[RecognizerResult],
    exclude_overlaps: bool = True,
    overlap_tolerance: int = 1
) -> List[RecognizerResult]:
    """
    Filter out overlapping entity detections based on proximity of spans.
    
    Args:
        results: List of RecognizerResult objects from analyzer
        exclude_overlaps: Whether to exclude overlapping entities
        overlap_tolerance: Number of characters to consider as overlap tolerance
        
    Returns:
        Filtered list of RecognizerResult objects
    """
    if not exclude_overlaps or len(results) <= 1:
        return results
    
    # Sort by confidence score (descending) to keep highest confidence entities
    sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
    filtered_results = []
    
    for result in sorted_results:
        # Check if this result overlaps with any already filtered result
        is_overlapping = False
        for filtered in filtered_results:
            # Check if different entity types with close or overlapping spans
            if result.entity_type != filtered.entity_type:
                start_proximity = abs(result.start - filtered.start) <= overlap_tolerance
                end_proximity = abs(result.end - filtered.end) <= overlap_tolerance
                span_overlap = (
                    (filtered.start <= result.start <= filtered.end) or
                    (filtered.start <= result.end <= filtered.end) or
                    (result.start <= filtered.start <= result.end) or
                    (result.start <= filtered.end <= result.end)
                )
                
                if (start_proximity or end_proximity or span_overlap):
                    is_overlapping = True
                    break
        
        if not is_overlapping:
            filtered_results.append(result)
    
    return filtered_results


def create_deny_list_recognizer(
    deny_list: List[str],
) -> Optional[PatternRecognizer]:
    """
    Create a pattern recognizer for a deny list.
    
    Args:
        deny_list: List of words to flag as PII
        
    Returns:
        A PatternRecognizer or None if the deny list is empty
    """
    if not deny_list:
        return None

    deny_list_recognizer = PatternRecognizer(
        supported_entity="GENERIC_PII", 
        deny_list=deny_list
    )
    return deny_list_recognizer


def create_regex_recognizer(
    regex: str,
    entity_type: str,
    score: float = 0.6,
    context: Optional[List[str]] = None
) -> Optional[PatternRecognizer]:
    """
    Create a pattern recognizer for a custom regex pattern.
    
    Args:
        regex: The regex pattern to match
        entity_type: The entity type to assign to matches
        score: The confidence score to assign to matches
        context: Context words that increase the confidence of matches
        
    Returns:
        A PatternRecognizer or None if the regex is invalid
    """
    if not regex:
        return None
    
    try:
        pattern = Pattern(name="Custom regex pattern", regex=regex, score=score)
        regex_recognizer = PatternRecognizer(
            supported_entity=entity_type, 
            patterns=[pattern], 
            context=context
        )
        return regex_recognizer
    except Exception as e:
        logger.error(f"Failed to create regex recognizer: {e}")
        return None
        return regex_recognizer
    except Exception as e:
        logger.error(f"Failed to create regex recognizer: {e}")
        return None

def create_default_regex_recognizers(analyzer):
    # yaml_file = "./recognizers.yaml"
    yaml_file = "./core/recognizers.yaml"
    analyzer.registry.add_recognizers_from_yaml(yaml_file)


def get_supported_entities(model_family: str, model_path: str) -> Dict[str, str]:
    """
    Get the supported entities for a given model along with their descriptions.

    Args:
        model_family: The model family (spacy, huggingface, gliner)
        model_path: The specific model path or name

    Returns:
        Dictionary mapping entity names to their descriptions
    """
    from config.config import ENTITY_DESCRIPTIONS, CORE_ENTITIES

    # Get all supported entities from the analyzer
    analyzer, _ = initialize_analyzer_engine(model_family, model_path)
    supported_entities = set(analyzer.get_supported_entities())

    # Add GENERIC_PII
    supported_entities.add("GENERIC_PII")

    # Filter to core entities for UI simplicity, but keep any others the model supports
    result_entities = {}

    # First add core entities that are supported
    for entity in CORE_ENTITIES:
        if entity in supported_entities:
            result_entities[entity] = ENTITY_DESCRIPTIONS.get(entity, "")

    # Then add any other supported entities
    for entity in supported_entities:
        if entity not in result_entities:
            result_entities[entity] = ENTITY_DESCRIPTIONS.get(entity, "")

    return result_entities
