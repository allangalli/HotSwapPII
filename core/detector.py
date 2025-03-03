"""
Core PII detection logic for the HotSwapPII.
"""
import logging
from typing import Any, Dict, List, Optional, Set, Union

import streamlit as st
from presidio_analyzer import (AnalyzerEngine, Pattern, PatternRecognizer,
                               RecognizerResult)

from models.model_factory import get_analyzer_engine

logger = logging.getLogger(__name__)

@st.cache_data
def analyze_text(
    model_family: str,
    model_path: str,
    text: str,
    entities: Optional[List[str]] = None,
    language: str = "en",
    score_threshold: float = 0.35,
    return_decision_process: bool = False,
    allow_list: Optional[List[str]] = None,
    deny_list: Optional[List[str]] = None,
    regex_pattern: Optional[str] = None,
    regex_entity_type: Optional[str] = None,
    regex_score: Optional[float] = None,
    regex_context: Optional[List[str]] = None,
) -> List[RecognizerResult]:
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
    if not text:
        return []
    
    # Get the analyzer engine
    analyzer = get_analyzer_engine(model_family, model_path)
    
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
    
    # Analyze the text
    results = analyzer.analyze(
        text=text,
        entities=entities,
        language=language,
        score_threshold=score_threshold,
        return_decision_process=return_decision_process,
        allow_list=allow_list,
        ad_hoc_recognizers=ad_hoc_recognizers if ad_hoc_recognizers else None,
    )
    
    return results


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
