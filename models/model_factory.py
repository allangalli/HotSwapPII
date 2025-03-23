"""
Factory for creating NLP engines and analyzer engines based on the selected model type.
"""
import logging
from typing import Dict, Optional, Tuple, Union

import streamlit as st
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngine

from models.spacy_engine import create_spacy_engine
from models.transformers_engine import create_transformers_engine
from models.gliner_engine import create_gliner_engine
from models.custom_recognizers import (
    create_canadian_postal_code_recognizer,
    create_canadian_drivers_license_recognizer
)

logger = logging.getLogger(__name__)

@st.cache_resource
def get_nlp_engine_and_registry(
    model_family: str, 
    model_path: str
) -> Tuple[NlpEngine, RecognizerRegistry]:
    """
    Factory method to create an NLP engine and registry based on model family.
    
    Args:
        model_family: The model family (spacy, huggingface, gliner)
        model_path: The specific model path or name
        
    Returns:
        A tuple of (NlpEngine, RecognizerRegistry)
        
    Raises:
        ValueError: If the model family is not supported
    """
    logger.info(f"Creating NLP engine for {model_family}/{model_path}")
    
    model_family = model_family.lower()
    
    # Get NLP engine and registry based on model family
    if "spacy" in model_family:
        nlp_engine, registry = create_spacy_engine(model_path)
    elif "huggingface" in model_family:
        nlp_engine, registry = create_transformers_engine(model_path)
    elif "gliner" in model_family:
        nlp_engine, registry = create_gliner_engine(model_path)
    else:
        raise ValueError(f"Unsupported model family: {model_family}")
    
    # Add custom recognizers
    logger.info("Adding custom recognizers")
    
    # Add Canadian postal code recognizer
    canadian_postal_code_recognizer = create_canadian_postal_code_recognizer()
    registry.add_recognizer(canadian_postal_code_recognizer)
    
    # Add Canadian driver's license recognizer
    canadian_drivers_license_recognizer = create_canadian_drivers_license_recognizer()
    registry.add_recognizer(canadian_drivers_license_recognizer)
    
    return nlp_engine, registry


@st.cache_resource
def get_analyzer_engine(
    model_family: str, 
    model_path: str
) -> AnalyzerEngine:
    """
    Create an AnalyzerEngine instance for the specified model.
    
    Args:
        model_family: The model family (spacy, huggingface, gliner)
        model_path: The specific model path or name
        
    Returns:
        An initialized AnalyzerEngine
    """
    nlp_engine, registry = get_nlp_engine_and_registry(model_family, model_path)
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)
    return analyzer


def extract_model_info(model_selection: str) -> Tuple[str, str]:
    """
    Extract model family and path from a model selection string.
    
    Args:
        model_selection: Model selection in format 'Family/model_path'
        
    Returns:
        Tuple of (model_family, model_path)
    """
    if "/" not in model_selection:
        return model_selection, model_selection
        
    parts = model_selection.split("/", 1)
    model_family = parts[0]
    model_path = "/".join(parts[1:])
    
    return model_family, model_path


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
    analyzer = get_analyzer_engine(model_family, model_path)
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
