"""
Factory for creating NLP engines and analyzer engines based on the selected model type.
"""
import logging
from typing import Dict, Optional, Tuple, Union

import streamlit as st
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngine

from models.spacy_engine import create_spacy_engine, get_spacy_model
from models.transformers_engine import create_transformers_engine, get_huggingface_model
from models.gliner_engine import create_gliner_engine, get_gliner_model
from config.config import MODEL_DETAILS

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
    
    if "spacy" in model_family:
        return create_spacy_engine(model_path)
    elif "huggingface" in model_family:
        return create_transformers_engine(model_path)
    elif "gliner" in model_family:
        return create_gliner_engine(model_path)
    else:
        raise ValueError(f"Unsupported model family: {model_family}")


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


def extract_model_info(model_selection: str) -> Tuple[str, str, str]:
    """
    Extract model family and path from a model selection string.
    
    Args:
        model_selection: Model selection in format 'Family/model_path'
        
    Returns:
        Tuple of (base_model, model_family, model_path)
    """
    try:
        model_details = MODEL_DETAILS[model_selection]
    except KeyError as e:
        st.error(f"Model details don't exist for model {model_selection}: {e}")
        
    base_model = model_details["base_model"]
    model_family = model_details["model_family"]
    model_path = model_details["model_path"]
    
    return base_model, model_family, model_path

def get_independent_model(model_family: str, model_path: str):
    if model_family == "spaCy":
        model = get_spacy_model(model_path)
    elif model_family == "HuggingFace":
        model = get_huggingface_model(model_path)
    elif model_family == "GLiNER":
        model = get_gliner_model(model_path)
    return model