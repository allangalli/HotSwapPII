"""
Sidebar UI components for the HotSwapPII.
"""
import logging
import os
from typing import Dict, List, Optional, Tuple

import streamlit as st
from streamlit_tags import st_tags

from config.config import (ALLOW_CUSTOM_MODELS, DEFAULT_ANONYMIZATION_METHOD,
                           DEFAULT_ENTITY_SELECTION, DEFAULT_EXCLUDE_OVERLAPS,
                           DEFAULT_MASK_CHAR, DEFAULT_MASK_CHARS_COUNT,
                           DEFAULT_MODEL, DEFAULT_MODEL_INDEX,
                           DEFAULT_OVERLAP_TOLERANCE, DEFAULT_THRESHOLD,
                           ENTITY_DESCRIPTIONS, MODEL_OPTIONS,
                           OPENAI_DEFAULT_MODEL)
from models.model_factory import extract_model_info, get_supported_entities
from utils.synthetic_data import OpenAIParams

logger = logging.getLogger(__name__)

def render_sidebar() -> Dict:
    """
    Render the sidebar UI components.
    
    Returns:
        Dictionary with all sidebar settings
    """
    st.sidebar.title("Master Settings")
    
    # Create tabs for different settings categories
    tabs = st.sidebar.tabs(["Model", "Processing"])
    
    # Model tab
    with tabs[0]:
        model_settings = render_model_settings()
        anonymization_settings = render_anonymization_settings()
        advanced_settings = render_advanced_settings()
    
    # Processing tab
    with tabs[1]:
        processing_settings = render_processing_settings(model_settings)
    
    # Combine all settings
    settings = {
        **model_settings,
        **processing_settings,
        **anonymization_settings,
        **advanced_settings,
    }
    
    return settings


def render_model_settings() -> Dict:
    """
    Render model selection settings in the sidebar.
    
    Returns:
        Dictionary with model settings
    """
    st.subheader("Model Selection")
    
    # Model selection
    model_help = """
    Select which Named Entity Recognition (NER) model to use for PII detection, 
    in parallel to rule-based recognizers.
    """
    model_selection = st.selectbox(
        "NER model",
        options=MODEL_OPTIONS + (["Other"] if ALLOW_CUSTOM_MODELS else []),
        index=DEFAULT_MODEL_INDEX,
        help=model_help,
    )
    
    # Handle custom model selection
    if model_selection == "Other" and ALLOW_CUSTOM_MODELS:
        model_family = st.selectbox(
            "Model family",
            options=["spaCy", "HuggingFace", "GLiNER"],
            index=0,
        )
        model_path = st.text_input("Model name or path", value="")
    else:
        model_family, model_path = extract_model_info(model_selection)
    
    st.caption("Note: Models might take some time to download.")
    
    # Return settings
    return {
        "model_selection": model_selection,
        "model_family": model_family,
        "model_path": model_path,
    }


def render_processing_settings(model_settings: Dict) -> Dict:
    """
    Render processing settings in the sidebar.
    
    Args:
        model_settings: Dictionary with model settings
        
    Returns:
        Dictionary with processing settings
    """
    st.subheader("Detection Settings")
    
    # Confidence threshold
    threshold = st.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_THRESHOLD,
        step=0.05,
        help="Minimum confidence score required to include an entity detection",
    )
    
    # Entity selection
    st.subheader("Entity Selection")
    
    # Get supported entities for the selected model
    model_family = model_settings["model_family"]
    model_path = model_settings["model_path"]
    
    if model_family and model_path:
        try:
            entity_descriptions = get_supported_entities(model_family, model_path)
            
            # Display entity selection with tooltips
            selected_entities = []
            
            st.write("Select entities to detect:")
            for entity, description in entity_descriptions.items():
                if st.checkbox(
                    entity, 
                    value=entity in DEFAULT_ENTITY_SELECTION,
                    help=description
                ):
                    selected_entities.append(entity)
        except Exception as e:
            logger.error(f"Error getting supported entities: {e}")
            st.error(f"Error loading entity list: {e}")
            selected_entities = DEFAULT_ENTITY_SELECTION
    else:
        selected_entities = DEFAULT_ENTITY_SELECTION
    
    # Return settings
    return {
        "threshold": threshold,
        "selected_entities": selected_entities,
    }


def render_anonymization_settings() -> Dict:
    """
    Render anonymization settings in the sidebar.
    
    Returns:
        Dictionary with anonymization settings
    """
    st.subheader("Anonymization")
    
    # Anonymization method
    anonymization_method = st.selectbox(
        "De-identification approach",
        options=["redact", "replace", "mask", "hash", "encrypt", "synthesize"],
        index=1,
        help="""
        Select how to transform detected PII:
        - redact: Remove the PII completely
        - replace: Replace with entity type (e.g., <PERSON>)
        - mask: Replace characters with a mask character
        - hash: Replace with a hash of the text
        - encrypt: Encrypt the text (reversible)
        - highlight: Show the original text with highlighted PII
        - synthesize: Replace with realistic fake values (requires OpenAI)
        """,
    )
    
    # Method-specific settings
    method_settings = {}
    
    if anonymization_method == "mask":
        method_settings["mask_char"] = st.text_input(
            "Mask character", value=DEFAULT_MASK_CHAR, max_chars=1
        )
        method_settings["mask_chars_count"] = st.number_input(
            "Number of characters to mask",
            value=DEFAULT_MASK_CHARS_COUNT,
            min_value=0,
            max_value=100,
        )
    elif anonymization_method == "encrypt":
        method_settings["encrypt_key"] = st.text_input(
            "Encryption key",
            value="WmZq4t7w!z%C&F)J",
            type="password",
            help="Key used for AES encryption",
        )
    elif anonymization_method == "synthesize":
        # OpenAI settings for synthetic data
        openai_key = os.getenv("OPENAI_API_KEY", "")
        openai_key = st.text_input(
            "OpenAI API Key",
            value=openai_key,
            type="password",
            help="API key for OpenAI services",
        )
        openai_model = st.text_input(
            "OpenAI Model",
            value=OPENAI_DEFAULT_MODEL,
            help="Model to use for synthetic data generation",
        )
        
        method_settings["openai_params"] = OpenAIParams(
            api_key=openai_key,
            model=openai_model,
        )
    
    # Return settings
    return {
        "anonymization_method": anonymization_method,
        **method_settings,
    }


def render_advanced_settings() -> Dict:
    """
    Render advanced settings in the sidebar.
    
    Returns:
        Dictionary with advanced settings
    """
    st.subheader("Advanced Settings")
    
    # Overlap handling
    exclude_overlaps = st.checkbox(
        "Exclude overlapping entities",
        value=DEFAULT_EXCLUDE_OVERLAPS,
        help="Filter out entities that overlap with higher confidence entities",
    )
    
    # Only show overlap tolerance if exclude_overlaps is checked
    overlap_tolerance = DEFAULT_OVERLAP_TOLERANCE
    if exclude_overlaps:
        overlap_tolerance = st.number_input(
            "Overlap tolerance (characters)",
            value=DEFAULT_OVERLAP_TOLERANCE,
            min_value=0,
            max_value=20,
            help="Number of characters to consider as overlap between entities",
        )
    
    # Decision process
    return_decision_process = st.checkbox(
        "Show detection reasoning",
        value=False,
        help="Add detection reasoning to the findings table",
    )
    
    # Allow and deny lists
    with st.expander("Allow and Deny Lists", expanded=False):
        allow_list = st_tags(
            label="Allowlist (never flag these)",
            text="Enter word and press enter",
            value=[],
        )
        st.caption(
            "Allowlists contain words that should never be flagged as PII"
        )
        
        deny_list = st_tags(
            label="Denylist (always flag these)",
            text="Enter word and press enter",
            value=[],
        )
        st.caption(
            "Denylists contain words that should always be flagged as PII"
        )
    
    # Custom regex patterns
    with st.expander("Custom Regex Pattern", expanded=False):
        regex_pattern = st.text_input(
            "Regex pattern",
            value="",
            help="Custom regex pattern to match additional entities",
        )
        
        if regex_pattern:
            regex_entity_type = st.selectbox(
                "Entity type",
                options=list(ENTITY_DESCRIPTIONS.keys()),
                index=0,
                help="Entity type to assign to regex matches",
            )
            
            regex_score = st.slider(
                "Confidence score",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Confidence score to assign to regex matches",
            )
            
            regex_context = st.text_input(
                "Context words (comma-separated)",
                value="",
                help="Words that, if nearby, increase the confidence of matches",
            ).split(",") if st.checkbox("Use context words", value=False) else None
        else:
            regex_entity_type = None
            regex_score = None
            regex_context = None
    
    # Return settings
    return {
        "exclude_overlaps": exclude_overlaps,
        "overlap_tolerance": overlap_tolerance,
        "return_decision_process": return_decision_process,
        "allow_list": allow_list,
        "deny_list": deny_list,
        "regex_pattern": regex_pattern,
        "regex_entity_type": regex_entity_type,
        "regex_score": regex_score,
        "regex_context": regex_context,
    }
