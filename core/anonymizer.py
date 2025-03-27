"""
Text anonymization functionality for the HotSwapPII.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import streamlit as st
from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_anonymizer.entities import RecognizerResult as AnonymizerResult

logger = logging.getLogger(__name__)

@st.cache_resource
def get_anonymizer_engine() -> AnonymizerEngine:
    """
    Get or create an AnonymizerEngine instance.
    
    Returns:
        An AnonymizerEngine instance
    """
    return AnonymizerEngine()


def anonymize_text(
    text: str,
    entities: List[RecognizerResult],
    anonymization_method: str = "replace",
    mask_char: str = "*",
    mask_chars_count: int = 15,
    encrypt_key: str = None,
) -> Dict[str, Any]:
    """
    Anonymize identified PII entities in text.
    
    Args:
        text: The text to anonymize
        entities: List of detected entities to anonymize
        anonymization_method: Anonymization method to use (redact, replace, mask, hash, encrypt)
        mask_char: Character to use for masking
        mask_chars_count: Number of characters to mask
        encrypt_key: Key for encryption
    
    Returns:
        Dictionary with anonymized text and anonymized entities
    """
    if not text or not entities:
        return {"text": text, "items": []}
        
    # Configure operator based on selected method
    operator_config = None
    operator = anonymization_method
    
    if anonymization_method == "mask":
        operator_config = {
            "type": "mask",
            "masking_char": mask_char,
            "chars_to_mask": mask_chars_count,
            "from_end": False,
        }
    elif anonymization_method == "encrypt":
        operator_config = {"key": encrypt_key}
    elif anonymization_method == "highlight":
        operator_config = {"lambda": lambda x: x}
        operator = "custom"
    elif anonymization_method == "synthesize":
        operator = "replace"
    
    # Anonymize the text
    engine = get_anonymizer_engine()
    result = engine.anonymize(
        text=text,
        analyzer_results=entities,
        operators={"DEFAULT": OperatorConfig(operator, operator_config)},
    )
    
    return {
        "text": result.text,
        "items": result.items
    }


def create_annotated_tokens(
    text: str,
    entities: List[RecognizerResult]
) -> List:
    """
    Create tokens for the annotated text component to highlight entities.
    
    Args:
        text: The original text
        entities: List of detected entities
        
    Returns:
        List of tokens for the annotated_text component
    """
    if not text or not entities:
        return [text]
        
    # Use the anonymizer to resolve overlaps
    result = anonymize_text(
        text=text,
        entities=entities,
        anonymization_method="highlight",
    )
    
    # Sort by start index
    items = sorted(result["items"], key=lambda x: x.start)
    
    tokens = []
    current_pos = 0
    
    for item in items:
        # Add text before the entity
        if item.start > current_pos:
            tokens.append(text[current_pos:item.start])
        
        # Add the entity as a tuple (text, entity_type)
        tokens.append((text[item.start:item.end], item.entity_type))
        
        current_pos = item.end
    
    # Add any remaining text
    if current_pos < len(text):
        tokens.append(text[current_pos:])
    
    return tokens
