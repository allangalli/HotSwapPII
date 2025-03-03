"""
SpaCy NLP engine implementation for the HotSwapPII.
"""
import logging
from typing import Tuple

from presidio_analyzer import RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngine, NlpEngineProvider

logger = logging.getLogger(__name__)

def create_spacy_engine(model_path: str) -> Tuple[NlpEngine, RecognizerRegistry]:
    """
    Create an NLP engine and registry using SpaCy.
    
    Args:
        model_path: The SpaCy model to use (e.g., 'en_core_web_lg')
        
    Returns:
        A tuple of (NlpEngine, RecognizerRegistry)
    """
    logger.info(f"Initializing SpaCy NLP engine with model: {model_path}")
    
    # Configure the SpaCy NLP engine
    nlp_configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": model_path}],
        "ner_model_configuration": {
            "model_to_presidio_entity_mapping": {
                "PER": "PERSON",
                "PERSON": "PERSON",
                "NORP": "NRP",
                "FAC": "FACILITY",
                "LOC": "LOCATION",
                "GPE": "LOCATION",
                "LOCATION": "LOCATION",
                "ORG": "ORGANIZATION",
                "ORGANIZATION": "ORGANIZATION",
                "DATE": "DATE_TIME",
                "TIME": "DATE_TIME",
            },
            "low_confidence_score_multiplier": 0.4,
            "low_score_entity_names": ["ORG", "ORGANIZATION"],
        },
    }

    # Create the NLP engine and registry
    nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine)

    return nlp_engine, registry
