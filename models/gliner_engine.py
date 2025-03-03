"""
GLiNER-based NLP engine implementation for the HotSwapPII.
"""
import logging
from typing import Tuple

from presidio_analyzer import RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngine, NlpEngineProvider
from presidio_analyzer.predefined_recognizers import GLiNERRecognizer

from config.config import GLINER_ENTITY_MAPPING

logger = logging.getLogger(__name__)

def create_gliner_engine(model_path: str) -> Tuple[NlpEngine, RecognizerRegistry]:
    """
    Create an NLP engine and registry using GLiNER.
    
    Args:
        model_path: The GLiNER model to use (e.g., 'urchade/gliner_multi_pii-v1')
        
    Returns:
        A tuple of (NlpEngine, RecognizerRegistry)
    """
    logger.info(f"Initializing GLiNER NLP engine with model: {model_path}")
    
    # Load a small spaCy model as we don't need spaCy's NER
    nlp_configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    }

    # Create the NLP engine
    nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
    
    # Create the registry and load predefined recognizers
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine)
    
    # Create and add the GLiNER recognizer
    gliner_recognizer = GLiNERRecognizer(
        model_name=model_path,
        entity_mapping=GLINER_ENTITY_MAPPING,
        flat_ner=False,
        multi_label=True,
        map_location="cpu",
    )
    
    # Add the GLiNER recognizer to the registry
    registry.add_recognizer(gliner_recognizer)
    
    # Remove SpaCy recognizer to avoid conflicts
    registry.remove_recognizer("SpacyRecognizer")
    
    return nlp_engine, registry
