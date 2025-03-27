"""
SpaCy NLP engine implementation for the HotSwapPII.
"""
import logging
from typing import Tuple

from presidio_analyzer import RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngine, NlpEngineProvider
import spacy
import torch
from utils.data_processing import get_standardized_pii_label
from core.result import AnalyzerResult

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

def get_spacy_model(model_path: str):
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using SpaCy with model: {model_path}")
    if not spacy.util.is_package(model_path):
        spacy.cli.download(model_path)

    # Load models
    nlp_spacy = spacy.load(model_path)
    print(nlp_spacy._path)
    print(nlp_spacy.get_pipe("ner"))
    print(nlp_spacy.get_pipe("ner").labels)

    nlp_spacy.analyze = lambda text, entities, language, score_threshold=None, return_decision_process=None, allow_list=None, ad_hoc_recognizers=None: spacy_pii_detection(nlp_spacy, text)

    return nlp_spacy

# Spacy PII Detection
def spacy_pii_detection(model, text: str):
    doc = model(text)

    # for val in doc.ents[0].lefts:
    #     print(val)
    entities = [AnalyzerResult(ent.label_, ent.start_char, ent.end_char, 0, ent.ents) for ent in doc.ents]

    return entities

