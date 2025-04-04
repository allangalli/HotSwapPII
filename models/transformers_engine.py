"""
Transformers-based NLP engine implementation for the HotSwapPII.
"""
import logging
from typing import Tuple

from utils.data_processing import get_standardized_pii_label

from presidio_analyzer import RecognizerRegistry, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngine, NlpEngineProvider
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

from core.result import AnalyzerResult

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_transformers_engine(model_path: str) -> Tuple[NlpEngine, RecognizerRegistry]:
    """
    Create an NLP engine and registry using HuggingFace Transformers.

    Args:
        model_path: The HuggingFace model to use (e.g., 'dslim/bert-base-NER')

    Returns:
        A tuple of (NlpEngine, RecognizerRegistry)
    """
    logger.info(f"Initializing Transformers NLP engine with model: {model_path}")

    # Configure the Transformers NLP engine with a small spaCy model for text processing
    nlp_configuration = {
        "nlp_engine_name": "transformers",
        "models": [
            {
                "lang_code": "en",
                "model_name": {"spacy": "en_core_web_sm", "transformers": model_path},
            }
        ],
        "ner_model_configuration": {
            "model_to_presidio_entity_mapping": {
                "PER": "PERSON",
                "PERSON": "PERSON",
                "LOC": "LOCATION",
                "LOCATION": "LOCATION",
                "GPE": "LOCATION",
                "ORG": "ORGANIZATION",
                "ORGANIZATION": "ORGANIZATION",
                "NORP": "NRP",
                "AGE": "AGE",
                "ID": "ID",
                "EMAIL": "EMAIL_ADDRESS",
                "PATIENT": "PERSON",
                "STAFF": "PERSON",
                "HOSP": "ORGANIZATION",
                "PATORG": "ORGANIZATION",
                "DATE": "DATE_TIME",
                "TIME": "DATE_TIME",
                "PHONE": "PHONE_NUMBER",
                "HCW": "PERSON",
                "HOSPITAL": "ORGANIZATION",
                "FACILITY": "LOCATION",
            },
            "low_confidence_score_multiplier": 0.4,
            "low_score_entity_names": ["ID"],
            "labels_to_ignore": [
                "CARDINAL",
                "EVENT",
                "LANGUAGE",
                "LAW",
                "MONEY",
                "ORDINAL",
                "PERCENT",
                "PRODUCT",
                "QUANTITY",
                "WORK_OF_ART",
            ],
        },
    }

    # Create the NLP engine and registry
    nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine)

    return nlp_engine, registry

def get_huggingface_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
    model.analyze = lambda text, entities, language, score_threshold=None, return_decision_process=None, allow_list=None, ad_hoc_recognizers=None: piiranha_pii_detection(model, tokenizer, text)
    return model

def combine_adjacent_labels(entries):
    if not entries:
        return []

    # Sort entries by start position
    sorted_entries = sorted(entries, key=lambda x: x['start'])

    result = []
    current = sorted_entries[0].copy()

    for entry in sorted_entries[1:]:
        # If current entry ends where next begins (off by 1) and labels match
        if current['end'] + 1 == entry['start'] and current['label'] == entry['label']:
            # Extend current entry
            current['end'] = entry['end']
            # Average the confidence scores if available
            if 'score' in current and 'score' in entry:
                current['score'] = (current['score'] + entry['score']) / 2
        else:
            # Add current to result and start new current
            result.append(current)
            current = entry.copy()

    # Don't forget to add the last entry
    result.append(current)

    return result

# Piiranha PII Detection with max_length to prevent truncation warnings
def piiranha_pii_detection(model, tokenizer, text):
    # Set a consistent max length for all tokenization
    max_length = 2048
    
    # First tokenization for model input
    inputs = tokenizer(
        text=text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=max_length
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Second tokenization to get offset mapping - using SAME parameters
    encoded_inputs = tokenizer.encode_plus(
        text=text, 
        return_offsets_mapping=True, 
        add_special_tokens=True,
        truncation=True,
        max_length=max_length
    )
    offset_mapping = encoded_inputs['offset_mapping']
    
    # Run model inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Apply softmax to convert logits to probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Get predicted class and its probability for each token
    predictions = torch.argmax(outputs.logits, dim=-1)
    confidence_scores = torch.max(probs, dim=-1).values
    
    is_pii_started = False
    pii_type_start = 0
    current_pii_type = ''
    current_confidence = 0.0  # Track confidence for current entity

    entities = []
    
    # Get the length of predictions (should match offset_mapping length, minus special tokens)
    pred_length = predictions.shape[1]

    for i, (start, end) in enumerate(offset_mapping):
        if start == end:  # Special token
            continue
            
        # Skip if index is out of bounds (shouldn't happen now with aligned tokenization)
        if i >= pred_length:
            logger.warning(f"Token index {i} exceeds prediction length {pred_length}")
            break

        # Get prediction and confidence for this token
        label = predictions[0][i].item()
        confidence = confidence_scores[0][i].item()

        if label != model.config.label2id['O']:  # Non-O label
            pii_type = model.config.id2label[label]
            if not is_pii_started:
                is_pii_started = True
                pii_type_start = start  # Remove the +1 offset
                current_pii_type = pii_type
                current_confidence = confidence
            elif pii_type != current_pii_type:
                # End current pii type and start a new one
                entities.append({
                    "start": pii_type_start, 
                    "end": start, 
                    "label": get_standardized_pii_label(current_pii_type),
                    "score": current_confidence
                })
                pii_type_start = start
                current_pii_type = pii_type
                current_confidence = confidence
            else:
                # Same entity continues, update confidence to average
                current_confidence = (current_confidence + confidence) / 2
        else:
            if is_pii_started:
                entities.append({
                    "start": pii_type_start, 
                    "end": start, 
                    "label": get_standardized_pii_label(current_pii_type),
                    "score": current_confidence
                })
                is_pii_started = False
                current_confidence = 0.0

    # Handle case where PII is at the end of the text
    if is_pii_started:
        entities.append({
            "start": pii_type_start, 
            "end": len(text) if end >= len(text) else end, 
            "label": get_standardized_pii_label(current_pii_type),
            "score": current_confidence
        })
    
    # Combine adjacent entities with the same label
    combined_entities = combine_adjacent_labels(entities)
    
    # Convert to AnalyzerResult objects
    results = []
    for entity in combined_entities:
        results.append(
           AnalyzerResult(
                entity_type=entity["label"],
                start=entity["start"],
                end=entity["end"],
                score=entity.get("score", 0)
            )
        )
    
    return results