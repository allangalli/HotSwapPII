"""
GLiNER-based NLP engine implementation for the HotSwapPII.
"""
import logging
from typing import Tuple
from typing import List, Dict

from gliner import GLiNER
from presidio_analyzer import RecognizerRegistry, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngine, NlpEngineProvider
from presidio_analyzer.predefined_recognizers import GLiNERRecognizer

from config.config import GLINER_ENTITY_MAPPING, GLINER_LABELS
from core.result import AnalyzerResult

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
        flat_ner=True,
        # flat_ner=False,
        multi_label=False,
        # multi_label=True,
        # map_location="cpu",
        threshold=0.5,
        # threshold=0.3,
        map_location="gpu",
    )

    gliner_recognizer.presidio_analyze = gliner_recognizer.analyze

    def _predict_chunked_text(text: str, entities: List[str], chunk_size: int = 384) -> \
            List[RecognizerResult]:

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        chunks = _split_text_into_chunks(gliner_recognizer.gliner, text, chunk_size)
        offsets = _calculate_offsets(chunks)

        all_entities = []
        chunk_entities_list = []
        for chunk in chunks:
            chunk_entities_list.append(gliner_recognizer.presidio_analyze(chunk, entities))

        for chunk_entities, offset in zip(chunk_entities_list, offsets):
            adjusted_entities = _adjust_indices_presidio(chunk_entities, offset)
            all_entities.extend(adjusted_entities)

        return all_entities

    def _adjust_indices_presidio(entities: List[RecognizerResult], offset: int) -> List[RecognizerResult]:
        for entity in entities:
            entity.start += offset
            entity.end += offset
        return entities

    gliner_recognizer.analyze = \
        lambda text, entities, language = "en", score_threshold=None, return_decision_process=None, allow_list=None, ad_hoc_recognizers=None, nlp_artifacts = None: (
            _predict_chunked_text(text, entities))

    # Add the GLiNER recognizer to the registry
    registry.add_recognizer(gliner_recognizer)

    # Remove SpaCy recognizer to avoid conflicts
    registry.remove_recognizer("SpacyRecognizer")

    print('engine', nlp_engine)
    return nlp_engine, registry

def get_gliner_model(model_path: str, options: Dict[str, any] = None):
    gliner_model = GLiNER.from_pretrained(model_path)

    # options["labels"]
    gliner_model.analyze = lambda text, entities, language, score_threshold=None, return_decision_process=None, allow_list=None, ad_hoc_recognizers=None: gliner_pii_detection(gliner_model, text, GLINER_LABELS)

    return gliner_model

def _split_text_into_chunks(model, text: str, chunk_size: int) -> List[str]:
    tokens = model.data_processor.words_splitter(text)
    chunks = []
    text_indices = [0]
    for token, start, end in tokens:
        chunks.append(token)
        if len(chunks) == chunk_size:
            text_indices.append(end)
            chunks = []
    text_indices.append(len(text))

    return [text[text_indices[i]:text_indices[i + 1]] for i in range(len(text_indices) - 1)]

def _calculate_offsets(chunks: List[str]) -> List[int]:
    offset = 0
    offsets = []
    for chunk in chunks:
        offsets.append(offset)
        offset += len(chunk) + 1  # +1 for the space that was removed during split
    return offsets

def _adjust_indices(entities: List[Dict], offset: int) -> List[Dict]:
    for entity in entities:
        entity["start"] += offset
        entity["end"] += offset
    return entities

def _predict_long_text(model, text: str, labels: List[str], chunk_size: int = 384, flat_ner=True, threshold=0.5) -> \
List[Dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    chunks = _split_text_into_chunks(model, text, chunk_size)
    offsets = _calculate_offsets(chunks)

    all_entities = []
    chunk_entities_list = model.batch_predict_entities(chunks, labels, flat_ner=flat_ner, threshold=threshold)

    for chunk_entities, offset in zip(chunk_entities_list, offsets):
        adjusted_entities = _adjust_indices(chunk_entities, offset)
        all_entities.extend(adjusted_entities)

    return all_entities

# GLiNER PII Detection
def gliner_pii_detection(model, text: str, labels: List[str], threshold=0.5, nested_ner=False, long_text=True):
    if long_text:
        result = _predict_long_text(model, text, labels, chunk_size=384, flat_ner=not nested_ner, threshold=threshold)

    else:
        result = model.predict_entities(
            text, labels, flat_ner=not nested_ner, threshold=threshold
        )

    entities = [AnalyzerResult(entity['label'], entity['start'], entity['end'], entity['score'], entity['text']) for entity in result]
    return entities
