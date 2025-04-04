"""
Model-related configuration settings for the HotSwapPII.
"""
from typing import Dict

# Model configurations
DEFAULT_MODEL_INDEX = 0
DEFAULT_THRESHOLD = 0.40

# Default for filtering overlapping entities
DEFAULT_EXCLUDE_OVERLAPS = True
DEFAULT_OVERLAP_TOLERANCE = 5

# Default for adding results to benchmarks
DEFAULT_ADD_TO_BENCHMARK = True

# Available models
MODEL_OPTIONS = [
    "Presidio (default)",
    "Presidio (gliner_multi_pii-v1)",
    "Presidio (deid_roberta_i2b2)",
    "Presidio (stanford-deidentifier-base)",
    "Presidio (piiranha-v1-detect-personal-information)",
    "SpaCy (en_core_web_lg)",
    "SpaCy (en_core_web_trf)",
    # "HuggingFace Transformer (deid_roberta_i2b2)",
    # "HuggingFace Transformer (stanford-deidentifier-base)",
    "GLiNER (gliner_multi_pii-v1)",
    "Piiranha (piiranha-v1-detect-personal-information)"
]

MODEL_DETAILS: Dict[str, Dict[str, str]] = {
    "Presidio (default)": {
        "base_model": "presidio",
        "model_family": "spaCy",
        "model_path": "en_core_web_lg",
    },
    "Presidio (gliner_multi_pii-v1)": {
        "base_model": "presidio",
        "model_family": "GLiNER",
        "model_path": "urchade/gliner_multi_pii-v1",
    },
    "Presidio (deid_roberta_i2b2)": {
        "base_model": "presidio",
        "model_family": "HuggingFace",
        "model_path": "obi/deid_roberta_i2b2",
    },
    "Presidio (stanford-deidentifier-base)": {
        "base_model": "presidio",
        "model_family": "HuggingFace",
        "model_path": "StanfordAIMI/stanford-deidentifier-base",
    },
    "Presidio (piiranha-v1-detect-personal-information)": {
        "base_model": "presidio",
        "model_family": "HuggingFace",
        "model_path": "iiiorg/piiranha-v1-detect-personal-information",
    },
    "SpaCy (en_core_web_lg)": {
        "base_model": "independent",
        "model_family": "spaCy",
        "model_path": "en_core_web_lg",
    },
    "SpaCy (en_core_web_trf)": {
        "base_model": "independent",
        "model_family": "spaCy",
        "model_path": "en_core_web_lg",
    },
    # "HuggingFace Transformer (deid_roberta_i2b2)": {
    #     "base_model": "independent",
    #     "model_family": "HuggingFace",
    #     "model_path": "obi/deid_roberta_i2b2",
    # },
    # "HuggingFace Transformer (stanford-deidentifier-base)": {
    #     "base_model": "independent",
    #     "model_family": "HuggingFace",
    #     "model_path": "StanfordAIMI/stanford-deidentifier-base",
    # },
    "GLiNER (gliner_multi_pii-v1)": {
        "base_model": "independent",
        "model_family": "GLiNER",
        "model_path": "urchade/gliner_multi_pii-v1",
    },
    "Piiranha (piiranha-v1-detect-personal-information)": {
        "base_model": "independent",
        "model_family": "HuggingFace",
        "model_path": "iiiorg/piiranha-v1-detect-personal-information",
    }
}

MODEL_OPTIONS_TO_BENHCMARKS_KEY: Dict[str, str] = {
    "Presidio (default)": "presidio",
    "Presidio (gliner_multi_pii-v1)": "presidio+gliner",
    "Presidio (deid_roberta_i2b2)": "presidio+roberta",
    "Presidio (stanford-deidentifier-base)": "presidio+stanford",
    "Presidio (piiranha-v1-detect-personal-information)": "presidio+piiranha",
    "SpaCy (en_core_web_lg)": "spacy-lg",
    "SpaCy (en_core_web_trf)": "spacy-trf",
    # "HuggingFace Transformer (deid_roberta_i2b2)": "roberta",
    # "HuggingFace Transformer (stanford-deidentifier-base)": "stanford",
    "GLiNER (gliner_multi_pii-v1)": "gliner",
    "Piiranha (piiranha-v1-detect-personal-information)": "piiranha",
} 