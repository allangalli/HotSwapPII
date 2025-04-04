"""
Configuration settings for the HotSwapPII.

This file centralizes imports from individual configuration modules
to maintain backward compatibility while splitting the configuration
into more manageable specialized modules.
"""

# Application settings
from config.app_config import (
    APP_TITLE, 
    APP_DESCRIPTION, 
    DEFAULT_DEMO_TEXT_PATH,
    DEFAULT_ANONYMIZATION_METHOD,
    DEFAULT_MASK_CHAR,
    DEFAULT_MASK_CHARS_COUNT,
    DEFAULT_ENCRYPT_KEY,
    OPENAI_DEFAULT_MODEL,
    ALLOW_CUSTOM_MODELS,
    LOG_LEVEL
)

# Model settings
from config.model_config import (
    DEFAULT_MODEL_INDEX,
    DEFAULT_THRESHOLD,
    DEFAULT_EXCLUDE_OVERLAPS,
    DEFAULT_OVERLAP_TOLERANCE,
    DEFAULT_ADD_TO_BENCHMARK,
    MODEL_OPTIONS,
    MODEL_DETAILS,
    MODEL_OPTIONS_TO_BENHCMARKS_KEY
)

# Entity settings
from config.entity_config import (
    DEFAULT_ENTITY_SELECTION,
    ENTITY_DESCRIPTIONS,
    CORE_ENTITIES,
    GLINER_ENTITY_MAPPING,
    GLINER_LABELS,
    MODEL_ENTITIES_TO_STANDARDIZED_ENTITY_MAPPING
)

# Benchmark settings
from config.benchmark_config import (
    DATASET_OPTIONS,
    DATASET_DESCRIPTIONS,
    DATASET_SAMPLE_FILE_PATH,
    DATASET_BENCHMARK_RESULTS_FILE_PATH
)

# Evaluation settings
from config.evaluation_config import (
    DEFAULT_OVERLAP_THRESHOLD,
    GROUND_TRUTH_TAGS,
    VALIDATION_TO_SYSTEM_MAPPING,
    SYSTEM_TO_VALIDATION_MAPPING
)
