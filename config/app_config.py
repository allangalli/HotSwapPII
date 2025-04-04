"""
Application-level configuration settings for the HotSwapPII.
"""
import os

# Application settings
APP_TITLE = "HotSwapPII"
APP_DESCRIPTION = "Identify and anonymize personally identifiable information (PII) in text."

# Default text loaded on application start
DEFAULT_DEMO_TEXT_PATH = "data/demo_text.txt"

# Default anonymization settings
DEFAULT_ANONYMIZATION_METHOD = "replace"
DEFAULT_MASK_CHAR = "*"
DEFAULT_MASK_CHARS_COUNT = 15
DEFAULT_ENCRYPT_KEY = "WmZq4t7w!z%C&F)J"

# OpenAI settings (for synthetic data generation)
OPENAI_DEFAULT_MODEL = "gpt-3.5-turbo-instruct"

# Allow using custom models (controlled by env var)
ALLOW_CUSTOM_MODELS = os.getenv("ALLOW_CUSTOM_MODELS", "False").lower() in ("true", "1", "t")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO") 