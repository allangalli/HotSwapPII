"""
Configuration settings for the HotSwapPII.
"""
import os
from typing import Dict, List, Set

# Application settings
APP_TITLE = "HotSwapPII"
APP_DESCRIPTION = "Identify and anonymize personally identifiable information (PII) in text."
GITHUB_URL = "https://github.com/allangalli/HotSwapPII"

# Default text loaded on application start
DEFAULT_DEMO_TEXT_PATH = "data/demo_text.txt"

# Model configurations
DEFAULT_MODEL = "GLiNER/urchade/gliner_multi_pii-v1"
DEFAULT_MODEL_INDEX = 1  # Index in the model list dropdown
DEFAULT_THRESHOLD = 0.40

# Available models
MODEL_OPTIONS = [
    "spaCy/en_core_web_lg",
    "HuggingFace/obi/deid_roberta_i2b2",
    "HuggingFace/StanfordAIMI/stanford-deidentifier-base",
    "GLiNER/urchade/gliner_multi_pii-v1",
]

# Allow using custom models (controlled by env var)
ALLOW_CUSTOM_MODELS = os.getenv("ALLOW_CUSTOM_MODELS", "False").lower() in ("true", "1", "t")

# Default anonymization settings
DEFAULT_ANONYMIZATION_METHOD = "replace"
DEFAULT_MASK_CHAR = "*"
DEFAULT_MASK_CHARS_COUNT = 15
DEFAULT_ENCRYPT_KEY = "WmZq4t7w!z%C&F)J"

# OpenAI settings (for synthetic data generation)
OPENAI_DEFAULT_MODEL = "gpt-3.5-turbo-instruct"

# Entity selection
DEFAULT_ENTITY_SELECTION = [
    "NAME", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", 
    "US_SSN", "DATE_TIME", "DRIVER_LICENSE_NUMBER", "POSTAL_CODE"
]

# Supported entity types with descriptions
ENTITY_DESCRIPTIONS: Dict[str, str] = {
    "NAME": "Names of individuals",
    "EMAIL_ADDRESS": "Email addresses",
    "PHONE_NUMBER": "Telephone and fax numbers",
    "CREDIT_CARD": "Credit card numbers",
    "DATE_TIME": "Dates and times",
    "US_SSN": "US Social Security Numbers (SSN)",
    "DRIVER_LICENSE_NUMBER": "Driver's License numbers (US and Canadian)",
    "POSTAL_CODE": "Postal codes (including Canadian format)",
    "GENERIC_PII": "Other personal identifiable information"
}

# Core allowed entities (limited for UI simplicity)
CORE_ENTITIES: Set[str] = {
    "NAME", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", 
    "US_SSN", "DATE_TIME", "DRIVER_LICENSE_NUMBER", "POSTAL_CODE"
}

# Default for filtering overlapping entities
DEFAULT_EXCLUDE_OVERLAPS = True
DEFAULT_OVERLAP_TOLERANCE = 5

# Validation dataset mapping
VALIDATION_TO_SYSTEM_MAPPING: Dict[str, str] = {
    'date': 'DATE_TIME',
    'time': 'DATE_TIME',
    'name': 'NAME',
    'email': 'EMAIL_ADDRESS',
    'phone_number': 'PHONE_NUMBER',
    'ssn': 'US_SSN',
    'credit_card_number': 'CREDIT_CARD',
    'driver_license_number': 'DRIVER_LICENSE_NUMBER',
    'postal_code': 'POSTAL_CODE',
}

# System to validation entity mapping (reverse of the above)
SYSTEM_TO_VALIDATION_MAPPING: Dict[str, str] = {v: k for k, v in VALIDATION_TO_SYSTEM_MAPPING.items()}

# GLiNER entity mapping
GLINER_ENTITY_MAPPING: Dict[str, str] = {
    "person": "NAME",
    "name": "NAME",
    "organization": "GENERIC_PII",
    "location": "GENERIC_PII",
    "email": "EMAIL_ADDRESS",
    "credit card number": "CREDIT_CARD",
    "phone number": "PHONE_NUMBER",
    "social security number": "US_SSN",
    "passport number": "GENERIC_PII",
    "driver's license number": "DRIVER_LICENSE_NUMBER",
    "individual taxpayer identification number": "GENERIC_PII",
    "ip address": "GENERIC_PII",
    "International Bank Account Number": "GENERIC_PII",
    "Age": "GENERIC_PII",
    "debit card number": "GENERIC_PII",
    "client card number": "GENERIC_PII",
    "canadian social insurance number": "US_SSN",
    "canadian passport number": "GENERIC_PII",
    "canadian driver's license number": "DRIVER_LICENSE_NUMBER",
    "canadian postal code": "POSTAL_CODE",
    "postal code": "POSTAL_CODE",
    "zip code": "POSTAL_CODE",
    "MAC Address": "GENERIC_PII",
    "Personal Address": "GENERIC_PII",
    "Address": "GENERIC_PII",
    "Company": "GENERIC_PII",
    "Password": "GENERIC_PII",
    "Job Title": "GENERIC_PII",
    "Job Position": "GENERIC_PII",
    "Account Number": "GENERIC_PII",
    "date": "DATE_TIME",
    "date of birth": "DATE_TIME",
    "birthdate": "DATE_TIME",
}

# Evaluation settings
DEFAULT_OVERLAP_THRESHOLD = 0.5  # For entity matching in evaluation

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
