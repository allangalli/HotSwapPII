"""
Configuration settings for the HotSwapPII.
"""
import os
from typing import Dict, List, Set

# Application settings
APP_TITLE = "HotSwapPII"
APP_DESCRIPTION = "Identify and anonymize personally identifiable information (PII) in text."
GITHUB_URL = "https://github.com/yourorganization/pii-detection"

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
    "PERSON", "ORGANIZATION", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER", 
    "CREDIT_CARD", "US_SSN", "DATE_TIME", "CAN_DRIVER_LICENSE", "ADDRESS", 
    "US_PASSPORT", "US_DRIVER_LICENSE", "PASSWORD",
]

# Supported entity types with descriptions
ENTITY_DESCRIPTIONS: Dict[str, str] = {
    "PERSON": "Names of individuals",
    "ORGANIZATION": "Companies, agencies, institutions",
    "LOCATION": "Physical locations, cities, countries",
    "EMAIL_ADDRESS": "Email addresses",
    "PHONE_NUMBER": "Telephone and fax numbers",
    "CREDIT_CARD": "Credit card numbers",
    "DATE_TIME": "Dates and times",
    "US_SSN": "US Social Security Numbers",
    "US_PASSPORT": "US Passport numbers",
    "US_DRIVER_LICENSE": "US Driver's License numbers",
    "CAN_DRIVER_LICENSE": "Canadian Driver's License numbers",
    "IP_ADDRESS": "IP addresses",
    "ADDRESS": "Street addresses",
    "IBAN_CODE": "International Bank Account Numbers",
    "NRP": "Nationalities, religious, and political groups",
    "URL": "Web URLs",
    "AGE": "Age information",
    "CRYPTO": "Cryptocurrency addresses",
    "MEDICAL_LICENSE": "Medical license numbers",
    "US_ITIN": "US Individual Taxpayer Identification Numbers",
    "US_BANK_NUMBER": "US Bank account numbers",
    "PASSWORD": "Passwords",
    "GENERIC_PII": "Other personal identifiable information"
}

# Core allowed entities (limited for UI simplicity)
CORE_ENTITIES: Set[str] = {
    "PERSON", "ORGANIZATION", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER", 
    "CREDIT_CARD", "US_SSN", "DATE_TIME", "IP_ADDRESS", "ADDRESS",
    "US_PASSPORT", "US_DRIVER_LICENSE", "CAN_DRIVER_LICENSE", "PASSWORD",
}

# Default for filtering overlapping entities
DEFAULT_EXCLUDE_OVERLAPS = True
DEFAULT_OVERLAP_TOLERANCE = 5

# Validation dataset mapping
VALIDATION_TO_SYSTEM_MAPPING: Dict[str, str] = {
    'date': 'DATE_TIME',
    'time': 'DATE_TIME',
    'name': 'PERSON',
    'street_address': 'ADDRESS',
    'email': 'EMAIL_ADDRESS',
    'phone_number': 'PHONE_NUMBER',
    'ssn': 'US_SSN',
    'credit_card_number': 'CREDIT_CARD',
    'passport_number': 'US_PASSPORT',
    'driver_license_number': 'US_DRIVER_LICENSE',
    'driver_license_CA': 'CAN_DRIVER_LICENSE',
    'password': 'PASSWORD',
    'company': 'ORGANIZATION',
    'job_title': 'JOB_TITLE',
    'account_number': 'ACCOUNT_NUMBER',
}

# System to validation entity mapping (reverse of the above)
SYSTEM_TO_VALIDATION_MAPPING: Dict[str, str] = {v: k for k, v in VALIDATION_TO_SYSTEM_MAPPING.items()}

# GLiNER entity mapping
GLINER_ENTITY_MAPPING: Dict[str, str] = {
    "person": "PERSON",
    "name": "PERSON",
    "organization": "ORGANIZATION",
    "location": "LOCATION",
    "email": "EMAIL_ADDRESS",
    "credit card number": "CREDIT_CARD",
    "phone number": "PHONE_NUMBER",
    "social security number": "US_SSN",
    "passport number": "US_PASSPORT",
    "driver's license number": "US_DRIVER_LICENSE",
    "individual taxpayer identification number": "US_ITIN",
    "ip address": "IP_ADDRESS",
    "International Bank Account Number": "IBAN_CODE",
    "Age": "AGE",
    "debit card number": "CLIENT_CARD",
    "client card number": "CLIENT_CARD",
    "canadian social insurance number": "CAN_SIN",
    "canadian passport number": "CAN_PASSPORT",
    "canadian driver's license number": "CAN_DRIVER_LICENSE",
    "canadian postal code": "CAN_POSTAL_CODE",
    "MAC Address": "MAC_ADDRESS",
    "Personal Address Excluding City, Province, Postal Code, Country": "ADDRESS",
    "Company": "ORGANIZATION",
    "Password": "PASSWORD",
    "Job Title": "JOB_TITLE",
    "Job Position": "JOB_TITLE",
    "Account Number": "ACCOUNT_NUMBER",
}

# Evaluation settings
DEFAULT_OVERLAP_THRESHOLD = 0.5  # For entity matching in evaluation

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
