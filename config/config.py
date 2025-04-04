"""
Configuration settings for the HotSwapPII.
"""
import os
from typing import Dict, List, Set

# Application settings
APP_TITLE = "HotSwapPII"
APP_DESCRIPTION = "Identify and anonymize personally identifiable information (PII) in text."

# Default text loaded on application start
DEFAULT_DEMO_TEXT_PATH = "data/demo_text.txt"

# Model configurations
DEFAULT_MODEL_INDEX = 0
DEFAULT_THRESHOLD = 0.40


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

# Available benchmark datasets
DATASET_OPTIONS = [
    "1. Gretel AI Generated PII data",
    "2. Different Gretel AI Generated PII data",
    "3. Simple generated data",
    "4. Generated data with variation",
    "5. Enron email data",
]

# Available benchmark datasets
DATASET_DESCRIPTIONS = [
    "500 rows from a Gretel AI Generated PII dataset. Gretel AI is a data generation software.",
    "Different Gretel AI Generated PII data",
    "Simple generated data",
    "Generated data with variation",
    "Enron email data",
]

# Available benchmark datasets
DATASET_SAMPLE_FILE_PATH = [
    "./data/benchmark_datasets/1_original_gretel_ai_conformance_data_500.csv",
    "./data/benchmark_datasets/2_another_gretel_ai_data_500.csv",
    "./data/benchmark_datasets/3_simple_generated_data_500.csv",
    "./data/benchmark_datasets/4_fake_data_with_variance_final.csv",
    "./data/benchmark_datasets/5_enron_data.csv"
]

DATASET_BENCHMARK_RESULTS_FILE_PATH = [
    "./data/benchmark_results/1_original_gretel_ai_conformance_data_500_results.json",
    "./data/benchmark_results/2_second_gretel_ai_data_500_results.json",
    "./data/benchmark_results/3_simple_generated_data_500_results.json",
    "./data/benchmark_results/4_fake_data_with_variance_final_results.json",
    "./data/benchmark_results/5_enron_data_results.json"
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
    "PERSON",
    # "ORGANIZATION",
    "LOCATION",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "CA_SIN",
    "DATE_TIME",
    "CAN_DRIVER_LICENSE",
    # "ADDRESS",
    "MAC_ADDRESS",
    "POSTAL_CODE",
    "ZIP_CODE",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    # "PASSWORD",
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
    # "US_ITIN": "US Individual Taxpayer Identification Numbers",
    "US_BANK_NUMBER": "US Bank account numbers",
    "PASSWORD": "Passwords",
    "GENERIC_PII": "Other personal identifiable information",
    "MAC_ADDRESS": "MAC Address",
    "CA_SIN": "Canadian Social Insurance Number",
    "POSTAL_CODE": "Canadian Postal Code",
    "ZIP_CODE": "US Zip Code",
    # "PROVINCE": "Canadian Province",
    # "STATE": "US State"
}

# Core allowed entities (limited for UI simplicity)
CORE_ENTITIES: Set[str] = {
    "PERSON", "ORGANIZATION", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER", 
    "CREDIT_CARD", "US_SSN", "DATE_TIME", "IP_ADDRESS", "ADDRESS",
    "US_PASSPORT", "US_DRIVER_LICENSE", "CAN_DRIVER_LICENSE", "PASSWORD",
    "MAC_ADDRESS", "CA_SIN", "POSTAL_CODE", "ZIP_CODE", "PROVINCE", "STATE"
}

# Default for filtering overlapping entities
DEFAULT_EXCLUDE_OVERLAPS = True
DEFAULT_OVERLAP_TOLERANCE = 5

# Ground truth tags
GROUND_TRUTH_TAGS = [
    'date',
    'time',
    'name',
    'email',
    'phone_number',
    'ssn',
    'credit_card_number',
    'passport_number',
    'driver_license_number',
    'password',
    'company',
    'account_number',
    'mac_address',
    'postal_code',
    'zip_code',
    'sin',
    'street_address'
]

# Validation dataset mapping
VALIDATION_TO_SYSTEM_MAPPING: Dict[str, str] = {
    'date': 'DATE_TIME',
    'time': 'DATE_TIME',
    'name': 'PERSON',
    # 'street_address': 'ADDRESS',
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
    'mac_address': 'MAC_ADDRESS',
    # 'province': 'PROVINCE',
    # 'state': 'STATE',
    'postal_code': 'POSTAL_CODE',
    'zip_code': 'ZIP_CODE',
    # 'sin': 'CA_SIN',
    'sin': 'CA_SIN',
    # 'location': 'LOCATION',
    'street_address': 'LOCATION',
}

# System to validation entity mapping (reverse of the above)
SYSTEM_TO_VALIDATION_MAPPING: Dict[str, str] = {v: k for k, v in VALIDATION_TO_SYSTEM_MAPPING.items()}

# GLiNER entity mapping
GLINER_ENTITY_MAPPING: Dict[str, str] = {
    "person": "PERSON",
    "organization": "ORGANIZATION",
    "address": "ADDRESS",
    "phone number": "PHONE_NUMBER",
    "passport number": "US_PASSPORT",
    "email": "EMAIL_ADDRESS",
    "credit card number": "CREDIT_CARD",
    "social security number": "US_SSN",
    "health insurance id number": "ID_NUMBER",
    "date of birth": "DATE_TIME",
    "mobile phone number": "PHONE_NUMBER",
    "bank account number": "ACCOUNT_NUMBER",
    "driver's license number": "US_DRIVER_LICENSE",
    "tax identification number": "ID_NUMBER",
    "identity card number": "ID_NUMBER",
    "national id number": "ID_NUMBER",
    "email address": "EMAIL_ADDRESS",
    "health insurance number": "ID_NUMBER",
    "student id number": "ID_NUMBER",
    "landline phone number": "PHONE_NUMBER",
    "postal code": "POSTAL_CODE",
    "passport_number": "US_PASSPORT",
    "identity document number": "ID_NUMBER",
    "social_security_number": "US_SSN",
    "name": "PERSON",
    "location": "LOCATION",
    "individual taxpayer identification number": "ID_NUMBER",
    "ip address": "IP_ADDRESS",
    "International Bank Account Number": "IBAN_CODE",
    "Age": "AGE",
    "debit card number": "CLIENT_CARD",
    "client card number": "CLIENT_CARD",
    "canadian social insurance number": "CA_SIN",
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

GLINER_LABELS = [
    "person",
    "organization",
    "address",
    "phone number",
    "passport number",
    "email",
    "credit card number",
    "social security number",
    "health insurance id number",
    "date of birth",
    "mobile phone number",
    "bank account number",
    "driver's license number",
    "tax identification number",
    "identity card number",
    "national id number",
    "email address",
    "health insurance number",
    "student id number",
    "landline phone number",
    "postal code",
    "passport_number",
    "identity document number",
    "social_security_number"
]

MODEL_ENTITIES_TO_STANDARDIZED_ENTITY_MAPPING: Dict[str, str] = {
    # Presidio Entities
    'DATE_TIME': 'date',
    'DATE': 'date',
    'TIME': 'date',
    'PERSON': 'name',
    'PHONE_NUMBER': 'phone_number',
    'EMAIL_ADDRESS': 'email',
    'ORGANIZATION': 'organization',
    'LOCATION': 'street_address',
    'ADDRESS': 'street_address',
    'POSTAL_CODE': 'postal_code',
    'ZIP_CODE': 'zip_code',
    'CREDIT_CARD': 'credit_card_number',
    'CLIENT_CARD': 'client_card_number',
    'MAC_ADDRESS': 'mac_address',
    'CA_SIN': 'sin',
    'US_SSN': 'ssn',
    'PROVINCE': 'province',
    'US_PASSPORT': 'passport_number',
    'CAN_PASSPORT': 'passport_number',
    'ID_NUMBER': 'id_number',
    'CAN_DRIVER_LICENSE': 'driver_license_number',
    'US_DRIVER_LICENSE': 'driver_license_number',
    'ACCOUNT_NUMBER': 'account_number',
    'IP_ADDRESS': 'ip_address',
    'JOB_TITLE': 'job_title',
    'CAN_POSTAL_CODE': 'postal_code',
    'IBAN_CODE': 'iban_code',
    'AGE': 'age',
    'PASSWORD': 'password',
    # GLiNER Entities
    "date of birth": "date",
    "mobile phone number": "phone_number",
    "bank account number": "account_number",
    "driver's license number": "driver_license_number",
    "tax identification number": "id_number",
    "identity card number": "id_number",
    "national id number": "id_number",
    "email address": "email",
    "health insurance number": "id_number",
    "student id number": "id_number",
    "landline phone number": "phone_number",
    "postal code": "postal_code",
    "identity document number": "id_number",
    'address': 'street_address',
    'email': 'email',
    'organization': 'company',
    'person': 'name',
    "phone number": "phone_number",
    "passport number": "passport_number",
    "credit card number": "credit_card_number",
    "social security number": "ssn",
    "health insurance id number": "id_number",
    # Piiranha Entities
    'I-ZIPCODE': 'postal_code',
    'I-SURNAME': 'name',
    'I-GIVENNAME': 'name',
    'I-USERNAME': 'id_number',
    'I-IDCARDNUM': 'id_number',
    'I-CITY': 'city',
    'I-STREET': 'street_address',
    'I-BUILDINGNUM': 'street_address',
    'I-EMAIL': 'email',
    'I-TELEPHONENUM': 'phone_number',
    'I-DATEOFBIRTH': 'date',
    'I-SOCIALNUM': 'ssn',
    'I-DRIVERLICENSENUM': 'driver_license_number',
    'I-ACCOUNTNUM': 'account_number',
    'I-TAXNUM': 'id_number',
    # 'ORG': 'name',
    'GPE': 'street_address',
    'LOC': 'street_address',
    "social_security_number": "ssn",
    # Gretel Entities (standardized expected ground truth labels)
    "phone_number": "phone_number",
    "passport_number": "passport_number",
    'postal_code': 'postal_code',
    'ssn': 'ssn',
    'id_number': 'id_number',
    'driver_license_number': 'driver_license_number',
    'street_address': 'street_address',
    "name": "name",
    "city": "city", # ?
    'account_number': 'account_number',
    'date': 'date'

}

# Evaluation settings
DEFAULT_OVERLAP_THRESHOLD = 0.5  # For entity matching in evaluation

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
