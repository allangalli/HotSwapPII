"""
Evaluation-related configuration settings for the HotSwapPII.
"""
from typing import Dict, List

# Evaluation settings
DEFAULT_OVERLAP_THRESHOLD = 0.5  # For entity matching in evaluation

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