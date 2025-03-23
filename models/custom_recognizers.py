"""
Custom recognizers for the PII detection system.
"""
import logging
from typing import List, Optional, Tuple

from presidio_analyzer import PatternRecognizer, Pattern

logger = logging.getLogger(__name__)

def create_canadian_postal_code_recognizer() -> PatternRecognizer:
    """
    Create a recognizer for Canadian postal codes.
    
    Canadian postal code format:
    - Format: A1A 1A1 (where A is a letter and 1 is a digit)
    - Space separates third and fourth characters (space is optional in the regex)
    - Cannot include letters D, F, I, O, Q, or U
    - First position cannot use W or Z
    
    Returns:
        PatternRecognizer for Canadian postal codes
    """
    # Define regex pattern for Canadian postal codes
    # ^(?!.*[DFIOQU])[A-VXY][0-9][A-Z] ?[0-9][A-Z][0-9]$
    patterns = [
        Pattern(
            name="canadian_postal_code",
            regex=r"(?i)(?<!\w)(?!.*[DFIOQU])[A-VXY][0-9][A-Z]\s?[0-9][A-Z][0-9](?!\w)",
            score=0.85
        )
    ]
    
    # Define context words to increase confidence
    context = [
        "postal", "code", "zip", "address", "mailing", "canada", "canadian",
        "province", "mail", "post"
    ]
    
    return PatternRecognizer(
        supported_entity="POSTAL_CODE",
        patterns=patterns,
        context=context,
        name="CanadianPostalCodeRecognizer"
    )


def create_canadian_drivers_license_recognizer() -> PatternRecognizer:
    """
    Create a recognizer for Canadian driver's licenses.
    
    Different provinces have different formats, but we'll implement the common ones:
    - Alberta: 9 digits
    - British Columbia: 7 digits, followed by a check digit
    - Ontario: A-NNNNN-NNNNN-NN or ANNNNN-NNNNN-NN
    - Quebec: A#### ######-##
    
    Returns:
        PatternRecognizer for Canadian driver's licenses
    """
    patterns = [
        # Alberta (9 digits)
        Pattern(
            name="alberta_dl",
            regex=r"(?<!\w)[0-9]{9}(?!\w)",
            score=0.6  # Lower initial score due to potential false positives
        ),
        
        # British Columbia (7 digits + check digit)
        Pattern(
            name="bc_dl",
            regex=r"(?<!\w)[0-9]{7}[0-9](?!\w)",
            score=0.6
        ),
        
        # Ontario format
        Pattern(
            name="ontario_dl",
            regex=r"(?<!\w)[A-Z]-\d{5}-\d{5}-\d{2}(?!\w)",
            score=0.85
        ),
        
        # Ontario alternative format
        Pattern(
            name="ontario_dl_alt",
            regex=r"(?<!\w)[A-Z]\d{5}-\d{5}-\d{2}(?!\w)",
            score=0.85
        ),
        
        # Quebec format
        Pattern(
            name="quebec_dl",
            regex=r"(?<!\w)[A-Z]\d{4}\s\d{6}-\d{2}(?!\w)",
            score=0.85
        )
    ]
    
    # Define context words to increase confidence
    context = [
        "driver", "license", "licence", "drivers", "driver's", "dl", "driving",
        "canada", "canadian", "alberta", "ontario", "quebec", "british columbia",
        "bc", "identification", "id"
    ]
    
    return PatternRecognizer(
        supported_entity="DRIVER_LICENSE_NUMBER",
        patterns=patterns,
        context=context,
        name="CanadianDriversLicenseRecognizer"
    )