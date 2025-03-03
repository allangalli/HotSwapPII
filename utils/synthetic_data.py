"""
Synthetic data generation for the HotSwapPII using OpenAI.
"""
import logging
import os
from dataclasses import dataclass
from typing import Optional

import streamlit as st
from openai import OpenAI

logger = logging.getLogger(__name__)

@dataclass
class OpenAIParams:
    """
    Parameters for OpenAI API calls.
    """
    api_key: str
    model: str
    max_tokens: int = 512
    temperature: float = 0.7


@st.cache_data
def generate_synthetic_text(
    anonymized_text: str,
    openai_params: OpenAIParams
) -> str:
    """
    Generate synthetic text with fake PII values using OpenAI.
    
    Args:
        anonymized_text: Text with placeholders for PII (e.g., "<PERSON>")
        openai_params: OpenAI API parameters
        
    Returns:
        Text with fake PII values
    """
    if not openai_params.api_key:
        return "Error: Please provide an OpenAI API key"
    
    try:
        client = OpenAI(api_key=openai_params.api_key)
        
        # Create prompt
        prompt = create_synthetic_data_prompt(anonymized_text)
        
        # Call OpenAI
        response = client.completions.create(
            model=openai_params.model,
            prompt=prompt,
            max_tokens=openai_params.max_tokens,
            temperature=openai_params.temperature,
        )
        
        return response.choices[0].text.strip()
    except Exception as e:
        logger.error(f"Error generating synthetic text: {e}")
        return f"Error generating synthetic text: {e}"


def create_synthetic_data_prompt(anonymized_text: str) -> str:
    """
    Create a prompt for synthetic data generation.
    
    Args:
        anonymized_text: Text with placeholders
        
    Returns:
        Prompt for OpenAI
    """
    return f"""
    Your role is to create synthetic text based on de-identified text with placeholders instead of Personally Identifiable Information (PII).
    Replace the placeholders (e.g. <PERSON>, <DATE_TIME>, <EMAIL_ADDRESS>) with fake values.

    Instructions:

    a. Use completely random numbers, so every digit is drawn between 0 and 9.
    b. Use realistic names that come from diverse genders, ethnicities and countries.
    c. If there are no placeholders, return the text as is.
    d. Keep the formatting as close to the original as possible.
    e. If PII exists in the input, replace it with fake values in the output.
    f. Remove whitespace before and after the generated text
    
    Examples:
    input: [[TEXT STARTS]] My credit card number is <CREDIT_CARD> and it expires on <DATE_TIME>. [[TEXT ENDS]]
    output: My credit card number is 4539 1867 2497 5592 and it expires on March 12, 2027.
    
    input: [[TEXT STARTS]] <PERSON> was the chief science officer at <ORGANIZATION>. [[TEXT ENDS]]
    output: Maria Rodriguez was the chief science officer at NeuralTech Systems.
    
    input: [[TEXT STARTS]] {anonymized_text} [[TEXT ENDS]]
    output:"""
