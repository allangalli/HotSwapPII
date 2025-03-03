"""
Data processing utilities for the HotSwapPII.
"""
import json
import logging
from typing import Dict, List, Optional, Tuple, Union

import streamlit as st
from presidio_analyzer import RecognizerResult

logger = logging.getLogger(__name__)

def load_demo_text(file_path: str) -> str:
    """
    Load demo text from file.
    
    Args:
        file_path: Path to the demo text file
        
    Returns:
        Demo text as string
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading demo text from {file_path}: {e}")
        return "Hello, my name is John Smith. My email is john.smith@example.com and my phone number is (555) 123-4567."


def format_results_as_json(text: str, results: List[RecognizerResult]) -> str:
    """
    Format analyzer results as JSON.
    
    Args:
        text: The analyzed text
        results: List of RecognizerResult objects
        
    Returns:
        Formatted JSON string
    """
    if not results:
        return json.dumps([], indent=2)
    
    json_results = []
    
    for res in results:
        # Add each entity to the result list
        json_results.append({
            "start": int(res.start),
            "end": int(res.end),
            "text": str(text[res.start:res.end]),
            "entity_type": str(res.entity_type),
            "score": float(res.score)
        })
    
    # Sort by start position
    json_results.sort(key=lambda x: x["start"])
    
    return json.dumps(json_results, indent=2)


def get_file_upload_info(uploaded_file) -> Dict[str, str]:
    """
    Get information about an uploaded file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Dictionary with file information
    """
    if not uploaded_file:
        return {}
    
    try:
        return {
            "filename": uploaded_file.name,
            "type": uploaded_file.type,
            "size": f"{uploaded_file.size / 1024:.2f} KB",
        }
    except Exception as e:
        logger.error(f"Error getting file info: {e}")
        return {"error": str(e)}
