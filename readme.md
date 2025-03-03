# HotSwapPII

A comprehensive system for detecting and anonymizing personally identifiable information (PII) in text documents using various NER models.

## Overview

This application provides an interactive interface for detecting PII entities in text with features including:

- Support for multiple NER engines (SpaCy, HuggingFace Transformers, GLiNER)
- Multiple anonymization methods (redaction, replacement, masking, encryption)
- Synthetic data generation with OpenAI
- Model evaluation capabilities
- Entity-type performance metrics
- Customizable detection settings

## Installation

### Using Poetry (Recommended)

1. Clone this repository
2. Install dependencies with Poetry:

```bash
# Install Poetry if needed
pip install poetry

# Install dependencies
poetry install

# Run the application
poetry run streamlit run app.py
```

### Using pip

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Usage

### PII Detection

1. Enter or upload text in the input panel
2. Configure model and detection settings in the sidebar
3. Click "Analyze Text" to detect PII entities
4. View detection results and anonymized output

### Model Evaluation

1. Navigate to the "Model Evaluation" tab
2. Upload a CSV file with labeled data
3. Configure evaluation settings
4. Click "Evaluate Model" to assess model performance
5. View evaluation metrics and entity-type performance

## Features

### Detection Models

- **SpaCy**: Fast and efficient NER models for production use
- **HuggingFace Transformers**: Deep learning models with higher accuracy
- **GLiNER**: Generalist model for zero-shot named entity recognition

### Anonymization Methods

- **Redact**: Remove PII entities completely
- **Replace**: Replace with entity type (e.g., `<PERSON>`)
- **Mask**: Replace characters with a mask character (e.g., `*******`)
- **Hash**: Replace with a hash of the text
- **Encrypt**: Encrypt the text (reversible)
- **Highlight**: View original text with highlighted entities
- **Synthesize**: Replace with realistic fake values using OpenAI

### Advanced Features

- **Allow/Deny Lists**: Customize detection with word lists
- **Custom Regex Patterns**: Define your own entity patterns
- **Overlap Handling**: Configurable handling of overlapping entities
- **Entity Selection**: Choose which entity types to detect
- **Decision Process**: View reasoning behind detection decisions

## Data Format for Evaluation

The evaluation feature expects a CSV file with:

- A `text` column containing the text to evaluate
- A `label` column with JSON annotations in this format:

```json
[{"start": 58, "end": 89, "text": "John Smith", "labels": ["name"]}]
```

## Project Structure

```
pii-detection/
├── app.py                  # Main application entry point
├── config/                 # Configuration settings
├── models/                 # Model implementations
├── core/                   # Core detection and anonymization logic
├── ui/                     # UI components
├── utils/                  # Utility functions
├── data/                   # Sample data and resources
└── README.md               # Project documentation
```

## Dependencies

- presidio-analyzer: Microsoft's PII detection library
- presidio-anonymizer: Microsoft's PII anonymization library
- streamlit: Web application framework
- spacy: NLP toolkit
- transformers: Hugging Face's transformer models
- pandas: Data manipulation
- openai: OpenAI API for synthetic data

## Credits

- This application uses [Microsoft Presidio](https://github.com/microsoft/presidio) for PII detection and anonymization
- UI built with [Streamlit](https://streamlit.io/)
