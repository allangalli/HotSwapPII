"""
Benchmark-related configuration settings for the HotSwapPII.
"""
from typing import List

# Available benchmark datasets
DATASET_OPTIONS = [
    "1. Gretel AI Generated PII data",
    "2. Different Gretel AI Generated PII data",
    "3. Simple generated data",
    "4. Generated data with variation",
    "5. Enron email data",
]

# Descriptions for benchmark datasets
DATASET_DESCRIPTIONS = [
    "500 rows from a Gretel AI Generated PII dataset. Gretel AI is a data generation software.",
    "Different Gretel AI Generated PII data",
    "Simple generated data",
    "Generated data with variation",
    "Enron email data",
]

# File paths for benchmark dataset samples
DATASET_SAMPLE_FILE_PATH: List[str] = [
    "./data/benchmark_datasets/1_original_gretel_ai_conformance_data_500.csv",
    "./data/benchmark_datasets/2_another_gretel_ai_data_500.csv",
    "./data/benchmark_datasets/3_simple_generated_data_500.csv",
    "./data/benchmark_datasets/4_fake_data_with_variance_final.csv",
    "./data/benchmark_datasets/5_enron_data.csv"
]

# File paths for benchmark results
DATASET_BENCHMARK_RESULTS_FILE_PATH: List[str] = [
    "./data/benchmark_results/1_original_gretel_ai_conformance_data_500_results.json",
    "./data/benchmark_results/2_second_gretel_ai_data_500_results.json",
    "./data/benchmark_results/3_simple_generated_data_500_results.json",
    "./data/benchmark_results/4_fake_data_with_variance_final_results.json",
    "./data/benchmark_results/5_enron_data_results.json"
] 