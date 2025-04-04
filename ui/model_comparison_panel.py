"""
Evaluation panel UI components for the HotSwapPII.
"""
import logging
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from core.evaluator import (analyze_threshold_sensitivity, evaluate_model,
                            process_validation_data)
from utils.visualization import (create_f1_comparison_chart,
                                 create_metrics_heatmap)
from config.config import DATASET_OPTIONS, DATASET_DESCRIPTIONS, DATASET_SAMPLE_FILE_PATH, DATASET_BENCHMARK_RESULTS_FILE_PATH, MODEL_OPTIONS_TO_BENHCMARKS_KEY

logger = logging.getLogger(__name__)


def render_model_comparison_panel(settings: Dict) -> None:
    """
    Render the model benchmarks panel.

    Args:
        settings: Dictionary with application settings
    """
    st.header("Model Comparison")

    with st.expander("About Model Comparison", expanded=False):
        st.info('Placeholder')

    # Model selection
    dataset_help = """
    Select which dataset benchmark metrics to view. The metrics correspond to the performance of the selected model in
    the sidebar on this particular dataset along with other important metrics. Descriptions of the dataset and it's distinguishing features
    are available.
    """

    dataset_selection = st.selectbox(
        "Dataset",
        options=DATASET_OPTIONS,
        index=None,
        placeholder="Select dataset...",
        help=dataset_help,
        key="model_comparison_panel",
    )

    if dataset_selection:
        selected_dataset_index = DATASET_OPTIONS.index(dataset_selection)

        metric_evaluation_schemas = ["Type", "Partial", "Exact", "Strict"]
        metric_evaluation_schemas_keys = ["ent_type", "partial", "exact", "strict"]
        metric_evaluation_schema_tabs = st.tabs(metric_evaluation_schemas)

        with open(DATASET_BENCHMARK_RESULTS_FILE_PATH[selected_dataset_index]) as f:
            benchmark_results = json.load(f)
            for i in range(4):
                with metric_evaluation_schema_tabs[i]:
                    evaluation_metrics = ["Recall", "Precision", "F1"]
                    evaluation_metrics_keys = ["recall", "precision", "f1"]
                    evaluation_metrics_tabs = st.tabs(evaluation_metrics)
                    for j in range(3):
                        with evaluation_metrics_tabs[j]:
                            def filter_out_singular_metrics(pair):
                                unwanted_keys = ['rows', 'time_taken']
                                key, value = pair
                                if key in unwanted_keys:
                                    return False
                                else:
                                    return True

                            filtered_benchmark_results = dict(filter(filter_out_singular_metrics, benchmark_results.items()))
                            models = list(filtered_benchmark_results.keys())
                            data = []
                            BENCHMARK_TO_MODEL_OPTION = {v: k for k, v in MODEL_OPTIONS_TO_BENHCMARKS_KEY.items()}
                            columns = ["Entity Type"] + [BENCHMARK_TO_MODEL_OPTION[model] for model in models]
                            filtered_keys = dict(filter(filter_out_singular_metrics, filtered_benchmark_results[models[0]].items()))

                            for entity_type in filtered_keys:
                                row = [entity_type]
                                for model in models:
                                    row.append([
                                        f"{filtered_benchmark_results[model][entity_type][metric_evaluation_schemas_keys[i]][evaluation_metrics_keys[j]]:.2%}"
                                    ])
                                data.append(row)

                            # Rename columns
                            display_df = pd.DataFrame(
                                data=data,
                                columns=columns
                            )

                            # Display the table
                            st.dataframe(
                                display_df,
                                use_container_width=True,
                                hide_index=True,
                            )
                            
                            # Add download button for model comparison results
                            csv_comparison = display_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label=f"Download {evaluation_metrics[j]} Comparison ({metric_evaluation_schemas[i]}) (CSV)",
                                data=csv_comparison,
                                file_name=f"{dataset_selection}_{metric_evaluation_schemas_keys[i]}_{evaluation_metrics_keys[j]}_comparison.csv",
                                mime="text/csv",
                            )