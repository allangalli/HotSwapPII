"""
Visualization utilities for the HotSwapPII.
"""
import logging
from typing import Dict, List, Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st
from presidio_analyzer import RecognizerResult

logger = logging.getLogger(__name__)

def create_entity_barchart(
    entity_counts: Dict[str, int],
    title: str = "Entity Type Distribution",
) -> alt.Chart:
    """
    Create a bar chart of entity type counts.
    
    Args:
        entity_counts: Dictionary mapping entity types to counts
        title: Chart title
        
    Returns:
        Altair chart object
    """
    if not entity_counts:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame({
        "Entity Type": list(entity_counts.keys()),
        "Count": list(entity_counts.values())
    })
    
    # Sort by count descending
    df = df.sort_values("Count", ascending=False)
    
    # Create chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Entity Type:N", sort=None),
        y=alt.Y("Count:Q"),
        color=alt.Color("Entity Type:N", legend=None),
        tooltip=["Entity Type", "Count"]
    ).properties(
        title=title,
        width=600,
        height=300
    ).interactive()
    
    return chart


def create_confidence_histogram(
    results: List[RecognizerResult],
    bins: int = 10,
    title: str = "Confidence Score Distribution",
) -> alt.Chart:
    """
    Create a histogram of confidence scores.
    
    Args:
        results: List of RecognizerResult objects
        bins: Number of bins for the histogram
        title: Chart title
        
    Returns:
        Altair chart object
    """
    if not results:
        return None
    
    # Extract confidence scores
    scores = [result.score for result in results]
    
    # Create DataFrame
    df = pd.DataFrame({"Confidence": scores})
    
    # Create histogram
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Confidence:Q", bin=alt.Bin(maxbins=bins), title="Confidence Score"),
        y=alt.Y("count()", title="Count"),
        tooltip=["count()"]
    ).properties(
        title=title,
        width=600,
        height=300
    ).interactive()
    
    return chart


def create_entity_confidence_boxplot(
    results: List[RecognizerResult],
    title: str = "Confidence Scores by Entity Type",
) -> alt.Chart:
    """
    Create a box plot of confidence scores by entity type.
    
    Args:
        results: List of RecognizerResult objects
        title: Chart title
        
    Returns:
        Altair chart object
    """
    if not results:
        return None
    
    # Create DataFrame
    df = pd.DataFrame([
        {"Entity Type": result.entity_type, "Confidence": result.score}
        for result in results
    ])
    
    # Create box plot
    chart = alt.Chart(df).mark_boxplot().encode(
        x=alt.X("Entity Type:N", title="Entity Type"),
        y=alt.Y("Confidence:Q", title="Confidence Score"),
        color="Entity Type:N",
        tooltip=["Entity Type", "Confidence"]
    ).properties(
        title=title,
        width=600,
        height=300
    ).interactive()
    
    return chart


def create_metrics_heatmap(
    entity_metrics: pd.DataFrame,
    metric: str = "f1",
    title: Optional[str] = None,
) -> alt.Chart:
    """
    Create a heatmap of evaluation metrics by entity type.
    
    Args:
        entity_metrics: DataFrame with entity-type metrics
        metric: Metric to visualize ('precision', 'recall', or 'f1')
        title: Chart title (optional)
        
    Returns:
        Altair chart object
    """
    if entity_metrics.empty:
        return None
    
    # Validate metric
    if metric not in ["precision", "recall", "f1"]:
        metric = "f1"
    
    # Set title
    if title is None:
        metric_names = {"precision": "Precision", "recall": "Recall", "f1": "F1 Score"}
        title = f"{metric_names[metric]} by Entity Type"
    
    # Select relevant columns
    df = entity_metrics[["entity_type", metric]].copy()
    
    # Convert to long format
    df = df.rename(columns={"entity_type": "Entity Type", metric: "Value"})
    
    # Create heatmap
    chart = alt.Chart(df).mark_rect().encode(
        x=alt.X("Entity Type:N", title="Entity Type"),
        color=alt.Color(
            "Value:Q",
            scale=alt.Scale(domain=[0, 1], scheme="blues"),
            legend=alt.Legend(title="Value")
        ),
        tooltip=["Entity Type", "Value"]
    ).properties(
        title=title,
        width=600,
        height=100
    ).interactive()
    
    # Add text values
    text = chart.mark_text(baseline="middle").encode(
        text=alt.Text("Value:Q", format=".2f"),
        color=alt.condition(
            alt.datum.Value > 0.5,
            alt.value("white"),
            alt.value("black")
        )
    )
    
    return chart + text


def create_f1_comparison_chart(
    entity_metrics: pd.DataFrame,
    title: str = "Precision, Recall, and F1 by Entity Type",
) -> alt.Chart:
    """
    Create a grouped bar chart comparing precision, recall, and F1 by entity type.
    
    Args:
        entity_metrics: DataFrame with entity-type metrics
        title: Chart title
        
    Returns:
        Altair chart object
    """
    if entity_metrics.empty:
        return None
    
    # Select relevant columns
    df = entity_metrics[["entity_type", "precision", "recall", "f1"]].copy()
    
    # Melt to long format
    df_melted = pd.melt(
        df,
        id_vars=["entity_type"],
        value_vars=["precision", "recall", "f1"],
        var_name="Metric",
        value_name="Value"
    )
    
    # Rename columns
    df_melted = df_melted.rename(columns={"entity_type": "Entity Type"})
    
    # Title case metric names
    df_melted["Metric"] = df_melted["Metric"].str.title()
    
    # Create chart
    chart = alt.Chart(df_melted).mark_bar().encode(
        x=alt.X("Entity Type:N", title="Entity Type"),
        y=alt.Y("Value:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "Metric:N",
            scale=alt.Scale(
                domain=["Precision", "Recall", "F1"],
                range=["#1f77b4", "#ff7f0e", "#2ca02c"]
            )
        ),
        column=alt.Column("Metric:N", title=None),
        tooltip=["Entity Type", "Metric", "Value"]
    ).properties(
        title=title,
        width=200,
        height=300
    ).interactive()
    
    return chart
