import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def create_score_distribution(df: pd.DataFrame, metrics: list):
    """Create score distribution plot"""
    fig = go.Figure()
    for metric in metrics:
        fig.add_trace(go.Histogram(
            x=df[metric],
            name=metric,
            opacity=0.75
        ))
    
    fig.update_layout(
        title="Score Distribution",
        barmode='overlay'
    )
    return fig

def create_time_series(df: pd.DataFrame, metrics: list):
    """Create time series plot"""
    fig = px.line(
        df,
        x='timestamp',
        y=metrics,
        title="Score Trends Over Time"
    )
    return fig

def create_correlation_matrix(df: pd.DataFrame, metrics: list):
    """Create correlation matrix"""
    corr = df[metrics].corr()
    fig = px.imshow(
        corr,
        text=corr.round(2),
        title="Metric Correlations"
    )
    return fig