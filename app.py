import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

def load_specialty_data(file_path):
    """Load evaluation results for a single specialty"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Extract scores from dictionary fields
        metric_columns = [
            'accuracy', 'thoroughness', 'organized', 'synthesized',
            'hallucination', 'speciality_appropriateness', 'usefulness',
            'fairness', 'comprehensible', 'internally_consistent', 'succinct'
        ]
        
        for col in metric_columns:
            if col in df.columns:
                df[f'{col}_score'] = df[col].apply(lambda x: x['score'] if isinstance(x, dict) else x)
                df[f'{col}_justification'] = df[col].apply(lambda x: x['justification'] if isinstance(x, dict) else '')
        
        # Convert action types
        if 'final_action_type' in df.columns:
            df['final_action_type_binary'] = df['final_action_type'].map({1: 0, 2: 1})
            
        return df
        
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")
        return pd.DataFrame()

def analyze_specialty(df, specialty_name):
    """Create analysis section for a single specialty"""
    st.header(f"{specialty_name} Analysis")
    
    # Define score columns at the beginning
    score_columns = [col for col in df.columns if col.endswith('_score')]
    
    # Add Classification Score Overview at the top
    st.subheader("Classification Score Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_notes = len(df)
        st.metric("Total Notes", total_notes)
    
    with col2:
        avg_class_prob = df['classification_probability'].mean()
        st.metric("Average Classification Score", f"{avg_class_prob:.3f}")
    
    with col3:
        avg_accept_score = df[df['final_action_type'] == 1]['classification_probability'].mean()
        st.metric("Avg Score - Accepted Notes", f"{avg_accept_score:.3f}")
    
    with col4:
        avg_reject_score = df[df['final_action_type'] == 2]['classification_probability'].mean()
        st.metric("Avg Score - Rejected Notes", f"{avg_reject_score:.3f}")
    
    # Classification Score Distribution
    fig = go.Figure()
    
    # Add histogram for all notes
    fig.add_trace(go.Histogram(
        x=df['classification_probability'],
        name='All Notes',
        nbinsx=30,
        opacity=0.7
    ))
    
    # Add histograms for accepted and rejected notes
    fig.add_trace(go.Histogram(
        x=df[df['final_action_type'] == 1]['classification_probability'],
        name='Accepted Notes',
        nbinsx=30,
        opacity=0.7
    ))
    
    fig.add_trace(go.Histogram(
        x=df[df['final_action_type'] == 2]['classification_probability'],
        name='Rejected Notes',
        nbinsx=30,
        opacity=0.7
    ))
    
    fig.update_layout(
        title="Classification Score Distribution",
        xaxis_title="Classification Score",
        yaxis_title="Count",
        barmode='overlay',
        height=400
    )
    
    st.plotly_chart(fig)
    
    # Quality Metrics
    st.subheader("Quality Metrics")
    metrics = {
        'Accuracy': 'accuracy_score',
        'Thoroughness': 'thoroughness_score',
        'Organization': 'organized_score',
        'Synthesis': 'synthesized_score',
        'Hallucination': 'hallucination_score',
        'Specialty Appropriateness': 'speciality_appropriateness_score',
        'Usefulness': 'usefulness_score',
        'Fairness': 'fairness_score',
        'Comprehensibility': 'comprehensible_score',
        'Internal Consistency': 'internally_consistent_score',
        'Succinctness': 'succinct_score'
    }
    
    # Create metric columns dynamically
    cols = st.columns(4)
    for i, (metric_name, metric_col) in enumerate(metrics.items()):
        if metric_col in df.columns:
            with cols[i % 4]:
                avg_value = df[metric_col].mean()
                st.metric(metric_name, f"{avg_value:.2f}")
    
    # Score Distributions
    st.subheader("Score Distributions")
    
    # Box plot for all metrics
    fig = go.Figure()
    for metric_name, metric_col in metrics.items():
        if metric_col in df.columns:
            fig.add_trace(go.Box(
                y=df[metric_col],
                name=metric_name,
                boxmean=True
            ))
    
    fig.update_layout(
        title=f"Score Distribution - {specialty_name}",
        yaxis_title="Score",
        boxmode='group',
        height=600
    )
    st.plotly_chart(fig)
    
    # Correlation Matrix
    st.subheader("Metric Correlations")
    available_metrics = [col for col in metrics.values() if col in df.columns]
    corr = df[available_metrics].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr,
        x=[metric_name for metric_name, metric_col in metrics.items() if metric_col in df.columns],
        y=[metric_name for metric_name, metric_col in metrics.items() if metric_col in df.columns],
        text=corr.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        colorscale='RdBu'
    ))
    
    fig.update_layout(
        title=f"Correlation Matrix - {specialty_name}",
        height=800,
        width=800
    )
    st.plotly_chart(fig)
    
    # Classification Analysis
    st.subheader("Classification Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        if all(col in df.columns for col in ['classification', 'final_action_type_binary']):
            conf_matrix = confusion_matrix(
                df['final_action_type_binary'],
                df['classification']
            )
            
            fig = go.Figure(data=go.Heatmap(
                z=conf_matrix,
                x=['Predicted Accept', 'Predicted Reject'],
                y=['Actual Accept', 'Actual Reject'],
                text=conf_matrix,
                texttemplate='%{text}',
                textfont={"size": 12},
                colorscale='RdBu'
            ))
            
            fig.update_layout(
                title=f"Confusion Matrix - {specialty_name}",
                width=400,
                height=400
            )
            st.plotly_chart(fig)
            
            # Calculate metrics
            tn, fp, fn, tp = conf_matrix.ravel()
            metrics = {
                'Accuracy': (tp + tn) / (tp + tn + fp + fn),
                'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
            }
            metrics['F1-Score'] = 2 * (metrics['Precision'] * metrics['Recall']) / (metrics['Precision'] + metrics['Recall']) if (metrics['Precision'] + metrics['Recall']) > 0 else 0
            
            st.write("Classification Metrics:")
            metrics_df = pd.DataFrame({
                'Metric': list(metrics.keys()),
                'Value': [f"{v:.3f}" for v in metrics.values()]
            })
            st.dataframe(metrics_df)
    
    with col2:
        if 'classification_probability' in df.columns:
            # ROC Curve
            fpr, tpr, _ = roc_curve(
                df['final_action_type_binary'],
                df['classification_probability']
            )
            roc_auc = auc(fpr, tpr)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f'ROC curve (AUC = {roc_auc:.2f})',
                mode='lines'
            ))
            
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                name='Random',
                mode='lines',
                line=dict(dash='dash')
            ))
            
            fig.update_layout(
                title=f'ROC Curve - {specialty_name}',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                width=400,
                height=400
            )
            st.plotly_chart(fig)
    
    # Error Analysis
    st.subheader("Error Analysis")
    if all(col in df.columns for col in ['classification', 'final_action_type_binary']):
        misclassified = df[df['classification'] != df['final_action_type_binary']]
        
        st.write(f"Number of misclassified cases: {len(misclassified)}")
        
        if len(misclassified) > 0:
            # Show detailed analysis of misclassified cases
            st.write("### Misclassified Cases Analysis")
            
            # Average scores for misclassified vs correctly classified
            correct = df[df['classification'] == df['final_action_type_binary']]
            
            # Use score columns for comparison
            score_metrics = [col for col in df.columns if col.endswith('_score')]
            metric_names = [col.replace('_score', '').title() for col in score_metrics]
            
            comparison_df = pd.DataFrame({
                'Metric': metric_names,
                'Misclassified': [misclassified[col].mean() for col in score_metrics],
                'Correctly Classified': [correct[col].mean() for col in score_metrics]
            })
            
            fig = go.Figure(data=[
                go.Bar(name='Misclassified', x=comparison_df['Metric'], y=comparison_df['Misclassified']),
                go.Bar(name='Correctly Classified', x=comparison_df['Metric'], y=comparison_df['Correctly Classified'])
            ])
            
            fig.update_layout(
                title='Score Comparison: Misclassified vs Correctly Classified Cases',
                barmode='group',
                height=400,
                xaxis_tickangle=-45  # Angle the x-axis labels for better readability
            )
            st.plotly_chart(fig)
            
            # Display misclassified cases
            display_cols = ['request_id', 'classification_probability'] + score_metrics
            st.dataframe(misclassified[display_cols])

    
    # Justification Analysis
    st.subheader("Justification Examples")
    with st.expander("View Random Justifications"):
        justification_cols = [col for col in df.columns if col.endswith('_justification')]
        if justification_cols:
            sample_row = df.sample(n=1).iloc[0]
            for col in justification_cols:
                metric_name = col.replace('_justification', '')
                st.write(f"**{metric_name.title()}**: {sample_row[col]}")
    
    # Raw Data
    with st.expander("View Raw Data"):
        display_columns = ['request_id', 'classification_probability'] + score_columns
        st.dataframe(df[display_columns])
        
        # Download button
        csv = df[display_columns].to_csv(index=False)
        st.download_button(
            label=f"Download {specialty_name} Data",
            data=csv,
            file_name=f'evaluation_data_{specialty_name}.csv',
            mime='text/csv',
        )

def analyze_all_specialties(all_dfs):
    """Create analysis section combining all specialties"""
    st.header("Cross-Specialty Analysis")
    
    # Combine all dataframes
    df_combined = pd.concat(all_dfs, ignore_index=True)
    
    # Overall Metrics
    st.subheader("Overall Classification Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_notes = len(df_combined)
        st.metric("Total Notes Analyzed", total_notes)
    
    with col2:
        avg_class_prob = df_combined['classification_probability'].mean()
        st.metric("Average Classification Score", f"{avg_class_prob:.3f}")
    
    with col3:
        avg_accept_score = df_combined[df_combined['final_action_type'] == 1]['classification_probability'].mean()
        st.metric("Avg Score - Accepted Notes", f"{avg_accept_score:.3f}")
    
    with col4:
        avg_reject_score = df_combined[df_combined['final_action_type'] == 2]['classification_probability'].mean()
        st.metric("Avg Score - Rejected Notes", f"{avg_reject_score:.3f}")
    
    # Overall Quality Metrics Section - Now always visible
    st.subheader("Overall Quality Metrics (All Specialties)")
    
    # Get all score columns
    score_columns = [col for col in df_combined.columns if col.endswith('_score')]
    metric_names = [col.replace('_score', '').title() for col in score_columns]
    
    # Calculate overall averages and standard deviations
    overall_metrics = []
    cols = st.columns(4)  # Create 4 columns for metrics
    
    for i, (metric_col, metric_name) in enumerate(zip(score_columns, metric_names)):
        avg_value = df_combined[metric_col].mean()
        std_value = df_combined[metric_col].std()
        
        with cols[i % 4]:
            st.metric(
                metric_name,
                f"{avg_value:.2f}",
                f"±{std_value:.2f}"
            )
            
        overall_metrics.append({
            'Metric': metric_name,
            'Average': avg_value,
            'Std Dev': std_value,
            'Min': df_combined[metric_col].min(),
            'Max': df_combined[metric_col].max()
        })
    
    # Detailed metrics table - Now always visible
    st.subheader("Detailed Metrics Statistics")
    metrics_df = pd.DataFrame(overall_metrics)
    metrics_df = metrics_df.round(3)
    st.dataframe(metrics_df, use_container_width=True)
    
    # Distribution plot for all metrics
    fig = go.Figure()
    for metric_col in score_columns:
        fig.add_trace(go.Box(
            y=df_combined[metric_col],
            name=metric_col.replace('_score', '').title(),
            boxmean=True
        ))
    
    fig.update_layout(
        title="Distribution of All Quality Metrics",
        yaxis_title="Score",
        boxmode='group',
        height=500,
        showlegend=True
    )
    st.plotly_chart(fig)
    
    # Correlation matrix for all metrics
    corr_matrix = df_combined[score_columns].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=[col.replace('_score', '').title() for col in score_columns],
        y=[col.replace('_score', '').title() for col in score_columns],
        text=np.round(corr_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorscale='RdBu'
    ))
    
    fig.update_layout(
        title="Correlation Matrix of All Quality Metrics",
        height=800,
        width=800
    )
    st.plotly_chart(fig)
    
    # Specialty-wise Comparison
    st.subheader("Specialty-wise Comparison")
    
    # Classification scores by specialty
    fig = go.Figure()
    for specialty in df_combined['specialty'].unique():
        specialty_data = df_combined[df_combined['specialty'] == specialty]
        fig.add_trace(go.Box(
            y=specialty_data['classification_probability'],
            name=specialty,
            boxmean=True
        ))
    
    fig.update_layout(
        title="Classification Score Distribution by Specialty",
        yaxis_title="Classification Score",
        height=500
    )
    st.plotly_chart(fig)
    
    # Average Metrics by Specialty
    score_columns = [col for col in df_combined.columns if col.endswith('_score')]
    avg_metrics = df_combined.groupby('specialty')[score_columns].mean()
    
    # Create a heatmap of average scores
    fig = go.Figure(data=go.Heatmap(
        z=avg_metrics.values,
        x=[col.replace('_score', '').title() for col in score_columns],
        y=avg_metrics.index,
        text=np.round(avg_metrics.values, 2),
        texttemplate='%{text}',
        colorscale='RdYlGn'
    ))
    
    fig.update_layout(
        title="Average Quality Metrics by Specialty",
        height=600
    )
    st.plotly_chart(fig)
    
    # Action Type Distribution
    st.subheader("Action Type Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        # Overall distribution
        action_counts = df_combined['final_action_type'].value_counts()
        fig = px.pie(
            values=action_counts.values,
            names=action_counts.index.map({1: 'Accept', 2: 'Reject'}),
            title="Overall Action Type Distribution"
        )
        st.plotly_chart(fig)
    
    with col2:
        # Distribution by specialty
        action_by_specialty = pd.crosstab(df_combined['specialty'], df_combined['final_action_type'])
        action_by_specialty.columns = ['Accept', 'Reject']
        fig = px.bar(
            action_by_specialty,
            barmode='group',
            title="Action Types by Specialty"
        )
        st.plotly_chart(fig)
    
    # Quality Score Analysis
    st.subheader("Quality Score Analysis")
    
    # Calculate average quality score
    quality_cols = [col for col in score_columns if col != 'classification_probability']
    df_combined['avg_quality_score'] = df_combined[quality_cols].mean(axis=1)
    
    # Quality score vs Classification probability
    fig = px.scatter(
        df_combined,
        x='avg_quality_score',
        y='classification_probability',
        color='specialty',
        title="Quality Score vs Classification Probability",
        labels={
            'avg_quality_score': 'Average Quality Score',
            'classification_probability': 'Classification Probability'
        }
    )
    st.plotly_chart(fig)
    
    # Correlation Analysis
    st.subheader("Cross-Specialty Correlation Analysis")
    
    # Calculate correlations between classification probability and quality metrics
    correlations = []
    for specialty in df_combined['specialty'].unique():
        specialty_data = df_combined[df_combined['specialty'] == specialty]
        for metric in quality_cols:
            corr = specialty_data['classification_probability'].corr(specialty_data[metric])
            correlations.append({
                'Specialty': specialty,
                'Metric': metric.replace('_score', '').title(),
                'Correlation': corr
            })
    
    corr_df = pd.DataFrame(correlations)
    corr_pivot = corr_df.pivot(index='Specialty', columns='Metric', values='Correlation')
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_pivot.values,
        x=corr_pivot.columns,
        y=corr_pivot.index,
        text=np.round(corr_pivot.values, 2),
        texttemplate='%{text}',
        colorscale='RdBu'
    ))
    
    fig.update_layout(
        title="Correlation between Classification and Quality Metrics by Specialty",
        height=600
    )
    st.plotly_chart(fig)
    
    # Summary Statistics
    st.subheader("Summary Statistics")
    with st.expander("View Detailed Statistics"):
        summary_stats = df_combined.groupby('specialty').agg({
            'classification_probability': ['mean', 'std', 'count'],
            'avg_quality_score': ['mean', 'std']
        }).round(3)
        
        st.dataframe(summary_stats)
        
        # Download button for summary stats
        csv = summary_stats.to_csv()
        st.download_button(
            label="Download Summary Statistics",
            data=csv,
            file_name='summary_statistics.csv',
            mime='text/csv',
        )

def create_dashboard():
    st.set_page_config(page_title="Specialty Evaluation Dashboard", layout="wide")
    st.title("Specialty-wise Note Evaluation Dashboard")
    
    # Find all specialty files
    data_dir = Path('final_output/')
    eval_files = list(data_dir.glob('auto_eval_*.json'))
    
    if not eval_files:
        st.error("No evaluation files found!")
        return
    
    # Load all specialty data
    all_dfs = []
    specialty_names = []
    
    for file in eval_files:
        specialty_name = file.stem.replace('auto_eval_', '')
        df = load_specialty_data(file)
        if not df.empty:
            df['specialty'] = specialty_name
            all_dfs.append(df)
            specialty_names.append(specialty_name)
    
    # Create tabs
    tabs = ["Overview"] + specialty_names
    selected_tab = st.tabs(tabs)
    
    # Overview tab
    with selected_tab[0]:
        if all_dfs:
            analyze_all_specialties(all_dfs)
        else:
            st.error("No data available for analysis")
    
    # Individual specialty tabs
    for i, (tab, df) in enumerate(zip(selected_tab[1:], all_dfs)):
        with tab:
            analyze_specialty(df, specialty_names[i])

if __name__ == "__main__":
    create_dashboard()
