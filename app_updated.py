import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
from plotly.subplots import make_subplots

def load_and_process_data(file_path: str):
    """Load and process the evaluation data."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Debug counter
    note_quality_count = 0
    auto_eval_data = []
    human_eval_data = []
    
    for sample in data:
        specialty = sample.get('specialties', ['Unknown'])[0]
        request_id = sample.get('request_id')
        timestamp_note_gen = pd.to_datetime(sample.get('timestamp_note_gen'))
        
        # Process auto eval data
        if sample.get("auto_eval_info") and len(sample["auto_eval_info"]) > 0:
            auto_info = sample["auto_eval_info"][0]
            
            # Debug print for first few samples
            if note_quality_count < 3:
                print(f"Sample {note_quality_count + 1} auto_eval_info:", auto_info.get('note_quality_classification'))
            
            # Add note_quality_classification
            if auto_info.get('note_quality_classification'):
                note_quality_count += 1
                auto_eval_data.append({
                    'request_id': request_id,
                    'specialty': specialty,
                    'metric': 'note_quality_classification',
                    'score': float(auto_info['note_quality_classification']['acceptance_class']),
                    'justification': f"Probability: {auto_info['note_quality_classification'].get('probability', 'N/A')}",
                    'timestamp_note_gen': timestamp_note_gen
                })
            
            # Process other metrics
            for score in auto_info.get("scores", []):
                auto_eval_data.append({
                    'request_id': request_id,
                    'specialty': specialty,
                    'metric': score['metric'],
                    'score': float(score['score']),
                    'justification': score.get('justification', ''),
                    'timestamp_note_gen': timestamp_note_gen
                })
        
        # Process human eval data
        if sample.get("human_eval_info"):
            human_info = sample["human_eval_info"]
            
            # Add acceptance_class as note_quality_classification
            if 'acceptance_class' in human_info:
                human_score = float(human_info['acceptance_class'])
                human_eval_data.append({
                    'request_id': request_id,
                    'specialty': specialty,
                    'metric': 'note_quality_classification',
                    'score': human_score,
                    'justification': '',
                    'timestamp_note_gen': timestamp_note_gen
                })

                human_eval_data.append({
                    'request_id': request_id,
                    'specialty': specialty,
                    'metric': 'note_classification',
                    'score': human_score,
                    'justification': '',
                    'timestamp_note_gen': timestamp_note_gen
                })
            
            # Process other metrics
            for score in human_info.get("metric_scores", []):
                if score.get("score") is not None:
                    human_eval_data.append({
                        'request_id': request_id,
                        'specialty': specialty,
                        'metric': score['metric'],
                        'score': float(score['score']),
                        'justification': score.get('justification', ''),
                        'timestamp_note_gen': timestamp_note_gen
                    })
    
    auto_df = pd.DataFrame(auto_eval_data)
    human_df = pd.DataFrame(human_eval_data)
    return auto_df, human_df

def create_classification_comparison(auto_df, human_df):
    """Create specific analysis for classification comparisons."""
    st.subheader("Note Classification Analysis")
    
    # Create tabs for different classification types
    class_tab1, class_tab2 = st.tabs(["Logistic Regression Note Classification", "LLM-generated Note Classification"])
    
    with class_tab1:
        st.subheader("Logitic Regression Note Classification Comparison")
        
        # Show overall statistics first
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Auto Classifications", 
                     len(auto_df[auto_df['metric'] == 'note_quality_classification']))
        with col2:
            st.metric("Total Human Classifications", 
                     len(human_df[human_df['metric'] == 'note_quality_classification']))
        with col3:
            st.metric("Matching Pairs", 
                     len(set(auto_df[auto_df['metric'] == 'note_quality_classification']['request_id'])
                         .intersection(set(human_df[human_df['metric'] == 'note_quality_classification']['request_id']))))
        
        # Merge auto and human poor note classifications
        poor_classification_data = pd.merge(
            auto_df[auto_df['metric'] == 'note_quality_classification'][['request_id', 'score', 'justification', 'specialty']],
            human_df[human_df['metric'] == 'note_quality_classification'][['request_id', 'score']],
            on='request_id',
            suffixes=('_auto', '_human'),
            how='inner'
        )
        
        if len(poor_classification_data) > 0:
            # Calculate agreement metrics
            total_comparisons = len(poor_classification_data)
            agreements = (poor_classification_data['score_auto'] == poor_classification_data['score_human']).sum()
            agreement_rate = (agreements / total_comparisons * 100)
            
            # Display agreement metrics
            st.subheader("Agreement Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Agreements", agreements)
            with col2:
                st.metric("Agreement Rate", f"{agreement_rate:.1f}%")
            
            # Create confusion matrix
            confusion_matrix = pd.crosstab(
                poor_classification_data['score_human'],
                poor_classification_data['score_auto'],
                margins=True
            )
            
            # Display confusion matrix
            st.subheader("Confusion Matrix")
            st.dataframe(confusion_matrix)
            
            # Create probability analysis
            if 'probability' in ' '.join(poor_classification_data['justification']):
                st.subheader("Probability Analysis")
                probabilities = poor_classification_data['justification'].str.extract(r'Probability: ([\d.]+)').astype(float)
                fig = px.scatter(
                    x=probabilities[0],
                    y=poor_classification_data['score_human'],
                    color=poor_classification_data['specialty'],
                    title='Classification Probability vs Human Decision',
                    labels={'x': 'Auto Classification Probability', 'y': 'Human Decision'}
                )
                st.plotly_chart(fig)
            
            # Specialty-wise breakdown
            st.subheader("Specialty-wise Analysis")
            specialty_stats = poor_classification_data.groupby('specialty').agg({
                'request_id': 'count',
                'score_auto': ['mean', 'std'],
                'score_human': ['mean', 'std']
            }).round(3)
            st.dataframe(specialty_stats)
            
            # Sample data
            st.subheader("Sample Classifications")
            st.dataframe(poor_classification_data.head(10))
        else:
            st.warning("No matching poor note classification data found")
    
    with class_tab2:
        st.subheader("LLM-generated Note Classification")
        
        # Show overall statistics first
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Auto Classifications", 
                     len(auto_df[auto_df['metric'] == 'note_classification']))
        with col2:
            st.metric("Total Human Classifications", 
                     len(human_df[human_df['metric'] == 'note_classification']))
        with col3:
            st.metric("Matching Pairs", 
                     len(set(auto_df[auto_df['metric'] == 'note_classification']['request_id'])
                         .intersection(set(human_df[human_df['metric'] == 'note_classification']['request_id']))))
        
        # Merge auto and human poor note classifications
        classification_data = pd.merge(
            auto_df[auto_df['metric'] == 'note_classification'][['request_id', 'score', 'justification', 'specialty']],
            human_df[human_df['metric'] == 'note_classification'][['request_id', 'score']],
            on='request_id',
            suffixes=('_auto', '_human'),
            how='inner'
        )
        
        if len(classification_data) > 0:
            # Calculate agreement metrics
            total_comparisons = len(classification_data)
            agreements = (classification_data['score_auto'] == classification_data['score_human']).sum()
            agreement_rate = (agreements / total_comparisons * 100)
            
            # Display agreement metrics
            st.subheader("Agreement Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Agreements", agreements)
            with col2:
                st.metric("Agreement Rate", f"{agreement_rate:.1f}%")
            
            # Create confusion matrix
            confusion_matrix = pd.crosstab(
                classification_data['score_human'],
                classification_data['score_auto'],
                margins=True
            )
            
            # Display confusion matrix
            st.subheader("Confusion Matrix")
            st.dataframe(confusion_matrix)
            
            # Create probability analysis
            if 'probability' in ' '.join(classification_data['justification']):
                st.subheader("Probability Analysis")
                probabilities = classification_data['justification'].str.extract(r'Probability: ([\d.]+)').astype(float)
                fig = px.scatter(
                    x=probabilities[0],
                    y=classification_data['score_human'],
                    color=classification_data['specialty'],
                    title='Classification Probability vs Human Decision',
                    labels={'x': 'Auto Classification Probability', 'y': 'Human Decision'}
                )
                st.plotly_chart(fig)
            
            # Specialty-wise breakdown
            st.subheader("Specialty-wise Analysis")
            specialty_stats = classification_data.groupby('specialty').agg({
                'request_id': 'count',
                'score_auto': ['mean', 'std'],
                'score_human': ['mean', 'std']
            }).round(3)
            st.dataframe(specialty_stats)
            
            # Sample data
            st.subheader("Sample Classifications")
            st.dataframe(classification_data.head(10))
        else:
            st.warning("No matching note classification data found")


def create_specialty_analysis(auto_df, human_df, selected_metrics, selected_specialties):
    """Create specialty-wise analysis visualizations."""
    
    st.subheader("Specialty-wise Analysis")
    
    # Specialty-wise mean scores
    auto_specialty_means = auto_df[auto_df['metric'].isin(selected_metrics)].groupby(
        ['specialty', 'metric'])['score'].mean().reset_index()
    
    # Create heatmap of average scores by specialty
    fig = go.Figure(data=go.Heatmap(
        z=auto_specialty_means.pivot(index='specialty', columns='metric', values='score').values,
        x=selected_metrics,
        y=auto_specialty_means.pivot(index='specialty', columns='metric', values='score').index,
        colorscale='RdYlGn',
        text=np.round(auto_specialty_means.pivot(index='specialty', columns='metric', values='score').values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
    ))
    
    fig.update_layout(
        title="Average Scores by Specialty and Metric",
        height=400 + (len(selected_specialties) * 20)
    )
    st.plotly_chart(fig)
    
    # Box plots for score distribution by specialty
    col1, col2 = st.columns(2)
    
    with col1:
        metric_for_box = st.selectbox("Select metric for distribution", selected_metrics)
        fig = px.box(
            auto_df[auto_df['metric'] == metric_for_box],
            x='specialty',
            y='score',
            title=f'{metric_for_box} Score Distribution by Specialty'
        )
        st.plotly_chart(fig)
    
    with col2:
        # Correlation matrix for selected specialty
        selected_specialty = st.selectbox("Select specialty for correlation analysis", selected_specialties)
        specialty_data = auto_df[auto_df['specialty'] == selected_specialty]
        
        pivot_data = specialty_data.pivot(columns='metric', values='score')
        corr_matrix = pivot_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
        ))
        fig.update_layout(
            title=f"Metric Correlations for {selected_specialty}",
            height=500,
            width=500
        )
        st.plotly_chart(fig)
    
    # Detailed statistics table
    st.subheader("Detailed Statistics by Specialty")
    specialty_stats = auto_df[auto_df['metric'].isin(selected_metrics)].groupby(
        ['specialty', 'metric']).agg({
            'score': ['count', 'mean', 'std', 'min', 'max']
        }).round(3)
    st.dataframe(specialty_stats)

def create_dashboard():
    st.title("Note Evaluation Analytics Dashboard")
    
    # Load data
    file_path = "last_three_weeks.json"
    auto_df, human_df = load_and_process_data(file_path)
    
    # Sidebar filters
    st.sidebar.header("Filters")
    selected_metrics = st.sidebar.multiselect(
        "Select Metrics",
        options=sorted(auto_df['metric'].unique()),
        default=['accuracy', 'hallucination', 'usefulness', 'note_quality_classification']
    )
    
    selected_specialties = st.sidebar.multiselect(
        "Select Specialties",
        options=sorted(auto_df['specialty'].unique()),
        default=sorted(auto_df['specialty'].unique())
    )
    
    # Filter data based on selections
    auto_df_filtered = auto_df[
        (auto_df['metric'].isin(selected_metrics)) & 
        (auto_df['specialty'].isin(selected_specialties))
    ]
    human_df_filtered = human_df[
        (human_df['metric'].isin(selected_metrics)) & 
        (human_df['specialty'].isin(selected_specialties))
    ]
    
    # create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overall Analysis", 
        "Specialty Analysis", 
        "Auto vs Human Comparison",
        "Detailed Metrics",
        "Weekly Trend Analysis",
        "Classification Analysis"
    ])
    
    with tab1:
        st.header("Overall Analysis")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Notes Evaluated", len(auto_df['request_id'].unique()))
        with col2:
            st.metric("Notes with Human Evaluation", len(human_df['request_id'].unique()))
        with col3:
            st.metric("Number of Specialties", len(selected_specialties))
        
        # Overall score distributions for both auto and human evals
        col1, col2 = st.columns(2)
        
        with col1:
            fig_auto = px.box(
                auto_df_filtered,
                x='metric',
                y='score',
                color='specialty',
                title='Auto Evaluation Score Distribution'
            )
            st.plotly_chart(fig_auto)
        
        with col2:
            fig_human = px.box(
                human_df_filtered,
                x='metric',
                y='score',
                color='specialty',
                title='Human Evaluation Score Distribution'
            )
            st.plotly_chart(fig_human)
        
        # Mean scores comparison
        st.subheader("Mean Scores Comparison")
        
        # Calculate statistics for both auto and human evals
        auto_means = auto_df_filtered.groupby('metric')['score'].mean().round(2)
        auto_stds = auto_df_filtered.groupby('metric')['score'].std().round(2)
        auto_counts = auto_df_filtered.groupby('metric')['score'].count()
        
        human_means = human_df_filtered.groupby('metric')['score'].mean().round(2)
        human_stds = human_df_filtered.groupby('metric')['score'].std().round(2)
        human_counts = human_df_filtered.groupby('metric')['score'].count()
        
        # Create comparison table
        comparison_data = []
        for metric in selected_metrics:
            comparison_data.append({
                'Metric': metric,
                'Auto Mean': f"{auto_means.get(metric, 0):.2f} ± {auto_stds.get(metric, 0):.2f} (n={auto_counts.get(metric, 0)})",
                'Human Mean': f"{human_means.get(metric, 0):.2f} ± {human_stds.get(metric, 0):.2f} (n={human_counts.get(metric, 0)})",
                'Auto Mean (raw)': auto_means.get(metric, 0),  # For sorting
                'Human Mean (raw)': human_means.get(metric, 0)  # For sorting
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Plot means comparison
        fig_means = go.Figure()
        fig_means.add_trace(go.Bar(
            name='Auto Evaluation',
            x=comparison_df['Metric'],
            y=comparison_df['Auto Mean (raw)'],
            error_y=dict(
                type='data',
                array=auto_stds[selected_metrics],
                visible=True
            )
        ))
        fig_means.add_trace(go.Bar(
            name='Human Evaluation',
            x=comparison_df['Metric'],
            y=comparison_df['Human Mean (raw)'],
            error_y=dict(
                type='data',
                array=human_stds[selected_metrics],
                visible=True
            )
        ))
        
        fig_means.update_layout(
            title='Mean Scores: Auto vs Human Evaluation',
            barmode='group',
            xaxis_title='Metric',
            yaxis_title='Score'
        )
        st.plotly_chart(fig_means)
        
        # Display comparison table
        st.dataframe(
        comparison_df[['Metric', 'Auto Mean', 'Human Mean', 'Auto Mean (raw)']]
        .sort_values('Auto Mean (raw)', ascending=False)
        .drop(columns=['Auto Mean (raw)'])  # Drop the sorting column after sorting
        .set_index('Metric')
)
    
    with tab2:
        create_specialty_analysis(auto_df_filtered, human_df_filtered, selected_metrics, selected_specialties)
    
    with tab3:
        st.header("Auto vs Human Evaluation Comparison")
        
        # Calculate agreement statistics
        agreement_data = []
        for specialty in selected_specialties:
            for metric in selected_metrics:
                # Get scores for the metric and specialty
                auto_subset = auto_df[
                    (auto_df['metric'] == metric) & 
                    (auto_df['specialty'] == specialty)
                ][['request_id', 'score']]
                
                human_subset = human_df[
                    (human_df['metric'] == metric) & 
                    (human_df['specialty'] == specialty)
                ][['request_id', 'score']]
                
                # Merge on request_id to get matching pairs
                merged_df = pd.merge(auto_subset, human_subset, 
                                   on='request_id', 
                                   suffixes=('_auto', '_human'))
                
                if not merged_df.empty:
                    matches = sum(merged_df['score_auto'] == merged_df['score_human'])
                    total = len(merged_df)
                    agreement = (matches / total) * 100 if total > 0 else 0
                    
                    agreement_data.append({
                        'specialty': specialty,
                        'metric': metric,
                        'agreement_rate': agreement,
                        'matches': matches,
                        'total': total,
                        'mean_auto': merged_df['score_auto'].mean(),
                        'mean_human': merged_df['score_human'].mean()
                    })
        
        if agreement_data:
            agreement_df = pd.DataFrame(agreement_data)
            
            # Agreement heatmap
            fig = px.density_heatmap(
                agreement_df,
                x='metric',
                y='specialty',
                z='agreement_rate',
                title='Agreement Rate Heatmap',
                labels={'agreement_rate': 'Agreement Rate (%)'}
            )
            st.plotly_chart(fig)
            
            # Mean difference analysis
            agreement_df['score_diff'] = agreement_df['mean_auto'] - agreement_df['mean_human']
            fig = px.bar(
                agreement_df,
                x='metric',
                y='score_diff',
                color='specialty',
                title='Mean Score Difference (Auto - Human)',
                barmode='group'
            )
            st.plotly_chart(fig)
            
            # Detailed statistics
            st.dataframe(agreement_df.round(2))
    
    with tab4:
        st.header("Detailed Metrics Analysis")
        
        # Select metric for detailed analysis
        selected_metric = st.selectbox("Select metric for detailed analysis", selected_metrics)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Score distribution
            fig = px.histogram(
                auto_df_filtered[auto_df_filtered['metric'] == selected_metric],
                x='score',
                color='specialty',
                title=f'Score Distribution for {selected_metric}',
                marginal='box'
            )
            st.plotly_chart(fig)
        
        with col2:
            # Sample justifications
            st.subheader("Sample Justifications")
            specialty = st.selectbox("Select specialty", selected_specialties)
            samples = auto_df_filtered[
                (auto_df_filtered['metric'] == selected_metric) & 
                (auto_df_filtered['specialty'] == specialty)
            ][['score', 'justification']].head()
            st.dataframe(samples)
        
        # Correlation analysis
        st.subheader("Correlation with Other Metrics")
        pivot_data = auto_df_filtered.pivot_table(
            index='request_id',
            columns='metric',
            values='score'
        )
        corr_matrix = pivot_data.corr()
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            title="Metric Correlations"
        )
        st.plotly_chart(fig)

    with tab5:
        st.header("Trend Analysis")
        
        # Time range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=auto_df_filtered['timestamp_note_gen'].min().date(),
                min_value=auto_df_filtered['timestamp_note_gen'].min().date(),
                max_value=auto_df_filtered['timestamp_note_gen'].max().date()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=auto_df_filtered['timestamp_note_gen'].max().date(),
                min_value=auto_df_filtered['timestamp_note_gen'].min().date(),
                max_value=auto_df_filtered['timestamp_note_gen'].max().date()
            )
        
        # Filter data by date range
        mask = (auto_df_filtered['timestamp_note_gen'].dt.date >= start_date) & (auto_df_filtered['timestamp_note_gen'].dt.date <= end_date)
        date_filtered_auto = auto_df_filtered[mask]
        
        mask = (human_df_filtered['timestamp_note_gen'].dt.date >= start_date) & (human_df_filtered['timestamp_note_gen'].dt.date <= end_date)
        date_filtered_human = human_df_filtered[mask]
        
        # Volume analysis
        st.subheader("Weekly Evaluation Volume")

        # Calculate weekly volumes for unique request_ids
        weekly_auto_volume = (
            date_filtered_auto
            .groupby([pd.Grouper(key='timestamp_note_gen', freq='W'), 'metric'])['request_id']
            .nunique()
            .reset_index()
            .rename(columns={'request_id': 'auto_count'})
        )

        weekly_human_volume = (
            date_filtered_human
            .groupby([pd.Grouper(key='timestamp_note_gen', freq='W'), 'metric'])['request_id']
            .nunique()
            .reset_index()
            .rename(columns={'request_id': 'human_count'})
        )

        # Merge volumes
        weekly_volume = pd.merge(
            weekly_auto_volume,
            weekly_human_volume,
            on=['timestamp_note_gen', 'metric'],
            how='outer'
        ).fillna(0)

        # Create two subplots with shared x-axis
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Auto Evaluation Volume', 'Human Evaluation Volume')
        )

        # Add auto volume trace
        fig.add_trace(
            go.Bar(
                name='Auto Evaluations',
                x=weekly_volume['timestamp_note_gen'],
                y=weekly_volume['auto_count'],
                marker_color='blue',
                hovertemplate="Week: %{x}<br>Notes evaluated: %{y}<extra></extra>"
            ),
            row=1, col=1
        )

        # Add human volume trace
        fig.add_trace(
            go.Bar(
                name='Human Evaluations',
                x=weekly_volume['timestamp_note_gen'],
                y=weekly_volume['human_count'],
                marker_color='red',
                hovertemplate="Week: %{x}<br>Notes evaluated: %{y}<extra></extra>"
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            height=600,
            title='Weekly Evaluation Volume',
            showlegend=False,
            hovermode='x unified'
        )

        # Update y-axes labels
        fig.update_yaxes(title_text="Number of Notes", row=1, col=1)
        fig.update_yaxes(title_text="Number of Notes", row=2, col=1)
        fig.update_xaxes(title_text="Week", row=2, col=1)

        st.plotly_chart(fig)

        # Add summary statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Average Weekly Auto Evaluations", 
                f"{weekly_volume['auto_count'].mean():.1f}",
                f"Total: {weekly_volume['auto_count'].sum():.0f}"
            )
        with col2:
            st.metric(
                "Average Weekly Human Evaluations", 
                f"{weekly_volume['human_count'].mean():.1f}",
                f"Total: {weekly_volume['human_count'].sum():.0f}"
            )
        
        # Weekly score trends
        st.subheader("Weekly Score Trends")
        
        # Metric selector for detailed trend
        metric_for_trend = st.selectbox(
            "Select metric for trend analysis",
            selected_metrics,
            key="trend_metric"
        )
        
        # Weekly aggregation with volume info
        weekly_auto = (
            date_filtered_auto[date_filtered_auto['metric'] == metric_for_trend]
            .set_index('timestamp_note_gen')
            .groupby([pd.Grouper(freq='W')])
            .agg({
                'score': ['mean', 'std', 'count'],
                'request_id': 'nunique'
            })
            .reset_index()
        )
        weekly_auto.columns = ['timestamp_note_gen', 'mean', 'std', 'total_evals', 'unique_notes']
        
        weekly_human = (
            date_filtered_human[date_filtered_human['metric'] == metric_for_trend]
            .set_index('timestamp_note_gen')
            .groupby([pd.Grouper(freq='W')])
            .agg({
                'score': ['mean', 'std', 'count'],
                'request_id': 'nunique'
            })
            .reset_index()
        )
        weekly_human.columns = ['timestamp_note_gen', 'mean', 'std', 'total_evals', 'unique_notes']
        
        # Create trend plot with volume annotations
        fig = go.Figure()
        
        # Add auto evaluation trend
        fig.add_trace(go.Scatter(
            x=weekly_auto['timestamp_note_gen'],
            y=weekly_auto['mean'],
            name='Auto Evaluation',
            mode='lines+markers',
            error_y=dict(
                type='data',
                array=weekly_auto['std'],
                visible=True
            ),
            hovertemplate="<br>".join([
                "Week: %{x}",
                "Score: %{y:.2f}",
                "Notes evaluated: %{customdata[0]}",
                "Total evaluations: %{customdata[1]}"
            ]),
            customdata=np.column_stack((weekly_auto['unique_notes'], weekly_auto['total_evals']))
        ))
        
        # Add human evaluation trend
        fig.add_trace(go.Scatter(
            x=weekly_human['timestamp_note_gen'],
            y=weekly_human['mean'],
            name='Human Evaluation',
            mode='lines+markers',
            error_y=dict(
                type='data',
                array=weekly_human['std'],
                visible=True
            ),
            hovertemplate="<br>".join([
                "Week: %{x}",
                "Score: %{y:.2f}",
                "Notes evaluated: %{customdata[0]}",
                "Total evaluations: %{customdata[1]}"
            ]),
            customdata=np.column_stack((weekly_human['unique_notes'], weekly_human['total_evals']))
        ))
        
        fig.update_layout(
            title=f'Weekly {metric_for_trend} Score Trends',
            xaxis_title='Week',
            yaxis_title='Score',
            hovermode='x unified'
        )
        st.plotly_chart(fig)
        
        # Statistical summary with volume info
        st.subheader("Weekly Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Auto Evaluation Statistics")
            st.dataframe(weekly_auto.round(3))
        
        with col2:
            st.write("Human Evaluation Statistics")
            st.dataframe(weekly_human.round(3))
    
    with tab6:
        create_classification_comparison(auto_df_filtered, human_df_filtered)
        

    
if __name__ == "__main__":
    st.set_page_config(page_title="Note Evaluation Analytics", layout="wide")
    create_dashboard()