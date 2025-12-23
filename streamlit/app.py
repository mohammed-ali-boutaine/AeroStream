import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from sql.queries import (
    KPI_SUMMARY, SENTIMENT_DISTRIBUTION, TWEETS_BY_AIRLINE,
    NEGATIVE_REASONS, RECENT_TWEETS, TIME_SERIES, TOP_COMPLAINTS
)

# Configuration
DB_URL = "postgresql://ali:root@postgres_backend:5432/backend_db"

st.set_page_config(
    page_title="AeroStream - Analyse des Sentiments",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
        color: #000000;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #000000;
        text-align: center;
    }
    .kpi-container {
        background-color: #ffffff;
        border-radius: 1rem;
        border: 2px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #000000 !important;
    }
    .stMarkdown {
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# Database connection
@st.cache_resource
def get_db_engine():
    try:
        engine = create_engine(DB_URL, pool_pre_ping=True)
        return engine
    except Exception as e:
        st.error(f"Erreur: {e}")
        return None

# Data fetching functions
@st.cache_data(ttl=30)
def fetch_kpi_data():
    """Fetch KPI data for dashboard"""
    engine = get_db_engine()
    if engine:
        try:
            with engine.connect() as conn:
                result = conn.execute(text(KPI_SUMMARY)).fetchone()
            
            return {
                'total_tweets': result[0] or 0,
                'total_airlines': result[1] or 0,
                'negative_tweets': result[2] or 0,
                'negative_percentage': float(result[3]) if result[3] else 0,
                'positive_tweets': result[4] or 0,
                'neutral_tweets': result[5] or 0
            }
        except Exception as e:
            st.error(f"Erreur lors de la récupération des KPI: {e}")
            return None
    return None

@st.cache_data(ttl=30)
def fetch_sentiment_distribution():
    """Fetch sentiment distribution"""
    engine = get_db_engine()
    if engine:
        try:
            df = pd.read_sql(SENTIMENT_DISTRIBUTION, engine)
            return df
        except Exception as e:
            st.error(f"Erreur: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data(ttl=30)
def fetch_tweets_by_airline():
    """Fetch tweet volume by airline"""
    engine = get_db_engine()
    if engine:
        try:
            df = pd.read_sql(TWEETS_BY_AIRLINE, engine)
            return df
        except Exception as e:
            st.error(f"Erreur: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data(ttl=30)
def fetch_negative_reasons():
    """Fetch main negative reasons"""
    engine = get_db_engine()
    if engine:
        try:
            df = pd.read_sql(NEGATIVE_REASONS, engine)
            return df
        except Exception as e:
            st.error(f"Erreur: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data(ttl=30)
def fetch_recent_tweets(limit=50):
    """Fetch recent tweets"""
    engine = get_db_engine()
    if engine:
        try:
            df = pd.read_sql(text(RECENT_TWEETS), engine, params={'limit': limit})
            return df
        except Exception as e:
            st.error(f"Erreur: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data(ttl=30)
def fetch_time_series_data():
    """Fetch time series data for tweets"""
    engine = get_db_engine()
    if engine:
        try:
            df = pd.read_sql(TIME_SERIES, engine)
            return df
        except Exception as e:
            st.error(f"Erreur: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data(ttl=30)
def fetch_sentiment_by_airline():
    """Fetch sentiment distribution by airline with percentages"""
    engine = get_db_engine()
    if engine:
        try:
            df = pd.read_sql(TWEETS_BY_AIRLINE, engine)
            if not df.empty:
                # Calculate percentages
                df['total'] = df['positive'] + df['neutral'] + df['negative']
                df['positive_pct'] = (df['positive'] / df['total'] * 100).round(2)
                df['neutral_pct'] = (df['neutral'] / df['total'] * 100).round(2)
                df['negative_pct'] = (df['negative'] / df['total'] * 100).round(2)
                return df
            return df
        except Exception as e:
            st.error(f"Erreur: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# Main Page
st.markdown('<div class="main-header">Analyse des Sentiments des Compagnies Aériennes</div>', unsafe_allow_html=True)
    
# Fetch KPI data
kpi_data = fetch_kpi_data()

if kpi_data is None or kpi_data['total_tweets'] == 0:
    st.warning("Aucune donnée disponible. Veuillez exécuter le pipeline ETL d'abord.")
    st.info("Le pipeline ETL collecte les données en micro-batch, nettoie et prédit les sentiments, puis les stocke dans PostgreSQL.")
else:
    # KPI Cards in bordered container
    # st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
    st.markdown("### Indicateurs Clés de Performance (KPI)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Nombre Total de Tweets",
            value=f"{kpi_data['total_tweets']:,}",
            help="Volume total de tweets collectés et analysés"
        )
    
    with col2:
        st.metric(
            label="Nombre de Compagnies Aériennes",
            value=kpi_data['total_airlines'],
            help="Nombre de compagnies aériennes uniques dans les données"
        )
    
    with col3:
        st.metric(
            label="Pourcentage de Tweets Négatifs",
            value=f"{kpi_data['negative_percentage']:.1f}%",
            delta=f"-{kpi_data['negative_tweets']} tweets",
            delta_color="inverse",
            help="Proportion de tweets avec un sentiment négatif"
        )
    # st.markdown('</div>', unsafe_allow_html=True)
    
    # st.divider()
    
    # Sentiment Distribution and Time Series side by side
    st.markdown("### Répartition des Sentiments")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sentiment_df = fetch_sentiment_distribution()
        if not sentiment_df.empty:
            # Pie chart
            colors = {
                'positive': '#28a745',
                'neutral': '#ffc107',
                'negative': '#8b5cf6'
            }
            
            fig_pie = px.pie(
                sentiment_df,
                values='count',
                names='airline_sentiment',
                title="Distribution Globale des Sentiments",
                color='airline_sentiment',
                color_discrete_map=colors,
                hole=0.4
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(
                height=400, 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#000000'),
                title_font=dict(color='#000000'),
                legend=dict(font=dict(color='#000000'))
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        satisfaction_df = fetch_sentiment_by_airline()
        if not satisfaction_df.empty:
            # Calculate satisfaction rate (positive / total * 100)
            satisfaction_df['satisfaction_rate'] = (satisfaction_df['positive_pct']).round(2)
            satisfaction_sorted = satisfaction_df.sort_values('satisfaction_rate', ascending=True)
            
            fig_satisfaction = go.Figure()
            
            fig_satisfaction.add_trace(go.Bar(
                x=satisfaction_sorted['satisfaction_rate'],
                y=satisfaction_sorted['airline'],
                orientation='h',
                marker=dict(
                    color=satisfaction_sorted['satisfaction_rate'],
                    colorscale='Purples',
                    showscale=False
                ),
                text=satisfaction_sorted['satisfaction_rate'].apply(lambda x: f'{x:.1f}%'),
                textposition='outside'
            ))
            
            fig_satisfaction.update_layout(
                title="Calcul du taux de satisfaction par compagnie",
                xaxis_title="Taux de satisfaction (%)",
                yaxis_title="Compagnie",
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#000000'),
                title_font=dict(color='#000000'),
                xaxis=dict(title_font=dict(color='#000000'), tickfont=dict(color='#000000')),
                yaxis=dict(title_font=dict(color='#000000'), tickfont=dict(color='#000000')),
                showlegend=False
            )
            
            st.plotly_chart(fig_satisfaction, use_container_width=True)
    
    # st.divider()
    
    # Tweet Volume by Airline
    st.markdown("### Mesure du volume de tweets par compagnie")
    
    volume_df = fetch_sentiment_by_airline()
    if not volume_df.empty:
        volume_sorted = volume_df.sort_values('total', ascending=True)
        
        fig_volume = go.Figure()
        
        fig_volume.add_trace(go.Bar(
            x=volume_sorted['total'],
            y=volume_sorted['airline'],
            orientation='h',
            marker=dict(
                color=volume_sorted['total'],
                colorscale='Purples',
                showscale=False
            ),
            text=volume_sorted['total'],
            textposition='outside'
        ))
        
        fig_volume.update_layout(
            title="Volume de Tweets par Compagnie",
            xaxis_title="Nombre de Tweets",
            yaxis_title="Compagnie",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#000000'),
            title_font=dict(color='#000000'),
            xaxis=dict(title_font=dict(color='#000000'), tickfont=dict(color='#000000')),
            yaxis=dict(title_font=dict(color='#000000'), tickfont=dict(color='#000000')),
            showlegend=False
        )
        
        st.plotly_chart(fig_volume, use_container_width=True)
    
    # st.divider()
    
    # Sentiment Distribution by Airline
    st.markdown("### Distribution des Sentiments par Compagnie")
    
    airline_sentiment_df = fetch_sentiment_by_airline()
    if not airline_sentiment_df.empty:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Négatif',
            x=airline_sentiment_df['airline'],
            y=airline_sentiment_df['negative_pct'],
            marker_color='#8b5cf6'
        ))
        
        fig.add_trace(go.Bar(
            name='Neutre',
            x=airline_sentiment_df['airline'],
            y=airline_sentiment_df['neutral_pct'],
            marker_color='#808080'
        ))
        
        fig.add_trace(go.Bar(
            name='Positif',
            x=airline_sentiment_df['airline'],
            y=airline_sentiment_df['positive_pct'],
            marker_color='#28a745'
        ))
        
        fig.update_layout(
            title="Sentiment Distribution by Airline",
            xaxis_title="Airline",
            yaxis_title="Percentage",
            barmode='group',
            height=500,
            xaxis={'tickangle': 45, 'title_font': {'color': '#000000'}, 'tickfont': {'color': '#000000'}},
            yaxis={'title_font': {'color': '#000000'}, 'tickfont': {'color': '#000000'}},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#000000'),
            title_font=dict(color='#000000'),
            legend=dict(
                title="Sentiment",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color='#000000'),
                title_font=dict(color='#000000')
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Negative reasons
    st.markdown("### Principales Causes de Tweets Négatifs")
    
    negative_reasons = fetch_negative_reasons()
    if not negative_reasons.empty:
        fig_reasons = px.bar(
            negative_reasons,
            x='count',
            y='negativereason',
            orientation='h',
            title="Top 10 des Raisons de Mécontentement",
            labels={'count': 'Nombre de Mentions', 'negativereason': 'Raison'},
            color='count',
            color_continuous_scale='Purples'
        )
        fig_reasons.update_layout(
            height=500, 
            yaxis={'categoryorder': 'total ascending', 'title_font': {'color': '#000000'}, 'tickfont': {'color': '#000000'}},
            xaxis={'title_font': {'color': '#000000'}, 'tickfont': {'color': '#000000'}},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#000000'),
            title_font=dict(color='#000000')
        )
        st.plotly_chart(fig_reasons, use_container_width=True)
    else:
        st.info("Aucune raison négative enregistrée.")
