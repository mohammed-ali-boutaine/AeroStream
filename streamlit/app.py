import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import os
from datetime import datetime

# Configuration
DB_URL = "postgresql://ali:root@postgres_backend:5432/backend_db"

st.set_page_config(
    page_title="✈️ AeroStream - Analyse des Sentiments",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Database connection
@st.cache_resource
def get_db_engine():
    """Create SQLAlchemy engine"""
    try:
        engine = create_engine(DB_URL, pool_pre_ping=True)
        return engine
    except Exception as e:
        st.error(f"❌ Erreur de connexion à la base de données: {e}")
        return None

# Data fetching functions
@st.cache_data(ttl=30)
def fetch_kpi_data():
    """Fetch KPI data for dashboard"""
    engine = get_db_engine()
    if engine:
        try:
            query = """
                SELECT 
                    COUNT(*) as total_tweets,
                    COUNT(DISTINCT airline) as total_airlines,
                    SUM(CASE WHEN airline_sentiment = 'negative' THEN 1 ELSE 0 END) as negative_tweets,
                    ROUND(
                        100.0 * SUM(CASE WHEN airline_sentiment = 'negative' THEN 1 ELSE 0 END) / COUNT(*), 
                        2
                    ) as negative_percentage,
                    SUM(CASE WHEN airline_sentiment = 'positive' THEN 1 ELSE 0 END) as positive_tweets,
                    SUM(CASE WHEN airline_sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral_tweets
                FROM airline_tweets
            """
            with engine.connect() as conn:
                result = conn.execute(text(query)).fetchone()
            
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
            query = """
                SELECT airline_sentiment, COUNT(*) as count
                FROM airline_tweets
                GROUP BY airline_sentiment
                ORDER BY count DESC
            """
            df = pd.read_sql(query, engine)
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
            query = """
                SELECT 
                    airline,
                    COUNT(*) as tweet_count,
                    SUM(CASE WHEN airline_sentiment = 'positive' THEN 1 ELSE 0 END) as positive,
                    SUM(CASE WHEN airline_sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral,
                    SUM(CASE WHEN airline_sentiment = 'negative' THEN 1 ELSE 0 END) as negative,
                    ROUND(
                        100.0 * SUM(CASE WHEN airline_sentiment = 'positive' THEN 1 ELSE 0 END) / COUNT(*),
                        2
                    ) as satisfaction_rate
                FROM airline_tweets
                GROUP BY airline
                ORDER BY tweet_count DESC
            """
            df = pd.read_sql(query, engine)
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
            query = """
                SELECT 
                    negativereason,
                    COUNT(*) as count
                FROM airline_tweets
                WHERE negativereason IS NOT NULL AND negativereason != ''
                GROUP BY negativereason
                ORDER BY count DESC
                LIMIT 10
            """
            df = pd.read_sql(query, engine)
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
            query = f"""
                SELECT 
                    id, airline, airline_sentiment, negativereason,
                    text, clean_text, tweet_created, created_at
                FROM airline_tweets
                ORDER BY created_at DESC
                LIMIT {limit}
            """
            df = pd.read_sql(query, engine)
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
            query = """
                SELECT 
                    DATE_TRUNC('hour', created_at) as hour,
                    COUNT(*) as tweet_count,
                    SUM(CASE WHEN airline_sentiment = 'positive' THEN 1 ELSE 0 END) as positive,
                    SUM(CASE WHEN airline_sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral,
                    SUM(CASE WHEN airline_sentiment = 'negative' THEN 1 ELSE 0 END) as negative
                FROM airline_tweets
                GROUP BY hour
                ORDER BY hour DESC
                LIMIT 100
            """
            df = pd.read_sql(query, engine)
            return df
        except Exception as e:
            st.error(f"Erreur: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/airplane-take-off.png", width=80)
    st.title("🧭 Navigation")
    
    page = st.radio(
        "Sélectionner une page:",
        ["📊 Tableau de Bord KPI", "📈 Analytics Détaillées"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Database status
    st.subheader("🔧 État du Système")
    engine = get_db_engine()
    if engine:
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM airline_tweets")).fetchone()
                count = result[0]
            st.success(f"✅ Base de données connectée")
            st.info(f"📊 {count:,} tweets en base")
        except:
            st.error("❌ Erreur de requête")
    else:
        st.error("❌ Base de données déconnectée")
    
    st.divider()
    st.caption("AeroStream v1.0")
    st.caption(f"Dernière mise à jour: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# ==================== PAGE 1: TABLEAU DE BORD KPI ====================
if page == "📊 Tableau de Bord KPI":
    st.markdown('<div class="main-header">✈️ Tableau de Bord - Analyse des Sentiments des Compagnies Aériennes</div>', unsafe_allow_html=True)
    
    # Fetch KPI data
    kpi_data = fetch_kpi_data()
    
    if kpi_data is None or kpi_data['total_tweets'] == 0:
        st.warning("⚠️ Aucune donnée disponible. Veuillez exécuter le pipeline ETL d'abord.")
        st.info("💡 Le pipeline ETL collecte les données en micro-batch, nettoie et prédit les sentiments, puis les stocke dans PostgreSQL.")
    else:
        # KPI Cards
        st.markdown("### 📊 Indicateurs Clés de Performance (KPI)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="📝 Nombre Total de Tweets",
                value=f"{kpi_data['total_tweets']:,}",
                help="Volume total de tweets collectés et analysés"
            )
        
        with col2:
            st.metric(
                label="✈️ Nombre de Compagnies Aériennes",
                value=kpi_data['total_airlines'],
                help="Nombre de compagnies aériennes uniques dans les données"
            )
        
        with col3:
            st.metric(
                label="😞 Pourcentage de Tweets Négatifs",
                value=f"{kpi_data['negative_percentage']:.1f}%",
                delta=f"-{kpi_data['negative_tweets']} tweets",
                delta_color="inverse",
                help="Proportion de tweets avec un sentiment négatif"
            )
        
        st.divider()
        
        # Sentiment Distribution
        st.markdown("### 🎭 Répartition des Sentiments")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            sentiment_df = fetch_sentiment_distribution()
            if not sentiment_df.empty:
                # Pie chart
                colors = {
                    'positive': '#28a745',
                    'neutral': '#ffc107',
                    'negative': '#dc3545'
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
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.markdown("#### 📈 Statistiques Détaillées")
            
            # Positive
            positive_pct = (kpi_data['positive_tweets'] / kpi_data['total_tweets'] * 100) if kpi_data['total_tweets'] > 0 else 0
            st.metric(
                label="😊 Tweets Positifs",
                value=f"{kpi_data['positive_tweets']:,}",
                delta=f"{positive_pct:.1f}%"
            )
            
            # Neutral
            neutral_pct = (kpi_data['neutral_tweets'] / kpi_data['total_tweets'] * 100) if kpi_data['total_tweets'] > 0 else 0
            st.metric(
                label="😐 Tweets Neutres",
                value=f"{kpi_data['neutral_tweets']:,}",
                delta=f"{neutral_pct:.1f}%"
            )
            
            # Negative
            st.metric(
                label="😞 Tweets Négatifs",
                value=f"{kpi_data['negative_tweets']:,}",
                delta=f"{kpi_data['negative_percentage']:.1f}%",
                delta_color="inverse"
            )
        
        st.divider()
        
        # Volume by Airline
        st.markdown("### ✈️ Volume de Tweets par Compagnie")
        
        airline_df = fetch_tweets_by_airline()
        if not airline_df.empty:
            fig_volume = px.bar(
                airline_df,
                x='airline',
                y='tweet_count',
                title="Nombre de Tweets par Compagnie Aérienne",
                labels={'airline': 'Compagnie', 'tweet_count': 'Nombre de Tweets'},
                color='tweet_count',
                color_continuous_scale='Blues'
            )
            fig_volume.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_volume, use_container_width=True)
        
        st.divider()
        
        # Recent Tweets
        st.markdown("### 📋 Tweets Récents")
        
        recent_tweets = fetch_recent_tweets(20)
        if not recent_tweets.empty:
            display_df = recent_tweets[['created_at', 'airline', 'airline_sentiment', 'text']].copy()
            display_df.columns = ['Date', 'Compagnie', 'Sentiment', 'Tweet']
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Date": st.column_config.DatetimeColumn("Date", format="DD/MM/YYYY HH:mm"),
                    "Compagnie": st.column_config.TextColumn("Compagnie", width="small"),
                    "Sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                    "Tweet": st.column_config.TextColumn("Tweet", width="large")
                }
            )

# ==================== PAGE 2: ANALYTICS DÉTAILLÉES ====================
elif page == "📈 Analytics Détaillées":
    st.markdown('<div class="main-header">📈 Analyse Approfondie des Sentiments</div>', unsafe_allow_html=True)
    
    # Fetch data
    airline_df = fetch_tweets_by_airline()
    negative_reasons = fetch_negative_reasons()
    time_series = fetch_time_series_data()
    
    if airline_df.empty:
        st.warning("⚠️ Aucune donnée disponible.")
    else:
        # 1. Sentiment distribution by airline
        st.markdown("### 🎭 Répartition des Sentiments par Compagnie")
        
        # Prepare data for stacked bar chart
        airline_sentiment_df = airline_df[['airline', 'positive', 'neutral', 'negative']].copy()
        
        fig_sentiment_airline = go.Figure()
        
        fig_sentiment_airline.add_trace(go.Bar(
            name='Positif',
            x=airline_sentiment_df['airline'],
            y=airline_sentiment_df['positive'],
            marker_color='#28a745'
        ))
        
        fig_sentiment_airline.add_trace(go.Bar(
            name='Neutre',
            x=airline_sentiment_df['airline'],
            y=airline_sentiment_df['neutral'],
            marker_color='#ffc107'
        ))
        
        fig_sentiment_airline.add_trace(go.Bar(
            name='Négatif',
            x=airline_sentiment_df['airline'],
            y=airline_sentiment_df['negative'],
            marker_color='#dc3545'
        ))
        
        fig_sentiment_airline.update_layout(
            barmode='stack',
            title="Distribution des Sentiments par Compagnie Aérienne",
            xaxis_title="Compagnie",
            yaxis_title="Nombre de Tweets",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_sentiment_airline, use_container_width=True)
        
        st.divider()
        
        # 2. Satisfaction rate by airline
        st.markdown("### 📊 Taux de Satisfaction par Compagnie")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            fig_satisfaction = px.bar(
                airline_df.sort_values('satisfaction_rate', ascending=False),
                x='airline',
                y='satisfaction_rate',
                title="Taux de Satisfaction (% de Tweets Positifs)",
                labels={'airline': 'Compagnie', 'satisfaction_rate': 'Taux de Satisfaction (%)'},
                color='satisfaction_rate',
                color_continuous_scale='RdYlGn',
                range_color=[0, 100]
            )
            fig_satisfaction.update_layout(height=400)
            st.plotly_chart(fig_satisfaction, use_container_width=True)
        
        with col2:
            st.markdown("#### 🏆 Top Compagnies")
            top_airlines = airline_df.sort_values('satisfaction_rate', ascending=False).head(5)
            
            for idx, row in top_airlines.iterrows():
                st.metric(
                    label=row['airline'],
                    value=f"{row['satisfaction_rate']:.1f}%",
                    delta=f"{row['tweet_count']} tweets"
                )
        
        st.divider()
        
        # 3. Negative reasons
        st.markdown("### 😞 Principales Causes de Tweets Négatifs")
        
        if not negative_reasons.empty:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                fig_reasons = px.bar(
                    negative_reasons,
                    x='count',
                    y='negativereason',
                    orientation='h',
                    title="Top 10 des Raisons de Mécontentement",
                    labels={'count': 'Nombre de Mentions', 'negativereason': 'Raison'},
                    color='count',
                    color_continuous_scale='Reds'
                )
                fig_reasons.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_reasons, use_container_width=True)
            
            with col2:
                st.markdown("#### 📊 Classement")
                total_negative = negative_reasons['count'].sum()
                
                for idx, row in negative_reasons.head(10).iterrows():
                    percentage = (row['count'] / total_negative * 100) if total_negative > 0 else 0
                    st.metric(
                        label=row['negativereason'][:30],
                        value=row['count'],
                        delta=f"{percentage:.1f}%"
                    )
        else:
            st.info("Aucune raison négative enregistrée.")
        
        st.divider()
        
        # 4. Time series analysis
        if not time_series.empty:
            st.markdown("### 📅 Évolution Temporelle des Tweets")
            
            fig_time = go.Figure()
            
            fig_time.add_trace(go.Scatter(
                x=time_series['hour'],
                y=time_series['positive'],
                name='Positif',
                mode='lines+markers',
                line=dict(color='#28a745', width=2),
                fill='tonexty'
            ))
            
            fig_time.add_trace(go.Scatter(
                x=time_series['hour'],
                y=time_series['neutral'],
                name='Neutre',
                mode='lines+markers',
                line=dict(color='#ffc107', width=2),
                fill='tonexty'
            ))
            
            fig_time.add_trace(go.Scatter(
                x=time_series['hour'],
                y=time_series['negative'],
                name='Négatif',
                mode='lines+markers',
                line=dict(color='#dc3545', width=2),
                fill='tonexty'
            ))
            
            fig_time.update_layout(
                title="Évolution des Sentiments au Fil du Temps",
                xaxis_title="Date et Heure",
                yaxis_title="Nombre de Tweets",
                height=400,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_time, use_container_width=True)
        
        st.divider()
        
        # 5. Detailed airline comparison table
        st.markdown("### 📋 Comparaison Détaillée des Compagnies")
        
        comparison_df = airline_df.copy()
        comparison_df = comparison_df.sort_values('tweet_count', ascending=False)
        comparison_df.columns = ['Compagnie', 'Total Tweets', 'Positifs', 'Neutres', 'Négatifs', 'Taux de Satisfaction (%)']
        
        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Compagnie": st.column_config.TextColumn("Compagnie", width="medium"),
                "Total Tweets": st.column_config.NumberColumn("Total Tweets", format="%d"),
                "Positifs": st.column_config.NumberColumn("Positifs", format="%d"),
                "Neutres": st.column_config.NumberColumn("Neutres", format="%d"),
                "Négatifs": st.column_config.NumberColumn("Négatifs", format="%d"),
                "Taux de Satisfaction (%)": st.column_config.NumberColumn(
                    "Taux de Satisfaction",
                    format="%.1f%%",
                    help="Pourcentage de tweets positifs"
                )
            }
        )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        <p>🚀 AeroStream - Plateforme d'Analyse des Sentiments des Compagnies Aériennes</p>
        <p>Pipeline ETL: Collecte en micro-batch → Nettoyage → Prédiction → Stockage PostgreSQL → Visualisation</p>
    </div>
    """,
    unsafe_allow_html=True
)
