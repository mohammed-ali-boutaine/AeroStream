# AeroStream - Documentation Technique Complète

## 📋 Table des Matières
1. [Vue d'ensemble](#vue-densemble)
2. [Architecture du Système](#architecture-du-système)
3. [Technologies Utilisées](#technologies-utilisées)
4. [Concepts Clés](#concepts-clés)
5. [Pipeline de Données](#pipeline-de-données)
6. [Composants Détaillés](#composants-détaillés)
7. [Workflow Machine Learning](#workflow-machine-learning)
8. [Déploiement](#déploiement)
9. [Guide d'Installation](#guide-dinstallation)

---

## 🎯 Vue d'ensemble

**AeroStream** est une plateforme d'analyse de sentiments en temps réel pour les tweets concernant les compagnies aériennes américaines. Le système collecte, nettoie, analyse et visualise les sentiments des clients en utilisant des techniques avancées de Machine Learning et NLP.

### Objectifs du Projet
- Analyser les sentiments (positif, neutre, négatif) des tweets en temps réel
- Identifier les principales causes d'insatisfaction client
- Calculer les taux de satisfaction par compagnie aérienne
- Fournir un tableau de bord interactif pour la visualisation des KPIs
- Automatiser l'ensemble du pipeline via orchestration

### Dataset
- **Source**: Hugging Face - `7Xan7der7/us_airline_sentiment`
- **Volume**: ~14,640 tweets
- **Classes**: Negative, Neutral, Positive (données déséquilibrées)
- **Features**: 13 colonnes incluant texte, compagnie, sentiment, raisons négatives

---

## 🏗️ Architecture du Système

### Architecture Globale

```
┌─────────────────────────────────────────────────────────────┐
│                    AEROSTREAM ARCHITECTURE                   │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Jupyter    │─────▶│   ChromaDB   │─────▶│    Models    │
│  Notebooks   │      │  (Vector DB) │      │   Training   │
│ (EDA/Prep)   │      │              │      │              │
└──────────────┘      └──────────────┘      └──────────────┘
                                                    │
                                                    ▼
                      ┌─────────────────────────────────────┐
                      │         FastAPI Backend             │
                      │  ┌─────────────────────────────┐   │
                      │  │   ML Model (Best Model)     │   │
                      │  │   Sentence Transformers     │   │
                      │  │   Prediction Service        │   │
                      │  └─────────────────────────────┘   │
                      └─────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌──────────────┐ ┌──────────┐  ┌──────────────┐
            │   Airflow    │ │PostgreSQL│  │  Streamlit   │
            │     DAG      │ │ Backend  │  │  Dashboard   │
            │  (ETL Loop)  │ │   DB     │  │   (Viz)      │
            └──────────────┘ └──────────┘  └──────────────┘
                    │             ▲               ▲
                    └─────────────┴───────────────┘
                   Micro-batch Pipeline Flow
```

### Flux de Données

1. **Phase d'Entraînement** (Offline)
   ```
   Hugging Face → EDA → Preprocessing → Embeddings → ChromaDB → Model Training → Best Model
   ```

2. **Phase de Production** (Online - Streaming)
   ```
   Airflow → API (Fake Tweets) → Preprocessing → Prediction → PostgreSQL → Streamlit
   ```

---

## 💻 Technologies Utilisées

### 1. **Data Science & Machine Learning**

| Technologie | Version | Utilisation |
|------------|---------|-------------|
| **Python** | 3.13+ | Langage principal |
| **Sentence Transformers** | Latest | Génération d'embeddings (paraphrase-multilingual-MiniLM-L12-v2) |
| **Scikit-learn** | Latest | Modèles ML (Logistic Regression, Random Forest) |
| **XGBoost** | Latest | Modèle de boosting |
| **PyTorch** | Latest | Backend pour transformers & MLP |
| **Pandas** | Latest | Manipulation de données |
| **NumPy** | Latest | Calculs numériques |

### 2. **Natural Language Processing**

- **Sentence Transformers**: Embeddings de 384 dimensions
- **Regex (re)**: Nettoyage de texte
- **Emoji**: Conversion émojis → texte
- **NLTK**: Stopwords et analyse linguistique

### 3. **Bases de Données**

| Database | Type | Usage |
|----------|------|-------|
| **ChromaDB** | Vector Database | Stockage des embeddings (train/test collections) |
| **PostgreSQL** | Relational DB | Stockage des tweets prédits et métadonnées |

### 4. **Backend & API**

- **FastAPI**: Framework REST API moderne avec support async
- **Uvicorn**: Serveur ASGI haute performance
- **Pydantic**: Validation des schémas de données
- **Psycopg2**: Driver PostgreSQL

### 5. **Orchestration & Workflow**

- **Apache Airflow**: Orchestration du pipeline ETL
  - LocalExecutor pour exécution locale
  - DAGs Python pour définition du workflow
  - XCom pour communication inter-tâches

### 6. **Visualisation**

- **Streamlit**: Dashboard interactif
- **Plotly**: Graphiques interactifs (pie charts, bar charts, time series)
- **SQLAlchemy**: ORM pour requêtes SQL

### 7. **Containerisation**

- **Docker**: Containerisation des services
- **Docker Compose**: Orchestration multi-conteneurs
- **Networks**: Isolation réseau des services

---

## 📚 Concepts Clés

### 1. **Embeddings Sémantiques**

Les **embeddings** transforment du texte en vecteurs numériques qui capturent le sens sémantique.

```python
# Modèle utilisé: paraphrase-multilingual-MiniLM-L12-v2
# Caractéristiques:
- Dimensions: 384
- Multilingue (support de 50+ langues)
- Optimisé pour la similarité sémantique
- Préentraîné sur des paires de paraphrases
```

**Avantages**:
- Capture le contexte et les nuances linguistiques
- Robuste aux variations orthographiques
- Gère automatiquement la casse (pas besoin de lowercase)

### 2. **Vector Database (ChromaDB)**

ChromaDB est une base de données vectorielle optimisée pour la recherche de similarité.

**Collections créées**:
- `airline_sentiment_train`: 80% des données (~11,712 documents)
- `airline_sentiment_test`: 20% des données (~2,928 documents)

**Métadonnées stockées**:
```json
{
  "label": "negative",
  "airline": "United"
}
```

### 3. **Class Imbalance Handling**

Le dataset est déséquilibré (majorité de tweets négatifs). Solutions appliquées:

```python
# Calcul des poids de classe
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y),
    y=y
)

# Application dans les modèles
- Logistic Regression: class_weight='balanced'
- Random Forest: class_weight='balanced'
- XGBoost: sample_weight=class_weights
- MLP: sample_weight=class_weights
```

### 4. **Micro-batch Processing**

L'architecture utilise un traitement par micro-lots pour simuler le streaming temps réel.

```python
# Configuration Airflow
batch_size = 20  # 20 tweets par exécution
schedule_interval = '@hourly'  # Exécution toutes les heures
```

**Avantages**:
- Réduit la latence
- Optimise l'utilisation des ressources
- Facilite le monitoring et le debugging

**Sélection automatique**: Basée sur le F1-score (weighted) sur test set

### 6. **ETL Pipeline**

**Extract → Transform → Load**

```python
# Task 1: Extract
fetch_data_from_api() → API call → batch de tweets

# Task 2: Transform  
process_text_data() → Nettoyage + preprocessing

# Task 3: Load
store_in_database() → Bulk insert PostgreSQL
```

---

## 🔄 Pipeline de Données

### Phase 1: Préparation des Données (Offline)

#### 1.1 EDA (Exploratory Data Analysis)
**Notebook**: `1-EDA.ipynb`

```python
# Analyses effectuées:
- Distribution des classes (imbalanced)
- Statistiques descriptives (13 colonnes)
- Détection de doublons (1,041 trouvés)
- Valeurs manquantes
- Analyse des raisons négatives
- Wordclouds par sentiment
```

#### 1.2 Preprocessing
**Notebook**: `2-Preprocessing.ipynb`

**Étapes de nettoyage**:
```python
def preprocess_text(text):
    # 1. Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # 3. Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # 4. Keep hashtag content, remove #
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # 5. Convert emojis to text
    text = emoji.demojize(text)
    text = text.replace(":", " ").replace("_", " ")
    
    # 6. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
```

**Génération d'embeddings**:
```python
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
X_embeddings = model.encode(X, show_progress_bar=True, batch_size=32)
# Output: (n_samples, 384) numpy array
```

#### 1.3 ChromaDB Storage
**Notebook**: `3-db-setup.ipynb`

```python
# Train/Test Split
train_idx, test_idx = train_test_split(
    range(len(df)),
    test_size=0.2,
    random_state=42,
    stratify=df['airline_sentiment']
)

# Insertion par batch (5000 documents/batch)
train_collection.add(
    embeddings=batch_embeddings.tolist(),
    documents=batch_df['clean_text'].tolist(),
    metadatas=[...],
    ids=[f"train_{i}" for i in range(...)]
)
```

#### 1.4 Model Training
**Notebook**: `4-Modeling copy.ipynb`

**Modèles entraînés**:

```python
# 1. Logistic Regression
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced',
    C=1.0
)

# 2. Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',
    max_depth=20,
    n_jobs=-1
)

# 3. XGBoost
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='mlogloss',
    n_jobs=-1
)

# 4. MLP
mlp_model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=128,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=200,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10
)
```

**Évaluation**:
- Classification reports
- Confusion matrices
- F1-scores (macro, weighted)
- ROC curves & AUC
- Learning curves
- Overfitting analysis (train vs test gap)

### Phase 2: Production Pipeline (Online)

#### 2.1 Airflow DAG
**File**: `airflow/dags/ETL.py`

```python
# DAG Configuration
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
}

# Schedule: Hourly execution
schedule_interval='@hourly'

# Tasks Chain
fetch_data >> process_data >> store_data
```

**Task 1: Fetch Data**
```python
def fetch_data_from_api(**context):
    url = f"{API_BASE_URL}/fake-tweets?batch_size={batch_size}"
    response = requests.get(url, timeout=30)
    tweets = response.json()
    context['task_instance'].xcom_push(key='raw_tweets', value=tweets)
```

**Task 2: Process Data**
```python
def process_text_data(**context):
    tweets = context['task_instance'].xcom_pull(key='raw_tweets', ...)
    processed_data = []
    for tweet in tweets:
        clean_tweet_text = clean_text(tweet.get('text', ''))
        processed_tweet = {
            'airline': tweet.get('airline'),
            'airline_sentiment': tweet.get('airline_sentiment'),
            'negativereason': tweet.get('negativereason'),
            'tweet_created': tweet.get('tweet_created'),
            'text': tweet.get('text', ''),
            'clean_text': clean_tweet_text,
            'processed_at': datetime.now().isoformat()
        }
        processed_data.append(processed_tweet)
    context['task_instance'].xcom_push(key='processed_tweets', ...)
```

**Task 3: Store in Database**
```python
def store_in_database(**context):
    processed_tweets = context['task_instance'].xcom_pull(key='processed_tweets', ...)
    conn = psycopg2.connect(**DB_CONFIG)
    insert_query = """
        INSERT INTO airline_tweets 
        (airline_sentiment, negativereason, airline, text, tweet_created, clean_text)
        VALUES %s
    """
    execute_values(cursor, insert_query, values)
    conn.commit()
```

---

## 🔧 Composants Détaillés

### 1. Backend FastAPI

**Structure**:
```
backend/
├── main.py                 # API endpoints
├── services/
│   ├── predict.py         # ML prediction service
│   └── faker.py           # Fake tweet generator
├── schemas/
│   └── prediction.py      # Pydantic models
├── models/
│   ├── best_model.pkl     # Trained model
│   └── label_encoder.pkl  # Label encoder
└── database/
    └── database.py        # DB connection
```

**Endpoints**:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check + DB status |
| `/predict` | POST | Single text prediction |
| `/batch-predict` | POST | Multiple texts prediction |
| `/fake-tweets` | GET | Generate fake tweets |

**Prediction Service**:
```python
class AirlineSentimentService:
    def __init__(self):
        self.model = pickle.load(open(model_path, 'rb'))
        self.label_encoder = pickle.load(open(encoder_path, 'rb'))
        self.transformer = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    def predict(self, text):
        clean_text = self.preprocess_text(text)
        embedding = self.transformer.encode([clean_text])[0].reshape(1, -1)
        prediction = self.model.predict(embedding)[0]
        probabilities = self.model.predict_proba(embedding)[0]
        
        return {
            "predicted_sentiment": sentiment,
            "confidence": float(np.max(probabilities)),
            "probabilities": {...}
        }
```

### 2. PostgreSQL Database

**Schema**:
```sql
CREATE TABLE airline_tweets (
    id SERIAL PRIMARY KEY,
    airline_sentiment VARCHAR(20) NOT NULL,
    negativereason VARCHAR(100),
    airline VARCHAR(50) NOT NULL,
    text TEXT NOT NULL,
    tweet_created TIMESTAMP,
    clean_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT check_sentiment CHECK (airline_sentiment IN ('positive', 'negative', 'neutral'))
);

-- Indexes pour performance
CREATE INDEX idx_airline ON airline_tweets(airline);
CREATE INDEX idx_sentiment ON airline_tweets(airline_sentiment);
CREATE INDEX idx_tweet_created ON airline_tweets(tweet_created);
CREATE INDEX idx_created_at ON airline_tweets(created_at);
```

### 3. Streamlit Dashboard

**Structure**:
```
streamlit/
├── app.py              # Main dashboard
└── sql/
    └── queries.py      # SQL queries
```

**Pages**:
1. **Tableau de Bord KPI**: KPIs principaux + distributions
2. **Analytics Détaillées**: Analyses approfondies

**KPIs Affichés**:
- 📝 Nombre total de tweets
- ✈️ Nombre de compagnies aériennes
- 😞 Pourcentage de tweets négatifs
- 📊 Distribution des sentiments (pie chart)
- 📈 Volume par compagnie (bar chart)
- 😞 Top 10 raisons négatives
- 📅 Évolution temporelle (time series)
- 🏆 Taux de satisfaction par compagnie

**Caching Strategy**:
```python
@st.cache_data(ttl=30)  # Cache for 30 seconds
def fetch_kpi_data():
    # Fetch data from PostgreSQL
    ...

# Manual refresh
if st.button("🔄 Rafraîchir"):
    st.cache_data.clear()
    st.rerun()
```

### 4. Airflow

**Configuration**:
```yaml
# docker-compose.yml
airflow:
  environment:
    - AIRFLOW__CORE__EXECUTOR=LocalExecutor
    - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres_airflow:5432/airflow
  depends_on:
    - postgres_airflow
    - backend
```

**DAG Features**:
- Schedule: `@hourly`
- Retry: 2 attempts avec 3 minutes de délai
- XCom: Communication entre tâches
- Error handling: Try/except avec logging

---

## 🔬 Workflow Machine Learning

### 1. Feature Engineering

**Text → Embeddings Pipeline**:
```
Raw Text → Preprocessing → Sentence Transformer → 384D Vector
```

**Avantages des embeddings**:
- Capture du contexte sémantique
- Réduction de dimensionnalité (vocabulaire → 384 dimensions)
- Transfert learning (modèle pré-entraîné)
- Robustesse au bruit

### 2. Model Training

**Stratégie d'entraînement**:
```python
# 1. Data Split (stratifié)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Class Balancing
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

# 3. Training avec cross-validation
for model in [lr, rf, xgb, mlp]:
    model.fit(X_train, y_train, sample_weight=weights)
    
# 4. Evaluation
f1_scores = {model: f1_score(y_test, y_pred, average='weighted') for model in models}

# 5. Selection du meilleur modèle
best_model = max(models, key=lambda x: f1_scores[x])
```

### 3. Evaluation Metrics

**Metrics utilisées**:
- **F1-Score (Weighted)**: Métrique principale (gère le class imbalance)
- **Accuracy**: Précision globale
- **Confusion Matrix**: Distribution des prédictions
- **ROC-AUC**: Performance par classe
- **Learning Curves**: Détection d'overfitting

**Overfitting Analysis**:
```python
overfitting_gap = f1_train - f1_test
overfitting_pct = (overfitting_gap / f1_train) * 100
```

### 4. Model Deployment

**Sauvegarde**:
```python
# Best model
with open("../models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Label encoder (si nécessaire)
with open("../models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
```

**Chargement en production**:
```python
# backend/services/predict.py
self.model = pickle.load(open(model_path, 'rb'))
self.label_encoder = pickle.load(open(encoder_path, 'rb'))
self.transformer = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
```

---

## 🐳 Déploiement

### Docker Compose Architecture

**Services**:
```yaml
services:
  postgres_backend:   # Port 5432
  postgres_airflow:   # Port 5434
  backend:            # Port 8000
  airflow:            # Port 8080
  streamlit:          # Port 8501
```

**Network**:
```yaml
networks:
  aerostream:
    driver: bridge
```

**Volumes**:
```yaml
volumes:
  postgres_backend_data:  # Persistence PostgreSQL backend
  postgres_airflow_data:  # Persistence PostgreSQL Airflow
  airflow-logs:           # Logs Airflow
```

### Service Configuration

#### Backend (FastAPI)
```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

#### Airflow
```dockerfile
FROM apache/airflow:2.7.0-python3.11
USER root
# Install dependencies
USER airflow
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```

#### Streamlit
```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Health Checks

```yaml
postgres_backend:
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U ali -d backend_db"]
    interval: 10s
    timeout: 5s
    retries: 5

postgres_airflow:
  healthcheck:
    test: ["CMD", "pg_isready", "-U", "airflow"]
    interval: 5s
    retries: 5
```

---

## 📦 Guide d'Installation

### Prérequis
- Docker Desktop
- Docker Compose
- 8GB RAM minimum
- 10GB espace disque

### Installation

```bash
# 1. Clone le repository
git clone <repo_url>
cd AeroStream

# 2. Créer les répertoires nécessaires
mkdir -p data/raw data/processed data/embedding data/metadata
mkdir -p models chromadb

# 3. Lancer les services
docker-compose up --build

# 4. Vérifier les services
# - Airflow: http://localhost:8080 (admin/admin)
# - FastAPI: http://localhost:8000/docs
# - Streamlit: http://localhost:8501
```

### Préparation des Données (Première fois)

```bash
# 1. Exécuter les notebooks dans l'ordre:
1-EDA.ipynb           # Analyse exploratoire
2-Preprocessing.ipynb # Nettoyage + embeddings
3-db-setup.ipynb      # ChromaDB storage
4-Modeling.ipynb      # Training + evaluation

# 2. Vérifier que les fichiers sont créés:
- data/processed/data.csv
- data/embedding/embeddings.npy
- data/metadata/metadata.csv
- models/best_model.pkl
- chromadb/ (collections)

# 3. Initialiser la base PostgreSQL
# → Automatique au démarrage via init.sql

# 4. Activer le DAG Airflow
# → Aller sur http://localhost:8080
# → Activer "airline_sentiment_etl_pipeline"
```

### Configuration

**Variables d'environnement** (`.env`):
```env
# PostgreSQL Backend
POSTGRES_USER=ali
POSTGRES_PASSWORD=root
POSTGRES_DB=backend_db

# Airflow
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__WEBSERVER__SECRET_KEY=aliali

# API
API_PORT=8000
BACKEND_URL=http://backend:8000
```

---

## 📊 Monitoring & Observabilité

### Logs

**Airflow Logs**:
```bash
docker exec -it airflow bash
tail -f /opt/airflow/logs/dag_id=airline_sentiment_etl_pipeline/...
```

**Backend Logs**:
```bash
docker logs -f fastapi_backend
```

**Streamlit Logs**:
```bash
docker logs -f streamlit
```

### Métriques Clés

**Pipeline Health**:
- Taux de succès des DAG runs
- Temps d'exécution moyen par task
- Nombre de tweets traités/heure

**Model Performance**:
- Distribution des prédictions
- Confidence scores moyens
- Latence de prédiction

**Database**:
- Nombre total de tweets stockés
- Taux de croissance
- Requêtes lentes

---

## 🔒 Sécurité & Best Practices

### Sécurité

1. **Credentials**: Utiliser des secrets managers (pas de hardcode)
2. **Network Isolation**: Services isolés sur réseau Docker privé
3. **CORS**: Configuré sur FastAPI (à restreindre en production)
4. **SQL Injection**: Utilisation de parameterized queries
5. **Health Checks**: Surveillance continue des services

### Best Practices

1. **Code Quality**:
   - Type hints en Python
   - Docstrings pour fonctions
   - Error handling systématique

2. **Data Validation**:
   - Pydantic schemas pour API
   - Constraints SQL sur PostgreSQL
   - Validation des embeddings

3. **Performance**:
   - Batch processing (micro-batches)
   - Caching Streamlit (TTL 30s)
   - Indexes PostgreSQL
   - Connection pooling SQLAlchemy

4. **Scalabilité**:
   - Architecture microservices
   - Stateless API (FastAPI)
   - Horizontal scaling possible

---

## 🚀 Améliorations Futures

### Court Terme
- [ ] Ajouter auto-refresh automatique Streamlit (sans bouton)
- [ ] Changer schedule Airflow à `*/1 * * * *` (chaque minute)
- [ ] Ajouter monitoring Prometheus/Grafana
- [ ] Tests unitaires et d'intégration

### Moyen Terme
- [ ] Déploiement cloud (AWS/GCP/Azure)
- [ ] CI/CD avec GitHub Actions
- [ ] MLflow pour tracking des expériences
- [ ] A/B testing de modèles
- [ ] Real-time streaming avec Kafka

### Long Terme
- [ ] Fine-tuning du modèle transformer
- [ ] Multi-label classification (émotions)
- [ ] Détection d'anomalies
- [ ] Système de recommandations

---

## 📖 Références

### Documentation Technique
- [Sentence Transformers](https://www.sbert.net/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Airflow Documentation](https://airflow.apache.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Papers & Resources
- BERT: Pre-training of Deep Bidirectional Transformers
- Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
- XGBoost: A Scalable Tree Boosting System

---