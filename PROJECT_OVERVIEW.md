# AeroStream ✈️

## Plateforme d'Analyse de Sentiments des Compagnies Aériennes en Temps Réel

![Python](https://img.shields.io/badge/Python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![Airflow](https://img.shields.io/badge/Airflow-2.7.0-orange)

---

## 📋 Introduction

**AeroStream** est une plateforme complète d'analyse de sentiments dédiée aux compagnies aériennes américaines. Le projet exploite des techniques avancées de **Machine Learning** et de **Natural Language Processing (NLP)** pour analyser automatiquement les opinions des clients exprimées sur les réseaux sociaux, en particulier sur Twitter.

Dans un contexte où la satisfaction client est cruciale pour l'industrie aérienne, AeroStream permet aux compagnies de :
- **Comprendre en temps réel** le sentiment de leurs clients
- **Identifier rapidement** les sources d'insatisfaction
- **Prendre des décisions éclairées** basées sur des données concrètes
- **Améliorer continuellement** la qualité de service

Le système traite l'ensemble du cycle de vie des données, de la collecte à la visualisation, en passant par le nettoyage, l'analyse et le stockage, le tout orchestré de manière automatique via des pipelines ETL modernes.

### 🎯 Cas d'Usage

- **Monitoring de réputation** : Suivi en temps réel de l'image de marque
- **Gestion de crise** : Détection rapide de problèmes récurrents
- **Analyse comparative** : Benchmark entre compagnies aériennes
- **Business Intelligence** : Insights actionnables pour la direction

---

## ❓ Problématique

### Contexte

L'industrie aérienne est l'un des secteurs les plus scrutés sur les réseaux sociaux. Chaque jour, des milliers de passagers partagent leurs expériences, qu'elles soient positives ou négatives. Ces données non structurées représentent une mine d'informations précieuses, mais leur exploitation pose plusieurs défis majeurs.

### Défis Identifiés

#### 1. **Volume de Données Massif** 📈
Les compagnies aériennes reçoivent **des milliers de mentions quotidiennes** sur les réseaux sociaux. L'analyse manuelle de ces volumes est impossible et coûteuse en ressources humaines. Une automatisation intelligente est nécessaire pour traiter ce flux continu d'informations.

#### 2. **Données Non Structurées** 💬
Les tweets contiennent :
- Du **langage informel** (argot, abréviations)
- Des **émojis** et symboles (😊, 😡, ✈️)
- Des **fautes d'orthographe** et variations linguistiques
- Des **mentions** (@username) et **hashtags** (#BadService)
- Des **URLs** et liens externes
- Du **sarcasme** et de l'**ironie** difficiles à détecter

Ces caractéristiques rendent l'analyse textuelle complexe et nécessitent des techniques avancées de preprocessing et de NLP.

#### 3. **Déséquilibre des Classes** ⚖️
Dans le contexte de l'analyse de sentiments des compagnies aériennes :
- **~60-70% de tweets négatifs** : Les clients mécontents s'expriment davantage
- **~15-20% de tweets positifs** : Moins de retours spontanés positifs
- **~15-20% de tweets neutres** : Informations factuelles

Ce **déséquilibre prononcé** pose des problèmes pour l'entraînement de modèles ML classiques qui tendent à favoriser la classe majoritaire au détriment des autres.

#### 4. **Temps Réel et Réactivité** ⏱️
Les entreprises doivent pouvoir **réagir rapidement** aux situations critiques :
- Retards massifs et annulations
- Problèmes techniques (pannes, accidents)
- Controverses médiatiques
- Crises sanitaires ou sécuritaires

Un système en **temps réel** (ou quasi temps réel) est donc indispensable pour permettre une réponse appropriée.

#### 5. **Identification des Causes** 🔍
Au-delà du sentiment général (positif/négatif/neutre), il est **crucial d'identifier précisément** les raisons du mécontentement :
- ⏰ **Retards** et annulations de vols
- 🧳 **Perte** ou dommages de bagages
- 👥 **Service client** défaillant ou impoli
- 💺 **Confort** insuffisant (sièges, espace)
- 💰 **Tarifs** perçus comme excessifs
- 🍽️ **Qualité** des repas et services à bord

Cette granularité permet d'agir sur les **vrais problèmes** opérationnels.

#### 6. **Visualisation et Actionnabilité** 📊
Les données brutes, même analysées, ne sont utiles que si elles sont :
- **Accessibles** aux décideurs non-techniques
- **Visuelles** avec des graphiques clairs et intuitifs
- **Actionnables** avec des KPIs pertinents
- **Synthétiques** pour faciliter la prise de décision rapide

### Questions de Recherche

Le projet AeroStream répond aux questions suivantes :

1. **Comment automatiser l'analyse de sentiments à grande échelle ?**
   - → Utilisation de modèles de ML pré-entraînés et d'embeddings sémantiques

2. **Comment gérer le déséquilibre des classes dans un contexte de classification ?**
   - → Application de techniques de pondération et rééquilibrage

3. **Comment structurer un pipeline de données end-to-end robuste ?**
   - → Architecture microservices avec orchestration Airflow

4. **Comment stocker et requêter efficacement des données vectorielles ?**
   - → Utilisation de bases de données vectorielles spécialisées (ChromaDB)

5. **Comment fournir des insights exploitables en temps réel ?**
   - → Tableau de bord interactif avec rafraîchissement automatique

---

## 🎯 Objectifs Principaux

### Objectif Global

Développer une **plateforme complète de bout en bout** capable d'analyser automatiquement les sentiments exprimés dans les tweets concernant les compagnies aériennes, tout en fournissant des visualisations interactives et des insights actionnables pour améliorer la qualité de service.

### Objectifs Spécifiques

#### 1. **Collecte et Préparation des Données** 📊

**Objectif** : Constituer un dataset propre et exploitable pour l'entraînement de modèles ML.

**Tâches réalisées** :
- ✅ Chargement du dataset `7Xan7der7/us_airline_sentiment` depuis Hugging Face (~14,640 tweets)
- ✅ Analyse exploratoire (EDA) : distribution des classes, statistiques, doublons
- ✅ Nettoyage : suppression de 1,041 doublons, gestion valeurs manquantes
- ✅ Prétraitement texte : URLs, mentions, émojis, hashtags, ponctuation
- ✅ Normalisation et validation des données

**Livrables** :
- Dataset nettoyé et structuré
- Notebook d'EDA documenté
- Statistiques descriptives complètes

#### 2. **Feature Engineering avec Deep Learning** 🧠

**Objectif** : Transformer le texte brut en représentations numériques riches capturant le sens sémantique.

**Tâches réalisées** :
- ✅ Utilisation de **Sentence Transformers**
- ✅ Modèle : `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions)
- ✅ Génération de ~14,000 vecteurs d'embeddings
- ✅ Sauvegarde des métadonnées (labels, compagnies, raisons négatives)
- ✅ Stockage dans **ChromaDB** avec 2 collections (train/test 80/20)

**Livrables** :
- Fichiers d'embeddings (embeddings.npy)
- Métadonnées structurées (metadata.csv)
- Collections ChromaDB persistantes

#### 3. **Entraînement et Évaluation de Modèles ML** 🤖

**Objectif** : Développer des modèles de classification performants avec gestion du déséquilibre.

**Modèles entraînés** :
- ✅ **Logistic Regression** : Baseline linéaire rapide
- ✅ **Random Forest** : Ensemble learning avec 100 arbres
- ✅ **XGBoost** : Gradient boosting optimisé
- ✅ **MLP** : Réseau de neurones (256-128-64 neurones)

**Techniques appliquées** :
- ✅ Gestion du class imbalance (class_weight='balanced')
- ✅ Cross-validation et early stopping
- ✅ Métriques adaptées : F1-score (weighted), confusion matrix, ROC-AUC
- ✅ Analyse d'overfitting (train vs test gap)
- ✅ Sélection automatique du meilleur modèle

**Livrables** :
- Modèle optimal sauvegardé (best_model.pkl)
- Label encoder (label_encoder.pkl)
- Rapports d'évaluation complets
- Visualisations (learning curves, ROC curves)

#### 4. **Déploiement d'une API REST** 🚀

**Objectif** : Exposer le modèle via une API pour prédictions en temps réel.

**Endpoints implémentés** :
- ✅ `GET /` : Informations API
- ✅ `GET /health` : Health check + statut DB
- ✅ `POST /predict` : Prédiction unitaire
- ✅ `POST /batch-predict` : Prédictions multiples
- ✅ `GET /fake-tweets` : Génération de données test

**Technologies** :
- FastAPI (framework moderne asynchrone)
- Uvicorn (serveur ASGI haute performance)
- Pydantic (validation schémas)
- Documentation Swagger automatique

**Livrables** :
- API fonctionnelle (port 8000)
- Documentation interactive (/docs)
- Service de prédiction optimisé

#### 5. **Pipeline ETL Automatisé** 🔄

**Objectif** : Créer un pipeline de données automatisé pour traitement en continu.

**Architecture DAG Airflow** :
1. ✅ **Task Extract** : Récupération micro-batch API (20 tweets)
2. ✅ **Task Transform** : Nettoyage + preprocessing
3. ✅ **Task Load** : Stockage bulk PostgreSQL

**Caractéristiques** :
- ✅ Orchestration périodique (schedule : hourly)
- ✅ Gestion erreurs avec retry (2 tentatives)
- ✅ Communication inter-tâches (XCom)
- ✅ Logging centralisé

**Livrables** :
- DAG fonctionnel avec taux succès > 95%
- Interface web Airflow (port 8080)
- Pipeline résilient et scalable

#### 6. **Agrégation et Analyse des Données** 📈

**Objectif** : Calculer des métriques métier et KPIs pour aide à la décision.

**Requêtes SQL optimisées** :
- ✅ Volume de tweets par compagnie
- ✅ Distribution des sentiments (global + par compagnie)
- ✅ Taux de satisfaction (% positifs)
- ✅ Top 10 raisons de mécontentement
- ✅ Évolution temporelle (time series)
- ✅ Comparaisons inter-compagnies

**Optimisations** :
- Indexes sur colonnes clés
- Requêtes paramétrées (protection SQL injection)
- Agrégations efficaces

**Livrables** :
- Module queries.py avec 8+ requêtes
- Base PostgreSQL structurée
- Performances < 100ms

#### 7. **Dashboard Interactif** 📊

**Objectif** : Fournir une interface utilisateur intuitive pour exploration des insights.

**Pages développées** :
1. ✅ **Tableau de Bord KPI** : Métriques essentielles
2. ✅ **Analytics Détaillées** : Analyses approfondies

**KPIs affichés** :
- ✅ 📝 Nombre total de tweets
- ✅ ✈️ Nombre de compagnies aériennes  
- ✅ 😞 Pourcentage de tweets négatifs
- ✅ 📊 Distribution des sentiments (pie chart)
- ✅ 📈 Volume par compagnie (bar chart)
- ✅ 🏆 Taux de satisfaction
- ✅ 😡 Top raisons négatives
- ✅ 📅 Évolution temporelle (time series)

**Fonctionnalités** :
- Visualisations interactives Plotly
- Rafraîchissement manuel (bouton)
- Caching intelligent (TTL 30s)
- Interface responsive

**Livrables** :
- Application Streamlit (port 8501)
- 10+ visualisations interactives
- UX optimisée

#### 8. **Infrastructure et Déploiement** 🐳

**Objectif** : Containeriser pour faciliter déploiement et scalabilité.

**Services Docker** :
- ✅ **postgres_backend** : Base de données principale (port 5432)
- ✅ **postgres_airflow** : Base Airflow (port 5434)
- ✅ **backend** : API FastAPI (port 8000)
- ✅ **airflow** : Orchestrateur (port 8080)
- ✅ **streamlit** : Dashboard (port 8501)

**Configuration** :
- ✅ Réseau privé (aerostream)
- ✅ Volumes persistants (données + logs)
- ✅ Health checks sur DBs
- ✅ Restart policies
- ✅ Dépendances entre services

**Livrables** :
- docker-compose.yml complet
- Dockerfiles optimisés
- Déploiement en 1 commande

---

## 💻 Technologies Utilisées

### Architecture Technique Globale

```
┌─────────────────────────────────────────────────────────────┐
│                    STACK TECHNIQUE                           │
└─────────────────────────────────────────────────────────────┘

📊 DATA SCIENCE         🤖 MACHINE LEARNING       🗄️ DATABASES
├─ Python 3.13         ├─ Scikit-learn           ├─ PostgreSQL 15
├─ Pandas              ├─ XGBoost                ├─ ChromaDB
├─ NumPy               ├─ PyTorch                └─ SQLAlchemy
├─ Matplotlib          └─ Sentence Transformers
└─ Seaborn             

🌐 BACKEND & API       ⚙️ ORCHESTRATION          📈 VISUALIZATION
├─ FastAPI             ├─ Apache Airflow 2.7     ├─ Streamlit
├─ Uvicorn             ├─ LocalExecutor          ├─ Plotly
├─ Pydantic            └─ DAGs Python            └─ Plotly Express
└─ Psycopg2            

🐳 DEVOPS & INFRA      🧹 PREPROCESSING          📦 UTILITIES
├─ Docker              ├─ Regex (re)             ├─ Pickle
├─ Docker Compose      ├─ NLTK                   ├─ JSON
└─ Linux Alpine        ├─ Emoji                  └─ Requests
                       └─ String methods          
```

### Détail par Catégorie

#### 🐍 Langage & Core

| Technologie | Version | Rôle |
|------------|---------|------|
| **Python** | 3.13+ | Langage principal |
| **pip** | Latest | Gestionnaire de packages |
| **venv** | Built-in | Environnements virtuels |

**Justification** : Python est le langage de référence pour le Data Science et ML avec un écosystème riche et mature.

---

#### 🤖 Machine Learning & NLP

##### **Sentence Transformers**
- **Modèle** : `paraphrase-multilingual-MiniLM-L12-v2`
- **Dimensions** : 384
- **Usage** : Génération d'embeddings sémantiques
- **Avantages** :
  - Multilingue (50+ langues)
  - Pré-entraîné sur paraphrases
  - Rapide et léger
  - Capture le contexte sémantique
  - Pas besoin de lowercase (gestion automatique)

##### **Scikit-learn**
- **Modèles** : Logistic Regression, Random Forest, MLP
- **Utils** : train_test_split, class_weight, metrics
- **Usage** : Entraînement modèles classiques et évaluation

##### **XGBoost**
- **Type** : Gradient Boosting optimisé
- **Usage** : Modèle de classification performant
- **Avantages** : Gestion native des poids, haute performance

##### **PyTorch**
- **Usage** : Backend pour Sentence Transformers
- **Support GPU** : Optionnel (accélération possible)

##### **NLTK & Preprocessing**
- **Regex (re)** : Nettoyage texte (URLs, mentions)
- **Emoji** : Conversion émojis → texte descriptif
- **NLTK** : Stopwords, tokenization
- **String** : Normalisation, whitespace

---

#### 🗄️ Bases de Données

##### **ChromaDB** (Vector Database)
- **Type** : Base de données vectorielle
- **Usage** : Stockage embeddings + métadonnées
- **Collections** :
  - `airline_sentiment_train` : 80% données (~11,712 docs)
  - `airline_sentiment_test` : 20% données (~2,928 docs)
- **Features** :
  - Recherche par similarité (cosine)
  - Persistance disque
  - API Python native
  - Requêtes vectorielles rapides

##### **PostgreSQL**
- **Version** : 15-alpine
- **Usage** : 
  - Backend DB : Tweets prédits
  - Airflow DB : Métadonnées orchestration
- **Tables** :
  - `airline_tweets` : Tweets avec sentiments
- **Features** :
  - ACID compliance
  - Indexes optimisés (airline, sentiment, dates)
  - Constraints d'intégrité
  - Haute fiabilité

##### **SQLAlchemy**
- **Type** : ORM Python
- **Usage** : Connexion et requêtes SQL
- **Features** : Connection pooling, query builder

---

#### 🌐 Backend & API

##### **FastAPI**
- **Type** : Framework web moderne asynchrone
- **Usage** : Exposition du modèle ML via REST API
- **Features** :
  - Auto-documentation (Swagger/OpenAPI)
  - Validation automatique (Pydantic)
  - Support async/await
  - Haute performance (ASGI)
  - Type hints natifs

##### **Uvicorn**
- **Type** : Serveur ASGI
- **Usage** : Serveur pour FastAPI
- **Features** : Hot reload, performance élevée

##### **Pydantic**
- **Type** : Validation de données
- **Usage** : Schémas requêtes/réponses API
- **Features** : Type safety, sérialisation JSON

---

#### ⚙️ Orchestration

##### **Apache Airflow**
- **Version** : 2.7.0
- **Type** : Plateforme d'orchestration workflows
- **Usage** : Automatisation pipeline ETL
- **Components** :
  - **DAG** : Définition workflow (Extract → Transform → Load)
  - **Scheduler** : Exécution périodique (hourly)
  - **LocalExecutor** : Exécution locale parallèle
  - **WebUI** : Monitoring graphique (port 8080)
- **Features** :
  - Retry automatique
  - XCom (communication inter-tâches)
  - Logging centralisé
  - Alerting
  - Historique exécutions

---

#### 📈 Visualisation

##### **Streamlit**
- **Type** : Framework dashboards Python
- **Usage** : Interface utilisateur interactive
- **Features** :
  - Widgets natifs (buttons, sliders, selects)
  - Caching intelligent (@st.cache_data)
  - Layout responsive
  - Déploiement simple
  - Rafraîchissement manuel

##### **Plotly**
- **Type** : Librairie visualisation interactive
- **Usage** : Graphiques dans Streamlit
- **Types** :
  - Pie charts : Distribution sentiments
  - Bar charts : Comparaisons compagnies
  - Line charts : Time series
  - Stacked bars : Compositions
- **Features** :
  - Interactivité (zoom, hover)
  - Export images
  - Personnalisation CSS

---

#### 🐳 Infrastructure & DevOps

##### **Docker**
- **Usage** : Containerisation services
- **Images utilisées** :
  - `postgres:15-alpine` : PostgreSQL léger
  - `apache/airflow:2.7.0-python3.11` : Airflow
  - `python:3.13-slim` : Backend + Streamlit
- **Avantages** :
  - Isolation complète
  - Reproducibilité
  - Portabilité multi-OS

##### **Docker Compose**
- **Version** : V3
- **Usage** : Orchestration multi-conteneurs
- **Services** : 5 conteneurs interconnectés
- **Configuration** :
  ```yaml
  networks:
    aerostream:           # Réseau privé
      driver: bridge
  
  volumes:
    postgres_backend_data  # Persistence données
    postgres_airflow_data
    airflow-logs
  ```

---

### Pourquoi ces Technologies ?

#### ✅ **Sentence Transformers** plutôt que BERT classique
- Plus rapide à l'inférence
- Optimisé pour similarité sémantique
- Modèle léger (MiniLM)
- Multilingue out-of-the-box

#### ✅ **ChromaDB** plutôt que FAISS
- API Python plus simple
- Métadonnées natives
- Persistance automatique
- Pas besoin de serveur séparé

#### ✅ **FastAPI** plutôt que Flask
- Performance supérieure (async)
- Auto-documentation Swagger
- Validation automatique
- Type hints natifs

#### ✅ **Airflow** plutôt que Cron
- Interface graphique riche
- Gestion erreurs avancée
- Retry policies
- Monitoring intégré
- DAG versionnés (code as config)

#### ✅ **Streamlit** plutôt que Dash/Plotly
- Développement plus rapide
- Moins de boilerplate
- Caching natif
- Widgets prêts à l'emploi

#### ✅ **Docker Compose** plutôt que Kubernetes
- Plus simple pour dev/démo
- Moins de overhead
- Suffisant pour échelle moyenne
- Migration K8s possible si besoin

---

## 🏆 Résultats & Performances

### Métriques de Performance

| Métrique | Valeur Cible | Valeur Obtenue | Statut |
|----------|--------------|----------------|--------|
| F1-Score (Test) | > 0.70 | ✅ > 0.75 | ✅ Dépassé |
| Latence API | < 500ms | ✅ ~200-300ms | ✅ Dépassé |
| Uptime Services | > 95% | ✅ > 98% | ✅ Dépassé |
| Temps Chargement Dashboard | < 3s | ✅ ~1-2s | ✅ Dépassé |
| Taux Succès DAG | > 90% | ✅ > 95% | ✅ Dépassé |

### Livrables Principaux

1. ✅ **4 Notebooks Jupyter** : EDA, Preprocessing, ChromaDB, Modeling
2. ✅ **Modèle ML Production** : best_model.pkl + label_encoder.pkl
3. ✅ **API REST** : 5 endpoints documentés (FastAPI)
4. ✅ **Pipeline ETL** : DAG Airflow avec 3 tasks
5. ✅ **Dashboard** : 2 pages Streamlit avec 10+ visualisations
6. ✅ **Infrastructure** : 5 services Docker orchestrés
7. ✅ **Documentation** : README + DOCUMENTATION technique

---

## 🚀 Quick Start

### Prérequis
- Docker Desktop installé
- 8GB RAM minimum
- 10GB espace disque

### Installation

```bash
# 1. Cloner le repository
git clone <repo_url>
cd AeroStream

# 2. Lancer tous les services
docker-compose up --build

# 3. Vérifier les services
✓ Airflow:    http://localhost:8080 (admin/admin)
✓ FastAPI:    http://localhost:8000/docs
✓ Streamlit:  http://localhost:8501
```

### Premiers Pas

```bash
# 1. Activer le DAG Airflow
# → http://localhost:8080
# → Toggle ON "airline_sentiment_etl_pipeline"

# 2. Tester l'API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Amazing flight with Delta Airlines!"}'

# 3. Explorer le Dashboard
# → http://localhost:8501
# → Cliquer sur "🔄 Rafraîchir" pour mettre à jour
```

---

## 📁 Structure du Projet

```
AeroStream/
├── 📓 notebook/              # Jupyter notebooks
│   ├── 1-EDA.ipynb          # Analyse exploratoire
│   ├── 2-Preprocessing.ipynb # Nettoyage + embeddings
│   ├── 3-db-setup.ipynb     # ChromaDB setup
│   └── 4-Modeling.ipynb     # Training + evaluation
│
├── 🚀 backend/               # API FastAPI
│   ├── main.py              # Endpoints
│   ├── services/
│   │   ├── predict.py       # ML service
│   │   └── faker.py         # Fake data generator
│   └── schemas/             # Pydantic models
│
├── 📊 streamlit/             # Dashboard
│   ├── app.py               # Interface Streamlit
│   └── sql/queries.py       # SQL queries
│
├── ⚙️ airflow/               # Orchestration
│   └── dags/ETL.py          # DAG principal
│
├── 📦 data/                  # Données
│   ├── raw/                 # Données brutes
│   ├── processed/           # Données nettoyées
│   ├── embedding/           # Embeddings (.npy)
│   └── metadata/            # Métadonnées (.csv)
│
├── 🤖 models/                # Modèles ML
│   ├── best_model.pkl
│   └── label_encoder.pkl
│
├── 🗄️ chromadb/              # Base vectorielle
│   └── collections/
│
├── 🐳 docker-compose.yml     # Orchestration services
├── 📖 README.md              # Ce fichier
└── 📚 DOCUMENTATION.md       # Doc technique complète
```

---

## 📖 Documentation

- **PROJECT_OVERVIEW.md** : Introduction, problématique, objectifs, technologies (ce fichier)
- **DOCUMENTATION.md** : Documentation technique complète, architecture, concepts avancés

---

## 🔗 Ressources

### Dataset
- [Hugging Face - US Airline Sentiment](https://huggingface.co/datasets/7Xan7der7/us_airline_sentiment)

### Documentation Officielle
- [Sentence Transformers](https://www.sbert.net/)
- [ChromaDB](https://docs.trychroma.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Apache Airflow](https://airflow.apache.org/docs/)
- [Streamlit](https://docs.streamlit.io/)
- [Scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)

### Papers de Référence
- BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)
- Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (Reimers et al., 2019)
- XGBoost: A Scalable Tree Boosting System (Chen et al., 2016)

---

## 📄 Licence

Ce projet est sous licence MIT.

---

## 👥 Contribution

Les contributions sont les bienvenues ! Pour contribuer :
1. Fork le repository
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit les changes (`git commit -m 'Add AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

---

## 📧 Contact

Pour toute question sur le projet AeroStream, veuillez ouvrir une issue sur GitHub.

---

**🎉 Merci d'utiliser AeroStream !** ✈️

*Analyse de sentiments intelligente pour une aviation plus satisfaisante*
