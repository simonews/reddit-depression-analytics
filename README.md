# Reddit Depression Analytics

Scalable Big Data pipeline for semantic and behavioral analysis of Reddit posts aimed at identifying linguistic and temporal markers associated with depression risk.

This project was developed within the course “Big Data Models and Techniques” and implements an end-to-end architecture based on Apache Spark, advanced NLP techniques, and an interactive analytics dashboard.

---

## Overview

Reddit Depression Analytics is a distributed data processing system designed to analyze large-scale unstructured textual data (544,447 posts) from the eRisk dataset.

The system goes beyond binary classification (Depressed / Control) and focuses on:

- Early risk detection
- Semantic abstraction using word embeddings
- Behavioral pattern extraction
- Circadian rhythm analysis
- Information-theoretic cognitive metrics

The architecture follows a decoupled design inspired by the Batch Layer of the Lambda Architecture.

---

## Dataset

- Source: eRisk (Early Risk Prediction on the Internet)
- Volume: 544,447 Reddit posts
- Format: Hierarchical XML (per-user history)
- Class imbalance: ~92.5% control vs ~7.5% at-risk

Each user file contains:
- Timestamp (DATE)
- Title
- Text content
- Metadata

---

## System Architecture

### Backend (ETL, Processing & ML Layer)

Implemented in PySpark.

Pipeline components:

- Distributed XML ingestion (spark-xml)
- Data cleaning and preprocessing
- Random undersampling for class balancing
- Word2Vec embedding training
- Random Forest classification
- Shannon entropy computation per user
- Temporal feature extraction (hour, day of week)
- Results persisted in Parquet format

### Frontend (Serving & Visualization Layer)

Built with Streamlit.

Visualization stack:
- Plotly (interactive charts)
- PyVis (semantic knowledge graph)
- Scikit-learn (ROC curve, AUC, confusion matrix)

---

## Methodological Evolution

### Baseline Model (TF-IDF + Logistic Regression)

- F1-score ≈ 93%
- Strong overfitting on keyword frequency
- Limited generalization

### Advanced Model (Word2Vec + Random Forest)

- F1-score ≈ 82.88%
- AUC = 0.95
- High recall (21/22 at-risk users correctly identified)
- Better semantic abstraction and generalization

The second approach prioritizes recall over raw accuracy, consistent with medical screening objectives.

---

## Advanced Analytical Components

### 1. Semantic Knowledge Graph

Word2Vec embeddings are used to extract semantic similarity relationships between terms.
The model autonomously links clinical-related terms such as:

- anxiety → ptsd
- sadness → medication

These relationships are visualized as an interactive graph.

---

### 2. Circadian Rhythm Analysis

Distributed aggregation over timestamps shows:

- Control users exhibit physiological night inactivity (02:00–06:00)
- At-risk users maintain persistent nighttime activity

This supports literature on sleep disturbance as a depression marker.

---

### 3. Behavioral Analysis (Verbosity)

Average post length per user shows:

- At-risk users tend toward shorter messages
- Suggestive of psychomotor slowing or apathy

---

### 4. Cognitive Analysis (Shannon Entropy)

Entropy is computed fully in Spark using window functions and aggregation.

Results indicate:

- Lower interaction frequency
- High semantic density when posting
- Reduced lexical variability consistent with emotional inertia

---

## Performance

- AUC: 0.95
- High recall on minority class
- Distributed processing on large-scale XML data
- End-to-end Spark-based pipeline

---

## Technologies Used

- Apache Spark (PySpark)
- Word2Vec
- Random Forest (Spark MLlib)
- Streamlit
- Plotly
- PyVis
- Scikit-learn

---

## Disclaimer

This project is intended for academic and research purposes.  
It does not provide medical diagnosis and should not be used in clinical settings.
