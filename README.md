# 🚀 Agile PBI Tutor: MLOps & NLP for Story Point Estimation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3.0-orange)
![SBERT](https://img.shields.io/badge/SBERT-Transformers-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

An end-to-end Machine Learning pipeline and web application designed to automatically classify, evaluate, and predict Agile Story Points for Product Backlog Items (PBIs) using Deep Natural Language Processing and Gradient Boosting.

## 🧠 Project Overview

Estimating effort in Agile methodologies is often plagued by human bias, the Planning Fallacy, and inconsistent baselines. This project introduces a **Data-Centric AI architecture** to objectively predict effort (in Fibonacci Story Points) by understanding the deep semantics of software requirements.

This system was trained on a refined subset of the `giseldo/neodataset` (mined from GitLab) and employs a robust MLOps pipeline to filter out "human label noise" and extreme outliers.

## 🏗️ Architecture & Pipeline

The system operates in a 5-Phase pipeline:

1. **Data Ingestion & Cleaning (`pandas`, `pyarrow`):** Processing of historical open-source PBIs.
2. **Semantic Router & LLM Judge (`OpenAI`, `Pydantic`):** Uses an AI agent with strict JSON schemas to evaluate the structural quality of a User Story (INVEST/CIDEM frameworks) and extract categorical taxonomy.
3. **Mathematical Translation (`SE-BERT`, `PyTorch`):** Converts raw technical text into dense 768-dimensional mathematical vectors, capturing domain-specific software engineering context.
4. **Predictive Engine (`LightGBM`):** A gradient boosting regressor trained to map the semantic embeddings and quality scores to Fibonacci effort estimations.
5. **Interactive UI (`Streamlit`):** A front-end for Product Owners to paste requirements, get real-time feedback, and export planning sessions to CSV.

## 🔬 Advanced MLOps Techniques Implemented

To ensure enterprise-grade reliability, especially for critical software (SaMD), the model incorporates:
* **Label Filtering:** Complete exclusion of anomalous macro-epics (>40 SP) and severity-marker outliers (e.g., 100, 200 SP) that destroy algorithmic convergence.
* **Target Smoothing:** Algorithmic rounding of linear human estimations ("Ideal Hours" like 4.0 or 6.0) into discrete, conservative Fibonacci containers to prevent optimism bias.
* **MdAE Optimization:** Evaluated using the **Median Absolute Error (MdAE)** to provide mathematical invulnerability against the left-skewed, asymmetric nature of Agile estimations.

## 📊 Performance Metrics

On a test set of unseen data, the optimized model achieved:
* **MdAE (Median Error):** `1.14 SP` (The model predicts the exact or adjacent Fibonacci category 50% of the time).
* **MAE (Mean Error):** `1.83 SP`.

> *Feature Importance analysis revealed that the SBERT semantic embeddings dominate the decision tree, proving the algorithm predicts based on deep technical understanding rather than superficial metrics.*

## 💻 How to Run Locally

### 1. Setup Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt