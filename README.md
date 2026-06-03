# 🧠 Agile NLP Estimator: AI-Driven Story Point Prediction

A production-ready Natural Language Processing pipeline designed to eliminate human bias and latency in Agile software development estimation. 

This engine ingests raw Jira/Linear ticket descriptions and predicts structural complexity (Story Points) using state-of-the-art text embeddings and gradient boosting.

### 🏢 The Business Problem
Manual sprint planning is a high-latency process prone to psychological bias (anchoring, optimism). Poor estimation destroys burn rates and startup runways. This architecture automates the baseline estimation, allowing engineering leaders to focus on system design rather than administrative guessing.

### ⚙️ System Architecture & Stack

- **Text Vectorization (NLP):** `SBERT` (Sentence-BERT) for generating dense, semantically meaningful embeddings from technical requirements.
- **Predictive Engine:** `LightGBM` for high-efficiency, low-memory gradient boosting over the embedding space.
- **Language:** `Python 3.10+`

### 📊 Architecture Flow
*(Nota interna: Cuando subas el código, aquí debes habilitar el renderizado de este diagrama)*
```mermaid
graph TD
    A[Raw Ticket / User Story] --> B(Text Preprocessing)
    B --> C{SBERT Model}
    C -->|768-dim Vector| D[LightGBM Predictor]
    D --> E[Predicted Story Points]
