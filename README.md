# 🧠 Agile NLP Assessor: AI-Driven Story Point & Quality Gatekeeper

A production-ready Natural Language Processing pipeline designed to augment Agile sprint planning. It acts as an objective baseline vote during Planning Poker and enforces quality standards for User Story drafting.

This engine ingests raw Jira/Linear ticket descriptions and predicts structural complexity using state-of-the-art text embeddings (SBERT) and gradient boosting (LightGBM).

### 🏢 The Business Problem (Friction & Bias)
1. **The Anchoring Bias:** Manual estimation is highly susceptible to the first spoken number in a room.
2. **Garbage-In, Garbage-Out:** Poorly drafted User Stories lead to misaligned engineering execution and blown budgets.

### 🎯 The Architectural Solution
This system **does not replace human consensus**; it enhances it by providing:
- **An Objective "Vote":** A mathematically derived Story Point estimate that serves as an unbiased baseline at the Planning Poker table.
- **Justified Complexity:** Feature importance mapping to highlight *why* a ticket is complex.
- **Drafting Feedback:** Identifies ambiguities in the User Story (HU) text *before* it reaches the engineering team, enforcing high redaction standards.

### ⚙️ System Architecture & Stack

- **Text Vectorization (NLP):** `SBERT` (Sentence-BERT) for generating dense, semantically meaningful embeddings from technical requirements.
- **Predictive Engine:** `LightGBM` for high-efficiency, low-memory gradient boosting over the embedding space.
- **Language:** `Python 3.10+`

### 📊 Architecture Flow
```mermaid
graph TD
    A[Raw User Story Draft] --> B(Text Preprocessing & NLP)
    B --> C{SBERT Embeddings}
    C --> D[LightGBM Predictor]
    C --> E[Ambiguity/Quality Filter]
    D --> F[Objective SP Vote + Justification]
    E --> G[Redaction Improvement Suggestions]
    F --> H((Planning Poker Integration))
    G --> H
