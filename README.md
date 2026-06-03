# 🧠 Agile NLP Assessor: AI-Driven Story Point & Quality Gatekeeper

A production-ready NLP pipeline designed to eliminate human bias and latency in Agile software development. This system augments sprint planning by providing an objective "vote" for Planning Poker and enforcing quality standards (INVEST) for User Story drafting.

---

### 🏢 The Business Problem (Friction & Bias)
1. **The Anchoring Bias:** Manual estimation is highly susceptible to the first spoken number in a room.
2. **"Garbage-In, Garbage-Out":** Poorly drafted User Stories lead to misaligned engineering execution and blown sprint budgets.

### 🎯 The Architectural Solution
This system **does not replace human consensus**; it enhances it by providing:
- **An Objective Baseline:** A mathematically derived Story Point estimate that serves as an unbiased anchor during Planning Poker.
- **Justified Complexity:** Feature importance mapping to highlight *why* a ticket is deemed complex.
- **Redaction Guardrails:** Identifies ambiguities in the User Story text *before* it reaches the engineering team.

---

### ⚙️ Engineering Stack & Pipeline
*Built for production scale, utilizing a 5-phase data-centric pipeline:*

1. **Ingestion:** Pandas/PyArrow optimized data mining.
2. **Quality Audit:** LLM Judge (OpenAI/JSON Schema) evaluating PBI structural quality.
3. **Semantic Vectorization:** `SBERT` (Sentence-BERT) generating 768-dim dense embeddings.
4. **Predictive Engine:** `LightGBM` (Gradient Boosting Regressor) trained on refined datasets to predict Fibonacci containers.
5. **UI Layer:** Streamlit-based interactive interface for real-time feedback.

### 📊 System Architecture
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

---
📫 **Contact for B2B / Remote W-8BEN opportunities:** [diego.acevedok@gmail.com](mailto:diego.acevedok@gmail.com) | [LinkedIn](https://www.linkedin.com/in/diego-ignacio-acevedo-pizarro/)
