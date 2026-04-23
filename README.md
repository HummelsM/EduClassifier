
# 📘 Case Study: Educational Search Intent Classification System (NLP)

## 🎓 Overview

Modern educational platforms struggle to correctly interpret student queries, often treating all inputs as simple keyword searches. This leads to poor content routing and suboptimal learning experiences.

To address this, I built an **end-to-end NLP classification system** that identifies the *intent behind student queries* and routes them into meaningful learning categories.

The system is trained on a custom dataset of **~560 labeled educational queries** and fine-tuned using transformer-based language models.

---

# 🎯 Problem Statement

Given a natural language student query, classify its intent into one of the following categories:

* **CONCEPTUAL** → Understanding definitions and “what/why” questions
* **PROCEDURAL** → Step-by-step problem solving or implementation
* **ADVANCED** → Comparative, optimization, or decision-making queries
* **NAVIGATIONAL** → Requests for accessing specific learning resources

---

# 🧠 Why This Matters

Traditional search systems fail in educational contexts because they:

* Treat all queries as keyword search
* Cannot distinguish learning intent
* Provide irrelevant or poorly structured results

This project explores:

> How well transformer models can capture *intent-level semantics* in educational queries.

---

# 🏗️ System Architecture

```text id="arch1"
User Query
   ↓
Text Preprocessing
   ↓
Transformer Tokenization
   ↓
Fine-tuned DistilBERT Model
   ↓
Intent Classification
   ↓
Action Mapping (Learning Response Type)
```

---

# ⚙️ Tech Stack

* Python
* Hugging Face Transformers
* PyTorch
* Streamlit
* Scikit-learn (evaluation)

---

# 📊 Dataset Design

A custom dataset of **~560 labeled samples** was created, focusing on educational query patterns.

### Class Distribution:

* ADVANCED: 198
* NAVIGATIONAL: 186
* CONCEPTUAL: 183
* PROCEDURAL: 180

### Key Design Principle:

Instead of keyword-based labeling, the dataset was structured around **intent semantics**, not surface-level phrases.

---

# 🤖 Model Approach

A pretrained **DistilBERT** model was fine-tuned for sequence classification.

* Base Model: DistilBERT (uncased)
* Objective: Multi-class classification (4 labels)
* Loss Function: Cross-entropy
* Optimizer: AdamW

---

# ⚠️ Key Challenge: Label Ambiguity

One of the main challenges encountered was **semantic overlap between classes**, especially:

### 1. Procedural vs Advanced

* “Procedure for quicksort” → PROCEDURAL
* “When should I use quicksort?” → ADVANCED

### 2. Navigational vs Procedural

* “Use BFS algorithm” was incorrectly learned as NAVIGATIONAL in early iterations due to keyword bias.

---

# 🔬 Findings & Error Analysis

### Observed Issues:

* Model initially over-relied on keywords like *“use”, “procedure”, “algorithm”*
* NAVIGATIONAL class contamination due to ambiguous training examples
* Confusion between “how-to” and “when-to-use” queries

### Root Cause:

> Dataset lacked strict separation between **system navigation language** and **algorithmic usage language**

---

# 🛠️ Improvements Applied

To address these issues, the following were implemented:

### ✔️ Label Refinement

Strict definition enforcement for NAVIGATIONAL:

> Only UI/system movement actions (open, go to, show page)

### ✔️ Contrast Training Pairs

Explicitly added:

```text id="fix1"
What is quicksort → CONCEPTUAL  
Procedure for quicksort → PROCEDURAL  
When should I use quicksort → ADVANCED  
Open quicksort notes → NAVIGATIONAL  
```

### ✔️ Anti-pattern examples

To break keyword bias:

* “Use BFS to solve maze” → PROCEDURAL
* “Use menu to open slides” → NAVIGATIONAL

---

# 📈 Evaluation Metrics

Model evaluated using:

* Accuracy
* Precision
* Recall
* F1-score (weighted)

Additional qualitative evaluation was performed through **error case analysis**, focusing on ambiguous query types.

---

# 🖥️ Deployment

A real-time inference interface was built using:

* Streamlit

Features:

* Live query classification
* Confidence score display
* Suggested learning action based on intent

---

# 💡 Key Insights

* Dataset design has more impact than model architecture at this scale
* Most errors came from **semantic ambiguity, not model capacity**
* Contrast learning examples significantly improved boundary separation
* NAVIGATIONAL class is highly sensitive to wording contamination

---

# 🚀 Future Work

* Expand dataset to 5K+ queries
* Introduce multi-label classification for ambiguous queries
* Integrate with graph-based educational search system
* Explore LLM-based intent disambiguation layer

---

# 📌 Summary

This project demonstrates how transformer models can be applied beyond generic NLP tasks to solve **real educational system problems**, particularly in interpreting student intent and improving search intelligence.

