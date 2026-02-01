# Rana-Hammirdev_PS14
# 🎯 Intelligent Issue Insight Engine - Complete Documentation

## 📋 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Usage](#usage)
6. [File Structure](#file-structure)
7. [Advanced NLP Pipeline](#advanced-nlp-pipeline)
8. [ML Models](#ml-models)
9. [Rule Engine](#rule-engine)
10. [Deployment](#deployment)

---

## 🎯 Overview

An **explainable AI system** that transforms unstructured issue descriptions into structured, actionable insights for administrators.

### Key Capabilities:
- ✅ Automatic categorization & priority prediction
- ✅ Handles Hinglish, spelling errors, informal language
- ✅ Explainable decisions (ML + Rules hybrid)
- ✅ Recurring pattern detection
- ✅ Time-aware trend analysis
- ✅ Department load balancing

---

## 🎨 Features (10 USPs)

1. **Explainable Priority Prediction** - Every decision has clear reasoning
2. **Impact × Urgency Logic** - Multi-dimensional priority scoring
3. **Recurring Issue Detection** - Auto-escalation for systemic problems
4. **Similar Issue Clustering** - Root cause identification
5. **Time-Aware Insights** - Peak detection & trend analysis
6. **Department Load Index** - Resource allocation guidance
7. **Confidence-Based Routing** - Human review for low-confidence cases
8. **Lifecycle Simulation** - Issue progression tracking
9. **Rule vs ML Comparison** - Transparent decision-making
10. **Today's Action Panel** - Executive dashboard

---

## 🏗️ Architecture
```
User Input (Hinglish/Informal)
        ↓
[NLP Preprocessing Pipeline]
  • Normalization
  • Language Detection
  • Spelling Correction
  • Hinglish Transliteration
  • Formalization
        ↓
[ML Models]                    [Rule Engine]
  • TF-IDF Vectorization         • Keyword Rules
  • Category Classifier          • Time-Based Rules
  • Urgency Classifier           • Impact Assessment
        ↓                              ↓
            [Priority Intelligence]
            • ML + Rules Hybrid
            • Auto-Escalation
            • Department Load
                    ↓
        [Explainable Output]
        • Final Priority
        • Routing Decision
        • Complete Explanation
```

---

## 💻 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
```bash
# 1. Clone repository
git clone <repo-url>
cd intelligent-issue-engine

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate dataset
python generate_dataset.py

# 5. Process dataset
python modules/data_processor.py

# 6. Train models
python train_models.py

# 7. Run application
streamlit run app.py
```

**Or use quick start:**
```bash
chmod +x run_app.sh
./run_app.sh
```

---

## 🚀 Usage

### 1. Web Interface
```bash
streamlit run app.py
```

Navigate to `http://localhost:8501`

### 2. Programmatic Usage
```python
from modules.nlp_preprocessing import NLPPreprocessor
from modules.ml_models import ModelManager
from modules.rule_engine import CompleteRuleEngine

# Initialize
preprocessor = NLPPreprocessor()
model_manager = ModelManager()
model_manager.load_models()
rule_engine = CompleteRuleEngine()

# Process input
user_input = "mera wifi nahi chal raha urgent exam hai"
cleaned = preprocessor.preprocess(user_input)['cleaned_text']

# Get predictions
ml_result = model_manager.predict_issue(cleaned)
rule_result = rule_engine.analyze_issue(cleaned)

print(f"Category: {ml_result['category']}")
print(f"Priority: {ml_result['priority']}")
---
---
## 🧠 Advanced NLP Pipeline

### Pipeline Steps:

1. **Normalization** - Lowercase, remove URLs, clean spaces
2. **Language Detection** - Detect English/Hinglish/Mixed
3. **Spelling Correction** - Fix common typos
4. **Hinglish Transliteration** - Convert Hindi words to English
5. **Formalization** - Expand abbreviations (plz → please)
6. **Cleaning** - Remove noise, special characters
7. **Tokenization** - Split into words
8. **Lemmatization** - Reduce to base forms
9. **Stopword Removal** - Keep important words only

### Example Transformation:
```
Input:  "mera wifi nahi chal raha plz help asap"
Output: "wifi not work please help soon possible"

Input:  "laptop ka screen kharab hai urgent exam hai"
Output: "laptop screen broken urgent exam"
```

