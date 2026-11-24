# Forecasting Patent Innovation at Ericsson

**Machine learning forecast of annual patent output and technology mix, based on 28,941 patent filings (1976–2024)**

MSc Business Analytics group project , University of Manchester

`Python` · `scikit-learn` · `pandas` · `TF-IDF` · `K-Means` · `Random Forest` · `SHAP`

---

## 📌 Business question
Ericsson's R&D leadership needs answers to two questions:
1. **How many patents will Ericsson file next year?** The answer feeds budget and hiring plans.
2. **How is the technology mix shifting?** This shows where engineering effort is going relative to Huawei, Nokia and Qualcomm.

## 📊 Data
- 28,941 Ericsson patent records, 1976–2024 (the partial year 2025 was excluded)
- Title text plus 9 binary keyword flags (5G, AI/ML, cloud/edge, security, IoT, network, energy, antenna, data)
- Aggregated to **48 annual observations** after the lag-1 transformation

## 🔧 Method (CRISP-DM)
**1. Exploratory analysis**
- ACF/PACF show an **AR(1)** pattern, so last year's count (`lag_1`) is the main predictor.
- Keyword *proportions* (not raw counts) are used, to separate the technology mix from overall volume.
- Overlapping keywords were removed using Spearman correlation (ρ ≥ 0.65).

**2. Forecasting models** (`src/task2_baseline_models.py`)
- 7 features: `lag_1` plus 6 lagged keyword proportions.
- **Chronological 80/10/10 split**: Train 1977–2014, Validation 2015–2019, Test 2020–2024. There is no shuffling, so no future data leaks into training.
- **Decision Tree** (interpretable model), with `max_depth` tuned over {2, 4, 6}.
- **Random Forest** (complex model), with `n_estimators` tuned over {10, …, 50}.
- Benchmarked against a persistence baseline (ŷₜ = yₜ₋₁).
- The Random Forest was interpreted with **TreeSHAP**.

**3. Segmentation** (`src/task3_text_embedding_pipeline.py`)
- Patent titles were clustered with this pipeline: cleaning → singularisation → **TF-IDF** (1–2 grams, scikit-learn English and domain stopwords) → **Truncated SVD** (50) → L2 normalisation → **K-Means** (K = 3).
- Three themes emerged: *Radio & signal transmission* (61%), *Digital media & coding* (20%) and *Network architecture* (19%).
- A model was forecast for each cluster and the forecasts were summed. The result was compared with the single-series baseline.
- Note: the segmentation script was reconstructed from the report's specification. It reproduces the reported cluster shares within 1 percentage point. Keyword features use the previous year's shares to avoid look-ahead leakage.

## 📈 Key results

| Model | Test MAPE |
|---|---|
| Persistence baseline | 11.73% |
| **Decision Tree (depth = 2)** | **5.53%** |
| Random Forest (n = 35) | 6.77% |

- Both models **beat the persistence baseline by about half**.
- **2025 forecast: 1,433–1,489 patents**, a moderate decline of 4–7% from 1,549 in 2024.
- SHAP shows that `lag_1` dominates the predictions. Patent output is driven mainly by **inertia**, and the technology mix adds a small correction.
- Segmentation **did not improve accuracy**: summing the per-cluster forecasts gave a Test MAPE of about 8.6% (DT and RF), compared with 5.5–6.8% for the single-series models. It did, however, show strategic shifts:
  - Network-architecture patents grew from about 0% before 1994 to **19%** of filings.
  - Titles containing acronyms (5G, IoT and similar) rose from about 0% to **25%** by 2024 (reported analysis; code not included).

## 📁 Repository structure
```
├── ericsson_patent_forecasting.ipynb     # full pipeline (Task 2 + Task 3) with outputs
├── src/
│   ├── task2_baseline_models.py          # DT & RF forecasting, hyperparameter sweeps, 2025 forecast
│   └── task3_text_embedding_pipeline.py  # TF-IDF + SVD + K-Means segmentation forecasting
├── figures/                              # key charts
└── requirements.txt
```

## ▶️ How to run
```bash
pip install -r requirements.txt
# put ericsson_patent_rich_dataset.csv in the repo root, then:
jupyter notebook ericsson_patent_forecasting.ipynb
```

## 📂 Data availability
The dataset (`ericsson_patent_rich_dataset.csv`) was provided for coursework and is **not included** in this repository.

## ⚠️ Limitations
- Only 48 annual observations, so forecasts are limited to one year ahead.
- Keyword flags come from titles only, which is a noisy proxy for technology. Patent classification codes (CPC/IPC) would be better.
- Outside factors (macroeconomic conditions, competitors, regulation) are not modelled.

---
**Author:** Yi-Wen (Sophia) Su · [LinkedIn](https://www.linkedin.com/in/yi-wen-su-523a54433/)

