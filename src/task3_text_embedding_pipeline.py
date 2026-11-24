"""
================================================================================
BMAN60422 — Task 3: Text-Embedding Cluster-Forecasting Pipeline
================================================================================
End-to-end script that reproduces the locked text-segmentation pipeline and the
per-cluster + aggregate forecasting evaluation.

CRISP-DM phase: Phase 3 (Data Preparation) → Phase 4 (Modelling) → Phase 5 (Evaluation)

Pipeline overview
-----------------
1. Load Ericsson patent dataset (28,941 patents, 1976-2024)
2. Preprocess titles (lowercase, strip punctuation)
3. Apply auto-detected plural-to-singular mapping
4. TF-IDF vectorise (ngrams 1-2, min_df=5, max_df=0.5, sublinear tf, L2 norm)
   with scikit-learn English stopwords + domain stopwords:
   method, system, communication, apparatus, device
5. Truncated SVD to 50 components
6. L2-normalise SVD output
7. K-Means with K=3 (locked from K-selection analysis), random_state=123, n_init=10
8. Build per-cluster annual panel and engineer lag_1 + 6 lagged prop_kw features
9. Train RF n=35 and DT depth=2 per cluster, 80/10/10 chronological split
10. Aggregate cluster forecasts and compute aggregate train/val/test MAPE

The 6 keyword features used per cluster: prop_kw_ai_ml, prop_kw_security, prop_kw_iot,
prop_kw_network, prop_kw_energy, prop_kw_data — same as the locked Task 2 baseline.

Run:    python task3_text_embedding_pipeline.py
Inputs: ericsson_patent_rich_dataset.csv (same directory)
Output: console summary of cluster sizes and MAPE results

Note: reconstructed from the report specification; reproduces reported cluster
shares within ~1 percentage point.
================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION (locked values from K-selection analysis)
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH        = 'ericsson_patent_rich_dataset.csv'
YEAR_MAX         = 2024            # exclude partial-year 2025 from analysis
RANDOM_STATE     = 123
N_SVD_COMPONENTS = 50               # locked (report: 50 more stable than 100)
K_CLUSTERS       = 3                # locked K (report Section 4.1)
N_INIT_KMEANS    = 10
SPLIT_TRAIN_FRAC = 0.80             # chronological 80/10/10 split (matches Task 2)
SPLIT_VAL_FRAC   = 0.10
RF_N_ESTIMATORS  = 35               # locked from Task 2 baseline
DT_MAX_DEPTH     = 2                # locked from Task 2 baseline

# Stopwords for TF-IDF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
STOPWORDS_BASIC = list(ENGLISH_STOP_WORDS)   # scikit-learn English stopword list
STOPWORDS_DOMAIN = ['method','system','communication','apparatus','device']

# Keyword features (same 6 used in Task 2 locked baseline)
KW_FOR_FEATURES = ['kw_ai_ml','kw_security','kw_iot','kw_network','kw_energy','kw_data']
FEATS = ['lag_1'] + [f'prop_{k}' for k in KW_FOR_FEATURES]


# ──────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def safe_mape(y_true, y_pred):
    """MAPE excluding zero-actual years (division by zero is undefined)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def build_singularisation_map(titles, stopwords, min_df=5):
    """Auto-detect plural→singular pairs in vocabulary using rule-based suffix
    matching: 's', 'es', 'ies' endings. Each pair requires both forms to
    appear in the corpus at min_df=5."""
    cv = CountVectorizer(ngram_range=(1, 1), min_df=min_df,
                         stop_words=stopwords, lowercase=False, binary=True)
    cv.fit(titles)
    vocab = set(cv.get_feature_names_out())

    mapping = {}
    for w in cv.get_feature_names_out():
        if len(w) < 4:
            continue
        # Skip words that look plural but aren't (e.g., "wireless", "apparatus", "basis")
        if w.endswith('ss') or w.endswith('us') or w.endswith('is'):
            continue
        candidate_singular = None
        # Rule 1: 'ies' ending → 'y' (frequencies → frequency)
        if w.endswith('ies') and len(w) > 4:
            sing = w[:-3] + 'y'
            if sing in vocab:
                candidate_singular = sing
        # Rule 2: 'es' ending → drop 'es' (processes → process); fallback to drop 's'
        elif w.endswith('es') and len(w) > 4:
            sing = w[:-2]
            if sing in vocab:
                candidate_singular = sing
            else:
                sing_s = w[:-1]
                if sing_s in vocab:
                    candidate_singular = sing_s
        # Rule 3: 's' ending → drop 's'
        elif w.endswith('s'):
            sing = w[:-1]
            if sing in vocab:
                candidate_singular = sing
        if candidate_singular:
            mapping[w] = candidate_singular
    return mapping


def apply_singularisation(text, mapping):
    """Replace each plural token in text with its singular form per the mapping."""
    return ' '.join(mapping.get(t, t) for t in text.split())


def build_cluster_panel(cluster_df, year_range, kw_cols):
    """For one cluster's patents, build the annual panel of patent_count and
    prop_kw_* features. Years with zero patents get count=0 and prop_kw=0."""
    yearly = cluster_df.groupby('year').agg(
        patent_count=('patent_id', 'size'),
        **{f'prop_{k}': (k, 'mean') for k in kw_cols}
    )
    yearly = yearly.reindex(year_range, fill_value=0)
    for col in [f'prop_{k}' for k in kw_cols]:
        yearly[col] = yearly[col].fillna(0)
    return yearly.reset_index().rename(columns={'index': 'year'})


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 80)
print("STEP 1 — Loading and filtering dataset")
print("=" * 80)

df = pd.read_csv(DATA_PATH)
df = df[df['year'] <= YEAR_MAX].reset_index(drop=True)
print(f"Loaded {len(df)} patents (years {df['year'].min()}-{df['year'].max()})")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — TEXT PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("STEP 2 — Text preprocessing (lowercase + strip punctuation)")
print("=" * 80)

titles_clean = (df['patent_title'].astype(str)
                  .str.lower()
                  .str.replace(r'[^\w\s]', ' ', regex=True)
                  .str.replace(r'\s+', ' ', regex=True)
                  .str.strip())
print(f"Cleaned {len(titles_clean)} titles")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — SINGULARISATION (plural→singular pairs)
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("STEP 3 — Build and apply plural→singular mapping")
print("=" * 80)

singular_map = build_singularisation_map(titles_clean, STOPWORDS_BASIC)
print(f"Detected {len(singular_map)} plural→singular pairs")

titles_processed = titles_clean.apply(lambda t: apply_singularisation(t, singular_map))


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — TF-IDF VECTORISATION
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("STEP 4 — TF-IDF vectorisation (ngrams 1-2, min_df=5, max_df=0.5, L2 norm)")
print("=" * 80)

stopwords_full = STOPWORDS_BASIC + STOPWORDS_DOMAIN
vectoriser = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.5,
    sublinear_tf=True,
    norm='l2',
    stop_words=stopwords_full,
    lowercase=False,
)
X_tfidf = vectoriser.fit_transform(titles_processed)
print(f"TF-IDF matrix: {X_tfidf.shape[0]:,} docs × {X_tfidf.shape[1]:,} terms")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — TRUNCATED SVD (n=50)
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print(f"STEP 5 — Truncated SVD (n_components={N_SVD_COMPONENTS})")
print("=" * 80)

svd = TruncatedSVD(
    n_components=N_SVD_COMPONENTS,
    random_state=RANDOM_STATE,
    algorithm='randomized',
)
X_svd = svd.fit_transform(X_tfidf)
print(f"SVD output: {X_svd.shape}, cumulative variance: {svd.explained_variance_ratio_.sum():.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — L2-NORMALISE SVD OUTPUT
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("STEP 6 — L2-normalise SVD output")
print("=" * 80)

X_emb = normalize(X_svd, norm='l2')
print(f"Embedding shape: {X_emb.shape}, sample row norm: {np.linalg.norm(X_emb[0]):.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 7 — K-MEANS CLUSTERING (K=3)
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print(f"STEP 7 — K-Means clustering (K={K_CLUSTERS})")
print("=" * 80)

kmeans = KMeans(
    n_clusters=K_CLUSTERS,
    n_init=N_INIT_KMEANS,
    init='k-means++',
    random_state=42,
)
cluster_labels = kmeans.fit_predict(X_emb)

df['cluster'] = cluster_labels
cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
print(f"\nCluster sizes:")
for c, n in cluster_sizes.items():
    pct = n / len(df) * 100
    print(f"  Cluster {c}: {n:>6} patents ({pct:>5.1f}%)")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 8 — BUILD PER-CLUSTER ANNUAL PANELS WITH LAG_1 AND PROP_KW FEATURES
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("STEP 8 — Build per-cluster annual panels (lag_1 + 6 lagged prop_kw features)")
print("=" * 80)

year_range = range(int(df['year'].min()), int(df['year'].max()) + 1)
cluster_panels = {}
for c in range(K_CLUSTERS):
    panel = build_cluster_panel(df[df['cluster'] == c], year_range, KW_FOR_FEATURES)
    panel['lag_1'] = panel['patent_count'].shift(1)
    for k in KW_FOR_FEATURES:                       # previous year's keyword share (no look-ahead)
        panel[f'prop_{k}'] = panel[f'prop_{k}'].shift(1)
    panel = panel.dropna(subset=['lag_1']).reset_index(drop=True)
    cluster_panels[c] = panel

n_years_total = len(cluster_panels[0])
n_train = int(round(n_years_total * SPLIT_TRAIN_FRAC))
n_val   = int(round(n_years_total * SPLIT_VAL_FRAC))

print(f"Total usable years per cluster: {n_years_total}")
print(f"Chronological split: train={n_train}, val={n_val}, test={n_years_total - n_train - n_val}")
print(f"  Train: {cluster_panels[0]['year'].iloc[0]}-{cluster_panels[0]['year'].iloc[n_train-1]}")
print(f"  Val:   {cluster_panels[0]['year'].iloc[n_train]}-{cluster_panels[0]['year'].iloc[n_train+n_val-1]}")
print(f"  Test:  {cluster_panels[0]['year'].iloc[n_train+n_val]}-{cluster_panels[0]['year'].iloc[-1]}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 9 — TRAIN RF AND DT PER CLUSTER
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print(f"STEP 9 — Train RF n={RF_N_ESTIMATORS} and DT depth={DT_MAX_DEPTH} per cluster")
print("=" * 80)

per_cluster_predictions = {}

for c in range(K_CLUSTERS):
    p = cluster_panels[c]
    train = p.iloc[:n_train]
    val   = p.iloc[n_train:n_train + n_val]
    test  = p.iloc[n_train + n_val:]

    rf_model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RANDOM_STATE,
    ).fit(train[FEATS], train['patent_count'])
    dt_model = DecisionTreeRegressor(
        max_depth=DT_MAX_DEPTH,
        random_state=RANDOM_STATE,
    ).fit(train[FEATS], train['patent_count'])

    per_cluster_predictions[c] = {
        'years_tr':   train['year'].values,
        'years_va':   val['year'].values,
        'years_te':   test['year'].values,
        'actual_tr':  train['patent_count'].values,
        'actual_va':  val['patent_count'].values,
        'actual_te':  test['patent_count'].values,
        'rf_pred_tr': rf_model.predict(train[FEATS]),
        'rf_pred_va': rf_model.predict(val[FEATS]),
        'rf_pred_te': rf_model.predict(test[FEATS]),
        'dt_pred_tr': dt_model.predict(train[FEATS]),
        'dt_pred_va': dt_model.predict(val[FEATS]),
        'dt_pred_te': dt_model.predict(test[FEATS]),
    }

print(f"Trained {K_CLUSTERS} RF models and {K_CLUSTERS} DT models (one per cluster)")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 10 — AGGREGATE CLUSTER FORECASTS AND COMPUTE MAPE
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("STEP 10 — Aggregate cluster forecasts → aggregate train/val/test MAPE")
print("=" * 80)

def aggregate_split(split_suffix):
    """Sum actuals and predictions across all clusters for one split."""
    n_yrs = len(per_cluster_predictions[0][f'years_{split_suffix}'])
    actual_sum = np.zeros(n_yrs)
    rf_sum     = np.zeros(n_yrs)
    dt_sum     = np.zeros(n_yrs)
    for c in range(K_CLUSTERS):
        d = per_cluster_predictions[c]
        actual_sum += d[f'actual_{split_suffix}']
        rf_sum     += d[f'rf_pred_{split_suffix}']
        dt_sum     += d[f'dt_pred_{split_suffix}']
    years = per_cluster_predictions[0][f'years_{split_suffix}']
    return years, actual_sum, rf_sum, dt_sum

print(f"\n{'Split':<10}{'Years':<14}{'#yrs':>6}{'RF MAPE':>12}{'DT MAPE':>12}")
print("-" * 56)
for split, label in [('tr', 'Train'), ('va', 'Val'), ('te', 'Test')]:
    years, actual_sum, rf_sum, dt_sum = aggregate_split(split)
    rf_mape = safe_mape(actual_sum, rf_sum)
    dt_mape = safe_mape(actual_sum, dt_sum)
    yr_range = f"{years[0]}-{years[-1]}"
    print(f"{label:<10}{yr_range:<14}{len(years):>6}{rf_mape:>11.2f}%{dt_mape:>11.2f}%")

# Detailed test-year breakdown (the headline forecast comparison)
print(f"\nDetailed test-year breakdown (RF aggregate forecast):")
print(f"  {'Year':<6}{'Sum actual':>12}{'Sum RF pred':>14}{'|err|':>10}{'|err|/actual':>16}")
print("  " + "-" * 56)
years_te, actual_te, rf_te, dt_te = aggregate_split('te')
for y, a, r in zip(years_te, actual_te, rf_te):
    err = abs(a - r)
    rel = err / a if a != 0 else np.nan
    print(f"  {int(y):<6}{int(a):>12}{r:>14.2f}{err:>10.2f}{rel*100:>15.2f}%")

print(f"\n{'='*80}")
print("DONE")
print(f"{'='*80}")
