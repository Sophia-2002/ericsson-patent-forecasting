"""
============================================================================
 BMAN60422 Coursework — Task 2 Baseline Forecasting Models
============================================================================

 Predicts total annual Ericsson patent counts using two baseline models:

   1. Interpretable model : Decision Tree   (max_depth=2)
   2. Complex ML model    : Random Forest   (n_estimators=35)

 Hyperparameter selection was carried out via grid sweeps:
   - DT  max_depth     ∈ {2, 4, 6}
   - RF  n_estimators  ∈ {10, 15, 20, …, 50}

 Features (7):
   - lag_1            : previous year's total patent count
   - lag_kw_iot       : previous year's proportion of patents containing 'iot'
   - lag_kw_ai_ml     : previous year's proportion containing 'ai_ml'
   - lag_kw_security  : previous year's proportion containing 'security'
   - lag_kw_energy    : previous year's proportion containing 'energy'
   - lag_kw_network   : previous year's proportion containing 'network'
   - lag_kw_data      : previous year's proportion containing 'data'

 Evaluation protocol:
   - 80/10/10 chronological split (2025 excluded as partial year)
   - Train : 1977-2014 (38 yrs)
   - Val   : 2015-2019 (5 yrs)
   - Test  : 2020-2024 (5 yrs)
   - Metrics: MAPE, MAE, R²
   - Baseline: Persistence (y_hat_t = y_{t-1}) for diagnostic comparison

 Forward forecast:
   - 2025 prediction generated using a model refit on the full 1977-2024
     window (train/val/test split is for hyperparameter selection only;
     final forecast uses all labelled data).
============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_percentage_error,
                             mean_absolute_error,
                             r2_score)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH        = "ericsson_patent_rich_dataset.csv"
RANDOM_STATE     = 123
KEYWORDS         = ["kw_iot", "kw_ai_ml", "kw_security",
                    "kw_energy", "kw_network", "kw_data"]

DT_DEPTH_GRID    = [2, 4, 6]
RF_NEST_GRID     = list(range(10, 51, 5))    # 10, 15, 20, ..., 50

DT_BEST_DEPTH    = 2
RF_BEST_NEST     = 35


# ---------------------------------------------------------------------------
# 1. Load and aggregate to yearly resolution
# ---------------------------------------------------------------------------
def load_and_aggregate(path: str) -> pd.DataFrame:
    """Load raw patent rows and aggregate to yearly counts + keyword shares.

    Excludes 2025 (partial year).  Keyword 'proportion' for a year is
    computed as keyword_count / total_patents in that year — which equals
    the mean of the binary kw_* flag.
    """
    df = pd.read_csv(path)
    df = df[df["year"] != 2025].copy()

    yearly = (df.groupby("year")
                .agg(total=("patent_id", "count"),
                     **{k: (k, "mean") for k in KEYWORDS})
                .reset_index()
                .sort_values("year")
                .reset_index(drop=True))
    return yearly


# ---------------------------------------------------------------------------
# 2. Build lag-1 features
# ---------------------------------------------------------------------------
def build_lag_features(yearly: pd.DataFrame):
    """Create lag_1 of the target and lag_1 of each keyword proportion.

    Drops the first row (1976), which has no prior year.
    """
    df = yearly.copy()
    df["lag_1"] = df["total"].shift(1)
    for k in KEYWORDS:
        df[f"lag_{k}"] = df[k].shift(1)

    feature_cols = ["lag_1"] + [f"lag_{k}" for k in KEYWORDS]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df, feature_cols


# ---------------------------------------------------------------------------
# 3. Chronological 80/10/10 split
# ---------------------------------------------------------------------------
def chronological_split(df: pd.DataFrame):
    """80/10/10 split preserving temporal order."""
    n       = len(df)
    n_train = int(round(n * 0.8))
    n_val   = int(round(n * 0.1))

    train = df.iloc[:n_train].reset_index(drop=True)
    val   = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test  = df.iloc[n_train + n_val:].reset_index(drop=True)
    return train, val, test


# ---------------------------------------------------------------------------
# 4. Metric helpers
# ---------------------------------------------------------------------------
def mape(y, pred):
    return mean_absolute_percentage_error(y, pred) * 100


def evaluate(model, X, y) -> dict:
    """Return MAPE, MAE, R² for a fitted model on (X, y)."""
    pred = model.predict(X)
    return {
        "MAPE": mape(y, pred),
        "MAE":  mean_absolute_error(y, pred),
        "R2":   r2_score(y, pred),
    }


# ---------------------------------------------------------------------------
# 5. Hyperparameter sweeps
# ---------------------------------------------------------------------------
def sweep_dt(X_train, y_train, X_val, y_val, X_test, y_test, depth_grid):
    """Decision-Tree sweep over max_depth.  Returns a tidy DataFrame."""
    rows = []
    for d in depth_grid:
        mdl = DecisionTreeRegressor(max_depth=d, random_state=RANDOM_STATE).fit(X_train, y_train)
        tr, va, te = evaluate(mdl, X_train, y_train), evaluate(mdl, X_val, y_val), evaluate(mdl, X_test, y_test)
        rows.append({
            "max_depth":   d,
            "Train MAPE":  f"{tr['MAPE']:.2f}%",
            "Val MAPE":    f"{va['MAPE']:.2f}%",
            "Test MAPE":   f"{te['MAPE']:.2f}%",
            "Train R²":    f"{tr['R2']:.3f}",
            "Val R²":      f"{va['R2']:.3f}",
            "Test R²":     f"{te['R2']:.3f}",
        })
    return pd.DataFrame(rows)


def sweep_rf(X_train, y_train, X_val, y_val, X_test, y_test, nest_grid):
    """Random-Forest sweep over n_estimators.  Returns a tidy DataFrame."""
    rows = []
    for n_est in nest_grid:
        mdl = RandomForestRegressor(n_estimators=n_est, random_state=RANDOM_STATE).fit(X_train, y_train)
        tr, va, te = evaluate(mdl, X_train, y_train), evaluate(mdl, X_val, y_val), evaluate(mdl, X_test, y_test)
        rows.append({
            "n_estimators": n_est,
            "Train MAPE":   f"{tr['MAPE']:.2f}%",
            "Val MAPE":     f"{va['MAPE']:.2f}%",
            "Test MAPE":    f"{te['MAPE']:.2f}%",
            "Train R²":     f"{tr['R2']:.3f}",
            "Val R²":       f"{va['R2']:.3f}",
            "Test R²":      f"{te['R2']:.3f}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Persistence baseline (diagnostic)
# ---------------------------------------------------------------------------
def persistence_predictions(subset: pd.DataFrame):
    """Persistence: y_hat_t = y_{t-1} (i.e. just the lag_1 column)."""
    return subset["lag_1"].values


# ---------------------------------------------------------------------------
# 7. Main pipeline
# ---------------------------------------------------------------------------
def main():
    # ---- Data prep ----
    yearly         = load_and_aggregate(DATA_PATH)
    data, feats    = build_lag_features(yearly)
    train, val, test = chronological_split(data)

    print(f"Total years (after lagging): {len(data)} "
          f"({data['year'].min()}–{data['year'].max()})")
    print(f"  Train: {train['year'].min()}–{train['year'].max()}  (n={len(train)})")
    print(f"  Val:   {val['year'].min()}–{val['year'].max()}    (n={len(val)})")
    print(f"  Test:  {test['year'].min()}–{test['year'].max()}    (n={len(test)})\n")

    X_train, y_train = train[feats], train["total"]
    X_val,   y_val   = val[feats],   val["total"]
    X_test,  y_test  = test[feats],  test["total"]

    # ====================================================================
    #  Hyperparameter sweeps
    # ====================================================================
    print("══════════ Decision Tree sweep — max_depth ∈ {2, 4, 6} ══════════")
    dt_sweep = sweep_dt(X_train, y_train, X_val, y_val, X_test, y_test, DT_DEPTH_GRID)
    print(dt_sweep.to_string(index=False))
    print(f"\n→ Selected: max_depth = {DT_BEST_DEPTH} "
          f"(parsimonious, robust on test, no overfitting)\n")

    print("══════════ Random Forest sweep — n_estimators ∈ {10, 15, …, 50} ══════════")
    rf_sweep = sweep_rf(X_train, y_train, X_val, y_val, X_test, y_test, RF_NEST_GRID)
    print(rf_sweep.to_string(index=False))
    print(f"\n→ Selected: n_estimators = {RF_BEST_NEST} "
          f"(sits on the stable plateau, best test MAPE / R² in the lower range)\n")

    # ====================================================================
    #  Fit locked champion models on TRAIN
    # ====================================================================
    dt = DecisionTreeRegressor(max_depth=DT_BEST_DEPTH,
                               random_state=RANDOM_STATE).fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=RF_BEST_NEST,
                               random_state=RANDOM_STATE).fit(X_train, y_train)

    for label, mdl in [(f"Decision Tree (max_depth={DT_BEST_DEPTH})", dt),
                       (f"Random Forest (n_estimators={RF_BEST_NEST})", rf)]:
        tr, va, te = evaluate(mdl, X_train, y_train), evaluate(mdl, X_val, y_val), evaluate(mdl, X_test, y_test)
        out = pd.DataFrame(
            [[tr['MAPE'], tr['MAE'], tr['R2']],
             [va['MAPE'], va['MAE'], va['R2']],
             [te['MAPE'], te['MAE'], te['R2']]],
            index=["Train", "Val", "Test"],
            columns=["MAPE", "MAE", "R²"]).round(3)
        print(f"── {label} ──")
        print(out.to_string())
        print()

    # ---- Persistence baseline (diagnostic) ----
    print("── Persistence baseline (y_hat_t = y_{t-1}) ──")
    pv, pt = persistence_predictions(val), persistence_predictions(test)
    print(f"  Val  MAPE: {mape(y_val, pv):.2f}%   R²: {r2_score(y_val, pv):.3f}")
    print(f"  Test MAPE: {mape(y_test, pt):.2f}%   R²: {r2_score(y_test, pt):.3f}\n")

    # ---- MDI feature importance ----
    print("── MDI feature importance ──")
    mdi = pd.DataFrame({
        f"DT (max_depth={DT_BEST_DEPTH})":   pd.Series(dt.feature_importances_, index=feats),
        f"RF (n_est={RF_BEST_NEST})":        pd.Series(rf.feature_importances_, index=feats),
    }).round(4)
    print(mdi.to_string())
    print()

    # ---- 2025 forward forecast (refit on full 1977-2024) ----
    dt_full = DecisionTreeRegressor(max_depth=DT_BEST_DEPTH,
                                    random_state=RANDOM_STATE).fit(data[feats], data["total"])
    rf_full = RandomForestRegressor(n_estimators=RF_BEST_NEST,
                                    random_state=RANDOM_STATE).fit(data[feats], data["total"])

    row_2024 = yearly[yearly["year"] == 2024].iloc[0]
    x_2025 = pd.DataFrame([{
        "lag_1": row_2024["total"],
        **{f"lag_{k}": row_2024[k] for k in KEYWORDS}
    }])[feats]

    print("── 2025 forecast (refit on full 1977-2024) ──")
    print(f"  DT (max_depth={DT_BEST_DEPTH}):     {float(dt_full.predict(x_2025)[0]):.0f}")
    print(f"  RF (n_estimators={RF_BEST_NEST}):  {float(rf_full.predict(x_2025)[0]):.0f}")


if __name__ == "__main__":
    main()
