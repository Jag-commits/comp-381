"""
Fraud Anomaly Detection Pipeline
==================================
All models learn what "normal" looks like from unlabelled training data.
Anything that deviates from normal is flagged as a potential anomaly.

Ground truth (is_fraud) is used ONLY at evaluation time to score results —
it is never passed to any model during training.

Anomaly Types Covered:
  1. Point Anomalies      — individual transactions globally out of range
                            → Isolation Forest, Local Outlier Factor,
                              Unsupervised Random Forest (proximity-based)
  2. Contextual Anomalies — transactions unusual within their own context
                            → Z-score per merchant category
  3. Collective Anomalies — transactions that don't fit any normal cluster
                            → DBSCAN

Output convention (all models normalised to):
    1 = anomaly (predicted fraud)
    0 = normal  (predicted legit)
"""
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    f1_score, precision_score, recall_score,
)
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier, NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN


# ─────────────────────────────────────────
# 1. Load & Prepare
# ─────────────────────────────────────────

def load_and_prepare(filepath: str):
    """
    Loads CSV, encodes categoricals, scales all features.

    Features used:
        amount, transaction_type, merchant_category, country, hour

    Excluded:
        transaction_id, user_id       — identifiers, no signal
        device_risk_score             — zero-overlap synthetic artifact
        ip_risk_score                 — zero-overlap synthetic artifact

    Returns:
        X_scaled  : scaled feature array (n_samples, n_features)
        y         : ground truth labels — used only for evaluation
        df_raw    : original DataFrame — needed for contextual detector
    """
    df = pd.read_csv(filepath)
    y  = df["is_fraud"].values

    feature_cols = ["amount", "transaction_type", "merchant_category", "country", "hour"]
    X_raw = df[feature_cols].copy()

    encoders = {}
    for col in ["transaction_type", "merchant_category", "country"]:
        le = LabelEncoder()
        X_raw[col] = le.fit_transform(X_raw[col].astype(str))
        encoders[col] = le

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return X_scaled, y, df, scaler, encoders


def build_splits(X, y, test_size=0.25, random_state=42):
    """
    Stratified 75/25 split — replicates the paper methodology.
    75% of data used for training, 25% held out for testing.
    Stratified to preserve the 5% fraud ratio in both splits.
    y passed to stratify only — anomaly models never see it during training.
    """
    all_idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=test_size, random_state=random_state, stratify=y
    )
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], train_idx, test_idx


# ─────────────────────────────────────────
# 2. Shared utilities
# ─────────────────────────────────────────

def _remap(raw: np.ndarray) -> np.ndarray:
    """
    Normalises detector output to binary 0/1.
        -1  (sklearn anomaly / DBSCAN noise)  →  1  (fraud)
        1   (sklearn normal)                  →  0  (legit)
        0+  (DBSCAN cluster id)               →  0  (legit)
    """
    return np.where(raw == -1, 1, 0)


def _report(name: str, anomaly_type: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Prints metrics and returns a result dictionary."""
    acc  = accuracy_score(y_true, y_pred)
    cm   = confusion_matrix(y_true, y_pred)
    rep  = classification_report(y_true, y_pred,
                                 target_names=["Normal", "Anomaly"],
                                 zero_division=0)
    f1   = f1_score(y_true, y_pred,        average="binary", zero_division=0)
    prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
    rec  = recall_score(y_true, y_pred,    average="binary", zero_division=0)

    print(f"\n{'='*58}")
    print(f"  [{anomaly_type}]  {name}")
    print(f"{'='*58}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (anomaly class)")
    print(f"  Recall    : {rec:.4f}  (anomaly class)")
    print(f"  F1 Score  : {f1:.4f}  (anomaly class)")
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"                 Normal  Anomaly")
    print(f"  Actual Normal  {cm[0,0]:>6}  {cm[0,1]:>7}")
    print(f"  Actual Anomaly {cm[1,0]:>6}  {cm[1,1]:>7}")
    print(f"\n  Classification Report:\n{rep}")

    return {
        "name": name, "type": anomaly_type,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "confusion_matrix": cm,
    }


# ─────────────────────────────────────────
# 3. Point Anomaly Detectors
# ─────────────────────────────────────────
#
# A POINT ANOMALY is a single transaction that is globally unusual
# compared to the rest of the dataset — e.g. an unusually large amount
# or a rare combination of country + merchant category.
#
# These models are trained on X_train with NO labels.
# They build an internal model of "normal" and score each test
# transaction by how well it fits that model.
# ─────────────────────────────────────────

def run_isolation_forest(X_train: np.ndarray, X_test: np.ndarray,
                         y_test: np.ndarray) -> dict:
    """
    Isolation Forest — Point Anomaly Detector
    ------------------------------------------
    Builds an ensemble of random trees. Normal transactions require many
    splits to isolate; anomalous ones are isolated quickly (short path
    length). contamination=0.05 tells the model to expect ~5% anomalies,
    matching this dataset's known fraud rate.

    No labels used during training.
    """
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,   # expected anomaly rate
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)                      # learns normality from training data
    y_pred = _remap(model.predict(X_test))  # -1 → anomaly, 1 → normal
    _report("Isolation Forest", "POINT", y_test, y_pred)
    return model


def run_lof(X_train: np.ndarray, X_test: np.ndarray,
            y_test: np.ndarray) -> dict:
    """
    Local Outlier Factor — Point Anomaly Detector
    -----------------------------------------------
    Measures the local density of each transaction relative to its
    neighbours. A transaction in a sparse neighbourhood (far from similar
    transactions) scores as an anomaly. novelty=True enables scoring of
    new test points after fitting on training data.

    No labels used during training.
    """
    model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.05,
        novelty=True,         # required to call predict() on new data
        n_jobs=-1,
    )
    model.fit(X_train)
    y_pred = _remap(model.predict(X_test))
    return _report("Local Outlier Factor", "POINT", y_test, y_pred)


def run_unsupervised_rf(X_train: np.ndarray, X_test: np.ndarray,
                        y_test: np.ndarray) -> dict:
    """
    Unsupervised Random Forest — Point Anomaly Detector
    -----------------------------------------------------
    RF is traditionally supervised, but can detect anomalies without
    labels using the proximity / synthetic-complement method:

      1. Generate a synthetic "complement" dataset by randomly permuting
         each feature column independently. This destroys the real
         correlations while keeping marginal distributions intact.
      2. Label real transactions as class 0, synthetic as class 1.
      3. Train RF to distinguish real from synthetic.
      4. Extract the leaf co-occurrence proximity matrix: two transactions
         that land in the same leaf frequently are "close" to each other.
      5. Transactions with low average proximity to all others are outliers.

    This is the approach described in Breiman (2001) for unsupervised RF
    and referenced in IoT anomaly detection literature.

    No fraud labels used during training.
    """
    rng = np.random.default_rng(42)

    # Step 1 — build synthetic complement by column-wise permutation
    synthetic = np.column_stack([
        rng.permutation(X_train[:, col]) for col in range(X_train.shape[1])
    ])

    # Step 2 — combine and label: real=0, synthetic=1
    X_combined = np.vstack([X_train, synthetic])
    y_combined  = np.array([0] * len(X_train) + [1] * len(synthetic))

    # Step 3 — train RF to separate real from synthetic
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_combined, y_combined)

    # Step 4 — compute proximity: sample training set to avoid memory issues
    # with large datasets (full n_train × n_test matrix is too expensive)
    train_leaves = rf.apply(X_train)   # (n_train, n_trees)
    test_leaves  = rf.apply(X_test)    # (n_test,  n_trees)

    sample_size = min(1000, len(X_train))
    rng2 = np.random.default_rng(42)
    sample_idx  = rng2.choice(len(X_train), size=sample_size, replace=False)
    train_leaves_sample = train_leaves[sample_idx]

    proximity = np.mean(
        test_leaves[:, np.newaxis, :] == train_leaves_sample[np.newaxis, :, :],
        axis=2,
    )  # shape (n_test, sample_size)

    # Step 5 — outlier score = mean proximity to all training points
    # Low score → far from everything → anomaly
    mean_prox = proximity.mean(axis=1)

    # Threshold: flag bottom contamination% as anomalies
    threshold = np.percentile(mean_prox, 5)   # 5% contamination
    y_pred    = (mean_prox < threshold).astype(int)

    return _report("Unsupervised Random Forest (Proximity)", "POINT", y_test, y_pred)


# ─────────────────────────────────────────
# 4. Contextual Anomaly Detector
# ─────────────────────────────────────────
#
# A CONTEXTUAL ANOMALY is a transaction that is only suspicious within
# its specific context. A £500 spend is normal for Electronics but
# highly abnormal for Grocery. Looking at raw values alone would miss it.
#
# We compute per-group Z-scores using training data statistics only,
# then flag test transactions that deviate beyond a threshold within
# their own group.
# ─────────────────────────────────────────

def run_contextual_zscore(df_raw: pd.DataFrame, train_idx: np.ndarray,
                          test_idx: np.ndarray, y_test: np.ndarray,
                          z_threshold: float = 2.5) -> dict:
    """
    Contextual Z-score — Contextual Anomaly Detector
    --------------------------------------------------
    For each merchant_category, computes the mean and std of amount
    and hour using TRAINING rows only. Test transactions are then scored
    by how many standard deviations they fall from their own category mean.

    Flagging rule: anomaly if |z_amount| > threshold OR |z_hour| > threshold

    z_threshold=2.5 means a transaction must be more than 2.5 SDs from
    its category mean to be flagged. Lower = more sensitive, more false
    positives. Higher = more conservative, fewer false positives.

    No labels used to compute group statistics.
    """
    train_df = df_raw.iloc[train_idx].copy()
    test_df  = df_raw.iloc[test_idx].copy()

    # Compute group stats from training data only
    group_stats = train_df.groupby("merchant_category")[["amount", "hour"]].agg(["mean", "std"])
    group_stats.columns = ["amount_mean", "amount_std", "hour_mean", "hour_std"]
    group_stats["amount_std"] = group_stats["amount_std"].replace(0, 1)
    group_stats["hour_std"]   = group_stats["hour_std"].replace(0, 1)

    test_df = test_df.join(group_stats, on="merchant_category")

    test_df["z_amount"] = (test_df["amount"] - test_df["amount_mean"]) / test_df["amount_std"]
    test_df["z_hour"]   = (test_df["hour"]   - test_df["hour_mean"])   / test_df["hour_std"]

    y_pred = (
        (test_df["z_amount"].abs() > z_threshold) |
        (test_df["z_hour"].abs()   > z_threshold)
    ).astype(int).values

    return _report(
        f"Contextual Z-score (per merchant_category, z>{z_threshold})",
        "CONTEXTUAL", y_test, y_pred,
    )


# ─────────────────────────────────────────
# 5. Collective Anomaly Detector
# ─────────────────────────────────────────
#
# A COLLECTIVE ANOMALY is a transaction that is not unusual on its own
# but doesn't belong to any recognisable cluster of normal behaviour.
# DBSCAN groups transactions by density; points that cannot be assigned
# to any cluster are labelled noise (-1) and treated as anomalies.
# ─────────────────────────────────────────

def run_dbscan(X_train: np.ndarray, X_test: np.ndarray,
               y_test: np.ndarray,
               eps: float = 0.5, min_samples: int = 5) -> dict:
    """
    DBSCAN — Collective Anomaly Detector
    --------------------------------------
    Fits density clusters on training data. Test points are assigned
    to the nearest training core point's cluster. If a test point is
    farther than eps from every core point it is labelled noise → anomaly.

    eps         : radius of a point's neighbourhood (in scaled space).
                  Smaller = tighter clusters = more anomalies flagged.
    min_samples : minimum neighbours to be a core point.
                  Higher = only very dense regions form clusters.

    No labels used during training.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    train_labels = db.fit_predict(X_train)

    core_mask   = np.zeros(len(X_train), dtype=bool)
    core_mask[db.core_sample_indices_] = True
    core_points = X_train[core_mask]
    core_labels = train_labels[core_mask]

    if len(core_points) == 0:
        # No clusters found — flag everything
        y_pred = np.ones(len(y_test), dtype=int)
    else:
        nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
        nn.fit(core_points)
        distances, indices = nn.kneighbors(X_test)
        assigned = np.where(
            distances[:, 0] <= eps,
            core_labels[indices[:, 0]],
            -1,
        )
        y_pred = _remap(assigned)

    return _report(
        f"DBSCAN (eps={eps}, min_samples={min_samples})",
        "COLLECTIVE", y_test, y_pred,
    )


# ─────────────────────────────────────────
# 6. Supervised Models (for comparison)
# ─────────────────────────────────────────
#
# These models ARE given labels during training — included to replicate
# the paper methodology (supervised RF, 75/25 split) and provide a
# performance baseline against the unsupervised anomaly detectors above.
# ─────────────────────────────────────────

def run_supervised_rf(X_train, X_test, y_train, y_test) -> dict:
    """
    Supervised Random Forest — replicates the paper's methodology.
    Trained on labelled data (75%), tested on 25%.
    class_weight='balanced' handles the 95/5 class imbalance.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)          # labels ARE used here
    y_pred = model.predict(X_test)
    _report("Random Forest (Supervised)", "SUPERVISED", y_test, y_pred)
    return model


def run_logistic_regression(X_train, X_test, y_train, y_test) -> dict:
    """
    Logistic Regression — supervised baseline.
    Fits a linear decision boundary between fraud and legit classes.
    """
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    _report("Logistic Regression (Supervised)", "SUPERVISED", y_test, y_pred)
    return model


def run_knn(X_train, X_test, y_train, y_test) -> dict:
    """
    K-Nearest Neighbours — supervised baseline.
    Classifies each test point by majority vote of its 7 nearest
    training neighbours. Relies on scaled features for distance.
    """
    model = KNeighborsClassifier(n_neighbors=7, metric="euclidean", n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    _report("KNN k=7 (Supervised)", "SUPERVISED", y_test, y_pred)
    return model



# ─────────────────────────────────────────
# 7. Summary
# ─────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    grouped = {}
    for r in results:
        grouped.setdefault(r["type"], []).append(r)

    print("\n" + "="*72)
    print("  FULL COMPARISON SUMMARY  (sorted by F1 — anomaly/fraud class)")
    print("  Split: 75% train / 25% test (stratified)")
    print("="*72)

    for atype in ["SUPERVISED", "POINT", "CONTEXTUAL", "COLLECTIVE"]:
        if atype not in grouped:
            continue
        section = sorted(grouped[atype], key=lambda r: r["f1"], reverse=True)
        label = {
            "SUPERVISED":  "SUPERVISED  (labels used in training)",
            "POINT":       "POINT ANOMALIES  (no labels in training)",
            "CONTEXTUAL":  "CONTEXTUAL ANOMALIES  (no labels in training)",
            "COLLECTIVE":  "COLLECTIVE ANOMALIES  (no labels in training)",
        }[atype]
        print(f"\n  ── {label} ──")
        print(f"  {'Model':<48} {'Prec':>7} {'Rec':>7} {'F1':>7}")
        print("  " + "─"*69)
        for r in section:
            print(f"  {r['name']:<48} {r['precision']:>7.4f}"
                  f" {r['recall']:>7.4f} {r['f1']:>7.4f}")

    print("\n" + "="*72)
    print("  Precision/Recall/F1 are for the ANOMALY (fraud) class.")
    print("  Supervised models use labels during training.")
    print("  Anomaly detectors use labels only for evaluation.")
    print("="*72)

def save_artifacts(name, model, scaler, encoders, folder="saved_models"):
    os.makedirs(folder, exist_ok=True)

    joblib.dump(model, f"{folder}/{name}_model.pkl")
    joblib.dump(scaler, f"{folder}/{name}_scaler.pkl")
    joblib.dump(encoders, f"{folder}/{name}_encoders.pkl")

    print(f"[✓] Saved {name} artifacts to '{folder}/'")

    
# ─────────────────────────────────────────
# 8. Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = "Model Maker\Dataset\synthetic_fraud_dataset.csv"   # adjust path as needed

    print("Loading and preparing data...")
    X, y, df_raw, scaler, encoders = load_and_prepare(DATA_PATH)
    X_train, X_test, y_train, y_test, train_idx, test_idx = build_splits(X, y)
    
    # Example: Isolation Forest
    iso_model = run_isolation_forest(X_train, X_test, y_test)
    save_artifacts("isolation_forest", iso_model, scaler, encoders)

    # Supervised RF
    rf_model = run_supervised_rf(X_train, X_test, y_train, y_test)
    save_artifacts("rf_supervised", rf_model, scaler, encoders)
 
    knn_model = run_knn(X_train, X_test, y_train, y_test)
    save_artifacts("knn", knn_model, scaler, encoders)




    print(f"Split  : 75% train / 25% test (stratified)")
    print(f"Train  : {len(y_train)} rows  |  Test : {len(y_test)} rows")
    print(f"Fraud in test : {y_test.sum()} / {len(y_test)} ({y_test.mean()*100:.1f}%)\n")

    results = []

    # ── Supervised ───────────────────────────────────────────────────
    print("─"*58)
    print("  SUPERVISED MODELS  (labels used during training)")
    print("─"*58)
    results.append(run_supervised_rf(X_train, X_test, y_train, y_test))
    results.append(run_logistic_regression(X_train, X_test, y_train, y_test))
    results.append(run_knn(X_train, X_test, y_train, y_test))

    # ── Point Anomalies ──────────────────────────────────────────────
    print("\n" + "─"*58)
    print("  POINT ANOMALIES  (no labels during training)")
    print("  Flagging individual transactions globally out of range")
    print("─"*58)
    results.append(run_isolation_forest(X_train, X_test, y_test))
    results.append(run_lof(X_train, X_test, y_test))
    results.append(run_unsupervised_rf(X_train, X_test, y_test))

    # ── Contextual Anomalies ─────────────────────────────────────────
    print("\n" + "─"*58)
    print("  CONTEXTUAL ANOMALIES  (no labels during training)")
    print("  Flagging transactions unusual within their own category")
    print("─"*58)
    results.append(run_contextual_zscore(df_raw, train_idx, test_idx, y_test))

    # ── Collective Anomalies ─────────────────────────────────────────
    print("\n" + "─"*58)
    print("  COLLECTIVE ANOMALIES  (no labels during training)")
    print("  Flagging transactions that fit no normal cluster pattern")
    print("─"*58)
    results.append(run_dbscan(X_train, X_test, y_test))

    print_summary(results)