"""
Feature importance diagnostic for the acoustic flat-aggregate feature set,
run once on train only (never touches dev/test).

Three class-balanced scorers, since MELD's ~47% neutral class would
otherwise dominate an unweighted ranking and mask signal useful for the
rare classes (disgust/fear):
  - Information Gain (mutual information): sklearn's mutual_info_classif has
    no native class-weighting, so the balancing is done by resampling every
    class up to the majority-class count (with replacement) before scoring.
  - Random Forest importance: class_weight="balanced".
  - XGBoost importance: scale_pos_weight is binary-only; the multiclass
    equivalent is a per-sample weight (compute_sample_weight("balanced", y)),
    the same approach used for FocalLoss's class weights elsewhere in this
    project.

Reports whether GFCC dominates the rankings the way it did in prior
clean-corpus experiments (RAVDESS/EMOVO/SUBESCO/EMODB) -- a standalone check
of whether GFCC's noise-robustness claim holds on genuinely noisy TV audio.
"""

import os
import json

import numpy as np
import pandas as pd

import xgboost as xgb

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_sample_weight


EMBED_DIR = config.EMBEDDINGS_DIR

OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "acoustic_feature_importance")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = config.RANDOM_SEED


FAMILIES = [
    "mfcc_delta2", "mfcc_delta", "gfcc_delta2", "gfcc_delta",
    "mfcc", "gfcc", "chroma",
    "spectral_flux", "spectral_bandwidth", "spectral_centroid",
    "rms", "f0", "zcr"
]


def feature_family(flat_name):

    if flat_name.endswith("_mean"):
        base = flat_name[:-len("_mean")]
    else:
        base = flat_name[:-len("_std")]

    for family in sorted(FAMILIES, key=len, reverse=True):
        if base == family or base.startswith(family + "_"):
            return family

    return base


X_train = np.load(os.path.join(EMBED_DIR, "train_acoustic_flat.npy"))

train_meta = pd.read_csv(os.path.join(EMBED_DIR, "train_metadata.csv"))

with open(os.path.join(EMBED_DIR, "acoustic_feature_names.json")) as f:
    feature_names = json.load(f)["flat_features"]

assert X_train.shape[1] == len(feature_names)

encoder = LabelEncoder()
y_train = encoder.fit_transform(train_meta["emotion"])

print("Train acoustic features:", X_train.shape)
print("Classes:", list(encoder.classes_))
print("Class counts:", dict(zip(encoder.classes_, np.bincount(y_train))))


# ---------------------------------------------------------------------------
# 1. Information Gain (mutual information), class-balanced via resampling
# ---------------------------------------------------------------------------

majority_count = np.bincount(y_train).max()

balanced_X, balanced_y = [], []

for cls in np.unique(y_train):

    cls_mask = y_train == cls

    X_res, y_res = resample(
        X_train[cls_mask],
        y_train[cls_mask],
        replace=True,
        n_samples=majority_count,
        random_state=SEED
    )

    balanced_X.append(X_res)
    balanced_y.append(y_res)

balanced_X = np.concatenate(balanced_X)
balanced_y = np.concatenate(balanced_y)

print("\nComputing Information Gain (class-balanced via resampling)...")

mi_scores = mutual_info_classif(balanced_X, balanced_y, random_state=SEED)


# ---------------------------------------------------------------------------
# 2. Random Forest importance, class_weight="balanced"
# ---------------------------------------------------------------------------

print("Computing Random Forest importance (class_weight='balanced')...")

rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1
)

rf.fit(X_train, y_train)
rf_scores = rf.feature_importances_


# ---------------------------------------------------------------------------
# 3. XGBoost importance, class-balanced via sample_weight
# ---------------------------------------------------------------------------

print("Computing XGBoost importance (balanced sample_weight)...")

sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    objective="multi:softprob",
    num_class=len(encoder.classes_),
    random_state=SEED,
    n_jobs=-1,
    eval_metric="mlogloss"
)

xgb_model.fit(X_train, y_train, sample_weight=sample_weight)
xgb_scores = xgb_model.feature_importances_


# ---------------------------------------------------------------------------
# Consolidate
# ---------------------------------------------------------------------------

df = pd.DataFrame({
    "feature": feature_names,
    "family": [feature_family(name) for name in feature_names],
    "information_gain": mi_scores,
    "random_forest_importance": rf_scores,
    "xgboost_importance": xgb_scores
})

for col in ["information_gain", "random_forest_importance", "xgboost_importance"]:
    df[f"{col}_rank"] = df[col].rank(ascending=False).astype(int)

df["mean_rank"] = df[[
    "information_gain_rank",
    "random_forest_importance_rank",
    "xgboost_importance_rank"
]].mean(axis=1)

df = df.sort_values("mean_rank").reset_index(drop=True)

df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

print("\nTop 20 features by mean rank across all three scorers:")
print(df.head(20)[["feature", "family", "information_gain", "random_forest_importance", "xgboost_importance", "mean_rank"]].to_string(index=False))


# ---------------------------------------------------------------------------
# Per-family diagnostic: does GFCC dominate, as in prior clean-corpus work?
# ---------------------------------------------------------------------------

family_summary = df.groupby("family").agg(
    num_features=("feature", "count"),
    mean_information_gain=("information_gain", "mean"),
    mean_random_forest_importance=("random_forest_importance", "mean"),
    mean_xgboost_importance=("xgboost_importance", "mean"),
    mean_rank=("mean_rank", "mean"),
    top20_count=("mean_rank", lambda r: (r <= 20).sum())
).sort_values("mean_rank")

family_summary.to_csv(os.path.join(OUTPUT_DIR, "family_summary.csv"))

print("\nPer-family summary (sorted by mean rank, lower = more important):")
print(family_summary.to_string())

gfcc_families = {"gfcc", "gfcc_delta", "gfcc_delta2"}
mfcc_families = {"mfcc", "mfcc_delta", "mfcc_delta2"}

gfcc_rank = family_summary.loc[family_summary.index.isin(gfcc_families), "mean_rank"].mean()
mfcc_rank = family_summary.loc[family_summary.index.isin(mfcc_families), "mean_rank"].mean()

gfcc_dominates = bool(gfcc_rank < mfcc_rank)

print(f"\nGFCC family mean rank: {gfcc_rank:.1f}  |  MFCC family mean rank: {mfcc_rank:.1f}")
print(f"GFCC ranks above MFCC on this (noisy TV audio) diagnostic: {gfcc_dominates}")

with open(os.path.join(OUTPUT_DIR, "gfcc_diagnostic.json"), "w") as f:
    json.dump({
        "gfcc_family_mean_rank": float(gfcc_rank),
        "mfcc_family_mean_rank": float(mfcc_rank),
        "gfcc_ranks_above_mfcc": gfcc_dominates,
        "note": (
            "Prior clean-corpus experiments (RAVDESS/EMOVO/SUBESCO/EMODB) found "
            "GFCC dominating importance rankings due to its noise-robust Gammatone "
            "filterbank. This checks whether that holds on MELD's noisy TV audio "
            "(background score/laugh track), not just on clean speech."
        )
    }, f, indent=2)

print("\nOutputs saved to:", OUTPUT_DIR)
print("✓ feature_importance.csv")
print("✓ family_summary.csv")
print("✓ gfcc_diagnostic.json")
