"""
ClaimVision Machine Learning & Fraud Detection Pipeline.

Trains and evaluates XGBoost classification models with:
- Provider-level group split (no provider crosses train/test boundary)
- Dynamic class-imbalance weighting (scale_pos_weight from training data only)
- Stratified cross-validation on training providers
- Threshold tuning
- Comprehensive metric logging
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, precision_recall_curve,
)
import xgboost as xgb

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def provider_group_split(
    df_features: pd.DataFrame,
    target_col: str = "potential_fraud",
    test_size: float = 0.20,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]:
    """
    Provider-level group split with approximately 80/20 train/test separation
    and class-distribution preservation.

    Each provider is assigned ENTIRELY to train OR test. No provider appears
    in both sets, preventing leakage of provider-specific billing patterns.

    The target label (PotentialFraud) lives at provider grain — one row per
    provider — so a stratified group split is achieved by shuffling fraud and
    non-fraud provider lists separately and taking the first test_size fraction
    of each as the test set.

    Returns:
        X_train, X_test, y_train, y_test, train_provider_ids, test_provider_ids
    """
    assert "provider_id" in df_features.columns, "provider_id column required"
    df = df_features.copy().reset_index(drop=True)

    rng = np.random.default_rng(random_seed)

    fraud_ids = df.loc[df[target_col] == 1, "provider_id"].tolist()
    clean_ids = df.loc[df[target_col] == 0, "provider_id"].tolist()

    rng.shuffle(fraud_ids)
    rng.shuffle(clean_ids)

    n_test_fraud = max(1, int(len(fraud_ids) * test_size))
    n_test_clean = max(1, int(len(clean_ids) * test_size))

    test_prov_ids = set(fraud_ids[:n_test_fraud]) | set(clean_ids[:n_test_clean])
    train_prov_ids = set(df["provider_id"]) - test_prov_ids

    # Critical leakage assertion
    assert train_prov_ids.isdisjoint(test_prov_ids), (
        "CRITICAL: Provider IDs overlap between train and test — data leakage detected!"
    )

    feature_cols = [c for c in df.columns if c not in ("provider_id", target_col)]
    train_df = df[df["provider_id"].isin(train_prov_ids)].reset_index(drop=True)
    test_df  = df[df["provider_id"].isin(test_prov_ids)].reset_index(drop=True)

    X_train = train_df[feature_cols]
    X_test  = test_df[feature_cols]
    y_train = train_df[target_col].astype(int)
    y_test  = test_df[target_col].astype(int)

    logger.info(
        "Provider-level group split complete:\n"
        "  Train providers: %d (Fraud=%d, Non-fraud=%d, Fraud rate=%.3f)\n"
        "  Test  providers: %d (Fraud=%d, Non-fraud=%d, Fraud rate=%.3f)\n"
        "  Provider disjoint check: PASSED",
        len(train_prov_ids), y_train.sum(), len(y_train) - y_train.sum(), y_train.mean(),
        len(test_prov_ids),  y_test.sum(),  len(y_test) - y_test.sum(),  y_test.mean(),
    )

    return X_train, X_test, y_train, y_test, list(train_prov_ids), list(test_prov_ids)


class FraudModelPipeline:
    """
    End-to-end ML pipeline for healthcare provider fraud classification.

    Uses a provider-level group split with approximately 80/20 train/test separation
    and class-distribution preservation. No provider appears in both train and test.
    """

    def __init__(
        self,
        random_seed: int = 42,
        test_size: float = 0.20,
        n_folds: int = 5,
        target_col: str = "potential_fraud",
    ):
        self.random_seed = random_seed
        self.test_size = test_size
        self.n_folds = n_folds
        self.target_col = target_col
        self.scaler = StandardScaler()
        self.model = None
        self.best_threshold = 0.35
        self.feature_names: List[str] = []
        self.metrics: Dict[str, Any] = {}
        self.train_provider_ids: List[str] = []
        self.test_provider_ids: List[str] = []

    def prepare_data(
        self, df_features: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Apply provider-level group split.
        Each provider is assigned entirely to train OR test.
        The provider-disjoint constraint is enforced via assertion.
        """
        logger.info("Preparing feature matrices with provider-level group split...")
        self.feature_names = [
            c for c in df_features.columns
            if c not in ("provider_id", self.target_col)
        ]

        X_train, X_test, y_train, y_test, train_ids, test_ids = provider_group_split(
            df_features,
            target_col=self.target_col,
            test_size=self.test_size,
            random_seed=self.random_seed,
        )
        self.train_provider_ids = train_ids
        self.test_provider_ids = test_ids

        # Enforced assertion
        assert set(train_ids).isdisjoint(set(test_ids)), (
            "Provider IDs must be disjoint between train and test"
        )

        return X_train, X_test, y_train, y_test

    def train_and_evaluate_baselines(
        self,
        X_train: pd.DataFrame, X_test: pd.DataFrame,
        y_train: pd.Series, y_test: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """Train Dummy and Logistic Regression baselines for benchmark comparison."""
        logger.info("Training baseline models for benchmark comparison...")
        baselines = {}

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        dummy_preds = dummy.predict(X_test)
        baselines["Dummy"] = {
            "Precision": float(precision_score(y_test, dummy_preds, zero_division=0)),
            "Recall":    float(recall_score(y_test, dummy_preds, zero_division=0)),
            "F1":        float(f1_score(y_test, dummy_preds, zero_division=0)),
            "ROC-AUC":   0.50,
            "PR-AUC":    float(average_precision_score(y_test, [y_train.mean()] * len(y_test))),
        }

        X_tr_scaled = self.scaler.fit_transform(X_train)
        X_te_scaled = self.scaler.transform(X_test)
        lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=self.random_seed)
        lr.fit(X_tr_scaled, y_train)
        lr_probs = lr.predict_proba(X_te_scaled)[:, 1]
        lr_preds = (lr_probs >= 0.5).astype(int)
        baselines["LogisticRegression"] = {
            "Precision": float(precision_score(y_test, lr_preds, zero_division=0)),
            "Recall":    float(recall_score(y_test, lr_preds, zero_division=0)),
            "F1":        float(f1_score(y_test, lr_preds, zero_division=0)),
            "ROC-AUC":   float(roc_auc_score(y_test, lr_probs)),
            "PR-AUC":    float(average_precision_score(y_test, lr_probs)),
        }
        logger.info("Baselines evaluated: %s", baselines)
        return baselines

    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
        """
        Train XGBoost classifier.

        scale_pos_weight is derived from the negative-to-positive provider ratio
        in the TRAINING DATA ONLY — not the test set, not the full dataset.
        This prevents any leakage of class distribution from test providers.

        The fraud-positive provider class is imbalanced (~9.35% base rate), so
        scale_pos_weight corrects the gradient updates proportionally.
        """
        logger.info("Configuring and training XGBoost Classifier...")
        num_neg = int((y_train == 0).sum())
        num_pos = int((y_train == 1).sum())
        scale_pos_weight = num_neg / max(num_pos, 1)
        logger.info(
            "Dynamic scale_pos_weight: %.4f (Training providers: Neg=%d, Pos=%d)",
            scale_pos_weight, num_neg, num_pos
        )

        xgb_clf = xgb.XGBClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.80,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=self.random_seed,
            n_jobs=-1,
            verbosity=0,
        )

        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
        cv_results = cross_validate(
            xgb_clf, X_train, y_train, cv=cv,
            scoring=["roc_auc", "recall", "f1"],
            return_train_score=False,
        )
        logger.info(
            "%d-Fold CV Results: ROC-AUC=%.4f, Recall=%.4f, F1=%.4f",
            self.n_folds,
            cv_results["test_roc_auc"].mean(),
            cv_results["test_recall"].mean(),
            cv_results["test_f1"].mean(),
        )

        xgb_clf.fit(X_train, y_train)
        self.model = xgb_clf
        self.metrics["scale_pos_weight"] = round(scale_pos_weight, 4)
        return xgb_clf

    def tune_threshold(self, X_train: pd.DataFrame, y_train: pd.Series) -> float:
        """Tune classification threshold on training providers for optimal F1."""
        train_probs = self.model.predict_proba(X_train)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_train, train_probs)
        f1_scores = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-9)
        best_idx = np.argmax(f1_scores[:-1])
        self.best_threshold = float(thresholds[best_idx])
        logger.info(
            "Optimized operational threshold: %.4f "
            "(Train Precision: %.4f, Recall: %.4f, F1: %.4f)",
            self.best_threshold, precisions[best_idx], recalls[best_idx], f1_scores[best_idx]
        )
        return self.best_threshold

    def evaluate_on_test(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate trained XGBoost model on holdout test providers."""
        logger.info("Evaluating XGBoost model on holdout test providers...")
        probs = self.model.predict_proba(X_test)[:, 1]

        roc_auc = float(roc_auc_score(y_test, probs))
        pr_auc  = float(average_precision_score(y_test, probs))

        preds_50    = (probs >= 0.50).astype(int)
        preds_tuned = (probs >= self.best_threshold).astype(int)
        f1_tuned    = float(f1_score(y_test, preds_tuned, zero_division=0))

        logger.info(
            "Test Set Results: ROC-AUC=%.4f, PR-AUC=%.4f, Tuned F1=%.4f, Tuned Recall=%.4f",
            roc_auc, pr_auc, f1_tuned,
            float(recall_score(y_test, preds_tuned, zero_division=0))
        )

        metrics = {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "default_threshold_0.50": {
                "precision": float(precision_score(y_test, preds_50, zero_division=0)),
                "recall":    float(recall_score(y_test, preds_50, zero_division=0)),
                "f1_score":  float(f1_score(y_test, preds_50, zero_division=0)),
                "confusion_matrix": confusion_matrix(y_test, preds_50).tolist(),
            },
            "tuned_threshold": {
                "threshold_value": self.best_threshold,
                "precision": float(precision_score(y_test, preds_tuned, zero_division=0)),
                "recall":    float(recall_score(y_test, preds_tuned, zero_division=0)),
                "f1_score":  f1_tuned,
                "confusion_matrix": confusion_matrix(y_test, preds_tuned).tolist(),
            },
        }
        self.metrics.update(metrics)
        return metrics

    def save_model(self, model_dir: Optional[Path] = None) -> Path:
        """Serialize model, scaler, feature list, split metadata, and metrics."""
        if model_dir is None:
            model_dir = Path(__file__).resolve().parent.parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "xgboost_fraud_model.joblib"
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "threshold": self.best_threshold,
            "train_provider_ids": self.train_provider_ids,
            "test_provider_ids": self.test_provider_ids,
            "metrics": self.metrics,
        }, model_path)
        logger.info("Serialized model artifact to %s", model_path)

        metrics_path = Path(__file__).resolve().parent.parent / "reports" / "model_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        logger.info("Saved metric results to %s", metrics_path)
        return model_path

    def run(self, df_features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Execute the full ML pipeline end-to-end."""
        if df_features is None:
            fe = FeatureEngineer()
            df_features = fe.compute_features()

        X_train, X_test, y_train, y_test = self.prepare_data(df_features)
        self.train_and_evaluate_baselines(X_train, X_test, y_train, y_test)
        self.train_xgboost(X_train, y_train)
        self.tune_threshold(X_train, y_train)
        metrics = self.evaluate_on_test(X_test, y_test)
        self.save_model()

        print("\n=== FINAL EVALUATION METRICS ===")
        print(json.dumps(metrics, indent=2))
        return metrics


if __name__ == "__main__":
    pipeline = FraudModelPipeline()
    pipeline.run()
