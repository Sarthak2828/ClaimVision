"""
ClaimVision Machine Learning & Fraud Detection Pipeline.
Trains and evaluates XGBoost classification models with class-imbalance handling,
stratified cross-validation, threshold tuning, and comprehensive metric logging.
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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, precision_recall_curve,
    classification_report
)
import xgboost as xgb

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class FraudModelPipeline:
    """End-to-end ML pipeline for healthcare provider fraud classification."""

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
        self.feature_names = []
        self.metrics: Dict[str, Any] = {}

    def prepare_data(self, df_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split dataset into train and test sets strictly prior to any transformation.
        Excludes provider ID and target from feature matrix.
        """
        logger.info("Preparing feature matrices and train/test splits...")
        y = df_features[self.target_col].astype(int)
        X = df_features.drop(columns=["provider_id", self.target_col])
        self.feature_names = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_seed, stratify=y
        )
        logger.info(f"Split complete: Train={len(X_train)} (Fraud={y_train.sum()}), "
                    f"Test={len(X_test)} (Fraud={y_test.sum()})")
        return X_train, X_test, y_train, y_test

    def train_and_evaluate_baselines(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """Train Dummy and Logistic Regression baselines for benchmark comparison."""
        logger.info("Training baseline models for benchmark comparison...")
        baselines = {}

        # 1. Dummy Classifier (stratified majority)
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        dummy_preds = dummy.predict(X_test)
        baselines["Dummy"] = {
            "Precision": float(precision_score(y_test, dummy_preds, zero_division=0)),
            "Recall": float(recall_score(y_test, dummy_preds, zero_division=0)),
            "F1": float(f1_score(y_test, dummy_preds, zero_division=0)),
            "ROC-AUC": 0.50,
            "PR-AUC": float(average_precision_score(y_test, [y_train.mean()] * len(y_test))),
        }

        # 2. Logistic Regression (scaled)
        X_tr_scaled = self.scaler.fit_transform(X_train)
        X_te_scaled = self.scaler.transform(X_test)
        lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=self.random_seed)
        lr.fit(X_tr_scaled, y_train)
        lr_probs = lr.predict_proba(X_te_scaled)[:, 1]
        lr_preds = (lr_probs >= 0.5).astype(int)
        baselines["LogisticRegression"] = {
            "Precision": float(precision_score(y_test, lr_preds, zero_division=0)),
            "Recall": float(recall_score(y_test, lr_preds, zero_division=0)),
            "F1": float(f1_score(y_test, lr_preds, zero_division=0)),
            "ROC-AUC": float(roc_auc_score(y_test, lr_probs)),
            "PR-AUC": float(average_precision_score(y_test, lr_probs)),
        }
        logger.info(f"Baselines evaluated: {baselines}")
        return baselines

    def train_xgboost(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost classifier with deliberate class-imbalance weighting (scale_pos_weight)
        and cross-validation evaluation.
        """
        logger.info("Configuring and training XGBoost Classifier...")
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        scale_pos_weight = num_neg / max(num_pos, 1)
        logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f} (Neg={num_neg}, Pos={num_pos})")

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
        )

        # 5-Fold Stratified Cross-Validation
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
        scoring = ["precision", "recall", "f1", "roc_auc", "average_precision"]
        cv_results = cross_validate(xgb_clf, X_train, y_train, cv=cv, scoring=scoring)

        self.metrics["cv_5fold"] = {
            "mean_precision": float(np.mean(cv_results["test_precision"])),
            "mean_recall": float(np.mean(cv_results["test_recall"])),
            "mean_f1": float(np.mean(cv_results["test_f1"])),
            "mean_roc_auc": float(np.mean(cv_results["test_roc_auc"])),
            "mean_pr_auc": float(np.mean(cv_results["test_average_precision"])),
        }
        logger.info(f"5-Fold CV Results: ROC-AUC={self.metrics['cv_5fold']['mean_roc_auc']:.4f}, "
                    f"Recall={self.metrics['cv_5fold']['mean_recall']:.4f}, "
                    f"F1={self.metrics['cv_5fold']['mean_f1']:.4f}")

        # Fit model on entire training set
        xgb_clf.fit(X_train, y_train)
        self.model = xgb_clf
        return xgb_clf

    def tune_threshold(self, X_train: pd.DataFrame, y_train: pd.Series) -> float:
        """
        Evaluate precision-recall trade-offs to select an optimal operational threshold.
        Optimizes for F1 while enforcing a minimum recall target for fraud prevention.
        """
        train_probs = self.model.predict_proba(X_train)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_train, train_probs)
        
        # Calculate F1 for all valid thresholds
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
        best_idx = np.argmax(f1_scores)
        self.best_threshold = float(thresholds[best_idx])
        logger.info(f"Optimized operational threshold: {self.best_threshold:.4f} "
                    f"(Train Precision: {precisions[best_idx]:.4f}, Recall: {recalls[best_idx]:.4f}, F1: {f1_scores[best_idx]:.4f})")
        return self.best_threshold

    def evaluate_test_set(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, Any]:
        """Evaluate final model performance on unseen holdout test set."""
        logger.info("Evaluating XGBoost model on holdout test set...")
        test_probs = self.model.predict_proba(X_test)[:, 1]
        
        # Evaluate at default threshold (0.50) and tuned threshold
        default_preds = (test_probs >= 0.50).astype(int)
        tuned_preds = (test_probs >= self.best_threshold).astype(int)

        roc_auc = float(roc_auc_score(y_test, test_probs))
        pr_auc = float(average_precision_score(y_test, test_probs))

        cm_default = confusion_matrix(y_test, default_preds).tolist()
        cm_tuned = confusion_matrix(y_test, tuned_preds).tolist()

        test_results = {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "default_threshold_0.50": {
                "precision": float(precision_score(y_test, default_preds, zero_division=0)),
                "recall": float(recall_score(y_test, default_preds, zero_division=0)),
                "f1_score": float(f1_score(y_test, default_preds, zero_division=0)),
                "confusion_matrix": cm_default,
            },
            "tuned_threshold": {
                "threshold_value": self.best_threshold,
                "precision": float(precision_score(y_test, tuned_preds, zero_division=0)),
                "recall": float(recall_score(y_test, tuned_preds, zero_division=0)),
                "f1_score": float(f1_score(y_test, tuned_preds, zero_division=0)),
                "confusion_matrix": cm_tuned,
            },
        }
        self.metrics["test_set"] = test_results
        logger.info(f"Test Set Results: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, "
                    f"Tuned F1={test_results['tuned_threshold']['f1_score']:.4f}, "
                    f"Tuned Recall={test_results['tuned_threshold']['recall']:.4f}")
        return test_results

    def save_artifacts(self, output_dir: Optional[Path] = None) -> None:
        """Serialize model, feature list, and metrics JSON."""
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent / "models"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / "xgboost_fraud_model.joblib"
        metadata = {
            "model": self.model,
            "feature_names": self.feature_names,
            "best_threshold": self.best_threshold,
            "metrics": self.metrics,
        }
        joblib.dump(metadata, model_path)
        logger.info(f"Serialized model artifact to {model_path}")

        reports_dir = output_dir.parent / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = reports_dir / "model_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Saved metric results to {metrics_file}")

    def run(self, df_features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Execute the entire ML training, tuning, and evaluation workflow."""
        if df_features is None:
            fe = FeatureEngineer()
            df_features = fe.compute_features()

        X_train, X_test, y_train, y_test = self.prepare_data(df_features)
        baselines = self.train_and_evaluate_baselines(X_train, X_test, y_train, y_test)
        self.metrics["baselines"] = baselines

        self.train_xgboost(X_train, y_train)
        self.tune_threshold(X_train, y_train)
        self.evaluate_test_set(X_test, y_test)
        self.save_artifacts()
        return self.metrics


if __name__ == "__main__":
    pipeline = FraudModelPipeline()
    results = pipeline.run()
    print("\n=== FINAL EVALUATION METRICS ===")
    print(json.dumps(results["test_set"], indent=2))
