"""
ClaimVision Explainable AI (SHAP) Module.
Computes TreeExplainer attributions, global beeswarm summaries, local waterfall plots,
and translates raw SHAP values into business narratives for healthcare fraud analysts.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import shap.explainers._tree

# Apply compatibility fix for XGBoost 3.x bracketed base_score in UBJSON
orig_decode = shap.explainers._tree.decode_ubjson_buffer

def _patched_decode_ubjson_buffer(fd):
    jmodel = orig_decode(fd)
    try:
        lmp = jmodel.get("learner", {}).get("learner_model_param", {})
        if "base_score" in lmp:
            bs = lmp["base_score"]
            if isinstance(bs, str) and bs.startswith("[") and bs.endswith("]"):
                lmp["base_score"] = bs[1:-1]
    except Exception:
        pass
    return jmodel

shap.explainers._tree.decode_ubjson_buffer = _patched_decode_ubjson_buffer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Configure headless matplotlib
plt.switch_backend("Agg")


class FraudExplainability:
    """Manages SHAP model explainability workflows."""

    def __init__(self, model_path: Optional[Path] = None):
        if model_path is None:
            model_path = Path(__file__).resolve().parent.parent / "models" / "xgboost_fraud_model.joblib"
        
        logger.info(f"Loading trained model artifact from {model_path}...")
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.feature_names = artifact["feature_names"]
        self.best_threshold = artifact.get("best_threshold", 0.5)
        self.explainer = shap.TreeExplainer(self.model)

    def generate_explanations(
        self,
        df_features: Optional[pd.DataFrame] = None,
        figures_dir: Optional[Path] = None,
        sample_size: int = 1500
    ) -> Dict[str, Any]:
        """
        Generate global and local SHAP explanations and save high-resolution figures.
        """
        if df_features is None:
            fe = FeatureEngineer()
            df_features = fe.compute_features()

        if figures_dir is None:
            figures_dir = Path(__file__).resolve().parent.parent / "reports" / "figures"
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

        provider_ids = df_features["provider_id"].values
        y_true = df_features["potential_fraud"].values
        X = df_features[self.feature_names]

        # Use sample for SHAP if dataset is large to maintain rapid execution
        if len(X) > sample_size:
            idx_sample = np.random.RandomState(42).choice(len(X), size=sample_size, replace=False)
            X_eval = X.iloc[idx_sample].copy()
            prov_eval = provider_ids[idx_sample]
            y_eval = y_true[idx_sample]
        else:
            X_eval = X.copy()
            prov_eval = provider_ids
            y_eval = y_true

        logger.info(f"Computing SHAP values on {len(X_eval)} provider feature records...")
        shap_explanation = self.explainer(X_eval)
        shap_values = shap_explanation.values

        # 1. Global Feature Importance Bar Chart
        logger.info("Generating SHAP feature importance plot...")
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs_shap
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

        plt.figure(figsize=(10, 8), dpi=150)
        top_15 = feature_importance_df.head(15).iloc[::-1]
        plt.barh(top_15["feature"], top_15["mean_abs_shap"], color="#1f77b4", edgecolor="none")
        plt.xlabel("Mean |SHAP Value| (Average Impact on Fraud Log-Odds)")
        plt.title("ClaimVision — Global Feature Importance (Top 15 Drivers)", fontsize=13, fontweight="bold")
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()
        bar_plot_path = figures_dir / "shap_feature_importance.png"
        plt.savefig(bar_plot_path)
        plt.close()

        # 2. Global Beeswarm Summary Plot
        logger.info("Generating SHAP summary beeswarm plot...")
        plt.figure(figsize=(11, 8), dpi=150)
        shap.summary_plot(shap_values, X_eval, feature_names=self.feature_names, show=False, max_display=15)
        plt.title("ClaimVision — SHAP Summary Beeswarm (Feature Impact Distribution)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        beeswarm_path = figures_dir / "shap_summary_beeswarm.png"
        plt.savefig(beeswarm_path)
        plt.close()

        # 3. Local Waterfall Explanations for Suspicious Cases
        logger.info("Generating local waterfall explanations for representative cases...")
        pred_probs = self.model.predict_proba(X_eval)[:, 1]
        
        # Case A: Top Flagged Fraudulent Provider (Highest Risk)
        high_risk_idx = np.argmax(pred_probs)
        high_risk_prov = prov_eval[high_risk_idx]
        plt.figure(figsize=(10, 6), dpi=150)
        shap.plots.waterfall(shap_explanation[high_risk_idx], max_display=10, show=False)
        plt.title(f"Provider {high_risk_prov} — Fraud Attribution Waterfall (P={pred_probs[high_risk_idx]:.3f})",
                  fontsize=12, fontweight="bold")
        plt.tight_layout()
        waterfall_high_path = figures_dir / f"shap_waterfall_{high_risk_prov}.png"
        plt.savefig(waterfall_high_path)
        plt.close()

        # 4. Generate Structured Natural Language Narrative for Analyst
        narrative = self.explain_provider_narrative(
            provider_id=high_risk_prov,
            feature_row=X_eval.iloc[high_risk_idx],
            shap_row=shap_values[high_risk_idx],
            prob=pred_probs[high_risk_idx],
            true_label=int(y_eval[high_risk_idx])
        )

        # Save Feature Importance JSON
        reports_dir = figures_dir.parent
        importance_json_path = reports_dir / "shap_feature_importance.json"
        feature_importance_df.to_json(importance_json_path, orient="records", indent=2)

        summary_report = {
            "top_features": feature_importance_df.head(10).to_dict(orient="records"),
            "high_risk_case": narrative,
            "artifacts_generated": [
                str(bar_plot_path.name),
                str(beeswarm_path.name),
                str(waterfall_high_path.name),
            ]
        }
        logger.info("Explainability generation complete.")
        return summary_report

    def explain_provider_narrative(
        self,
        provider_id: str,
        feature_row: pd.Series,
        shap_row: np.ndarray,
        prob: float,
        true_label: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Translates raw mathematical SHAP attributions into a concise business rationale
        suitable for non-technical claims adjudicators and SIU investigators.
        """
        risk_tier = "CRITICAL / HIGH" if prob >= self.best_threshold else "LOW / MONITORED"
        
        # Sort SHAP attributions
        feat_shap = sorted(
            zip(self.feature_names, feature_row.values, shap_row),
            key=lambda x: abs(x[2]),
            reverse=True
        )

        top_increasing = [item for item in feat_shap if item[2] > 0][:4]
        top_decreasing = [item for item in feat_shap if item[2] < 0][:2]

        explanation_bullets = []
        for feat, val, s_val in top_increasing:
            explanation_bullets.append(
                f"+ High Risk Driver: '{feat}' (value: {val:.2f}) increased log-odds by +{s_val:.3f}"
            )
        for feat, val, s_val in top_decreasing:
            explanation_bullets.append(
                f"- Mitigating Factor: '{feat}' (value: {val:.2f}) reduced log-odds by {s_val:.3f}"
            )

        return {
            "provider_id": provider_id,
            "predicted_fraud_probability": round(float(prob), 4),
            "assigned_risk_tier": risk_tier,
            "ground_truth_label": "Fraud" if true_label == 1 else "Non-Fraud" if true_label == 0 else "Unknown",
            "decision_rationale": explanation_bullets,
        }


if __name__ == "__main__":
    xai = FraudExplainability()
    report = xai.generate_explanations()
    print("\n=== SHAP EXPLANATION REPORT ===")
    print(json.dumps(report["high_risk_case"], indent=2))
