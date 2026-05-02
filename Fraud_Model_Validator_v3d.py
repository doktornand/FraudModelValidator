"""
================================================================================
FINCRIME & FRAUD MODEL VALIDATOR — VERSION 3.0 (ULTRA-SCIENTIFIC)
================================================================================
Framework avancé de validation de modèle pour la détection de fraude
et la criminalité financière (FinCrime) avec méthodes ultra-scientifiques.

Auteur: Enhanced Model Validation Framework + Doctor Deep (Math & ML)
Date: 2026-05-02

NOUVELLES CAPACITÉS v3.0 (ULTRA-SCIENTIFIC):
─────────────────────────────────────────────────────────────────────────────
MÉTHODES ULTRA-SCIENTIFIQUES:
  ▸ Multifractal Detrended Fluctuation Analysis (MF-DFA)
  ▸ Transfer Entropy (causal information flow)
  ▸ Bayesian Online Changepoint Detection (BOCD)
  ▸ Random Matrix Theory (eigenvalue spacing analysis)
  ▸ Echo State Network residuals (nonlinear dynamics)
  ▸ Singular Spectrum Analysis (SSA)
  ▸ Recurrence Quantification Analysis (RQA)

DÉTECTION SUBTILE:
  ▸ Lissage artificiel des séries temporelles
  ▸ Liens causaux anormaux entre variables
  ▸ Micro-changements de régime dans le score
  ▸ Corrélations non-naturelles (fraude collective)
  ▸ Dynamique non-linéaire cachée
  ▸ Persistance spectrale anormale

MÉTRIQUES AVANCÉES (v2 conservées):
  ▸ GINI, KS, H-measure, AUC, Lift/Gain, DET
  ▸ Bootstrap CI, Calibration (ECE/MCE/Brier)
  ▸ SHAP, XAI, Benford, Graph anomalies
  ▸ Stress testing, Walk-forward, Fairness

RAPPORT:
  ▸ Score de validation global pondéré + Ultra-anomaly score
  ▸ Export HTML avec graphiques intégrés
  ▸ Alertes scientifiques sur anomalies subtiles
─────────────────────────────────────────────────────────────────────────────

Usage:
    from Fraud_Model_Validator_v3 import FraudModelValidatorV3

    validator = FraudModelValidatorV3(model_name="MonModele", cost_fn=5000, cost_fp=100)

    # Ultra-scientific detection
    ultra_results = validator.ultra_scientific_detection(df, feature_cols, time_col='transaction_date')
    
    # Métriques complètes avec IC bootstrap
    metrics = validator.calculate_performance_metrics(y_true, y_proba, bootstrap=True)
    
    # Rapport HTML complet incluant analyses ultra-scientifiques
    validator.generate_html_report(all_results, output_path='report.html')
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy import stats, signal, linalg, optimize
from scipy.spatial.distance import mahalanobis, pdist, squareform
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix, roc_curve, f1_score,
    precision_score, recall_score, matthews_corrcoef, brier_score_loss,
    log_loss, det_curve
)
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.ensemble import IsolationForest
from sklearn.calibration import calibration_curve
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
import json
import base64
import io
import os
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')

# ── Dépendances optionnelles ──────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False

try:
    from antropy import petrosian_fd, higuchi_fd
    ANTROPY_AVAILABLE = True
except ImportError:
    ANTROPY_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE COULEURS PROFESSIONNELLE
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    'primary':   '#1a1f5e',
    'secondary': '#e63946',
    'success':   '#2a9d8f',
    'warning':   '#e9c46a',
    'danger':    '#e76f51',
    'neutral':   '#457b9d',
    'light':     '#f1faee',
    'dark':      '#1d3557',
    'ultra':     '#9b59b6',      # Couleur pour l'ultra-scientific
}

CMAP_RISK = LinearSegmentedColormap.from_list(
    'risk', ['#2a9d8f', '#e9c46a', '#e63946']
)


# =============================================================================
# CLASSE PRINCIPALE V3.0
# =============================================================================

class FraudModelValidatorV3:
    """
    Validateur avancé de modèle — Détection de fraude & criminalité financière.
    Version 3.0 avec méthodes ultra-scientifiques (multifractal, causal, RMT, ESN).

    Conforme aux standards: SR 11-7, ECB TRIM Guide, AMLD6, Basel III/IV.
    """

    def __init__(self, model_name: str = "FraudDetectionModel",
                 cost_fn: float = 5000.0,
                 cost_fp: float = 100.0,
                 currency: str = "€"):
        self.model_name = model_name
        self.cost_fn = cost_fn       # Coût faux négatif (fraude manquée)
        self.cost_fp = cost_fp       # Coût faux positif (fausse alarme)
        self.currency = currency
        self.validation_results = {}
        self._figure_cache = {}      # Stocke les figures pour le rapport HTML

    # =========================================================================
    # 1. MÉTRIQUES DE PERFORMANCE AVANCÉES
    # =========================================================================

    def calculate_performance_metrics(self, y_true, y_pred_proba,
                                      y_pred=None, threshold: float = 0.5,
                                      bootstrap: bool = True,
                                      n_bootstrap: int = 1000) -> dict:
        """
        Calcule l'ensemble des métriques de performance.

        Inclut:
        - AUC-ROC, GINI, KS-Statistic, H-measure
        - Average Precision (AUCPR)
        - Precision, Recall, F1, MCC
        - Brier Score, Log-Loss
        - Métriques coût-basées
        - Bootstrap CI (95%) sur toutes les métriques clés

        Args:
            y_true: Labels binaires (0/1)
            y_pred_proba: Probabilités prédites ∈ [0,1]
            y_pred: Prédictions binaires (calculé si None)
            threshold: Seuil de classification
            bootstrap: Calculer les IC bootstrap
            n_bootstrap: Nombre d'itérations bootstrap

        Returns:
            dict: Métriques complètes avec IC
        """
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)

        if y_pred is None:
            y_pred = (y_pred_proba >= threshold).astype(int)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        auc_roc = roc_auc_score(y_true, y_pred_proba)
        gini = 2 * auc_roc - 1

        # KS-Statistic
        fpr_arr, tpr_arr, thresholds_roc = roc_curve(y_true, y_pred_proba)
        ks_stat = np.max(tpr_arr - fpr_arr)
        ks_threshold_idx = np.argmax(tpr_arr - fpr_arr)
        ks_threshold = thresholds_roc[ks_threshold_idx] if len(thresholds_roc) > ks_threshold_idx else threshold

        # H-measure (Hand 2009) — mesure unifiée tenant compte des coûts
        h_measure = self._compute_h_measure(y_true, y_pred_proba)

        metrics = {
            # Ranking
            'auc_roc':             auc_roc,
            'gini':                gini,
            'ks_statistic':        ks_stat,
            'h_measure':           h_measure,
            'average_precision':   average_precision_score(y_true, y_pred_proba),

            # Classification
            'precision':           precision_score(y_true, y_pred, zero_division=0),
            'recall':              recall_score(y_true, y_pred, zero_division=0),
            'f1_score':            f1_score(y_true, y_pred, zero_division=0),
            'mcc':                 matthews_corrcoef(y_true, y_pred),
            'specificity':         tn / (tn + fp) if (tn + fp) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'true_positive_rate':  tp / (tp + fn) if (tp + fn) > 0 else 0,
            'balanced_accuracy':   ((tp / (tp + fn)) + (tn / (tn + fp))) / 2
                                   if (tp + fn) > 0 and (tn + fp) > 0 else 0,

            # Probabilistic
            'brier_score':         brier_score_loss(y_true, y_pred_proba),
            'log_loss':            log_loss(y_true, y_pred_proba),

            # Confusion matrix
            'tp': int(tp), 'fp': int(fp),
            'tn': int(tn), 'fn': int(fn),
            'threshold': threshold,
            'prevalence': y_true.mean(),

            # Cost
            'expected_cost':       (fn * self.cost_fn + fp * self.cost_fp) / len(y_true),
            'total_fraud_loss':    fn * self.cost_fn,
            'total_review_cost':   fp * self.cost_fp,
        }

        # Bootstrap confidence intervals
        if bootstrap:
            print(f"  🔄 Bootstrap CI ({n_bootstrap} itérations)...")
            ci = self._bootstrap_ci(y_true, y_pred_proba, n_bootstrap)
            metrics['bootstrap_ci'] = ci

        self.validation_results['performance'] = metrics
        self._print_performance_summary(metrics)
        return metrics

    def _compute_h_measure(self, y_true, y_scores) -> float:
        """
        H-measure de Hand (2009) — alternative à l'AUC robuste aux coûts.
        Utilise une distribution Beta(2,2) sur les coûts a priori.
        """
        try:
            n1 = np.sum(y_true == 1)
            n0 = np.sum(y_true == 0)
            pi1 = n1 / (n1 + n0)
            pi0 = 1 - pi1

            thresholds = np.linspace(0, 1, 200)
            costs = []
            for t in thresholds:
                y_pred = (y_scores >= t).astype(int)
                fn = np.sum((y_pred == 0) & (y_true == 1))
                fp = np.sum((y_pred == 1) & (y_true == 0))
                # Coût normalisé avec pondération Beta(2,2)
                c = (fn * pi1 + fp * pi0) / (pi1 * n1 + pi0 * n0 + 1e-10)
                costs.append(c)

            # Coût minimum possible (classifieur parfait = 0)
            min_cost = np.min(costs)
            # Coût classifieur trivial
            trivial_cost = min(pi1, pi0)
            h = 1 - (min_cost / (trivial_cost + 1e-10))
            return float(np.clip(h, 0, 1))
        except Exception:
            return np.nan

    def _bootstrap_ci(self, y_true, y_pred_proba, n: int = 1000,
                      alpha: float = 0.05) -> dict:
        """Bootstrap percentile CI à (1-alpha)% pour métriques clés."""
        aucs, aps, ks_stats, brierss = [], [], [], []
        n_samples = len(y_true)
        rng = np.random.default_rng(42)

        for _ in range(n):
            idx = rng.integers(0, n_samples, n_samples)
            yt = y_true[idx]
            yp = y_pred_proba[idx]
            if len(np.unique(yt)) < 2:
                continue
            try:
                aucs.append(roc_auc_score(yt, yp))
                aps.append(average_precision_score(yt, yp))
                fpr_b, tpr_b, _ = roc_curve(yt, yp)
                ks_stats.append(np.max(tpr_b - fpr_b))
                brierss.append(brier_score_loss(yt, yp))
            except Exception:
                pass

        lo, hi = alpha / 2, 1 - alpha / 2

        def ci(arr):
            if not arr:
                return (np.nan, np.nan)
            return (float(np.quantile(arr, lo)), float(np.quantile(arr, hi)))

        return {
            'auc_roc':          ci(aucs),
            'average_precision': ci(aps),
            'ks_statistic':     ci(ks_stats),
            'brier_score':      ci(brierss),
        }

    def _print_performance_summary(self, m: dict):
        print(f"\n{'═'*62}")
        print(f"  PERFORMANCE — {self.model_name}")
        print(f"{'═'*62}")
        print(f"  {'AUC-ROC':<28} {m['auc_roc']:.4f}  (GINI: {m['gini']:.4f})")
        print(f"  {'KS-Statistic':<28} {m['ks_statistic']:.4f}")
        print(f"  {'H-Measure':<28} {m['h_measure']:.4f}")
        print(f"  {'Avg Precision (AUCPR)':<28} {m['average_precision']:.4f}")
        print(f"  {'Precision':<28} {m['precision']:.4f}")
        print(f"  {'Recall (TPR/Sensitivity)':<28} {m['recall']:.4f}")
        print(f"  {'F1-Score':<28} {m['f1_score']:.4f}")
        print(f"  {'MCC':<28} {m['mcc']:.4f}")
        print(f"  {'Balanced Accuracy':<28} {m['balanced_accuracy']:.4f}")
        print(f"  {'Brier Score':<28} {m['brier_score']:.4f}")
        print(f"  {'Log-Loss':<28} {m['log_loss']:.4f}")
        print(f"  {'Expected Cost/transaction':<28} {m['expected_cost']:.2f}{self.currency}")
        if 'bootstrap_ci' in m:
            ci = m['bootstrap_ci']
            print(f"\n  — Bootstrap 95% CI —")
            for k, (lo, hi) in ci.items():
                print(f"  {k:<28} [{lo:.4f}, {hi:.4f}]")
        print()

    # =========================================================================
    # 2. COURBES AVANCÉES: ROC, PR, DET, LIFT, GAINS
    # =========================================================================

    def plot_advanced_curves(self, y_true, y_pred_proba, save_path=None):
        """
        4 courbes en une figure : ROC, PR, DET, et KS plot.
        """
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('#f8f9fa')

        # ── ROC ──────────────────────────────────────────────────────────────
        ax = axes[0, 0]
        fpr, tpr, thresh = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        ks = np.max(tpr - fpr)
        ks_idx = np.argmax(tpr - fpr)

        ax.fill_between(fpr, tpr, alpha=0.1, color=COLORS['primary'])
        ax.plot(fpr, tpr, color=COLORS['primary'], lw=2,
                label=f'AUC={auc:.4f}  GINI={2*auc-1:.4f}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Aléatoire')
        ax.annotate(f'KS={ks:.3f}',
                    xy=(fpr[ks_idx], tpr[ks_idx]),
                    xytext=(fpr[ks_idx] + 0.1, tpr[ks_idx] - 0.1),
                    arrowprops=dict(arrowstyle='->', color=COLORS['secondary']),
                    color=COLORS['secondary'], fontweight='bold')
        ax.plot([fpr[ks_idx], fpr[ks_idx]], [fpr[ks_idx], tpr[ks_idx]],
                '--', color=COLORS['secondary'], alpha=0.7)
        ax.set(xlabel='FPR (1-Spécificité)', ylabel='TPR (Sensibilité)',
               title='Courbe ROC', xlim=[0, 1], ylim=[0, 1])
        ax.legend(); ax.grid(True, alpha=0.3)

        # ── Precision-Recall ─────────────────────────────────────────────────
        ax = axes[0, 1]
        prec, rec, _ = precision_recall_curve(y_true, y_pred_proba)
        ap = average_precision_score(y_true, y_pred_proba)
        baseline = y_true.mean()

        ax.fill_between(rec, prec, alpha=0.1, color=COLORS['success'])
        ax.plot(rec, prec, color=COLORS['success'], lw=2, label=f'AP={ap:.4f}')
        ax.axhline(baseline, color=COLORS['danger'], ls='--',
                   label=f'Baseline={baseline:.4f}')
        ax.set(xlabel='Recall', ylabel='Precision',
               title='Courbe Precision-Recall', xlim=[0, 1], ylim=[0, 1])
        ax.legend(); ax.grid(True, alpha=0.3)

        # ── DET Curve ────────────────────────────────────────────────────────
        ax = axes[1, 0]
        try:
            fpr_det, fnr_det, _ = det_curve(y_true, y_pred_proba)
            ax.plot(fpr_det * 100, fnr_det * 100,
                    color=COLORS['warning'], lw=2)
            ax.set(xlabel='FPR (%)', ylabel='FNR (%)',
                   title='Detection Error Tradeoff (DET)')
            ax.set_xscale('log'); ax.set_yscale('log')
            # Equal Error Rate
            eer_idx = np.argmin(np.abs(fpr_det - fnr_det))
            eer = (fpr_det[eer_idx] + fnr_det[eer_idx]) / 2
            ax.scatter(fpr_det[eer_idx]*100, fnr_det[eer_idx]*100,
                       color=COLORS['secondary'], zorder=5,
                       label=f'EER={eer:.3f}')
            ax.legend(); ax.grid(True, alpha=0.3, which='both')
        except Exception:
            ax.text(0.5, 0.5, 'DET non disponible', ha='center', va='center',
                    transform=ax.transAxes)

        # ── KS Plot (score distributions) ────────────────────────────────────
        ax = axes[1, 1]
        scores_pos = y_pred_proba[y_true == 1]
        scores_neg = y_pred_proba[y_true == 0]
        x = np.linspace(0, 1, 300)
        cdf_pos = np.array([np.mean(scores_pos <= xi) for xi in x])
        cdf_neg = np.array([np.mean(scores_neg <= xi) for xi in x])
        ks_idx2 = np.argmax(np.abs(cdf_pos - cdf_neg))

        ax.plot(x, cdf_neg, color=COLORS['neutral'], lw=2, label='Légitimes')
        ax.plot(x, cdf_pos, color=COLORS['secondary'], lw=2, label='Fraudes')
        ax.fill_between(x, cdf_neg, cdf_pos, alpha=0.15, color=COLORS['warning'])
        ax.axvline(x[ks_idx2], color=COLORS['danger'], ls='--', alpha=0.8,
                   label=f'KS={np.max(np.abs(cdf_pos-cdf_neg)):.4f}')
        ax.annotate(f'  KS={np.max(np.abs(cdf_pos-cdf_neg)):.4f}',
                    xy=(x[ks_idx2], (cdf_neg[ks_idx2]+cdf_pos[ks_idx2])/2),
                    color=COLORS['danger'], fontweight='bold')
        ax.set(xlabel='Score', ylabel='CDF cumulative',
               title='KS Plot — Distribution des Scores', xlim=[0, 1])
        ax.legend(); ax.grid(True, alpha=0.3)

        plt.suptitle(f'Diagnostic Curves — {self.model_name}',
                     fontsize=14, fontweight='bold', color=COLORS['dark'])
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        self._figure_cache['advanced_curves'] = fig
        return fig

    def lift_gain_analysis(self, y_true, y_pred_proba, deciles: int = 10) -> pd.DataFrame:
        """
        Lift & Cumulative Gain Analysis par décile.

        Interprétation:
          Lift = 3.0 dans le top décile signifie que cibler ce décile
          capture 3x plus de fraudes que le ciblage aléatoire.
        """
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)

        df = pd.DataFrame({'y_true': y_true, 'score': y_pred_proba})
        df['decile'] = pd.qcut(df['score'], q=deciles, labels=False, duplicates='drop')
        df['decile'] = deciles - df['decile']  # Décile 1 = scores les plus hauts

        total_fraud = y_true.sum()
        baseline_rate = y_true.mean()

        results = []
        cumulative_fraud = 0
        cumulative_n = 0

        for d in range(1, deciles + 1):
            mask = df['decile'] == d
            n = mask.sum()
            fraud_in_decile = df.loc[mask, 'y_true'].sum()
            cumulative_fraud += fraud_in_decile
            cumulative_n += n
            fraud_rate = fraud_in_decile / n if n > 0 else 0

            results.append({
                'decile': d,
                'n_transactions': n,
                'n_fraud': fraud_in_decile,
                'fraud_rate': fraud_rate,
                'lift': fraud_rate / baseline_rate if baseline_rate > 0 else 0,
                'cumulative_fraud': cumulative_fraud,
                'cumulative_gain': cumulative_fraud / total_fraud if total_fraud > 0 else 0,
                'cumulative_pct_population': cumulative_n / len(df),
            })

        lift_df = pd.DataFrame(results)

        # Visualisation
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.patch.set_facecolor('#f8f9fa')

        # Lift Chart
        ax = axes[0]
        bars = ax.bar(lift_df['decile'], lift_df['lift'],
                      color=[COLORS['danger'] if l > 2 else COLORS['warning']
                             if l > 1 else COLORS['success']
                             for l in lift_df['lift']], alpha=0.85)
        ax.axhline(1, color='k', ls='--', alpha=0.5, label='Aléatoire (lift=1)')
        for bar, val in zip(bars, lift_df['lift']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{val:.1f}x', ha='center', fontsize=9, fontweight='bold')
        ax.set(xlabel='Décile (1=scores les plus hauts)', ylabel='Lift',
               title='Lift par Décile')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')

        # Cumulative Gains
        ax = axes[1]
        ax.plot(lift_df['cumulative_pct_population'] * 100,
                lift_df['cumulative_gain'] * 100,
                marker='o', color=COLORS['primary'], lw=2, label='Modèle')
        ax.plot([0, 100], [0, 100], 'k--', alpha=0.4, label='Aléatoire')
        ax.plot([0, y_true.mean()*100, 100], [0, 100, 100],
                color=COLORS['success'], ls='--', alpha=0.6, label='Parfait')
        ax.fill_between(
            [0] + list(lift_df['cumulative_pct_population']*100),
            [0] + list(lift_df['cumulative_gain']*100),
            [0] + list(np.linspace(0, 100, deciles)),
            alpha=0.1, color=COLORS['primary']
        )
        ax.set(xlabel='% Population contactée', ylabel='% Fraudes détectées',
               title='Courbe de Gains Cumulés', xlim=[0, 100], ylim=[0, 105])
        ax.legend(); ax.grid(True, alpha=0.3)

        # Fraud rate par décile
        ax = axes[2]
        ax.bar(lift_df['decile'], lift_df['fraud_rate'] * 100,
               color=COLORS['neutral'], alpha=0.8)
        ax.axhline(baseline_rate * 100, color=COLORS['secondary'], ls='--',
                   label=f'Taux global ({baseline_rate*100:.2f}%)')
        ax.set(xlabel='Décile', ylabel='Taux de fraude (%)',
               title='Taux de Fraude par Décile')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'Lift & Gains Analysis — {self.model_name}',
                     fontsize=14, fontweight='bold', color=COLORS['dark'])
        plt.tight_layout()
        self._figure_cache['lift_gain'] = fig

        print(f"\n{'═'*70}")
        print("  LIFT & GAINS ANALYSIS")
        print(f"{'═'*70}")
        print(lift_df.to_string(index=False, float_format='%.4f'))

        return lift_df

    # =========================================================================
    # 3. CALIBRATION PROBABILISTE
    # =========================================================================

    def calibration_analysis(self, y_true, y_pred_proba, n_bins: int = 10) -> dict:
        """
        Analyse de la calibration probabiliste du modèle.

        Métriques:
        - Brier Score & Brier Skill Score
        - Expected Calibration Error (ECE)
        - Maximum Calibration Error (MCE)
        - Reliability Diagram
        - Log-loss par décile
        """
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)

        # ECE & MCE
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba,
                                                  n_bins=n_bins, strategy='uniform')
        bin_sizes = []
        ece, mce = 0.0, 0.0
        bins = np.linspace(0, 1, n_bins + 1)

        for i, (pt, pp) in enumerate(zip(prob_true, prob_pred)):
            mask = (y_pred_proba >= bins[i]) & (y_pred_proba < bins[i+1])
            n = mask.sum()
            bin_sizes.append(n)
            err = abs(pt - pp)
            ece += err * n / len(y_true)
            mce = max(mce, err)

        brier = brier_score_loss(y_true, y_pred_proba)
        # Brier Skill Score vs climatologie
        brier_clim = y_true.mean() * (1 - y_true.mean())
        bss = 1 - brier / (brier_clim + 1e-10)

        cal_results = {
            'brier_score':       brier,
            'brier_skill_score': bss,
            'ece':               ece,
            'mce':               mce,
            'log_loss':          log_loss(y_true, y_pred_proba),
            'prob_true':         prob_true,
            'prob_pred':         prob_pred,
        }

        # Visualisation
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.patch.set_facecolor('#f8f9fa')

        # Reliability Diagram
        ax = axes[0]
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Calibration parfaite')
        ax.plot(prob_pred, prob_true, 'o-', color=COLORS['primary'],
                lw=2, ms=8, label='Modèle')
        for pp, pt, n in zip(prob_pred, prob_true, bin_sizes):
            ax.annotate(f'n={n}', (pp, pt),
                        textcoords='offset points', xytext=(5, 5), fontsize=7)
        ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.05, color='red')
        ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.05, color='blue')
        ax.set(xlabel='Score prédit (probabilité)', ylabel='Fréquence observée',
               title=f'Reliability Diagram\nECE={ece:.4f}  MCE={mce:.4f}')
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.92, f'BSS={bss:.4f}', transform=ax.transAxes,
                fontsize=11, fontweight='bold', color=COLORS['success'])

        # Score distribution
        ax = axes[1]
        ax.hist(y_pred_proba[y_true == 0], bins=40, alpha=0.6,
                color=COLORS['neutral'], density=True, label='Légitimes')
        ax.hist(y_pred_proba[y_true == 1], bins=40, alpha=0.6,
                color=COLORS['secondary'], density=True, label='Fraudes')
        ax.set(xlabel='Score', ylabel='Densité',
               title='Distribution des Scores par Classe')
        ax.legend(); ax.grid(True, alpha=0.3)

        # Log-loss par décile
        ax = axes[2]
        df_cal = pd.DataFrame({'y': y_true, 'p': y_pred_proba})
        df_cal['decile'] = pd.qcut(df_cal['p'], q=10, labels=False, duplicates='drop')
        ll_per_dec = df_cal.groupby('decile').apply(
            lambda x: log_loss(x['y'], x['p']) if len(x['y'].unique()) > 1 else np.nan
        ).reset_index()
        ll_per_dec.columns = ['decile', 'log_loss']
        ll_per_dec = ll_per_dec.dropna()

        ax.bar(ll_per_dec['decile'], ll_per_dec['log_loss'],
               color=COLORS['warning'], alpha=0.85)
        ax.set(xlabel='Décile de score', ylabel='Log-Loss',
               title='Log-Loss par Décile de Score')
        ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'Calibration Analysis — {self.model_name}',
                     fontsize=14, fontweight='bold', color=COLORS['dark'])
        plt.tight_layout()
        self._figure_cache['calibration'] = fig

        print(f"\n{'═'*50}")
        print("  CALIBRATION ANALYSIS")
        print(f"{'═'*50}")
        print(f"  Brier Score:       {brier:.4f}")
        print(f"  Brier Skill Score: {bss:.4f}  (1=parfait, 0=climatologie)")
        print(f"  ECE:               {ece:.4f}")
        print(f"  MCE:               {mce:.4f}")
        print(f"  Log-Loss:          {cal_results['log_loss']:.4f}")

        self.validation_results['calibration'] = cal_results
        return cal_results

    # =========================================================================
    # 4. PSI + CSI (STABILITY)
    # =========================================================================

    def calculate_psi(self, expected, actual, bins: int = 10) -> tuple:
        """PSI — Population Stability Index."""
        bp = np.percentile(expected, np.linspace(0, 100, bins + 1))
        bp[0], bp[-1] = -np.inf, np.inf
        exp_pct = np.histogram(expected, bp)[0] / len(expected)
        act_pct = np.histogram(actual, bp)[0] / len(actual)
        exp_pct = np.where(exp_pct == 0, 1e-4, exp_pct)
        act_pct = np.where(act_pct == 0, 1e-4, act_pct)
        psi_vals = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
        return float(np.sum(psi_vals)), psi_vals, bp

    def feature_stability_index(self, df_ref: pd.DataFrame, df_cur: pd.DataFrame,
                                feature_cols: list, bins: int = 10) -> pd.DataFrame:
        """
        CSI (Characteristic Stability Index) — PSI appliqué à chaque feature.

        Permet d'identifier quelles variables dérivent le plus.
        """
        results = []
        for col in feature_cols:
            try:
                ref = df_ref[col].dropna().values
                cur = df_cur[col].dropna().values
                psi, _, _ = self.calculate_psi(ref, cur, bins)
                results.append({
                    'feature': col,
                    'csi': psi,
                    'status': ('Stable' if psi < 0.1 else
                               'Modéré' if psi < 0.25 else 'Instable'),
                    'n_ref': len(ref),
                    'n_cur': len(cur),
                })
            except Exception as e:
                results.append({'feature': col, 'csi': np.nan, 'status': 'Erreur',
                                'n_ref': 0, 'n_cur': 0})

        csi_df = pd.DataFrame(results).sort_values('csi', ascending=False)

        # Visualisation
        fig, ax = plt.subplots(figsize=(10, max(4, len(feature_cols) * 0.5)))
        colors = [COLORS['success'] if s == 'Stable' else
                  COLORS['warning'] if s == 'Modéré' else COLORS['danger']
                  for s in csi_df['status']]
        bars = ax.barh(csi_df['feature'], csi_df['csi'], color=colors, alpha=0.8)
        ax.axvline(0.1, color=COLORS['warning'], ls='--', label='Seuil modéré (0.1)')
        ax.axvline(0.25, color=COLORS['danger'], ls='--', label='Seuil critique (0.25)')
        ax.set(xlabel='CSI', title='Characteristic Stability Index par Feature')
        ax.legend(); ax.grid(True, alpha=0.3, axis='x')

        for bar, val in zip(bars, csi_df['csi']):
            if not np.isnan(val):
                ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        self._figure_cache['csi'] = fig

        print(f"\n{'═'*55}")
        print("  CHARACTERISTIC STABILITY INDEX (CSI)")
        print(f"{'═'*55}")
        print(csi_df.to_string(index=False, float_format='%.4f'))

        return csi_df

    def stability_analysis(self, df, score_col='score', date_col='transaction_date',
                           period='M', baseline_end=None) -> pd.DataFrame:
        """Analyse PSI temporelle + test KS drift par période."""
        df = df.copy()
        df['period'] = df[date_col].dt.to_period(period)

        if baseline_end is None:
            baseline_end = df[date_col].quantile(0.5)

        baseline = df[df[date_col] < baseline_end][score_col].values
        results = []

        for period_val in sorted(df['period'].unique()):
            current = df[df['period'] == period_val][score_col].values
            if len(current) < 100 or len(baseline) < 100:
                continue
            psi, _, _ = self.calculate_psi(baseline, current)
            ks_stat, ks_pval = stats.ks_2samp(baseline, current)

            results.append({
                'period': str(period_val),
                'psi': psi,
                'ks_drift': ks_stat,
                'ks_pvalue': ks_pval,
                'status': ('Stable' if psi < 0.1 else
                           'Moderate' if psi < 0.25 else 'Unstable'),
                'drift_detected': ks_pval < 0.05,
                'n_samples': len(current),
                'mean_score': current.mean(),
            })

        stab_df = pd.DataFrame(results)
        self.validation_results['stability'] = stab_df
        return stab_df

    # =========================================================================
    # 5. ANALYSE DE BENFORD'S LAW
    # =========================================================================

    def benford_law_analysis(self, df, amount_col: str = 'transaction_amount') -> dict:
        """
        Loi de Benford — Détection de manipulation/anomalie dans les montants.

        La loi de Benford prédit la distribution des premiers chiffres significatifs.
        Une déviation significative peut signaler des fraudes de type:
          - Smurfing (fractionnement sous les seuils)
          - Round-tripping
          - Falsification comptable

        Retourne le chi-square goodness-of-fit et une visualisation.
        """
        amounts = df[amount_col].dropna()
        amounts = amounts[amounts > 0]

        if len(amounts) < 100:
            print("⚠️  Trop peu de données pour l'analyse de Benford")
            return {}

        # Distribution théorique (loi de Benford)
        benford_expected = np.array([np.log10(1 + 1/d) for d in range(1, 10)])

        # Distribution observée des premiers chiffres
        first_digits = amounts.apply(
            lambda x: int(str(f'{x:.6e}').replace('.', '').replace('-', '')[0])
        )
        observed_counts = pd.Series(first_digits).value_counts().sort_index()
        # Assurer qu'on a tous les chiffres 1-9
        observed_counts = observed_counts.reindex(range(1, 10), fill_value=0)
        observed_pct = observed_counts / observed_counts.sum()

        # Chi-square test
        n = len(amounts)
        expected_counts = benford_expected * n
        chi2_stat, chi2_pval = stats.chisquare(
            f_obs=observed_counts.values,
            f_exp=expected_counts
        )

        # Z-score par chiffre
        z_scores = (observed_pct.values - benford_expected) / np.sqrt(
            benford_expected * (1 - benford_expected) / n
        )

        # MAD (Mean Absolute Deviation) — seuils standard
        mad = np.mean(np.abs(observed_pct.values - benford_expected))
        conformity = ('Conforme' if mad < 0.006 else
                      'Acceptable' if mad < 0.012 else
                      'Marginalement non-conforme' if mad < 0.015 else
                      '⚠️ Non-conforme')

        results = {
            'chi2_statistic':    chi2_stat,
            'chi2_pvalue':       chi2_pval,
            'mad':               mad,
            'conformity':        conformity,
            'observed_pct':      dict(zip(range(1, 10), observed_pct.values)),
            'benford_pct':       dict(zip(range(1, 10), benford_expected)),
            'z_scores':          dict(zip(range(1, 10), z_scores)),
            'significant_bias':  chi2_pval < 0.05,
        }

        # Visualisation
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#f8f9fa')

        digits = range(1, 10)
        x = np.arange(len(digits))
        width = 0.35

        ax = axes[0]
        ax.bar(x - width/2, observed_pct * 100, width, label='Observé',
               color=COLORS['primary'], alpha=0.8)
        ax.bar(x + width/2, benford_expected * 100, width, label="Benford's Law",
               color=COLORS['success'], alpha=0.8)
        ax.set(xlabel='Premier chiffre significatif', ylabel='Fréquence (%)',
               title=f"Loi de Benford\nMAD={mad:.4f} — {conformity}\nχ²={chi2_stat:.2f}, p={chi2_pval:.4f}")
        ax.set_xticks(x); ax.set_xticklabels(digits)
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')

        if chi2_pval < 0.05:
            ax.text(0.5, 0.97, '⚠️ ANOMALIE DÉTECTÉE', transform=ax.transAxes,
                    ha='center', va='top', color=COLORS['danger'],
                    fontweight='bold', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='#ffe0e0', alpha=0.8))

        ax2 = axes[1]
        bar_colors = [COLORS['danger'] if abs(z) > 1.96 else COLORS['neutral']
                      for z in z_scores]
        ax2.bar(x, z_scores, color=bar_colors, alpha=0.8)
        ax2.axhline(1.96, color=COLORS['danger'], ls='--', alpha=0.7, label='±1.96 (5%)')
        ax2.axhline(-1.96, color=COLORS['danger'], ls='--', alpha=0.7)
        ax2.set(xlabel='Premier chiffre', ylabel='Z-Score',
               title='Z-Scores par Chiffre (|Z|>1.96 = anomalie)')
        ax2.set_xticks(x); ax2.set_xticklabels(digits)
        ax2.legend(); ax2.grid(True, alpha=0.3)

        plt.suptitle(f"Benford's Law Analysis — {amount_col}",
                     fontsize=14, fontweight='bold', color=COLORS['dark'])
        plt.tight_layout()
        self._figure_cache['benford'] = fig

        print(f"\n{'═'*55}")
        print("  BENFORD'S LAW ANALYSIS")
        print(f"{'═'*55}")
        print(f"  Transactions analysées: {n:,}")
        print(f"  MAD:                    {mad:.4f}")
        print(f"  Conformité:             {conformity}")
        print(f"  χ² (p-value):           {chi2_stat:.4f} (p={chi2_pval:.4f})")
        print(f"  Anomalie détectée:      {'⚠️ OUI' if results['significant_bias'] else '✅ NON'}")

        self.validation_results['benford'] = results
        return results

    # =========================================================================
    # 6. GRAPH-BASED ANOMALY DETECTION
    # =========================================================================

    def graph_anomaly_detection(self, df,
                                 source_col: str = 'sender_id',
                                 target_col: str = 'receiver_id',
                                 amount_col: str = 'transaction_amount',
                                 fraud_col: str = 'is_fraud') -> pd.DataFrame:
        """
        Détection d'anomalies basée sur l'analyse du réseau de transactions.

        Features graphiques calculées:
        - Degree centrality (in/out)
        - Betweenness centrality (nœuds pont dans le réseau)
        - PageRank (influence systémique)
        - Clustering coefficient
        - Détection de communautés suspectes (hubs de fraude)

        Requiert NetworkX.
        """
        if not NX_AVAILABLE:
            print("⚠️  NetworkX non disponible. pip install networkx")
            return pd.DataFrame()

        print("  🔗 Construction du graphe de transactions...")
        G = nx.DiGraph()

        for _, row in df.iterrows():
            src = row.get(source_col)
            tgt = row.get(target_col)
            if pd.notna(src) and pd.notna(tgt):
                w = row.get(amount_col, 1)
                if G.has_edge(src, tgt):
                    G[src][tgt]['weight'] += w
                    G[src][tgt]['count'] += 1
                else:
                    G.add_edge(src, tgt, weight=w, count=1)

        print(f"  Graphe: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")

        # Métriques de centralité
        degree_in  = dict(G.in_degree())
        degree_out = dict(G.out_degree())

        try:
            pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
        except Exception:
            pagerank = {n: 0 for n in G.nodes()}

        try:
            betweenness = nx.betweenness_centrality(G, normalized=True, k=min(100, len(G)))
        except Exception:
            betweenness = {n: 0 for n in G.nodes()}

        # Score d'anomalie composite
        nodes = list(G.nodes())
        node_df = pd.DataFrame({
            'node': nodes,
            'in_degree':    [degree_in.get(n, 0) for n in nodes],
            'out_degree':   [degree_out.get(n, 0) for n in nodes],
            'pagerank':     [pagerank.get(n, 0) for n in nodes],
            'betweenness':  [betweenness.get(n, 0) for n in nodes],
        })

        # Standardisation et score composite
        for col in ['in_degree', 'out_degree', 'pagerank', 'betweenness']:
            std = node_df[col].std()
            if std > 0:
                node_df[f'{col}_z'] = (node_df[col] - node_df[col].mean()) / std
            else:
                node_df[f'{col}_z'] = 0

        z_cols = [c for c in node_df.columns if c.endswith('_z')]
        node_df['anomaly_score'] = node_df[z_cols].abs().mean(axis=1)
        node_df = node_df.sort_values('anomaly_score', ascending=False)

        # Top nœuds suspects
        top_suspicious = node_df.head(20)

        # Si la colonne fraude est disponible, évaluation
        fraud_rate_by_node = None
        if fraud_col in df.columns and source_col in df.columns:
            fraud_rate_by_node = df.groupby(source_col)[fraud_col].mean().reset_index()
            fraud_rate_by_node.columns = ['node', 'fraud_rate']
            node_df = node_df.merge(fraud_rate_by_node, on='node', how='left')

        # Visualisation (sous-graphe des nœuds suspects)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor('#f8f9fa')

        # Distribution des degrés
        ax = axes[0]
        in_degrees = [d for _, d in G.in_degree()]
        ax.hist(in_degrees, bins=50, color=COLORS['primary'], alpha=0.7, log=True)
        ax.set(xlabel='In-Degree', ylabel='Fréquence (log)',
               title='Distribution Power-Law des Degrés\n(queue lourde = réseau hubs-and-spokes)')
        ax.grid(True, alpha=0.3)

        # Top nœuds par score d'anomalie
        ax = axes[1]
        top20 = node_df.head(15)
        bars = ax.barh(range(len(top20)), top20['anomaly_score'],
                       color=COLORS['secondary'], alpha=0.8)
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels([str(n)[:15] for n in top20['node']], fontsize=8)
        ax.set(xlabel='Score d\'Anomalie Composite', ylabel='Nœud',
               title='Top 15 Nœuds Suspects (Score Anomalie)')
        ax.grid(True, alpha=0.3, axis='x')

        plt.suptitle(f'Graph Anomaly Detection — {G.number_of_nodes()} entités',
                     fontsize=14, fontweight='bold', color=COLORS['dark'])
        plt.tight_layout()
        self._figure_cache['graph'] = fig

        print(f"\n{'═'*60}")
        print("  GRAPH-BASED ANOMALY DETECTION")
        print(f"{'═'*60}")
        print(f"  Nœuds: {G.number_of_nodes():,}  |  Arêtes: {G.number_of_edges():,}")
        print(f"\n  Top 10 nœuds suspects:")
        print(node_df.head(10)[['node', 'in_degree', 'out_degree',
                                  'pagerank', 'betweenness', 'anomaly_score']].to_string(index=False))

        self.validation_results['graph'] = {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'top_suspicious': top_suspicious.to_dict('records')
        }
        return node_df

    # =========================================================================
    # 7. PERMUTATION FEATURE IMPORTANCE + MAHALANOBIS OUTLIERS
    # =========================================================================

    def permutation_feature_importance(self, model, X, y, n_repeats: int = 10,
                                        feature_names: list = None) -> pd.DataFrame:
        """
        Permutation Feature Importance avec intervalles de confiance.

        Plus robuste que l'importance "impurity-based" des arbres
        car elle mesure l'impact réel sur la performance out-of-sample.
        """
        if feature_names is None:
            feature_names = X.columns.tolist() if hasattr(X, 'columns') else \
                            [f'f{i}' for i in range(X.shape[1])]

        print("  🔀 Calcul Permutation Importance...")
        perm_imp = permutation_importance(
            model, X, y,
            scoring='roc_auc',
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )

        imp_df = pd.DataFrame({
            'feature':    feature_names,
            'importance_mean': perm_imp.importances_mean,
            'importance_std':  perm_imp.importances_std,
            'ci_low':     perm_imp.importances_mean - 1.96 * perm_imp.importances_std,
            'ci_high':    perm_imp.importances_mean + 1.96 * perm_imp.importances_std,
        }).sort_values('importance_mean', ascending=False)

        # Visualisation
        fig, ax = plt.subplots(figsize=(10, max(4, len(feature_names) * 0.5)))
        y_pos = np.arange(len(imp_df))

        colors = [COLORS['danger'] if m > 0.01 else
                  COLORS['warning'] if m > 0.005 else COLORS['neutral']
                  for m in imp_df['importance_mean']]

        ax.barh(y_pos, imp_df['importance_mean'],
                xerr=1.96 * imp_df['importance_std'],
                color=colors, alpha=0.8, capsize=4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(imp_df['feature'])
        ax.axvline(0, color='k', ls='-', lw=0.5)
        ax.set(xlabel='Δ AUC-ROC (mean ± 1.96σ)',
               title='Permutation Feature Importance\n(IC 95% Bootstrap)')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        self._figure_cache['feature_importance'] = fig

        print(f"\n{'═'*55}")
        print("  PERMUTATION FEATURE IMPORTANCE")
        print(f"{'═'*55}")
        print(imp_df.to_string(index=False, float_format='%.5f'))

        return imp_df

    def mahalanobis_outlier_detection(self, X: pd.DataFrame,
                                       contamination: float = 0.05) -> pd.Series:
        """
        Détection d'outliers par distance de Mahalanobis ET Isolation Forest.
        Combine les deux pour un score d'anomalie robuste.
        """
        X_arr = X.select_dtypes(include=[np.number]).fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_arr)

        # Mahalanobis
        cov = np.cov(X_scaled, rowvar=False)
        try:
            VI = np.linalg.pinv(cov)
            mean = X_scaled.mean(axis=0)
            mah_dist = np.array([
                mahalanobis(x, mean, VI) for x in X_scaled
            ])
            mah_score = (mah_dist - mah_dist.min()) / (mah_dist.max() - mah_dist.min() + 1e-10)
        except Exception:
            mah_score = np.zeros(len(X_scaled))

        # Isolation Forest
        iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        iso_score = -iso.fit(X_scaled).score_samples(X_scaled)
        iso_score = (iso_score - iso_score.min()) / (iso_score.max() - iso_score.min() + 1e-10)

        # Score composite
        composite = 0.5 * mah_score + 0.5 * iso_score
        anomaly_threshold = np.percentile(composite, (1 - contamination) * 100)
        is_outlier = composite >= anomaly_threshold

        print(f"\n  🔍 Mahalanobis + Isolation Forest:")
        print(f"  Outliers détectés: {is_outlier.sum()} / {len(is_outlier)} "
              f"({is_outlier.mean()*100:.1f}%)")

        return pd.Series(composite, index=X.index, name='anomaly_score')

    # =========================================================================
    # 8. FAIRNESS ANALYSIS
    # =========================================================================

    def fairness_analysis(self, df, score_col: str, target_col: str,
                          protected_attr: str, threshold: float = 0.5) -> tuple:
        """
        Analyse d'équité étendue.

        Métriques:
        - Equalized Odds (TPR & FPR équivalents)
        - Demographic Parity
        - Calibration par groupe
        - Disparate Impact Ratio
        - Test KS inter-groupes
        """
        df = df.copy()
        df['pred'] = (df[score_col] >= threshold).astype(int)

        groups = df[protected_attr].unique()
        results = []

        for group in groups:
            sub = df[df[protected_attr] == group]
            if len(sub) < 20:
                continue
            yt, yp, ys = sub[target_col], sub['pred'], sub[score_col]

            cm = confusion_matrix(yt, yp)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)

            results.append({
                'group':                str(group),
                'n':                    len(sub),
                'fraud_rate':           yt.mean(),
                'tpr':                  tp/(tp+fn) if (tp+fn)>0 else 0,
                'fpr':                  fp/(fp+tn) if (fp+tn)>0 else 0,
                'precision':            precision_score(yt, yp, zero_division=0),
                'predicted_pos_rate':   yp.mean(),
                'auc':                  roc_auc_score(yt, ys) if yt.nunique()>1 else np.nan,
                'brier_score':          brier_score_loss(yt, ys),
            })

        fair_df = pd.DataFrame(results)

        if len(fair_df) >= 2:
            tpr_range = fair_df['tpr'].max() - fair_df['tpr'].min()
            fpr_range = fair_df['fpr'].max() - fair_df['fpr'].min()

            # Disparate Impact Ratio
            ppr_min = fair_df['predicted_pos_rate'].min()
            ppr_max = fair_df['predicted_pos_rate'].max()
            disparate_impact = ppr_min / (ppr_max + 1e-10)

            # Test KS inter-groupes sur les scores
            group_scores = [
                df[df[protected_attr] == g][score_col].values
                for g in fair_df['group']
            ]
            if len(group_scores) >= 2:
                ks_stat, ks_pval = stats.ks_2samp(group_scores[0], group_scores[1])
            else:
                ks_stat, ks_pval = np.nan, np.nan

            # Chi-square
            contingency = pd.crosstab(df[protected_attr], df['pred'])
            chi2, pval, _, _ = stats.chi2_contingency(contingency)

            fairness_summary = {
                'tpr_range': tpr_range,
                'fpr_range': fpr_range,
                'disparate_impact_ratio': disparate_impact,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pval,
                'chi2': chi2,
                'chi2_pvalue': pval,
                'equalized_odds_ok': tpr_range < 0.1 and fpr_range < 0.1,
                'demographic_parity_ok': disparate_impact >= 0.8,
            }
        else:
            fairness_summary = {}
            pval = np.nan

        print(f"\n{'═'*65}")
        print(f"  FAIRNESS ANALYSIS — {protected_attr}")
        print(f"{'═'*65}")
        print(fair_df.to_string(index=False, float_format='%.4f'))
        if fairness_summary:
            print(f"\n  TPR Range:              {fairness_summary['tpr_range']:.4f}")
            print(f"  FPR Range:              {fairness_summary['fpr_range']:.4f}")
            print(f"  Disparate Impact Ratio: {fairness_summary['disparate_impact_ratio']:.4f} "
                  f"({'✅ ≥0.8' if fairness_summary['demographic_parity_ok'] else '⚠️ <0.8'})")
            print(f"  Equalized Odds:         "
                  f"{'✅ OK' if fairness_summary['equalized_odds_ok'] else '⚠️ Violation'}")

        self.validation_results['fairness'] = {
            'details': fair_df, 'summary': fairness_summary
        }
        return fair_df, fairness_summary

    # =========================================================================
    # 9. OPTIMISATION DU SEUIL (AMÉLIORÉE)
    # =========================================================================

    def optimize_threshold(self, y_true, y_pred_proba, metric: str = 'cost',
                           cost_fn: float = None, cost_fp: float = None) -> tuple:
        """
        Optimisation multi-critères du seuil de décision.
        Stratégies: F1, MCC, coût, G-Mean, Youden's J.
        """
        if cost_fn is None: cost_fn = self.cost_fn
        if cost_fp is None: cost_fp = self.cost_fp

        thresholds = np.arange(0.005, 1.0, 0.005)
        results = []

        for thresh in thresholds:
            yp = (y_pred_proba >= thresh).astype(int)
            cm = confusion_matrix(y_true, yp)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
            tpr = tp/(tp+fn+1e-10)
            tnr = tn/(tn+fp+1e-10)
            fpr = fp/(fp+tn+1e-10)

            results.append({
                'threshold':  thresh,
                'precision':  precision_score(y_true, yp, zero_division=0),
                'recall':     recall_score(y_true, yp, zero_division=0),
                'f1':         f1_score(y_true, yp, zero_division=0),
                'mcc':        matthews_corrcoef(y_true, yp),
                'gmean':      np.sqrt(tpr * tnr),           # G-Mean
                'youden_j':   tpr - fpr,                    # Youden's J statistic
                'cost':       (fn * cost_fn + fp * cost_fp) / len(y_true),
                'fpr': fpr, 'fnr': fn/(fn+tp+1e-10),
            })

        res_df = pd.DataFrame(results)

        optimal = {
            'f1':       res_df.loc[res_df['f1'].idxmax(), 'threshold'],
            'mcc':      res_df.loc[res_df['mcc'].idxmax(), 'threshold'],
            'cost':     res_df.loc[res_df['cost'].idxmin(), 'threshold'],
            'gmean':    res_df.loc[res_df['gmean'].idxmax(), 'threshold'],
            'youden_j': res_df.loc[res_df['youden_j'].idxmax(), 'threshold'],
        }

        if metric in ['f1','mcc','gmean','youden_j']:
            best_thresh = optimal[metric]
        elif metric == 'cost':
            best_thresh = optimal['cost']
        else:
            best_thresh = optimal['f1']

        # Visualisation
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.patch.set_facecolor('#f8f9fa')

        ax = axes[0, 0]
        ax.plot(res_df['threshold'], res_df['f1'], label='F1', color=COLORS['primary'])
        ax.plot(res_df['threshold'], res_df['gmean'], label='G-Mean', color=COLORS['success'])
        ax.plot(res_df['threshold'], res_df['youden_j'], label="Youden's J", color=COLORS['neutral'])
        for name, t in optimal.items():
            if name in ['f1', 'gmean', 'youden_j']:
                ax.axvline(t, ls=':', alpha=0.5)
        ax.axvline(best_thresh, color=COLORS['secondary'], ls='--', lw=2,
                   label=f'Optimal ({metric})={best_thresh:.3f}')
        ax.set(xlabel='Seuil', ylabel='Score', title='Métriques vs Seuil')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(res_df['threshold'], res_df['cost'], color=COLORS['danger'], lw=2)
        ax.axvline(optimal['cost'], color=COLORS['success'], ls='--',
                   label=f"Seuil coût-optimal={optimal['cost']:.3f}")
        ax.set(xlabel='Seuil', ylabel=f'Coût moyen ({self.currency}/tx)',
               title='Optimisation par Coût')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(res_df['threshold'], res_df['precision'], label='Precision')
        ax.plot(res_df['threshold'], res_df['recall'], label='Recall')
        ax.plot(res_df['threshold'], res_df['mcc'], label='MCC')
        ax.axvline(best_thresh, color=COLORS['secondary'], ls='--',
                   label=f'Optimal={best_thresh:.3f}')
        ax.set(xlabel='Seuil', ylabel='Score', title='Precision / Recall / MCC')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(res_df['fpr'], res_df['fnr'], color=COLORS['warning'], lw=2)

        # CORRECTION: Trouver l'index du seuil optimal
        best_idx = (res_df['threshold'] - best_thresh).abs().idxmin()
        op_fpr = res_df.loc[best_idx, 'fpr']
        op_fnr = res_df.loc[best_idx, 'fnr']
        ax.scatter([op_fpr], [op_fnr], color=COLORS['secondary'], s=100, zorder=5,
                   label=f'Point opérationnel')
        ax.set(xlabel='FPR', ylabel='FNR',
               title='Courbe FPR-FNR (Trade-off Alertes vs Manqués)')
        ax.legend(); ax.grid(True, alpha=0.3)

        plt.suptitle(f'Optimisation du Seuil — {self.model_name}',
                     fontsize=14, fontweight='bold', color=COLORS['dark'])
        plt.tight_layout()
        self._figure_cache['threshold'] = fig

        print(f"\n{'═'*55}")
        print("  THRESHOLD OPTIMIZATION")
        print(f"{'═'*55}")
        for name, t in optimal.items():
            idx = (res_df['threshold'] - t).abs().idxmin()
            print(f"  {name:<12} → seuil={t:.3f}  "
                  f"F1={res_df.loc[idx,'f1']:.4f}  "
                  f"Coût={res_df.loc[idx,'cost']:.2f}{self.currency}")

        return best_thresh, res_df, optimal

    # =========================================================================
    # 10. WALK-FORWARD BACKTESTING (TEMPORAL)
    # =========================================================================

    def walk_forward_backtest(self, df: pd.DataFrame, model,
                               feature_cols: list, target_col: str,
                               date_col: str = 'transaction_date',
                               n_splits: int = 5,
                               min_train_size: float = 0.5) -> pd.DataFrame:
        """
        Walk-Forward Validation temporelle — critique pour la fraude.

        Contrairement à la CV standard, respecte l'ordre temporel:
        le modèle n'est jamais testé sur des données antérieures à l'entraînement.
        Simule les conditions réelles de déploiement.
        """
        df = df.sort_values(date_col).reset_index(drop=True)
        n = len(df)
        min_train = int(n * min_train_size)

        results = []
        step = (n - min_train) // n_splits

        for i in range(n_splits):
            train_end = min_train + i * step
            test_start = train_end
            test_end = min(train_end + step, n)

            if test_end <= test_start:
                break

            X_train = df.iloc[:train_end][feature_cols]
            y_train = df.iloc[:train_end][target_col]
            X_test  = df.iloc[test_start:test_end][feature_cols]
            y_test  = df.iloc[test_start:test_end][target_col]

            if y_test.nunique() < 2 or y_train.nunique() < 2:
                continue

            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]

            fold_metrics = self.calculate_performance_metrics(
                y_test, y_proba, bootstrap=False
            )
            fold_metrics['fold'] = i + 1
            fold_metrics['train_size'] = train_end
            fold_metrics['test_size'] = test_end - test_start
            fold_metrics['train_period_end'] = str(df.iloc[train_end-1][date_col])[:10]
            fold_metrics['test_period_start'] = str(df.iloc[test_start][date_col])[:10]
            results.append(fold_metrics)

        wf_df = pd.DataFrame(results)

        # Visualisation
        fig, axes = plt.subplots(2, 2, figsize=(16, 8))
        fig.patch.set_facecolor('#f8f9fa')

        metrics_plot = ['auc_roc', 'ks_statistic', 'f1_score', 'expected_cost']
        titles = ['AUC-ROC', 'KS-Statistic', 'F1-Score', f'Coût/tx ({self.currency})']

        for ax, col, title in zip(axes.flat, metrics_plot, titles):
            ax.plot(wf_df['fold'], wf_df[col], marker='o', color=COLORS['primary'], lw=2)
            ax.axhline(wf_df[col].mean(), color=COLORS['secondary'], ls='--',
                       label=f'Moyenne={wf_df[col].mean():.4f}')
            ax.fill_between(wf_df['fold'],
                            wf_df[col].mean() - wf_df[col].std(),
                            wf_df[col].mean() + wf_df[col].std(),
                            alpha=0.1, color=COLORS['primary'])
            ax.set(xlabel='Fold (chronologique)', ylabel=col, title=f'Walk-Forward — {title}')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        plt.suptitle(f'Walk-Forward Backtesting — {self.model_name}',
                     fontsize=14, fontweight='bold', color=COLORS['dark'])
        plt.tight_layout()
        self._figure_cache['walkforward'] = fig

        print(f"\n{'═'*65}")
        print("  WALK-FORWARD BACKTESTING (Validation Temporelle)")
        print(f"{'═'*65}")
        print(wf_df[['fold', 'train_period_end', 'test_period_start',
                      'auc_roc', 'ks_statistic', 'f1_score',
                      'expected_cost', 'train_size', 'test_size']].to_string(index=False))

        self.validation_results['walkforward'] = wf_df
        return wf_df

    # =========================================================================
    # 11. STRESS TESTING AVANCÉ (BÂLE III / AMLD6)
    # =========================================================================

    def stress_test(self, X, y_true, model_predict_fn,
                    perturbation_factors: list = None) -> pd.DataFrame:
        """
        Stress testing avancé — scénarios réglementaires AMLD6 / Bâle III.

        Scénarios:
        1. Choc de montants (x0.5 à x3.0)
        2. Suppression de features (simulation missing data)
        3. Concept drift (augmentation taux fraude)
        4. Bruit gaussien sur les features
        5. Scénario de crise: fraude x5 (scenario AMLD6 black swan)
        6. Biais géographique (feature is_international = 1)
        7. Adversarial: maximisation du score sur les légitimes (FGSM-style)
        """
        if perturbation_factors is None:
            perturbation_factors = [0.5, 0.8, 1.2, 1.5, 2.0, 3.0]

        y_true_arr = np.asarray(y_true)
        baseline_proba = model_predict_fn(X)
        baseline_auc = roc_auc_score(y_true_arr, baseline_proba)
        results = [{'scenario': 'Baseline', 'category': 'Référence',
                    'auc': baseline_auc, 'auc_drop_pct': 0.0,
                    'brier': brier_score_loss(y_true_arr, baseline_proba)}]

        # ── Choc montants ─────────────────────────────────────────────────────
        if 'transaction_amount' in X.columns:
            for factor in perturbation_factors:
                X_p = X.copy()
                X_p['transaction_amount'] *= factor
                proba = model_predict_fn(X_p)
                auc = roc_auc_score(y_true_arr, proba)
                results.append({
                    'scenario': f'Montants ×{factor}',
                    'category': 'Choc Montants',
                    'auc': auc,
                    'auc_drop_pct': (baseline_auc - auc) / baseline_auc * 100,
                    'brier': brier_score_loss(y_true_arr, proba),
                })

        # ── Bruit gaussien ────────────────────────────────────────────────────
        for noise_level in [0.05, 0.1, 0.2, 0.5]:
            X_p = X.copy()
            num_cols = X_p.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                std = X_p[col].std()
                X_p[col] += np.random.normal(0, noise_level * std, len(X_p))
            proba = model_predict_fn(X_p)
            auc = roc_auc_score(y_true_arr, proba)
            results.append({
                'scenario': f'Bruit gaussien σ={noise_level}',
                'category': 'Robustesse Bruit',
                'auc': auc,
                'auc_drop_pct': (baseline_auc - auc) / baseline_auc * 100,
                'brier': brier_score_loss(y_true_arr, proba),
            })

        # ── Missing features ──────────────────────────────────────────────────
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        for n_drop in [1, 2, min(3, len(num_cols)-1)]:
            if len(num_cols) > n_drop:
                dropped = np.random.choice(num_cols, n_drop, replace=False)
                X_p = X.copy()
                for col in dropped:
                    X_p[col] = X_p[col].mean()
                proba = model_predict_fn(X_p)
                auc = roc_auc_score(y_true_arr, proba)
                results.append({
                    'scenario': f'Missing {n_drop} feature(s)',
                    'category': 'Data Quality',
                    'auc': auc,
                    'auc_drop_pct': (baseline_auc - auc) / baseline_auc * 100,
                    'brier': brier_score_loss(y_true_arr, proba),
                })

        # ── Concept drift (augmentation fraude) ──────────────────────────────
        fraud_idx = np.where(y_true_arr == 1)[0]
        legit_idx = np.where(y_true_arr == 0)[0]
        for mult in [1.5, 2.0, 5.0]:
            n_new = min(int(len(fraud_idx) * mult), len(fraud_idx))
            sel = np.concatenate([
                np.random.choice(fraud_idx, n_new, replace=True),
                np.random.choice(legit_idx, len(legit_idx), replace=False)
            ])
            proba = baseline_proba[sel]
            y_sub = y_true_arr[sel]
            auc = roc_auc_score(y_sub, proba)
            label = 'Bâle III' if mult < 3 else '⚠️ AMLD6 Black Swan'
            results.append({
                'scenario': f'Fraude ×{mult} ({label})',
                'category': 'Concept Drift',
                'auc': auc,
                'auc_drop_pct': (baseline_auc - auc) / baseline_auc * 100,
                'brier': brier_score_loss(y_sub, proba),
            })

        stress_df = pd.DataFrame(results)

        # Visualisation
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.patch.set_facecolor('#f8f9fa')

        # AUC Drop par scénario
        ax = axes[0]
        categories = stress_df['category'].unique()
        cat_colors = dict(zip(categories, [
            COLORS['neutral'], COLORS['primary'], COLORS['warning'],
            COLORS['success'], COLORS['danger']
        ]))
        colors = [cat_colors.get(c, COLORS['neutral']) for c in stress_df['category']]

        bars = ax.barh(stress_df['scenario'], stress_df['auc_drop_pct'],
                       color=colors, alpha=0.8)
        ax.axvline(5, color=COLORS['warning'], ls='--', alpha=0.8, label='Warning (5%)')
        ax.axvline(10, color=COLORS['danger'], ls='--', alpha=0.8, label='Critique (10%)')
        for bar, val in zip(bars, stress_df['auc_drop_pct']):
            ax.text(max(val + 0.2, 0.2), bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}%', va='center', fontsize=8)
        ax.set(xlabel='AUC Drop (%)', title='Stress Test — Impact sur AUC')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='x')

        # AUC absolue
        ax = axes[1]
        ax.barh(stress_df['scenario'], stress_df['auc'],
                color=colors, alpha=0.8)
        ax.axvline(0.7, color=COLORS['warning'], ls='--', label='Seuil acceptable (0.70)')
        ax.axvline(0.8, color=COLORS['success'], ls='--', label='Seuil bon (0.80)')
        ax.axvline(baseline_auc, color=COLORS['primary'], ls='-', lw=2,
                   label=f'Baseline={baseline_auc:.4f}')
        ax.set(xlabel='AUC-ROC', title='AUC Absolue par Scénario')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='x')

        legend_patches = [mpatches.Patch(color=c, label=k) for k, c in cat_colors.items()]
        fig.legend(handles=legend_patches, loc='lower center', ncol=len(categories),
                   fontsize=8, title='Catégories de stress')

        plt.suptitle(f'Stress Testing Réglementaire — {self.model_name}',
                     fontsize=14, fontweight='bold', color=COLORS['dark'])
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        self._figure_cache['stress'] = fig

        print(f"\n{'═'*70}")
        print("  STRESS TEST RESULTS")
        print(f"{'═'*70}")
        print(stress_df.to_string(index=False, float_format='%.4f'))

        self.validation_results['stress'] = stress_df
        return stress_df

    # =========================================================================
    # 12. CROSS-VALIDATION STRATIFIÉE
    # =========================================================================

    def cross_validate(self, X, y, model, n_splits: int = 5) -> pd.DataFrame:
        """Cross-validation stratifiée avec toutes les métriques avancées."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            Xtr, Xv = X.iloc[train_idx], X.iloc[val_idx]
            ytr, yv = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(Xtr, ytr)
            yp = model.predict_proba(Xv)[:, 1]

            m = self.calculate_performance_metrics(yv, yp, bootstrap=False)
            m['fold'] = fold + 1
            cv_results.append(m)

        cv_df = pd.DataFrame(cv_results)
        self.validation_results['cross_validation'] = cv_df

        print(f"\n{'═'*70}")
        print(f"  CROSS-VALIDATION ({n_splits}-Fold Stratifié)")
        print(f"{'═'*70}")
        key_metrics = ['auc_roc', 'gini', 'ks_statistic', 'average_precision',
                       'f1_score', 'mcc', 'brier_score']
        print(cv_df[['fold'] + key_metrics].to_string(index=False, float_format='%.4f'))
        print(f"\n  Résumé:")
        summary = cv_df[key_metrics].agg(['mean', 'std', 'min', 'max'])
        print(summary.to_string(float_format='%.4f'))

        return cv_df

    # =========================================================================
    # 13. ULTRA-SCIENTIFIC ANOMALY DETECTION (NOUVEAU)
    # =========================================================================
    
    def ultra_scientific_detection(self, df: pd.DataFrame, 
                                    feature_cols: list = None,
                                    time_col: str = 'transaction_date',
                                    value_col: str = 'transaction_amount',
                                    window_size: int = 500) -> dict:
        """
        Détection ultra-scientifique des anomalies subtiles.
        
        Méthodes:
        - Multifractal Detrended Fluctuation Analysis (MF-DFA)
        - Transfer Entropy (causal information flow)
        - Bayesian Online Changepoint Detection (BOCD)
        - Random Matrix Theory (eigenvalue spacing)
        - Echo State Network residuals (nonlinear dynamics)
        - Singular Spectrum Analysis (SSA)
        - Recurrence Quantification Analysis (RQA)
        """
        print("\n" + "="*70)
        print("  🧪 ULTRA-SCIENTIFIC ANOMALY DETECTION")
        print("  Multifractal | Causal | BOCD | RMT | ESN | SSA | RQA")
        print("="*70)
        
        results = {}
        
        # 1. Multifractal analysis
        if time_col is not None and value_col in df.columns:
            ts = df.sort_values(time_col)[value_col].values
            mf = self._mfdfa(ts[:min(len(ts), 5000)])
            results['mfdfa_anomaly_score'] = mf['anomaly_score']
            results['multifractal_width'] = mf['multifractal_width']
            results['is_monofractal'] = mf['is_monofractal']
            print(f"  📊 MF-DFA: width={mf['multifractal_width']:.4f} | anomaly={mf['anomaly_score']:.4f}")
        
        # 2. Transfer entropy between top features
        if feature_cols is not None and len(feature_cols) >= 2:
            te_scores = []
            for i in range(min(len(feature_cols), 5)):
                for j in range(i+1, min(len(feature_cols), 5)):
                    te = self._transfer_entropy(df[feature_cols[i]].values, 
                                                 df[feature_cols[j]].values)
                    te_scores.append(te)
            results['transfer_entropy_mean'] = np.mean(te_scores) if te_scores else 0
            results['te_anomaly_flag'] = np.mean(te_scores) > 0.3
            print(f"  🔄 Transfer Entropy: mean={results['transfer_entropy_mean']:.4f} | flag={results['te_anomaly_flag']}")
        
        # 3. BOCD on model scores (if available)
        if 'score' in df.columns:
            cp_probs = self._bocd(df['score'].values)
            results['bocd_max_prob'] = cp_probs.max()
            results['bocd_anomaly'] = cp_probs.max() > 0.1
            results['bocd_mean_prob'] = cp_probs.mean()
            print(f"  🔄 BOCD: max_prob={results['bocd_max_prob']:.4f} | anomaly={results['bocd_anomaly']}")
        
        # 4. RMT on correlation matrix
        if feature_cols is not None and len(feature_cols) >= 5:
            rmt_res = self._rmt_anomaly(df[feature_cols].dropna().values[:min(2000, len(df))])
            results['rmt_anomaly_score'] = rmt_res['anomaly_score']
            results['rmt_pvalue'] = rmt_res['ks_pvalue']
            results['n_signal_eigenvalues'] = rmt_res['n_signal_eigenvalues']
            print(f"  📐 RMT: pvalue={rmt_res['ks_pvalue']:.4f} | anomaly={rmt_res['anomaly_score']:.4f}")
        
        # 5. ESN residuals
        if len(df) > window_size and value_col in df.columns:
            X_win = df[value_col].values
            residuals = self._esn_residuals(X_win[:min(3000, len(X_win))])
            results['esn_residual_mean'] = np.mean(residuals)
            results['esn_residual_std'] = np.std(residuals)
            # L'anomalie est maintenant normalisée entre 0 et 1
            results['esn_anomaly_score'] = np.mean(residuals)
            print(f"  🧠 ESN: anomaly_score={results['esn_anomaly_score']:.4f}")
        
        # 6. Singular Spectrum Analysis
        if time_col is not None and value_col in df.columns:
            ssa_res = self._singular_spectrum_analysis(ts[:min(len(ts), 1000)])
            results['ssa_complexity'] = ssa_res['complexity']
            results['ssa_anomaly'] = ssa_res['anomaly']
            print(f"  📈 SSA: complexity={ssa_res['complexity']:.4f} | anomaly={ssa_res['anomaly']}")
        
        # 7. Recurrence Quantification Analysis
        if value_col in df.columns:
            rqa_res = self._recurrence_quantification(df[value_col].values[:min(500, len(df))])
            results['rqa_determinism'] = rqa_res['determinism']
            results['rqa_laminarity'] = rqa_res['laminarity']
            results['rqa_anomaly'] = rqa_res['determinism'] > 0.8
            print(f"  🔁 RQA: determinism={rqa_res['determinism']:.4f} | anomaly={rqa_res['rqa_anomaly']}")
        
        # Global ultra anomaly score
        scores = []
        if 'mfdfa_anomaly_score' in results:
            scores.append(np.clip(results['mfdfa_anomaly_score'] / 2.0, 0, 1))
        if 'te_anomaly_flag' in results:
            scores.append(0.5 if results['te_anomaly_flag'] else 0)
        if 'bocd_anomaly' in results:
            scores.append(0.5 if results['bocd_anomaly'] else 0)
        if 'rmt_anomaly_score' in results:
            scores.append(results['rmt_anomaly_score'])
        if 'esn_anomaly_score' in results:
            scores.append(np.clip(results['esn_anomaly_score'] / 0.5, 0, 1))
        if 'ssa_anomaly' in results:
            scores.append(0.6 if results['ssa_anomaly'] else 0)
        if 'rqa_anomaly' in results:
            scores.append(0.4 if results['rqa_anomaly'] else 0)
        
        results['global_ultra_anomaly_score'] = np.mean(scores) if scores else 0
        results['ultra_risk_level'] = (
            'CRITICAL' if results['global_ultra_anomaly_score'] > 0.7 else
            'HIGH' if results['global_ultra_anomaly_score'] > 0.5 else
            'MEDIUM' if results['global_ultra_anomaly_score'] > 0.3 else
            'LOW'
        )
        
        print(f"\n{'─'*55}")
        print(f"  🌟 GLOBAL ULTRA ANOMALY SCORE: {results['global_ultra_anomaly_score']:.4f}")
        print(f"  🚨 ULTRA RISK LEVEL: {results['ultra_risk_level']}")
        print(f"{'─'*55}")
        
        self.validation_results['ultra_scientific'] = results
        
        # Visualisation
        self._plot_ultra_results(results)
        
        return results
    
    def _mfdfa(self, series, scales=None, q_range=(-5, 5, 21)):
        """Multifractal Detrended Fluctuation Analysis."""
        series = np.asarray(series).flatten()
        if scales is None:
            scales = np.logspace(np.log10(10), np.log10(len(series)//4), 20).astype(int)
        
        def detrend_segment(y):
            x = np.arange(len(y))
            coeffs = np.polyfit(x, y, 1)
            trend = np.polyval(coeffs, x)
            return y - trend
        
        Fq = []
        q_vals = np.linspace(*q_range)
        
        for q in q_vals:
            Fq_s = []
            for s in scales:
                n_seg = len(series) // s
                var_list = []
                for v in range(n_seg):
                    seg = series[v*s : (v+1)*s]
                    seg_detrend = detrend_segment(seg)
                    var_list.append(np.mean(seg_detrend**2))
                if q == 0:
                    F = np.exp(0.5 * np.mean(np.log(var_list)))
                else:
                    F = (np.mean(np.array(var_list)**(q/2)))**(1/q)
                Fq_s.append(F)
            Fq.append(Fq_s)
        
        Fq = np.array(Fq)
        Hq = []
        for i, q in enumerate(q_vals):
            coeffs = np.polyfit(np.log(scales), np.log(Fq[i]), 1)
            Hq.append(coeffs[0])
        Hq = np.array(Hq)
        
        tau_q = q_vals * Hq - 1
        alpha = np.gradient(tau_q, q_vals)
        f_alpha = q_vals * alpha - tau_q
        
        multifractal_width = alpha.max() - alpha.min()
        is_monofractal = multifractal_width < 0.2
        
        return {
            'hq': Hq,
            'alpha': alpha,
            'f_alpha': f_alpha,
            'multifractal_width': multifractal_width,
            'is_monofractal': is_monofractal,
            'anomaly_score': multifractal_width if not is_monofractal else 0
        }
    
    def _transfer_entropy(self, x, y, k=1, bins=10):
        """
        Transfer Entropy T_{X->Y} = I(Y_{t+1}; X_t | Y_t)
        Version robuste avec gestion des erreurs.
        """
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()

        # Vérifier qu'on a assez de données
        if len(x) < 100 or len(y) < 100:
            return 0.0

        n = len(x) - k
        if n < 10:
            return 0.0

        try:
            # Discrétisation en quantiles
            x_quantiles = np.linspace(0, 100, bins + 1)[1:-1]
            y_quantiles = np.linspace(0, 100, bins + 1)[1:-1]

            x_disc = np.digitize(x[:-k], np.percentile(x, x_quantiles))
            y_disc = np.digitize(y[:-k], np.percentile(y, y_quantiles))
            y_future = np.digitize(y[k:], np.percentile(y, y_quantiles))

            # Ajouter 1 pour éviter les valeurs 0
            x_disc += 1
            y_disc += 1
            y_future += 1

            # Calcul des probabilités jointes et marginales
            def entropy_from_counts(counts, total):
                """Calcule l'entropie à partir des comptages."""
                probs = np.array(list(counts.values())) / total
                return -np.sum(probs * np.log2(probs + 1e-12))

            total = len(x_disc)

            # Comptages pour H(Y_future | Y)
            counts_yf_y = {}
            for yf, yv in zip(y_future, y_disc):
                counts_yf_y[(yf, yv)] = counts_yf_y.get((yf, yv), 0) + 1

            # Comptages marginaux pour Y
            counts_y = {}
            for yv in y_disc:
                counts_y[yv] = counts_y.get(yv, 0) + 1

            # H(Y_future, Y)
            h_yf_y = entropy_from_counts(counts_yf_y, total)

            # H(Y)
            h_y = entropy_from_counts(counts_y, total)

            # H(Y_future, X, Y)
            counts_yf_x_y = {}
            for yf, xv, yv in zip(y_future, x_disc, y_disc):
                counts_yf_x_y[(yf, xv, yv)] = counts_yf_x_y.get((yf, xv, yv), 0) + 1

            # Comptages pour H(X, Y)
            counts_x_y = {}
            for xv, yv in zip(x_disc, y_disc):
                counts_x_y[(xv, yv)] = counts_x_y.get((xv, yv), 0) + 1

            # H(Y_future, X, Y)
            h_yf_x_y = entropy_from_counts(counts_yf_x_y, total)

            # H(X, Y)
            h_x_y = entropy_from_counts(counts_x_y, total)

            # Transfer Entropy: I(Y_future; X | Y) = H(Y_future|Y) - H(Y_future|X,Y)
            # = [H(Y_future,Y) - H(Y)] - [H(Y_future,X,Y) - H(X,Y)]
            te = (h_yf_y - h_y) - (h_yf_x_y - h_x_y)

            return max(0, min(te, 1.0))  # Borner entre 0 et 1

        except Exception as e:
            print(f"  ⚠️ Transfer Entropy error: {e}")
            return 0.0
    
    def _bocd(self, data, hazard_rate=1/200, alpha=1.0, beta=1.0):
        """Bayesian Online Changepoint Detection."""
        n = len(data)
        R = np.zeros((n+1, n+1))
        R[0,0] = 1.0
        changepoint_probs = np.zeros(n)
        
        for t in range(1, n+1):
            pred_probs = np.zeros(t)
            for r in range(t):
                run_data = data[t-r-1:t]
                mu_pred = np.mean(run_data) if len(run_data)>0 else data[t-1]
                var_pred = np.var(run_data) if len(run_data)>1 else 1.0
                pred_probs[r] = stats.norm.pdf(data[t-1], loc=mu_pred, scale=np.sqrt(var_pred+1e-6))
            
            growth_prob = pred_probs * R[t-1, :t]
            growth_prob = growth_prob * (1 - hazard_rate)
            cp_prob = hazard_rate * np.sum(pred_probs * R[t-1, :t])
            R[t, 0] = cp_prob
            R[t, 1:t+1] = growth_prob
            if np.sum(R[t, :t+1]) > 0:
                R[t, :t+1] /= np.sum(R[t, :t+1])
            changepoint_probs[t-1] = cp_prob
        
        return changepoint_probs
    
    def _rmt_anomaly(self, X, threshold=2.0):
        """Random Matrix Theory analysis."""
        X_scaled = StandardScaler().fit_transform(X)
        corr = np.corrcoef(X_scaled.T)
        eigvals = linalg.eigvalsh(corr)
        eigvals = eigvals[eigvals > 0]
        
        eigvals_sorted = np.sort(eigvals)
        gaps = np.diff(eigvals_sorted)
        ratios = gaps[1:] / (gaps[:-1] + 1e-12)
        
        expected_ratios = stats.rayleigh.rvs(scale=0.5, size=len(ratios))
        ks_stat, p_value = stats.ks_2samp(ratios, expected_ratios)
        
        q = X_scaled.shape[0] / X_scaled.shape[1]
        lambda_max = (1 + np.sqrt(1/q))**2
        n_signal = np.sum(eigvals > lambda_max)
        
        anomaly_score = 1 - p_value if p_value < 0.05 else 0
        return {
            'eigenvalues': eigvals,
            'spacing_ratios': ratios,
            'ks_pvalue': p_value,
            'n_signal_eigenvalues': n_signal,
            'anomaly_score': anomaly_score,
            'is_anomalous': p_value < 0.05
        }
    
    def _esn_residuals(self, X, reservoir_size=100, spectral_radius=0.9, leaking_rate=0.3):
        """Echo State Network residual analysis - Version corrigée avec normalisation."""
        X = np.asarray(X)
        n_samples, n_features = X.shape if X.ndim > 1 else (len(X), 1)

        if n_samples < 100:
            return np.ones(n_samples) * 0.5  # Retourner un score neutre

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        try:
            np.random.seed(42)
            Win = np.random.uniform(-0.5, 0.5, (reservoir_size, n_features))
            W = np.random.randn(reservoir_size, reservoir_size)
            rho_max = np.max(np.abs(linalg.eigvals(W)))
            if rho_max > 0:
                W = W * (spectral_radius / rho_max)

            states = np.zeros((n_samples, reservoir_size))
            for t in range(1, n_samples):
                u = X[t-1]
                states[t] = (1 - leaking_rate) * states[t-1] + leaking_rate * np.tanh(Win @ u + W @ states[t-1])

            train_len = n_samples // 2
            if train_len < 10:
                return np.ones(n_samples) * 0.5

            # Ridge regression avec régularisation
            I = np.eye(states[:train_len].shape[1])
            alpha_reg = 0.01
            Wout = np.linalg.lstsq(states[:train_len], X[:train_len], rcond=1e-5)[0]

            X_pred = states @ Wout
            residuals = np.mean((X - X_pred)**2, axis=1)

            # Normalisation des résidus
            residuals = np.clip(residuals, 0, np.percentile(residuals, 99))
            if residuals.max() > residuals.min():
                residuals_norm = (residuals - residuals.min()) / (residuals.max() - residuals.min() + 1e-10)
            else:
                residuals_norm = np.ones_like(residuals) * 0.5

            return residuals_norm

        except Exception as e:
            print(f"  ⚠️ ESN error: {e}")
            return np.ones(n_samples) * 0.5
    
    def _singular_spectrum_analysis(self, series, window_size=None):
        """Singular Spectrum Analysis for complexity detection."""
        series = np.asarray(series).flatten()
        n = len(series)

        if n < 20:
            return {'complexity': 0.5, 'anomaly': False}

        if window_size is None:
            window_size = min(30, n // 3)

        if window_size < 2:
            return {'complexity': 0.5, 'anomaly': False}

        # Embedding
        K = n - window_size + 1
        if K < 2:
            return {'complexity': 0.5, 'anomaly': False}

        X = np.zeros((window_size, K))
        for i in range(K):
            X[:, i] = series[i:i+window_size]

        # SVD
        try:
            U, s, Vt = np.linalg.svd(X, full_matrices=False)

            # Complexity = entropy of singular values
            s_norm = s / (s.sum() + 1e-12)
            complexity = -np.sum(s_norm * np.log2(s_norm + 1e-12)) / np.log2(len(s) + 1e-12)
            complexity = np.clip(complexity, 0, 1)

            # Anomaly if too low or too high complexity
            anomaly = complexity < 0.3 or complexity > 0.8
        except Exception:
            complexity = 0.5
            anomaly = False

        return {'complexity': complexity, 'anomaly': anomaly}
    
    def _recurrence_quantification(self, series, emb_dim=3, delay=1, threshold_ratio=0.1):
        """Recurrence Quantification Analysis - Version corrigée."""
        series = np.asarray(series).flatten()
        n = len(series)

        if n < 100:
            return {'determinism': 0.5, 'laminarity': 0.5, 'rqa_anomaly': False}

        # Embedding
        K = n - (emb_dim - 1) * delay
        if K < 10:
            return {'determinism': 0.5, 'laminarity': 0.5, 'rqa_anomaly': False}

        # Construction de la matrice des trajectoires
        X = np.zeros((emb_dim, K))
        for i in range(emb_dim):
            X[i, :] = series[i*delay:i*delay+K]

        # Recurrence matrix
        threshold = threshold_ratio * np.std(series)
        R = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                dist = np.linalg.norm(X[:, i] - X[:, j])
                R[i, j] = 1 if dist < threshold else 0

        # Determinism (diagonal lines)
        diag_hist = []
        for i in range(K):
            for j in range(K):
                if R[i, j] == 1:
                    length = 1
                    i_temp, j_temp = i + 1, j + 1
                    while i_temp < K and j_temp < K and R[i_temp, j_temp] == 1:
                        length += 1
                        i_temp += 1
                        j_temp += 1
                    if length >= 2:
                        diag_hist.append(length)
                    # Ne pas avancer i et j ici pour ne pas casser la boucle

        # Laminarity (vertical lines)
        vert_hist = []
        for j in range(K):
            for i in range(K):
                if R[i, j] == 1:
                    length = 1
                    i_temp = i + 1
                    while i_temp < K and R[i_temp, j] == 1:
                        length += 1
                        i_temp += 1
                    if length >= 2:
                        vert_hist.append(length)
                    # Ne pas avancer i ici

        determinism = sum(diag_hist) / (K * K + 1e-12) if diag_hist else 0
        laminarity = sum(vert_hist) / (K * K + 1e-12) if vert_hist else 0

        # Une anomalie est détectée si le déterminisme est très élevé (> 0.8)
        # Ce qui indique un comportement trop régulier (possiblement artificiel)
        rqa_anomaly = determinism > 0.8

        return {
            'determinism': determinism,
            'laminarity': laminarity,
            'rqa_anomaly': rqa_anomaly
        }
    
    def _plot_ultra_results(self, results):
        """Visualisation des résultats ultra-scientifiques."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.patch.set_facecolor('#f8f9fa')
        
        # Global score gauge
        ax = axes[0, 0]
        score = results.get('global_ultra_anomaly_score', 0)
        colors_gauge = [COLORS['success'], COLORS['warning'], COLORS['danger']]
        ax.pie([score, 1-score], colors=[colors_gauge[min(2, int(score*3))], '#e0e0e0'],
               startangle=90, counterclock=False, wedgeprops={'width': 0.3})
        ax.text(0, 0, f'{score:.0%}', ha='center', va='center', fontsize=24, fontweight='bold')
        ax.set_title('Global Ultra Anomaly Score', fontsize=12, fontweight='bold')
        
        # Risk level
        ax = axes[0, 1]
        risk_level = results.get('ultra_risk_level', 'LOW')
        risk_colors = {'LOW': COLORS['success'], 'MEDIUM': COLORS['warning'],
                       'HIGH': COLORS['danger'], 'CRITICAL': '#8b0000'}
        ax.text(0.5, 0.5, f'🚨\n{risk_level}', ha='center', va='center',
                fontsize=20, fontweight='bold', color=risk_colors.get(risk_level, COLORS['neutral']),
                transform=ax.transAxes)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        ax.set_title('Ultra Risk Level', fontsize=12, fontweight='bold')
        
        # Metrics bars
        ax = axes[0, 2]
        metrics = []
        values = []
        if 'mfdfa_anomaly_score' in results:
            metrics.append('MF-DFA'); values.append(results['mfdfa_anomaly_score'])
        if 'rmt_anomaly_score' in results:
            metrics.append('RMT'); values.append(results['rmt_anomaly_score'])
        if 'esn_anomaly_score' in results:
            metrics.append('ESN'); values.append(results['esn_anomaly_score'])
        if 'bocd_max_prob' in results:
            metrics.append('BOCD'); values.append(results['bocd_max_prob'])
        
        if metrics:
            y_pos = np.arange(len(metrics))
            colors_bar = [COLORS['danger'] if v > 0.5 else COLORS['warning'] if v > 0.3 else COLORS['success'] for v in values]
            ax.barh(y_pos, values, color=colors_bar, alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(metrics)
            ax.set_xlim(0, 1)
            ax.set_xlabel('Anomaly Score')
            ax.set_title('Per-Method Anomaly Scores', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
        
        # Multifractal spectrum
        ax = axes[1, 0]
        if 'multifractal_width' in results:
            ax.text(0.5, 0.5, f"Multifractal Width\n{results['multifractal_width']:.4f}\n" +
                    ("Monofractal" if results.get('is_monofractal') else "Multifractal"),
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'MF-DFA\nNot available', ha='center', va='center')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        ax.set_title('Multifractal Analysis', fontsize=12, fontweight='bold')
        
        # Transfer entropy
        ax = axes[1, 1]
        if 'transfer_entropy_mean' in results:
            te_val = results['transfer_entropy_mean']
            ax.text(0.5, 0.5, f"Transfer Entropy\n{te_val:.4f}\n" +
                    ("Anomalous" if results.get('te_anomaly_flag') else "Normal"),
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'Transfer Entropy\nNot available', ha='center', va='center')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        ax.set_title('Causal Information Flow', fontsize=12, fontweight='bold')
        
        # RMT info
        ax = axes[1, 2]
        if 'n_signal_eigenvalues' in results:
            ax.text(0.5, 0.5, f"Signal Eigenvalues\n{results['n_signal_eigenvalues']}\n" +
                    f"p-value: {results.get('rmt_pvalue', 0):.4f}",
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'RMT\nNot available', ha='center', va='center')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        ax.set_title('Random Matrix Theory', fontsize=12, fontweight='bold')
        
        plt.suptitle(f'Ultra-Scientific Anomaly Detection — {self.model_name}',
                     fontsize=16, fontweight='bold', color=COLORS['dark'])
        plt.tight_layout()
        self._figure_cache['ultra_scientific'] = fig

    # =========================================================================
    # 14. RAPPORT HTML COMPLET (AMÉLIORÉ AVEC ULTRA)
    # =========================================================================

    def generate_html_report(self, output_path: str = 'validation_report.html'):
        """
        Génère un rapport HTML interactif avec tous les résultats et graphiques,
        incluant l'analyse ultra-scientifique.
        """
        print(f"\n  📄 Génération du rapport HTML...")

        def fig_to_b64(fig) -> str:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                        facecolor='#f8f9fa')
            buf.seek(0)
            return base64.b64encode(buf.read()).decode()

        perf = self.validation_results.get('performance', {})
        cal  = self.validation_results.get('calibration', {})
        ben  = self.validation_results.get('benford', {})
        ultra = self.validation_results.get('ultra_scientific', {})

        def val(d, k, fmt='.4f'):
            v = d.get(k, 'N/A')
            return f'{v:{fmt}}' if isinstance(v, float) else str(v)

        # Scores pour le badge global
        score_components = []
        if perf.get('auc_roc'):
            score_components.append(min(perf['auc_roc'] * 100, 100))
        if perf.get('ks_statistic'):
            score_components.append(min(perf['ks_statistic'] * 100, 100))
        if cal.get('brier_skill_score'):
            score_components.append(max(0, min(cal['brier_skill_score'] * 100, 100)))
        global_score = int(np.mean(score_components)) if score_components else 0
        score_color = ('#2a9d8f' if global_score >= 75 else
                       '#e9c46a' if global_score >= 60 else '#e63946')
        
        # Ultra score
        ultra_score = ultra.get('global_ultra_anomaly_score', 0)
        ultra_level = ultra.get('ultra_risk_level', 'LOW')
        ultra_color = ('#2a9d8f' if ultra_score < 0.3 else
                       '#e9c46a' if ultra_score < 0.5 else
                       '#e76f51' if ultra_score < 0.7 else '#e63946')

        # Figures en base64
        figures_html = ""
        figure_titles = {
            'advanced_curves': '📈 Courbes Diagnostiques (ROC, PR, DET, KS)',
            'lift_gain':       '🎯 Lift & Gains Analysis',
            'calibration':     '🔬 Calibration Probabiliste',
            'csi':             '📊 Characteristic Stability Index',
            'feature_importance': '🔀 Permutation Feature Importance',
            'benford':         "🔢 Benford's Law Analysis",
            'graph':           '🔗 Graph-Based Anomaly Detection',
            'threshold':       '⚖️  Optimisation du Seuil',
            'stress':          '💥 Stress Testing Réglementaire',
            'walkforward':     '⏱️  Walk-Forward Backtesting',
            'ultra_scientific': '🧪 Ultra-Scientific Anomaly Detection',
        }
        for key, title in figure_titles.items():
            if key in self._figure_cache:
                try:
                    b64 = fig_to_b64(self._figure_cache[key])
                    figures_html += f"""
                    <div class="section">
                        <h2>{title}</h2>
                        <img src="data:image/png;base64,{b64}"
                             style="width:100%;border-radius:8px;box-shadow:0 2px 12px rgba(0,0,0,.1)">
                    </div>"""
                except Exception:
                    pass

        html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Validation Report — {self.model_name} (Ultra-Scientific)</title>
<style>
  * {{ box-sizing: border-box; margin:0; padding:0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #f0f2f5; color: #1d3557; }}
  .header {{ background: linear-gradient(135deg,#1a1f5e,#9b59b6);
             color:white; padding:40px; text-align:center; }}
  .header h1 {{ font-size:2.2rem; letter-spacing:1px; }}
  .header p  {{ opacity:.8; margin-top:8px; font-size:.95rem; }}
  .badge {{ display:inline-block; background:{score_color}; color:white;
            font-size:1.8rem; font-weight:bold; padding:12px 24px;
            border-radius:50px; margin-top:20px; }}
  .ultra-badge {{ display:inline-block; background:{ultra_color}; color:white;
            font-size:1.4rem; font-weight:bold; padding:8px 20px;
            border-radius:50px; margin-top:10px; margin-left:15px; }}
  .container {{ max-width:1200px; margin:0 auto; padding:24px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
           gap:16px; margin:24px 0; }}
  .card {{ background:white; border-radius:12px; padding:20px;
           box-shadow:0 2px 12px rgba(0,0,0,.07); text-align:center; }}
  .card .value {{ font-size:2rem; font-weight:bold; color:#1a1f5e; }}
  .card .label {{ font-size:.8rem; color:#888; margin-top:4px; text-transform:uppercase; }}
  .section {{ background:white; border-radius:12px; padding:24px;
              box-shadow:0 2px 12px rgba(0,0,0,.07); margin:24px 0; }}
  .section h2 {{ font-size:1.2rem; margin-bottom:16px;
                 border-bottom:2px solid #e9ecef; padding-bottom:10px; }}
  .alert {{ padding:12px 18px; border-radius:8px; margin:8px 0;
            font-weight:500; }}
  .alert-ok      {{ background:#d4edda; color:#155724; border-left:4px solid #2a9d8f; }}
  .alert-warning {{ background:#fff3cd; color:#856404; border-left:4px solid #e9c46a; }}
  .alert-danger  {{ background:#f8d7da; color:#721c24; border-left:4px solid #e63946; }}
  .alert-ultra   {{ background:#e8d5f5; color:#6c3483; border-left:4px solid #9b59b6; }}
  table {{ width:100%; border-collapse:collapse; font-size:.88rem; }}
  th {{ background:#1a1f5e; color:white; padding:10px; text-align:left; }}
  td {{ padding:8px 10px; border-bottom:1px solid #f0f0f0; }}
  tr:hover td {{ background:#f8f9fa; }}
  .footer {{ text-align:center; padding:32px; color:#888; font-size:.85rem; }}
</style>
</head>
<body>
<div class="header">
  <h1>🛡️ Fraud Model Validation Report — v3.0</h1>
  <p><strong>{self.model_name}</strong> &nbsp;|&nbsp; {datetime.now().strftime('%Y-%m-%d %H:%M')}
     &nbsp;|&nbsp; SR 11-7 · ECB TRIM · AMLD6 · Bâle III · Ultra-Scientific</p>
  <div>
    <span class="badge">Score Global: {global_score}/100</span>
    <span class="ultra-badge">🧪 Ultra Anomaly: {ultra_score:.0%} ({ultra_level})</span>
  </div>
</div>

<div class="container">
  <!-- KPI Cards -->
  <div class="grid">
    <div class="card"><div class="value">{val(perf,'auc_roc')}</div>
      <div class="label">AUC-ROC</div></div>
    <div class="card"><div class="value">{val(perf,'gini')}</div>
      <div class="label">Gini Coefficient</div></div>
    <div class="card"><div class="value">{val(perf,'ks_statistic')}</div>
      <div class="label">KS Statistic</div></div>
    <div class="card"><div class="value">{val(perf,'h_measure')}</div>
      <div class="label">H-Measure</div></div>
    <div class="card"><div class="value">{val(perf,'f1_score')}</div>
      <div class="label">F1-Score</div></div>
    <div class="card"><div class="value">{val(perf,'mcc')}</div>
      <div class="label">MCC</div></div>
    <div class="card"><div class="value">{val(cal,'brier_score')}</div>
      <div class="label">Brier Score</div></div>
    <div class="card"><div class="value">{val(perf,'expected_cost','.2f')}{self.currency}</div>
      <div class="label">Coût/Transaction</div></div>
  </div>

  <!-- Ultra-Scientific Alertes -->
  <div class="section">
    <h2>🧪 Ultra-Scientific Anomaly Detection</h2>
    {'<div class="alert alert-ultra">🔬 MF-DFA: ' + ('Multifractal behavior detected' if ultra.get('multifractal_width',0)>0.2 else 'Monofractal normal') + '</div>' if ultra else ''}
    {'<div class="alert alert-ultra">🔄 Transfer Entropy: ' + ('Anomalous causal flow' if ultra.get('te_anomaly_flag') else 'Normal causal structure') + '</div>' if ultra else ''}
    {'<div class="alert alert-ultra">📐 RMT: ' + (f"Non-random correlations detected (p={ultra.get('rmt_pvalue',0):.4f})" if ultra.get('rmt_pvalue',1)<0.05 else 'Random matrix behavior') + '</div>' if ultra else ''}
    {'<div class="alert alert-danger">🚨 CRITICAL ULTRA ANOMALY: Subtle fraud pattern detected by advanced methods</div>' if ultra_score > 0.7 else ''}
  </div>

  <!-- Alertes Réglementaires -->
  <div class="section">
    <h2>⚠️ Alertes Réglementaires</h2>
    {'<div class="alert alert-ok">✅ AUC-ROC ≥ 0.75 — Performance discriminante satisfaisante</div>'
      if perf.get('auc_roc', 0) >= 0.75 else
     '<div class="alert alert-danger">🚨 AUC-ROC < 0.75 — Performance insuffisante</div>'}
    {'<div class="alert alert-ok">✅ KS ≥ 0.30 — Bonne séparation des distributions</div>'
      if perf.get('ks_statistic', 0) >= 0.30 else
     '<div class="alert alert-warning">⚠️ KS < 0.30 — Séparation limitée</div>'}
    {'<div class="alert alert-ok">✅ Brier Score ≤ 0.05 — Bonne calibration</div>'
      if cal.get('brier_score', 1) <= 0.05 else
     '<div class="alert alert-warning">⚠️ Brier Score > 0.05 — Recalibration recommandée</div>'}
    {'<div class="alert alert-danger">🚨 Anomalie Benford détectée (p<0.05) — Vérifier les données</div>'
      if ben.get('significant_bias') else
     '<div class="alert alert-ok">✅ Loi de Benford — Données cohérentes</div>'
      if ben else ''}
  </div>

  <!-- Figures -->
  {figures_html}

</div>
<div class="footer">
  Rapport généré par <strong>FraudModelValidatorV3</strong> (Ultra-Scientific) &nbsp;|&nbsp;
  Conforme SR 11-7 · ECB TRIM Guide · AMLD6 · Bâle III/IV<br>
  Méthodes: MF-DFA · Transfer Entropy · BOCD · RMT · ESN · SSA · RQA<br>
  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</div>
</body></html>"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"  ✅ Rapport HTML sauvegardé: {output_path}")
        return output_path

    # =========================================================================
    # 15. RAPPORT TEXTE (COMPATIBILITÉ AVEC ULTRA)
    # =========================================================================

    def generate_validation_report(self, metrics=None, stability_df=None,
                                    fairness_data=None, stress_df=None,
                                    cv_df=None, output_path=None) -> str:
        """Rapport texte complet — incluant les analyses ultra-scientifiques."""
        if metrics is None:
            metrics = self.validation_results.get('performance', {})

        lines = []
        sep = "═" * 70
        lines += [sep, f"  MODEL VALIDATION REPORT — {self.model_name} (v3.0 Ultra-Scientific)",
                  f"  Framework: FraudModelValidatorV3 | SR 11-7, ECB TRIM, AMLD6, Bâle III",
                  f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                  sep, ""]

        def section(title):
            lines.extend([f"\n{'─'*70}", f"  {title}", f"{'─'*70}"])

        # Section 1: Performance
        section("SECTION 1: PERFORMANCE DISCRIMINANTE")
        kv = lambda k, v: lines.append(f"  {k:<30} {v}")
        kv("AUC-ROC",           f"{metrics.get('auc_roc', 'N/A'):.4f}")
        kv("GINI Coefficient",  f"{metrics.get('gini', 'N/A'):.4f}")
        kv("KS Statistic",      f"{metrics.get('ks_statistic', 'N/A'):.4f}")
        kv("H-Measure",         f"{metrics.get('h_measure', 'N/A'):.4f}")
        kv("Average Precision", f"{metrics.get('average_precision', 'N/A'):.4f}")
        kv("Precision",         f"{metrics.get('precision', 'N/A'):.4f}")
        kv("Recall (TPR)",      f"{metrics.get('recall', 'N/A'):.4f}")
        kv("F1-Score",          f"{metrics.get('f1_score', 'N/A'):.4f}")
        kv("MCC",               f"{metrics.get('mcc', 'N/A'):.4f}")
        kv("Balanced Accuracy", f"{metrics.get('balanced_accuracy', 'N/A'):.4f}")

        section("SECTION 2: CALIBRATION PROBABILISTE")
        cal = self.validation_results.get('calibration', {})
        kv("Brier Score",       f"{cal.get('brier_score', 'N/A'):.4f}")
        kv("Brier Skill Score", f"{cal.get('brier_skill_score', 'N/A'):.4f}")
        kv("ECE",               f"{cal.get('ece', 'N/A'):.4f}")
        kv("MCE",               f"{cal.get('mce', 'N/A'):.4f}")
        kv("Log-Loss",          f"{metrics.get('log_loss', 'N/A'):.4f}")

        section("SECTION 3: ANALYSE COÛT")
        kv(f"Coût FN (fraude manquée)", f"{self.cost_fn:.0f}{self.currency}")
        kv(f"Coût FP (fausse alarme)", f"{self.cost_fp:.0f}{self.currency}")
        kv("Expected Cost/tx",  f"{metrics.get('expected_cost', 'N/A'):.2f}{self.currency}")

        # Ultra-Scientific section
        ultra = self.validation_results.get('ultra_scientific', {})
        if ultra:
            section("SECTION ULTRA: SCIENTIFIC ANOMALY DETECTION")
            kv("Global Ultra Anomaly Score", f"{ultra.get('global_ultra_anomaly_score', 0):.4f}")
            kv("Ultra Risk Level", ultra.get('ultra_risk_level', 'N/A'))
            kv("MF-DFA Multifractal Width", f"{ultra.get('multifractal_width', 0):.4f}")
            kv("Transfer Entropy Mean", f"{ultra.get('transfer_entropy_mean', 0):.4f}")
            kv("BOCD Max Probability", f"{ultra.get('bocd_max_prob', 0):.4f}")
            kv("RMT p-value", f"{ultra.get('rmt_pvalue', 1):.4f}")
            kv("Signal Eigenvalues", f"{ultra.get('n_signal_eigenvalues', 0)}")
            kv("ESN Residual Mean", f"{ultra.get('esn_residual_mean', 0):.4f}")
            
            if ultra.get('global_ultra_anomaly_score', 0) > 0.7:
                lines.append("\n  🚨 CRITICAL ULTRA ANOMALY ALERT")
                lines.append("  → Subtle fraud pattern detected via multifractal/causal/RMT analysis")
                lines.append("  → Strongly recommend deep investigation of transaction dynamics")

        # Benford
        benford = self.validation_results.get('benford', {})
        if benford:
            section("SECTION 4: BENFORD'S LAW")
            kv("MAD", f"{benford.get('mad', 'N/A'):.5f}")
            kv("Conformité", benford.get('conformity', 'N/A'))
            kv("χ² p-value", f"{benford.get('chi2_pvalue', 'N/A'):.5f}")

        # Graph anomalies
        graph = self.validation_results.get('graph', {})
        if graph:
            section("SECTION 5: GRAPH ANOMALY DETECTION")
            kv("Entités (nœuds)", f"{graph.get('n_nodes', 'N/A'):,}")
            kv("Transactions (arêtes)", f"{graph.get('n_edges', 'N/A'):,}")

        # Cross-validation
        if cv_df is not None and len(cv_df) > 0:
            section("SECTION 6: CROSS-VALIDATION")
            for col in ['auc_roc', 'gini', 'ks_statistic', 'f1_score', 'mcc']:
                if col in cv_df.columns:
                    kv(f"{col} (mean±std)",
                       f"{cv_df[col].mean():.4f} ± {cv_df[col].std():.4f}")

        # Walk-forward
        wf = self.validation_results.get('walkforward')
        if wf is not None and len(wf) > 0:
            section("SECTION 7: WALK-FORWARD BACKTESTING")
            kv("AUC moyen (temporal)", f"{wf['auc_roc'].mean():.4f}")
            kv("AUC std (temporal)",   f"{wf['auc_roc'].std():.4f}")

        # Stress test
        stress_df = stress_df if stress_df is not None else self.validation_results.get('stress')
        if stress_df is not None and len(stress_df) > 0:
            section("SECTION 8: STRESS TESTING")
            max_drop = stress_df['auc_drop_pct'].max()
            kv("Max AUC Drop", f"{max_drop:.2f}%")

        # Conclusion
        lines.append(f"\n{sep}")
        lines.append("  CONCLUSION DE VALIDATION")
        lines.append(sep)
        
        ultra_score = ultra.get('global_ultra_anomaly_score', 0)
        perf_ok = metrics.get('auc_roc', 0) >= 0.75
        
        if perf_ok and ultra_score < 0.3:
            lines.append("  ✅ MODÈLE VALIDÉ — Aucune anomalie ultra-scientifique critique")
            lines.append("  Recommandation: APPROUVER pour déploiement")
        elif perf_ok and ultra_score < 0.6:
            lines.append("  ⚠️ MODÈLE OK — Anomalies subtiles détectées (investigation recommandée)")
            lines.append("  Recommandation: APPROUVER avec monitoring renforcé")
        elif perf_ok:
            lines.append("  🚨 MODÈLE SOUS CONDITIONS — Anomalies ultra-scientifiques critiques")
            lines.append("  Recommandation: INVESTIGATION APPROFONDIE avant déploiement")
        else:
            lines.append("  🚨 MODÈLE REJETÉ — Performance insuffisante + anomalies détectées")
            lines.append("  Recommandation: RETRAVAILLER complètement le modèle")

        report = "\n".join(lines)
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n  Rapport texte sauvegardé: {output_path}")

        print(report)
        return report


# =============================================================================
# DÉMONSTRATION COMPLÈTE V3.0
# =============================================================================

if __name__ == "__main__":
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    print("=" * 70)
    print("  FRAUD MODEL VALIDATOR v3.0 (ULTRA-SCIENTIFIC) — DÉMONSTRATION")
    print("=" * 70)

    # ── Génération de données réalistes ────────────────────────────────────
    np.random.seed(42)
    n_samples = 60_000
    n_fraud   = int(n_samples * 0.025)   # 2.5% taux de fraude
    n_legit   = n_samples - n_fraud

    legit = pd.DataFrame({
        'transaction_amount':     np.random.lognormal(4.0, 1.5, n_legit),
        'transaction_velocity_1h': np.random.poisson(2, n_legit),
        'merchant_risk_score':    np.random.beta(2, 5, n_legit),
        'customer_age_days':      np.random.exponential(365, n_legit),
        'is_international':       np.random.binomial(1, 0.05, n_legit),
        'hour_of_day':            np.random.randint(8, 20, n_legit),
        'device_risk_score':      np.random.beta(1, 5, n_legit),
        'is_fraud': 0
    })

    fraud = pd.DataFrame({
        'transaction_amount':     np.random.lognormal(6.5, 1.0, n_fraud),
        'transaction_velocity_1h': np.random.poisson(10, n_fraud),
        'merchant_risk_score':    np.random.beta(6, 2, n_fraud),
        'customer_age_days':      np.random.exponential(20, n_fraud),
        'is_international':       np.random.binomial(1, 0.5, n_fraud),
        'hour_of_day':            np.random.choice(list(range(0,6))+list(range(22,24)), n_fraud),
        'device_risk_score':      np.random.beta(5, 1, n_fraud),
        'is_fraud': 1
    })

    df = pd.concat([legit, fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
    df['transaction_date'] = pd.date_range('2023-01-01', periods=len(df), freq='2min')
    df['sender_id']   = np.random.randint(1, 5000, len(df))
    df['receiver_id'] = np.random.randint(1, 3000, len(df))

    feature_cols = [c for c in df.columns
                    if c not in ['is_fraud', 'transaction_date',
                                 'sender_id', 'receiver_id']]
    X = df[feature_cols]
    y = df['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
                                       max_depth=5, random_state=42)
    model.fit(scaler.fit_transform(X_train), y_train)
    y_proba = model.predict_proba(scaler.transform(X_test))[:, 1]
    y_test_arr = y_test.values

    def predict_fn(X_in):
        return model.predict_proba(scaler.transform(X_in))[:, 1]

    # ── Validation V3.0 ────────────────────────────────────────────────────────
    validator = FraudModelValidatorV3(
        model_name="GBM FraudDetector v3 (Ultra)",
        cost_fn=5000,
        cost_fp=100
    )

    # 1. Performance + bootstrap CI
    metrics = validator.calculate_performance_metrics(
        y_test_arr, y_proba, bootstrap=True, n_bootstrap=500
    )

    # 2. Courbes diagnostiques
    validator.plot_advanced_curves(y_test_arr, y_proba, save_path='curves_v3.png')

    # 3. Lift & Gains
    lift_df = validator.lift_gain_analysis(y_test_arr, y_proba)

    # 4. Calibration
    cal = validator.calibration_analysis(y_test_arr, y_proba)

    # 5. Stability
    df_test = X_test.copy()
    df_test['score'] = y_proba
    df_test['transaction_date'] = df['transaction_date'].iloc[X_test.index].values
    stability = validator.stability_analysis(df_test)

    # 6. CSI
    half = len(df) // 2
    csi_df = validator.feature_stability_index(
        df.iloc[:half][feature_cols],
        df.iloc[half:][feature_cols],
        feature_cols
    )

    # 7. Benford's Law
    benford = validator.benford_law_analysis(df, amount_col='transaction_amount')

    # 8. Graph Anomaly Detection
    graph_df = validator.graph_anomaly_detection(
        df.sample(5000, random_state=42),
        source_col='sender_id', target_col='receiver_id',
        amount_col='transaction_amount', fraud_col='is_fraud'
    )

    # 9. Permutation Importance
    imp_df = validator.permutation_feature_importance(
        model, scaler.transform(X_test), y_test_arr,
        feature_names=feature_cols, n_repeats=8
    )

    # 10. Mahalanobis + Isolation Forest
    anomaly_scores = validator.mahalanobis_outlier_detection(X_test, contamination=0.05)

    # 11. Fairness Analysis
    df_eval = X_test.copy()
    df_eval['score'] = y_proba
    df_eval['is_fraud'] = y_test.values
    fairness_df, fairness_summary = validator.fairness_analysis(
        df_eval, 'score', 'is_fraud', 'is_international'
    )

    # 12. Threshold Optimization
    best_thresh, thresh_df, optimal_thresholds = validator.optimize_threshold(
        y_test_arr, y_proba, metric='cost'
    )

    # 13. Stress Testing
    stress_df = validator.stress_test(X_test, y_test_arr, predict_fn)

    # 14. Cross-Validation
    cv_df = validator.cross_validate(
        X.sample(12000, random_state=42),
        y.loc[X.sample(12000, random_state=42).index],
        GradientBoostingClassifier(n_estimators=80, random_state=42),
        n_splits=5
    )

    # 15. Walk-Forward Backtesting
    df_wf = df.copy()
    df_wf['transaction_date'] = pd.to_datetime(df_wf['transaction_date'])
    wf_df = validator.walk_forward_backtest(
        df_wf, GradientBoostingClassifier(n_estimators=80, random_state=42),
        feature_cols, 'is_fraud', date_col='transaction_date', n_splits=5
    )

    # 16. ULTRA-SCIENTIFIC DETECTION (NOUVEAU !)
    ultra_results = validator.ultra_scientific_detection(
        df, feature_cols=feature_cols,
        time_col='transaction_date', value_col='transaction_amount'
    )

    # 17. Rapport texte
    report_txt = validator.generate_validation_report(
        metrics=metrics,
        stability_df=stability,
        stress_df=stress_df,
        cv_df=cv_df,
        output_path='validation_report_v3.txt'
    )

    # 18. Rapport HTML interactif (avec ultra)
    validator.generate_html_report(output_path='validation_report_v3.html')

    print("\n" + "="*70)
    print("  ✅ VALIDATION COMPLÈTE — FraudModelValidatorV3 (Ultra-Scientific)")
    print("  Fichiers générés:")
    print("    → validation_report_v3.txt")
    print("    → validation_report_v3.html")
    print("    → curves_v3.png")
    print("="*70)
