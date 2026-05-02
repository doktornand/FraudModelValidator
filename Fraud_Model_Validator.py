
"""
================================================================================
FINCRIME & FRAUD MODEL VALIDATOR
================================================================================
Framework complet de validation de modèle pour la détection de fraude
et la criminalité financière (FinCrime).

Auteur: Generated for Model Validation
Date: 2026-05-02

Couverture:
- Performance metrics (AUC, Precision, Recall, F1, MCC)
- Stability testing (PSI - Population Stability Index)
- Bias & Fairness analysis
- Threshold optimization (cost-based)
- Stress testing
- Cross-validation stratifiée
- Rapport de validation réglementaire

Usage:
    from fraud_model_validator import FraudModelValidator

    validator = FraudModelValidator(model_name="MonModele")
    metrics = validator.calculate_performance_metrics(y_true, y_proba)
    validator.plot_roc_pr_curves(y_true, y_proba)
    stability = validator.stability_analysis(df, score_col='score')
    fairness, pval = validator.fairness_analysis(df, 'score', 'is_fraud', 'segment')
    best_thresh, _ = validator.optimize_threshold(y_true, y_proba)
    stress = validator.stress_test(X, y_true, model_predict_fn)
    cv = validator.cross_validate(X, y, model, n_splits=5)
    report = validator.generate_validation_report(metrics, stability, fairness, stress, cv)
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix, roc_curve, f1_score,
    precision_score, recall_score, matthews_corrcoef
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class FraudModelValidator:
    """
    Validateur de modèle pour la détection de fraude et criminalité financière.

    Conforme aux standards réglementaires (SR 11-7, ECB Guide, etc.)
    """

    def __init__(self, model_name="FraudDetectionModel"):
        self.model_name = model_name
        self.validation_results = {}

    # =========================================================================
    # 1. MÉTRIQUES DE PERFORMANCE
    # =========================================================================

    def calculate_performance_metrics(self, y_true, y_pred_proba, y_pred=None, threshold=0.5):
        """
        Calcule l'ensemble des métriques de performance requises.

        Args:
            y_true: Labels réels (0/1)
            y_pred_proba: Probabilités prédites
            y_pred: Prédictions binaires (optionnel)
            threshold: Seuil de décision

        Returns:
            dict: Dictionnaire de métriques
        """
        if y_pred is None:
            y_pred = (y_pred_proba >= threshold).astype(int)

        metrics = {
            'auc_roc': roc_auc_score(y_true, y_pred_proba),
            'average_precision': average_precision_score(y_true, y_pred_proba),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'threshold': threshold
        }

        # Matrice de confusion détaillée
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0

        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Cost-based metric
        fraud_cost = 1000   # Coût moyen d'une fraude non détectée
        review_cost = 50    # Coût d'une revue manuelle (FP)
        metrics['expected_cost'] = (fn * fraud_cost + fp * review_cost) / len(y_true)

        return metrics

    def plot_roc_pr_curves(self, y_true, y_pred_proba, save_path=None):
        """
        Génère les courbes ROC et Precision-Recall.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        axes[0].plot(fpr, tpr, label=f'AUC = {auc:.4f}', linewidth=2)
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        axes[1].plot(recall, precision, label=f'AP = {avg_precision:.4f}', linewidth=2)
        axes[1].axhline(y=y_true.mean(), color='r', linestyle='--', 
                       label=f'Baseline = {y_true.mean():.4f}')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f'Model Performance - {self.model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

        return fig

    # =========================================================================
    # 2. STABILITÉ TEMPORELLE (PSI)
    # =========================================================================

    def calculate_psi(self, expected, actual, bins=10):
        """
        Calcule le Population Stability Index (PSI).

        Seuils:
            PSI < 0.1    : Pas de changement significatif
            0.1 <= PSI < 0.25 : Changement modéré (surveillance)
            PSI >= 0.25   : Changement significatif (action requise)
        """
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

        psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
        psi = np.sum(psi_values)

        return psi, psi_values, breakpoints

    def stability_analysis(self, df, score_col='score', date_col='transaction_date', 
                          period='M', baseline_end=None):
        """
        Analyse la stabilité temporelle du modèle par période.

        Args:
            df: DataFrame avec scores et dates
            score_col: Nom de la colonne de score
            date_col: Nom de la colonne de date
            period: 'D'=jour, 'W'=semaine, 'M'=mois
            baseline_end: Date de fin de la période de référence
        """
        df = df.copy()
        df['period'] = df[date_col].dt.to_period(period)

        if baseline_end is None:
            baseline_end = df[date_col].quantile(0.5)

        baseline = df[df[date_col] < baseline_end][score_col]
        periods = df['period'].unique()

        stability_results = []
        for period_val in sorted(periods):
            current = df[df['period'] == period_val][score_col]
            if len(current) > 100 and len(baseline) > 100:
                psi, psi_values, bins = self.calculate_psi(baseline.values, current.values)
                stability_results.append({
                    'period': str(period_val),
                    'psi': psi,
                    'status': 'Stable' if psi < 0.1 else 'Moderate' if psi < 0.25 else 'Unstable',
                    'n_samples': len(current)
                })

        return pd.DataFrame(stability_results)

    def plot_stability(self, stability_df, save_path=None):
        """Visualise l'évolution du PSI dans le temps."""
        fig, ax = plt.subplots(figsize=(12, 5))

        colors = {'Stable': 'green', 'Moderate': 'orange', 'Unstable': 'red'}
        bar_colors = [colors.get(s, 'gray') for s in stability_df['status']]

        bars = ax.bar(stability_df['period'], stability_df['psi'], color=bar_colors, alpha=0.7)
        ax.axhline(y=0.1, color='orange', linestyle='--', label='Seuil modéré (0.1)')
        ax.axhline(y=0.25, color='red', linestyle='--', label='Seuil critique (0.25)')

        ax.set_xlabel('Période')
        ax.set_ylabel('PSI')
        ax.set_title('Stabilité Temporelle du Modèle (PSI)')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)

        for bar, val in zip(bars, stability_df['psi']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

        return fig

    # =========================================================================
    # 3. ANALYSE DE BIAIS ET ÉQUITÉ (FAIRNESS)
    # =========================================================================

    def fairness_analysis(self, df, score_col, target_col, protected_attr, 
                         threshold=0.5, save_path=None):
        """
        Analyse l'équité du modèle selon un attribut protégé.

        Vérifie:
        - Equalized Odds (TPR et FPR similaires entre groupes)
        - Demographic Parity (taux de prédiction positifs similaires)
        """
        df = df.copy()
        df['pred'] = (df[score_col] >= threshold).astype(int)

        groups = df[protected_attr].unique()
        fairness_results = []

        for group in groups:
            subset = df[df[protected_attr] == group]
            if len(subset) < 10:
                continue

            y_true = subset[target_col]
            y_pred = subset['pred']
            y_score = subset[score_col]

            cm = confusion_matrix(y_true, y_pred)
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
            else:
                tn = fp = fn = tp = 0

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            pred_positive_rate = y_pred.mean()

            fairness_results.append({
                'group': group,
                'n_samples': len(subset),
                'fraud_rate': y_true.mean(),
                'predicted_positive_rate': pred_positive_rate,
                'tpr': tpr,
                'fpr': fpr,
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'auc': roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan
            })

        fairness_df = pd.DataFrame(fairness_results)

        # Test statistique: chi-square
        contingency = pd.crosstab(df[protected_attr], df['pred'])
        chi2, p_value, _, _ = stats.chi2_contingency(contingency)

        print(f"\n{'='*60}")
        print(f"FAIRNESS ANALYSIS - Protected Attribute: {protected_attr}")
        print(f"{'='*60}")
        print(f"Chi-square test p-value: {p_value:.6f}")
        print(f"\n{fairness_df.to_string(index=False)}")

        # Visualisation
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        fairness_df.plot(x='group', y=['tpr', 'fpr'], kind='bar', ax=axes[0])
        axes[0].set_title('TPR et FPR par Groupe')
        axes[0].set_ylabel('Rate')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        fairness_df.plot(x='group', y='predicted_positive_rate', kind='bar', 
                        ax=axes[1], color='coral')
        axes[1].set_title('Taux de Prédiction Positive par Groupe')
        axes[1].set_ylabel('Predicted Positive Rate')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)

        fairness_df.plot(x='group', y='auc', kind='bar', ax=axes[2], color='seagreen')
        axes[2].set_title('AUC par Groupe')
        axes[2].set_ylabel('AUC-ROC')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)

        plt.suptitle(f'Fairness Analysis - {protected_attr}', fontsize=12, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

        return fairness_df, p_value

    # =========================================================================
    # 4. OPTIMISATION DU SEUIL
    # =========================================================================

    def optimize_threshold(self, y_true, y_pred_proba, metric='f1', 
                          cost_fn=1000, cost_fp=50):
        """
        Optimise le seuil de décision selon différentes stratégies.

        Args:
            y_true: Labels réels
            y_pred_proba: Probabilités prédites
            metric: 'f1', 'mcc', ou 'cost'
            cost_fn: Coût d'un faux négatif
            cost_fp: Coût d'un faux positif
        """
        thresholds = np.arange(0.01, 1.0, 0.01)
        results = []

        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
            else:
                tn = fp = fn = tp = 0

            results.append({
                'threshold': thresh,
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'mcc': matthews_corrcoef(y_true, y_pred),
                'cost': (fn * cost_fn + fp * cost_fp) / len(y_true),
                'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0
            })

        results_df = pd.DataFrame(results)

        if metric == 'f1':
            best_idx = results_df['f1'].idxmax()
        elif metric == 'mcc':
            best_idx = results_df['mcc'].idxmax()
        elif metric == 'cost':
            best_idx = results_df['cost'].idxmin()
        else:
            best_idx = results_df['f1'].idxmax()

        best_threshold = results_df.loc[best_idx, 'threshold']

        # Visualisation
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(results_df['threshold'], results_df['precision'], label='Precision')
        axes[0].plot(results_df['threshold'], results_df['recall'], label='Recall')
        axes[0].plot(results_df['threshold'], results_df['f1'], label='F1-Score', linewidth=2)
        axes[0].axvline(x=best_threshold, color='r', linestyle='--', 
                       label=f'Optimal = {best_threshold:.2f}')
        axes[0].set_xlabel('Threshold')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Metrics vs Threshold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(results_df['threshold'], results_df['cost'], color='red', linewidth=2)
        axes[1].axvline(x=results_df.loc[results_df['cost'].idxmin(), 'threshold'], 
                       color='green', linestyle='--', 
                       label=f'Min Cost = {results_df["cost"].min():.1f}')
        axes[1].set_xlabel('Threshold')
        axes[1].set_ylabel('Expected Cost per Transaction')
        axes[1].set_title('Cost Optimization')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('Threshold Optimization', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

        return best_threshold, results_df

    # =========================================================================
    # 5. STRESS TESTING
    # =========================================================================

    def stress_test(self, X, y_true, model_predict_fn, 
                   perturbation_factors=[0.5, 0.8, 1.0, 1.2, 1.5, 2.0]):
        """
        Teste la robustesse du modèle face à des perturbations.

        Scénarios:
        - Perturbation des montants
        - Suppression de features
        - Concept drift (changement de taux de fraude)
        """
        baseline_proba = model_predict_fn(X)
        baseline_auc = roc_auc_score(y_true, baseline_proba)

        stress_results = [{'scenario': 'Baseline', 'auc': baseline_auc, 'auc_drop_pct': 0}]

        # Test 1: Perturbation des montants
        for factor in perturbation_factors:
            if 'transaction_amount' in X.columns:
                X_perturbed = X.copy()
                X_perturbed['transaction_amount'] *= factor
                proba = model_predict_fn(X_perturbed)
                auc = roc_auc_score(y_true, proba)
                stress_results.append({
                    'scenario': f'Amount x{factor}',
                    'auc': auc,
                    'auc_drop_pct': (baseline_auc - auc) / baseline_auc * 100
                })

        # Test 2: Suppression de features
        feature_cols = [c for c in X.columns if c not in ['is_fraud', 'transaction_date']]
        for n_drop in [1, 2, 3]:
            if len(feature_cols) > n_drop:
                dropped = np.random.choice(feature_cols, n_drop, replace=False)
                X_perturbed = X.copy()
                for col in dropped:
                    X_perturbed[col] = X_perturbed[col].mean()
                proba = model_predict_fn(X_perturbed)
                auc = roc_auc_score(y_true, proba)
                stress_results.append({
                    'scenario': f'Drop {n_drop} features',
                    'auc': auc,
                    'auc_drop_pct': (baseline_auc - auc) / baseline_auc * 100
                })

        # Test 3: Concept drift
        fraud_idx = np.where(y_true == 1)[0]
        legit_idx = np.where(y_true == 0)[0]

        for fraud_mult in [1.5, 2.0]:
            n_fraud_new = min(int(len(fraud_idx) * fraud_mult), len(fraud_idx))
            selected = np.concatenate([
                np.random.choice(fraud_idx, n_fraud_new, replace=True),
                np.random.choice(legit_idx, len(legit_idx), replace=False)
            ])
            proba = baseline_proba[selected]
            y_sub = y_true.iloc[selected] if hasattr(y_true, 'iloc') else y_true[selected]
            auc = roc_auc_score(y_sub, proba)
            stress_results.append({
                'scenario': f'Fraud rate x{fraud_mult}',
                'auc': auc,
                'auc_drop_pct': (baseline_auc - auc) / baseline_auc * 100
            })

        stress_df = pd.DataFrame(stress_results)

        print(f"\n{'='*60}")
        print("STRESS TEST RESULTS")
        print(f"{'='*60}")
        print(stress_df.to_string(index=False))

        # Visualisation
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['green' if x < 5 else 'orange' if x < 10 else 'red' 
                 for x in stress_df['auc_drop_pct']]
        bars = ax.barh(stress_df['scenario'], stress_df['auc_drop_pct'], color=colors, alpha=0.7)
        ax.set_xlabel('AUC Drop (%)')
        ax.set_title('Stress Test - Robustness Analysis')
        ax.axvline(x=5, color='orange', linestyle='--', label='Warning (5%)')
        ax.axvline(x=10, color='red', linestyle='--', label='Critical (10%)')
        ax.legend()

        for bar, val in zip(bars, stress_df['auc_drop_pct']):
            ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}%', va='center', fontsize=9)

        plt.tight_layout()
        plt.show()

        return stress_df

    # =========================================================================
    # 6. VALIDATION CROISÉE STRATIFIÉE
    # =========================================================================

    def cross_validate(self, X, y, model, n_splits=5):
        """
        Validation croisée stratifiée pour évaluer la stabilité des performances.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_val)[:, 1]

            metrics = self.calculate_performance_metrics(y_val, y_proba)
            metrics['fold'] = fold + 1
            cv_results.append(metrics)

        cv_df = pd.DataFrame(cv_results)

        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION RESULTS ({n_splits}-Fold)")
        print(f"{'='*60}")
        print(cv_df[['fold', 'auc_roc', 'precision', 'recall', 'f1_score', 'mcc']].to_string(index=False))

        print(f"\n--- Summary Statistics ---")
        summary = cv_df[['auc_roc', 'precision', 'recall', 'f1_score', 'mcc']].agg(['mean', 'std', 'min', 'max'])
        print(summary.to_string())

        # Visualisation
        fig, ax = plt.subplots(figsize=(10, 5))
        metrics_to_plot = ['auc_roc', 'precision', 'recall', 'f1_score', 'mcc']
        x = np.arange(len(metrics_to_plot))
        means = cv_df[metrics_to_plot].mean()
        stds = cv_df[metrics_to_plot].std()

        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_to_plot, rotation=15)
        ax.set_ylabel('Score')
        ax.set_title('Cross-Validation: Mean ± Std Dev')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)

        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.02, f'{m:.3f}±{s:.3f}', ha='center', fontsize=9)

        plt.tight_layout()
        plt.show()

        return cv_df

    # =========================================================================
    # 7. RAPPORT DE VALIDATION COMPLÈT
    # =========================================================================

    def generate_validation_report(self, metrics, stability_df, fairness_df, 
                                   stress_df, cv_df, output_path=None):
        """
        Génère un rapport de validation complet au format texte.
        """
        report = []
        report.append("="*70)
        report.append(f"  MODEL VALIDATION REPORT - {self.model_name}")
        report.append(f"  Domain: Financial Crime & Fraud Detection")
        report.append(f"  Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("="*70)

        # Section 1: Performance
        report.append("\n" + "-"*70)
        report.append("SECTION 1: MODEL PERFORMANCE")
        report.append("-"*70)
        report.append(f"  AUC-ROC:           {metrics['auc_roc']:.4f}")
        report.append(f"  Average Precision: {metrics['average_precision']:.4f}")
        report.append(f"  Precision:         {metrics['precision']:.4f}")
        report.append(f"  Recall (TPR):      {metrics['recall']:.4f}")
        report.append(f"  F1-Score:          {metrics['f1_score']:.4f}")
        report.append(f"  MCC:               {metrics['mcc']:.4f}")
        report.append(f"  FPR:               {metrics['false_positive_rate']:.4f}")
        report.append(f"  FNR:               {metrics['false_negative_rate']:.4f}")
        report.append(f"  Expected Cost:     {metrics['expected_cost']:.2f}€/transaction")

        # Section 2: Stability
        report.append("\n" + "-"*70)
        report.append("SECTION 2: TEMPORAL STABILITY (PSI)")
        report.append("-"*70)
        if stability_df is not None and len(stability_df) > 0:
            unstable = stability_df[stability_df['status'] == 'Unstable']
            moderate = stability_df[stability_df['status'] == 'Moderate']
            report.append(f"  Périodes analysées: {len(stability_df)}")
            report.append(f"  Périodes stables:   {len(stability_df) - len(unstable) - len(moderate)}")
            report.append(f"  Périodes modérées:  {len(moderate)}")
            report.append(f"  Périodes instables: {len(unstable)}")
            if len(unstable) > 0:
                report.append(f"  ⚠️  ALERTE: {len(unstable)} période(s) avec PSI > 0.25")

        # Section 3: Fairness
        report.append("\n" + "-"*70)
        report.append("SECTION 3: FAIRNESS & BIAS ANALYSIS")
        report.append("-"*70)
        if fairness_df is not None and len(fairness_df) > 0:
            report.append(fairness_df.to_string(index=False))
            tpr_range = fairness_df['tpr'].max() - fairness_df['tpr'].min()
            fpr_range = fairness_df['fpr'].max() - fairness_df['fpr'].min()
            report.append(f"\n  TPR Range: {tpr_range:.4f}")
            report.append(f"  FPR Range: {fpr_range:.4f}")
            if tpr_range > 0.1 or fpr_range > 0.1:
                report.append("  ⚠️  ALERTE: Différence significative de TPR/FPR entre groupes")

        # Section 4: Stress Test
        report.append("\n" + "-"*70)
        report.append("SECTION 4: STRESS TESTING")
        report.append("-"*70)
        if stress_df is not None and len(stress_df) > 0:
            max_drop = stress_df['auc_drop_pct'].max()
            report.append(f"  Max AUC Drop: {max_drop:.2f}%")
            if max_drop > 10:
                report.append("  ⚠️  ALERTE: AUC drop > 10% dans un scénario de stress")
            report.append(stress_df.to_string(index=False))

        # Section 5: Cross-Validation
        report.append("\n" + "-"*70)
        report.append("SECTION 5: CROSS-VALIDATION STABILITY")
        report.append("-"*70)
        if cv_df is not None and len(cv_df) > 0:
            auc_std = cv_df['auc_roc'].std()
            report.append(f"  AUC Mean: {cv_df['auc_roc'].mean():.4f}")
            report.append(f"  AUC Std:  {auc_std:.4f}")
            report.append(f"  AUC Min:  {cv_df['auc_roc'].min():.4f}")
            report.append(f"  AUC Max:  {cv_df['auc_roc'].max():.4f}")
            if auc_std > 0.02:
                report.append("  ⚠️  ALERTE: Haute variance entre folds (std > 0.02)")

        # Conclusion
        report.append("\n" + "="*70)
        report.append("VALIDATION CONCLUSION")
        report.append("="*70)

        alerts = [line for line in report if "⚠️  ALERTE" in line]
        if len(alerts) == 0:
            report.append("  ✅ Modèle VALIDÉ - Aucune alerte critique détectée")
        else:
            report.append(f"  ⚠️  Modèle SOUS RÉSERVE - {len(alerts)} alerte(s) détectée(s)")
            for alert in alerts:
                report.append(f"     {alert.strip()}")

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"\nRapport sauvegardé: {output_path}")

        print(report_text)
        return report_text


# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================
if __name__ == "__main__":
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    # Génération de données de démonstration
    np.random.seed(42)
    n_samples = 50000
    n_fraud = int(n_samples * 0.02)
    n_legit = n_samples - n_fraud

    legit = pd.DataFrame({
        'transaction_amount': np.random.lognormal(4, 1.5, n_legit),
        'transaction_velocity_1h': np.random.poisson(2, n_legit),
        'merchant_risk_score': np.random.beta(2, 5, n_legit),
        'customer_age_days': np.random.exponential(365, n_legit),
        'is_international': np.random.binomial(1, 0.05, n_legit),
        'is_fraud': 0
    })

    fraud = pd.DataFrame({
        'transaction_amount': np.random.lognormal(6, 1.2, n_fraud),
        'transaction_velocity_1h': np.random.poisson(8, n_fraud),
        'merchant_risk_score': np.random.beta(5, 2, n_fraud),
        'customer_age_days': np.random.exponential(30, n_fraud),
        'is_international': np.random.binomial(1, 0.4, n_fraud),
        'is_fraud': 1
    })

    df = pd.concat([legit, fraud]).sample(frac=1).reset_index(drop=True)
    df['transaction_date'] = pd.date_range(start='2024-01-01', periods=len(df), freq='min')

    # Split et entraînement
    feature_cols = [c for c in df.columns if c not in ['is_fraud', 'transaction_date']]
    X = df[feature_cols]
    y = df['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(scaler.fit_transform(X_train), y_train)

    y_proba = model.predict_proba(scaler.transform(X_test))[:, 1]

    # Validation
    validator = FraudModelValidator("DemoModel")

    # 1. Performance
    metrics = validator.calculate_performance_metrics(y_test, y_proba)
    validator.plot_roc_pr_curves(y_test, y_proba)

    # 2. Stability
    df_test = X_test.copy()
    df_test['score'] = y_proba
    df_test['transaction_date'] = df['transaction_date'].iloc[X_test.index].values
    stability = validator.stability_analysis(df_test)

    # 3. Fairness
    df_test['is_fraud'] = y_test.values
    fairness, _ = validator.fairness_analysis(df_test, 'score', 'is_fraud', 'is_international')

    # 4. Threshold
    best_thresh, _ = validator.optimize_threshold(y_test, y_proba)

    # 5. Stress test
    def predict_fn(X_in):
        return model.predict_proba(scaler.transform(X_in))[:, 1]
    stress = validator.stress_test(X_test, y_test, predict_fn)

    # 6. Cross-validation
    cv = validator.cross_validate(X.sample(10000), y.loc[X.sample(10000).index], 
                                  GradientBoostingClassifier(n_estimators=50, random_state=42))

    # 7. Rapport
    report = validator.generate_validation_report(metrics, stability, fairness, stress, cv)