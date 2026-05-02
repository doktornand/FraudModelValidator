# 🔍 FinCrime & Fraud Model Validator

> Framework de validation de modèle pour la détection de fraude et la criminalité financière (FinCrime). Conforme aux standards réglementaires **SR 11-7** (Federal Reserve), **ECB Guide to Internal Models**, et **EBA Guidelines on ML for IRB**.

---

## 📋 Table des matières

- [Description](#description)
- [Fonctionnalités](#fonctionnalités)
- [Installation](#installation)
- [Utilisation rapide](#utilisation-rapide)
- [Modules de validation](#modules-de-validation)
- [Seuils réglementaires](#seuils-réglementaires)
- [Exemples avancés](#exemples-avancés)
- [Architecture du projet](#architecture-du-projet)
- [Bonnes pratiques](#bonnes-pratiques)
- [Licence](#licence)

---

## 🎯 Description

Ce framework fournit un **pipeline complet de validation indépendante** pour les modèles de détection de fraude transactionnelle, blanchiment d'argent (AML), et criminalité financière. Il couvre l'ensemble des exigences réglementaires en matière de :

- **Développement** : performance, robustesse, interprétabilité
- **Validation** : tests indépendants, documentation, traçabilité
- **Surveillance** : stabilité temporelle, drift, fairness
- **Gouvernance** : rapports structurés, alertes automatisées

---

## ✨ Fonctionnalités

| Module | Métriques / Tests | Standard |
|--------|-------------------|----------|
| **Performance** | AUC-ROC, AP, Precision, Recall, F1, MCC, FPR, FNR, Specificity | SR 11-7 §III.B |
| **Stabilité temporelle** | PSI (Population Stability Index) par période | SR 11-7 §III.C.2 |
| **Équité (Fairness)** | TPR/FPR par groupe, Demographic Parity, Chi² | EBA GL on ML §4.2 |
| **Optimisation du seuil** | F1, MCC, ou coût pondéré (FN/FP) | ECB Guide §4.3 |
| **Stress testing** | Perturbation features, suppression, concept drift | SR 11-7 §III.C.3 |
| **Validation croisée** | Stratified K-Fold (stabilité inter-fold) | ECB Guide §4.4 |
| **Rapport** | Rapport texte structuré avec conclusion validée/sous réserve | SR 11-7 §IV |

---

## 🚀 Installation

### Prérequis

- Python ≥ 3.9
- pandas, numpy, scikit-learn, scipy, matplotlib

### Installation des dépendances

```bash
pip install pandas numpy scikit-learn scipy matplotlib
```

### Clonage du repo

```bash
git clone https://github.com/votre-org/fincrime-fraud-model-validator.git
cd fincrime-fraud-model-validator
```

---

## ⚡ Utilisation rapide

```python
from fraud_model_validator import FraudModelValidator
import pandas as pd

# --- 1. Charger vos données ---
# df doit contenir : features, 'is_fraud' (0/1), 'score' (probabilité), 'transaction_date'

df = pd.read_parquet("transactions_scored.parquet")

# --- 2. Initialiser le validateur ---
validator = FraudModelValidator(model_name="FraudDetection_v2.1")

# --- 3. Performance ---
metrics = validator.calculate_performance_metrics(
    y_true=df['is_fraud'],
    y_pred_proba=df['score'],
    threshold=0.5
)
validator.plot_roc_pr_curves(df['is_fraud'], df['score'])

# --- 4. Stabilité temporelle (PSI) ---
stability = validator.stability_analysis(
    df, score_col='score', date_col='transaction_date', period='W'
)
validator.plot_stability(stability)

# --- 5. Équité par segment ---
fairness, p_value = validator.fairness_analysis(
    df, score_col='score', target_col='is_fraud', protected_attr='pays'
)

# --- 6. Seuil optimal (coût-based) ---
best_thresh, _ = validator.optimize_threshold(
    df['is_fraud'], df['score'],
    metric='cost', cost_fn=1000, cost_fp=50
)

# --- 7. Stress testing ---
def model_predict_fn(X_input):
    return mon_modele.predict_proba(X_input)[:, 1]

stress = validator.stress_test(X_features, df['is_fraud'], model_predict_fn)

# --- 8. Rapport final ---
report = validator.generate_validation_report(
    metrics=metrics,
    stability_df=stability,
    fairness_df=fairness,
    stress_df=stress,
    cv_df=cv_results,
    output_path='rapport_validation.txt'
)
```

---

## 📊 Modules de validation

### 1. Métriques de performance

```python
metrics = validator.calculate_performance_metrics(y_true, y_proba, threshold=0.5)
```

Retourne un dictionnaire avec :
- `auc_roc` — Aire sous la courbe ROC
- `average_precision` — Average Precision (PR AUC)
- `precision`, `recall`, `f1_score` — Métriques classiques
- `mcc` — Matthews Correlation Coefficient (robuste au déséquilibre)
- `false_positive_rate`, `false_negative_rate` — Taux d'erreur
- `expected_cost` — Coût attendu par transaction (FN × coût_fraude + FP × coût_revue)

### 2. Stabilité temporelle (PSI)

```python
stability = validator.stability_analysis(
    df, score_col='score', date_col='transaction_date',
    period='W', baseline_end='2024-03-01'
)
```

Calcule le **Population Stability Index** entre la période de référence (baseline) et chaque période subséquente.

| PSI | Interprétation | Action |
|-----|----------------|--------|
| < 0.10 | Stable | Aucune action |
| 0.10 – 0.25 | Modéré | Surveillance renforcée |
| > 0.25 | Instable | Investigation + recalibration |

### 3. Analyse d'équité (Fairness)

```python
fairness, p_value = validator.fairness_analysis(
    df, score_col='score', target_col='is_fraud',
    protected_attr='segment_client', threshold=0.5
)
```

Analyse l'équité du modèle selon un attribut protégé (pays, segment, canal, etc.) :
- **Equalized Odds** : TPR et FPR comparables entre groupes
- **Demographic Parity** : Taux de prédiction positifs comparables
- Test du **Chi²** d'indépendance entre l'attribut protégé et la prédiction

> ⚠️ Alerte si l'écart de TPR ou FPR entre groupes dépasse **10 points de pourcentage**.

### 4. Optimisation du seuil

```python
best_thresh, results_df = validator.optimize_threshold(
    y_true, y_proba, metric='f1', cost_fn=1000, cost_fp=50
)
```

Optimise le seuil de décision selon trois stratégies :
- `metric='f1'` — Maximise le F1-Score
- `metric='mcc'` — Maximise le Matthews Correlation Coefficient
- `metric='cost'` — Minimise le coût total (faux négatifs × coût_fraude + faux positifs × coût_revue)

### 5. Stress testing

```python
stress = validator.stress_test(X, y_true, model_predict_fn)
```

Scénarios de stress appliqués :
- **Perturbation des montants** : ×0.5, ×0.8, ×1.2, ×1.5, ×2.0
- **Suppression de features** : 1, 2, 3 features remplacées par leur moyenne
- **Concept drift** : Augmentation artificielle du taux de fraude (×1.5, ×2.0)

> ⚠️ Alerte si le drop d'AUC dépasse **10%** dans un scénario.

### 6. Validation croisée stratifiée

```python
cv = validator.cross_validate(X, y, model, n_splits=5)
```

Évalue la stabilité des performances sur 5 folds stratifiés. 

> ⚠️ Alerte si l'écart-type de l'AUC entre folds dépasse **0.02** (surapprentissage potentiel).

### 7. Rapport de validation

```python
report = validator.generate_validation_report(
    metrics, stability_df, fairness_df, stress_df, cv_df,
    output_path='rapport_validation.txt'
)
```

Génère un rapport texte structuré avec :
- Synthèse des 5 sections (Performance, Stabilité, Fairness, Stress, CV)
- Détection automatique des alertes
- **Conclusion** : ✅ Validé / ⚠️ Sous réserve

---

## 📏 Seuils réglementaires

| Test | Seuil critique | Référence réglementaire |
|------|----------------|------------------------|
| AUC-ROC | < 0.70 | SR 11-7 §III.B.1 |
| PSI | > 0.25 | SR 11-7 §III.C.2 |
| Écart TPR/FPR (fairness) | > 10 pp | EBA GL on ML §4.2.3 |
| AUC drop (stress) | > 10% | SR 11-7 §III.C.3 |
| Std AUC (CV) | > 0.02 | ECB Guide §4.4.2 |
| FNR | > 5% | Politique interne (typique) |
| FPR | > 10% | Politique interne (typique) |

---

## 🔬 Exemples avancés

### Validation sur plusieurs segments

```python
segments = df['region'].unique()
all_reports = []

for seg in segments:
    sub = df[df['region'] == seg]
    metrics = validator.calculate_performance_metrics(sub['is_fraud'], sub['score'])
    all_reports.append({'segment': seg, **metrics})

pd.DataFrame(all_reports).to_csv("performance_par_segment.csv", index=False)
```

### Détection de drift continu (monitoring)

```python
# Exécuter mensuellement
stability = validator.stability_analysis(
    df_last_month, score_col='score', date_col='transaction_date', period='D'
)

if (stability['status'] == 'Unstable').any():
    send_alert_email("Drift détecté — PSI > 0.25")
```

### Comparaison de deux versions de modèle

```python
# Modèle A (actuel) vs Modèle B (candidat)
metrics_A = validator.calculate_performance_metrics(y_test, proba_A)
metrics_B = validator.calculate_performance_metrics(y_test, proba_B)

print(f"AUC A: {metrics_A['auc_roc']:.4f} | AUC B: {metrics_B['auc_roc']:.4f}")
print(f"Gain: {(metrics_B['auc_roc'] - metrics_A['auc_roc']) * 100:.2f} pp")
```

---

## 🏗️ Architecture du projet

```
fincrime-fraud-model-validator/
│
├── fraud_model_validator.py      # Classe principale (framework)
├── README.md                     # Ce fichier
├── requirements.txt              # Dépendances
│
├── examples/
│   ├── demo_validation.py        # Démonstration complète
│   └── demo_data_generation.py   # Générateur de données synthétiques
│
├── reports/
│   └── (rapports générés)
│
└── tests/
    ├── test_performance.py
    ├── test_stability.py
    └── test_fairness.py
```

---

## ✅ Bonnes pratiques

1. **Séparation des rôles** : Le validateur doit être indépendant de l'équipe de développement du modèle (SR 11-7 §II.A).

2. **Documentation** : Chaque test doit être documenté avec son objectif, sa méthodologie, et ses limites.

3. **Reproductibilité** : Fixer le `random_state` pour tous les tests stochastiques.

4. **Fréquence de validation** :
   - **Validation initiale** : Avant mise en production
   - **Validation périodique** : Annuelle (minimum)
   - **Validation ad-hoc** : En cas de drift, changement de données, ou incident

5. **Seuils personnalisables** : Adapter les seuils d'alerte à votre contexte métier et réglementaire.

6. **Traçabilité** : Archiver tous les rapports de validation avec la version du modèle et la date.

---

## 📚 Références réglementaires

- **SR 11-7** — Federal Reserve, *Guidance on Model Risk Management* (2011)
- **ECB Guide to Internal Models** — European Central Bank (2022)
- **EBA Guidelines on ML for IRB** — European Banking Authority (2024)
- **OCC 2011-12** — Office of the Comptroller of the Currency

---

## 📄 Licence

Ce projet est distribué sous licence **MIT**. Vous êtes libre de l'utiliser, le modifier et le distribuer dans vos projets internes ou commerciaux, sous réserve de mentionner la source.

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour proposer une amélioration :

1. Forkez le repo
2. Créez une branche (`git checkout -b feature/ma-feature`)
3. Commitez vos changements (`git commit -am 'Ajout de...'`)
4. Pushez (`git push origin feature/ma-feature`)
5. Ouvrez une Pull Request

---

> **Avertissement** : Ce framework est un outil d'aide à la validation. Il ne remplace pas le jugement expert d'un validateur de modèle qualifié ni l'avis des fonctions de conformité et de contrôle interne.
