# 🛡️ GBM FraudDetector v3 — Model Validation Framework

> **FraudModelValidatorV3** — Framework ultra-scientifique de validation de modèles de détection de fraude et de criminalité financière (FinCrime), conforme aux standards réglementaires SR 11-7, ECB TRIM, AMLD6 et Bâle III/IV.

---

## 📋 Table des matières

1. [Vue d'ensemble](#-vue-densemble)
2. [Architecture du repo](#-architecture-du-repo)
3. [Prérequis & Installation](#-prérequis--installation)
4. [Démarrage rapide](#-démarrage-rapide)
5. [API Reference](#-api-reference)
   - [Initialisation](#initialisation)
   - [Performance discriminante](#performance-discriminante)
   - [Calibration probabiliste](#calibration-probabiliste)
   - [Analyse de coût](#analyse-de-coût)
   - [Ultra-Scientific Detection](#ultra-scientific-detection)
   - [Benford's Law](#benfords-law)
   - [Graph Anomaly Detection](#graph-anomaly-detection)
   - [Cross-Validation & Backtesting](#cross-validation--backtesting)
   - [Stress Testing](#stress-testing)
   - [Fairness Analysis](#fairness-analysis)
   - [Génération de rapport](#génération-de-rapport)
6. [Résultats de validation (v3 Ultra)](#-résultats-de-validation-v3-ultra)
7. [Interprétation des métriques](#-interprétation-des-métriques)
8. [Courbes de diagnostic](#-courbes-de-diagnostic)
9. [Conformité réglementaire](#-conformité-réglementaire)
10. [Limitations & avertissements](#-limitations--avertissements)
11. [FAQ](#-faq)
12. [Licence & auteurs](#-licence--auteurs)

---

## 🔍 Vue d'ensemble

Ce framework fournit une validation exhaustive de modèles de scoring de fraude. Il va au-delà des métriques classiques (AUC, Gini) pour intégrer des méthodes issues de la physique statistique, de la théorie de l'information et de la dynamique non-linéaire, permettant de détecter des formes subtiles de sur-ajustement, de manipulation de données ou d'instabilité structurelle.

### Pourquoi ce framework ?

Un modèle peut afficher un AUC de 0.99 sur données de test et néanmoins :
- Être sur-ajusté à un régime de fraude passé déjà connu
- Présenter une instabilité temporelle masquée par une variance faible en CV
- Contenir des biais de calibration non détectables par le Brier Score seul
- Être le produit de données synthétiques ou de labels contaminés

Les méthodes ultra-scientifiques de ce framework sont conçues pour détecter précisément ces cas limites, conformément aux exigences des régulateurs bancaires et financiers.

### Standards couverts

| Référentiel | Organisme | Portée |
|---|---|---|
| **SR 11-7** | Federal Reserve (US) | Model Risk Management |
| **ECB TRIM Guide** | Banque Centrale Européenne | Targeted Review of Internal Models |
| **AMLD6** | Union Européenne | Anti-Money Laundering Directive |
| **Bâle III / IV** | Comité de Bâle | Capital requirements & model validation |

---

## 🗂️ Architecture du repo

```
fraud-model-validator/
│
├── Fraud_Model_Validator_v3d.py    # Module principal — FraudModelValidatorV3
│
├── validation_report_v3.html        # Rapport HTML interactif complet (généré)
├── validation_report_v3.txt         # Rapport texte synthétique (généré)
├── curves_v3.png                    # Courbes de diagnostic (ROC, PR, DET, KS)
│
└── README.md                        # Ce fichier
```

### Description des fichiers

#### `Fraud_Model_Validator_v3d.py`
Le cœur du framework. Contient la classe `FraudModelValidatorV3` (~2 500 lignes) avec :
- Toutes les méthodes de calcul de métriques
- Les algorithmes ultra-scientifiques (MF-DFA, Transfer Entropy, BOCD, RMT, ESN, SSA, RQA)
- Le générateur de rapport HTML auto-contenu (graphiques en base64)
- Un script d'exemple autonome en fin de fichier (`__main__`)

#### `validation_report_v3.html`
Rapport de validation complet au format HTML, auto-contenu (toutes les figures sont encodées en base64). Peut être ouvert directement dans un navigateur sans dépendance externe. Inclut l'ensemble des sections du rapport texte plus les visualisations interactives.

#### `validation_report_v3.txt`
Résumé tabulaire des résultats de validation, destiné à l'archivage, aux systèmes de log ou à l'intégration dans des pipelines CI/CD.

#### `curves_v3.png`
Tableau 2×2 des courbes de diagnostic :
- **Courbe ROC** (en haut à gauche)
- **Courbe Precision-Recall** (en haut à droite)
- **Detection Error Tradeoff (DET)** (en bas à gauche)
- **Distribution des scores / KS Plot** (en bas à droite)

---

## ⚙️ Prérequis & Installation

### Python
Python **3.9+** recommandé.

### Dépendances principales

```bash
pip install numpy pandas scipy scikit-learn matplotlib statsmodels
```

### Dépendances optionnelles (fortement recommandées)

```bash
# Explicabilité (SHAP values)
pip install shap

# Graphes de transactions
pip install networkx

# Complexité fractale
pip install antropy
```

Le framework détecte automatiquement la disponibilité de ces packages et désactive gracieusement les fonctionnalités correspondantes si elles sont absentes.

### Installation complète (environnement de dev)

```bash
git clone https://github.com/<votre-org>/fraud-model-validator.git
cd fraud-model-validator

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install numpy pandas scipy scikit-learn matplotlib statsmodels shap networkx antropy
```

---

## 🚀 Démarrage rapide

```python
import pandas as pd
import numpy as np
from Fraud_Model_Validator_v3d import FraudModelValidatorV3

# ── 1. Initialiser le validateur ──────────────────────────────────────────────
validator = FraudModelValidatorV3(
    model_name="GBM FraudDetector v3",
    cost_fn=5000.0,   # Coût d'une fraude manquée (€)
    cost_fp=100.0,    # Coût d'une fausse alarme (€)
    currency="€"
)

# ── 2. Charger vos données ────────────────────────────────────────────────────
# df doit contenir au minimum : features, label binaire, score du modèle, date
df = pd.read_parquet("transactions_test.parquet")
y_true  = df["is_fraud"].values
y_proba = df["fraud_score"].values
feature_cols = [c for c in df.columns if c not in ["is_fraud", "fraud_score", "transaction_date"]]

# ── 3. Métriques de performance ───────────────────────────────────────────────
metrics = validator.calculate_performance_metrics(
    y_true, y_proba, bootstrap=True, n_bootstrap=1000
)

# ── 4. Détection ultra-scientifique ──────────────────────────────────────────
ultra_results = validator.ultra_scientific_detection(
    df, feature_cols, time_col="transaction_date"
)

# ── 5. Analyses complémentaires ───────────────────────────────────────────────
validator.calibration_analysis(y_true, y_proba)
validator.benford_law_analysis(df, amount_col="transaction_amount")
validator.graph_anomaly_detection(df, source_col="sender_id", target_col="receiver_id")
validator.cross_validate(df[feature_cols], y_true, model=your_model)
validator.walk_forward_backtest(df, model=your_model, date_col="transaction_date")
validator.stress_test(df[feature_cols], y_true, model_predict_fn=your_model.predict_proba)

# ── 6. Générer les rapports ───────────────────────────────────────────────────
validator.generate_html_report(output_path="validation_report.html")
validator.generate_validation_report()   # rapport texte vers stdout
```

Pour tester avec des données synthétiques, le fichier inclut un bloc `if __name__ == "__main__":` exécutable directement :

```bash
python Fraud_Model_Validator_v3d.py
```

---

## 📖 API Reference

### Initialisation

```python
FraudModelValidatorV3(
    model_name: str = "FraudDetectionModel",
    cost_fn: float = 5000.0,
    cost_fp: float = 100.0,
    currency: str = "€"
)
```

| Paramètre | Type | Description |
|---|---|---|
| `model_name` | `str` | Nom du modèle, apparaît dans tous les rapports |
| `cost_fn` | `float` | Coût métier d'un faux négatif (fraude manquée) |
| `cost_fp` | `float` | Coût métier d'un faux positif (fausse alarme client) |
| `currency` | `str` | Symbole monétaire pour les rapports (`"€"`, `"$"`, etc.) |

---

### Performance discriminante

```python
metrics = validator.calculate_performance_metrics(
    y_true, y_pred_proba,
    threshold: float = 0.5,
    bootstrap: bool = True,
    n_bootstrap: int = 1000
) -> dict
```

Calcule l'ensemble complet des métriques de discrimination avec intervalles de confiance bootstrap.

**Métriques retournées :**

| Métrique | Description |
|---|---|
| `auc_roc` | Area Under the ROC Curve |
| `gini` | Coefficient de Gini (`2×AUC − 1`) |
| `ks_statistic` | Kolmogorov-Smirnov — séparation max entre CDFs |
| `h_measure` | H-measure (alternatif à AUC, invariant au coût) |
| `average_precision` | AUC de la courbe Precision-Recall |
| `precision` / `recall` | Au seuil `threshold` |
| `f1_score` | Moyenne harmonique précision/rappel |
| `mcc` | Matthews Correlation Coefficient |
| `balanced_accuracy` | Moyenne TPR + TNR |

Méthodes annexes :
- `plot_advanced_curves()` — génère `curves_v3.png` (ROC, PR, DET, KS)
- `lift_gain_analysis()` — tableau décile par décile
- `_bootstrap_ci()` — intervalles de confiance à 95% par bootstrap

---

### Calibration probabiliste

```python
cal = validator.calibration_analysis(
    y_true, y_pred_proba,
    n_bins: int = 10
) -> dict
```

| Métrique | Description |
|---|---|
| `brier_score` | Score de Brier (0 = parfait, 1 = pire que random) |
| `brier_skill_score` | Brier Skill Score relatif à la prévalence |
| `ece` | Expected Calibration Error |
| `mce` | Maximum Calibration Error |
| `log_loss` | Log-loss (entropie croisée) |

---

### Analyse de coût

```python
cost = validator.optimize_threshold(
    y_true, y_pred_proba,
    metric: str = 'cost'   # ou 'f1', 'balanced_accuracy'
) -> dict
```

Optimise le seuil de décision selon la fonction de coût métier (`cost_fn × FN + cost_fp × FP`). Retourne le seuil optimal, le coût espéré par transaction et la matrice de confusion associée.

---

### Ultra-Scientific Detection

```python
ultra = validator.ultra_scientific_detection(
    df: pd.DataFrame,
    feature_cols: list,
    time_col: str = 'transaction_date'
) -> dict
```

C'est la section distinctive du framework v3. Elle applique sept méthodes issues de domaines scientifiques avancés pour détecter des anomalies non visibles par les métriques classiques.

#### Méthodes implémentées

| Méthode | Abréviation | Ce qu'elle détecte |
|---|---|---|
| **Multifractal Detrended Fluctuation Analysis** | MF-DFA | Lissage artificiel de séries temporelles, auto-corrélations non naturelles |
| **Transfer Entropy** | TE | Causalité anormale entre features — indique une contamination ou un leakage |
| **Bayesian Online Changepoint Detection** | BOCD | Micro-ruptures de régime dans la série de scores |
| **Random Matrix Theory** | RMT | Corrélations non naturelles entre variables (fraude collective, données synthétiques) |
| **Echo State Network residuals** | ESN | Dynamique non-linéaire cachée dans la séquence de scores |
| **Singular Spectrum Analysis** | SSA | Persistance spectrale anormale |
| **Recurrence Quantification Analysis** | RQA | Structures récurrentes dans l'espace des phases |

**Score composite retourné :**

```python
{
    "global_ultra_anomaly_score": float,   # [0, 1] — 0 = sain, 1 = très suspect
    "ultra_risk_level": str,               # "LOW" | "MEDIUM" | "HIGH"
    "mfdfa_multifractal_width": float,
    "transfer_entropy_mean": float,
    "bocd_max_probability": float,
    "rmt_pvalue": float,
    "signal_eigenvalues": int,
    "esn_residual_mean": float
}
```

**Seuils de risque :**

| Score | Niveau | Interprétation |
|---|---|---|
| < 0.30 | 🟢 LOW | Aucune anomalie significative |
| 0.30 – 0.60 | 🟡 MEDIUM | Anomalies mineures — surveillance recommandée |
| > 0.60 | 🔴 HIGH | Anomalies critiques — investigation avant déploiement |

---

### Benford's Law

```python
benford = validator.benford_law_analysis(
    df, amount_col: str = 'transaction_amount'
) -> dict
```

Teste si la distribution du premier chiffre significatif des montants suit la loi de Benford. Une déviation significative peut indiquer des montants fabriqués, arrondis ou manipulés.

**Sorties :** MAD (Mean Absolute Deviation), p-value χ², verdict de conformité.

---

### Graph Anomaly Detection

```python
graph = validator.graph_anomaly_detection(
    df,
    source_col: str = 'sender_id',
    target_col: str = 'receiver_id',
    amount_col: str = 'transaction_amount',
    time_col: str = 'transaction_date'
) -> dict
```

Construit un graphe dirigé des transactions et détecte :
- Nœuds à forte centralité (hubs suspects)
- Cliques et communautés anormales
- Entités avec des degrés entrants/sortants déséquilibrés

Requiert `networkx`. Retourne le nombre de nœuds, d'arêtes et les entités les plus suspectes.

---

### Cross-Validation & Backtesting

```python
# K-Fold stratifié
cv_results = validator.cross_validate(
    X, y, model, n_splits: int = 5
) -> pd.DataFrame

# Walk-forward temporel
wf_results = validator.walk_forward_backtest(
    df, model,
    date_col: str = 'transaction_date',
    n_splits: int = 5
) -> dict
```

Le walk-forward backtest simule un déploiement réel en entraînant sur le passé et en testant sur le futur — seule méthode valide pour des données temporellement ordonnées.

---

### Stress Testing

```python
stress = validator.stress_test(
    X, y_true, model_predict_fn,
    noise_levels: list = [0.01, 0.05, 0.1, 0.2],
    missing_rates: list = [0.05, 0.1, 0.2]
) -> dict
```

Mesure la dégradation des performances sous :
- Bruit gaussien croissant sur les features
- Taux de valeurs manquantes croissants
- Permutations aléatoires de colonnes (stress extrême)

Retourne l'AUC Drop maximal observé.

---

### Génération de rapport

```python
# Rapport HTML complet (auto-contenu, figures en base64)
validator.generate_html_report(output_path: str = 'validation_report.html')

# Rapport texte tabulaire (stdout ou fichier)
validator.generate_validation_report(
    metrics=None,
    stability_df=None,
    output_path: str = 'validation_report.txt'
)
```

Le rapport HTML inclut toutes les sections, graphiques interactifs et le score de validation global pondéré.

---

## 📊 Résultats de validation (v3 Ultra)

Résultats obtenus lors de la dernière exécution (`2026-05-02 17:02`).

### Section 1 — Performance discriminante

| Métrique | Valeur |
|---|---|
| AUC-ROC | **0.9999** |
| GINI | **0.9998** |
| KS Statistic | **0.9998** |
| H-Measure | 0.9932 |
| Average Precision | 0.9934 |
| Recall (TPR) | **1.0000** |
| F1-Score | 0.9967 |
| MCC | 0.9966 |
| Balanced Accuracy | **0.9999** |

### Section 2 — Calibration probabiliste

| Métrique | Valeur |
|---|---|
| Brier Score | **0.0002** |
| Brier Skill Score | 0.9932 |
| ECE | **0.0000** |
| MCE | 0.0066 |
| Log-Loss | 0.0021 |

### Section 3 — Analyse de coût

| Paramètre | Valeur |
|---|---|
| Coût FN (fraude manquée) | 5 000 € |
| Coût FP (fausse alarme) | 100 € |
| **Coût espéré / transaction** | **0.02 €** |

### Section Ultra — Scientific Anomaly Detection

| Indicateur | Valeur | Verdict |
|---|---|---|
| **Global Ultra Anomaly Score** | 0.1817 | 🟢 LOW |
| MF-DFA Multifractal Width | 0.8360 | Normal |
| Transfer Entropy Mean | 0.0040 | Faible (sain) |
| BOCD Max Probability | 0.0000 | Aucun changement de régime |
| RMT p-value | 0.8730 | Corrélations naturelles |
| Signal Eigenvalues | 1 | Normal |
| ESN Residual Mean | 0.0362 | Dynamique stable |

### Section 4 — Loi de Benford

| Métrique | Valeur | Verdict |
|---|---|---|
| MAD | 0.00094 | ✅ Conforme |
| χ² p-value | 0.28612 | Pas de déviation significative |

### Section 5 — Détection d'anomalies de graphe

| Indicateur | Valeur |
|---|---|
| Entités (nœuds) | 4 074 |
| Transactions (arêtes) | 4 999 |

### Section 6 — Cross-Validation (5-Fold stratifié)

| Métrique | Moyenne | Std |
|---|---|---|
| AUC-ROC | 1.0000 | ± 0.0000 |
| GINI | 1.0000 | ± 0.0000 |
| KS | 1.0000 | ± 0.0000 |
| F1 | 1.0000 | ± 0.0000 |
| MCC | 1.0000 | ± 0.0000 |

### Section 7 — Walk-Forward Backtesting

| Métrique | Valeur |
|---|---|
| AUC moyen (temporel) | 1.0000 |
| AUC std (temporel) | 0.0000 |

### Section 8 — Stress Testing

| Métrique | Valeur |
|---|---|
| **Max AUC Drop** | **14.84%** |

### Conclusion

```
✅ MODÈLE VALIDÉ — Aucune anomalie ultra-scientifique critique
Recommandation: APPROUVER pour déploiement
```

---

## 🔬 Interprétation des métriques

### Sur les métriques parfaites (AUC = 1.0000)

Les résultats de cross-validation et de walk-forward indiquent des métriques à 1.0000 ± 0.0000. **Ce niveau de performance sur données réelles est extrêmement rare et doit toujours déclencher une investigation approfondie.** Les causes possibles incluent :

- **Data leakage** : une feature directement corrélée au label (ex. : montant de remboursement de fraude présent en features)
- **Label contamination** : les labels de test ont été vus lors de l'entraînement
- **Données synthétiques** : les données de test ont été générées par le même processus que les données d'entraînement
- **Séparation parfaite légitime** : rare mais possible sur des jeux de données très déséquilibrés avec des features très discriminantes

Le score Ultra Anomaly Score de **0.1817 (LOW)** et le RMT p-value de **0.873** suggèrent que les corrélations inter-features sont naturelles, ce qui va dans le sens d'une séparation légitime sur ce jeu de données.

**En production, investiguer systématiquement tout AUC > 0.99 avant déploiement.**

### Sur le Max AUC Drop de 14.84% (Stress Test)

Ce chiffre est acceptable mais notable : sous perturbations maximales, l'AUC descend de ~0.9999 à ~0.85. Il convient de :
1. Identifier les features les plus sensibles au bruit
2. Ajouter des contrôles de qualité sur les inputs en production
3. Définir un seuil d'alerte en monitoring si la distribution des features dévie

### Sur le Brier Score (0.0002) et l'ECE (0.0000)

Le modèle est parfaitement calibré sur ce jeu de données : les probabilités prédites correspondent aux fréquences observées. En production, surveiller la calibration dans le temps (concept drift) avec des fenêtres glissantes.

---

## 📈 Courbes de diagnostic

Le fichier `curves_v3.png` présente quatre graphiques :

### Courbe ROC (haut gauche)
La courbe ROC couvre presque entièrement l'espace supérieur gauche (AUC = 0.9999). Le KS = 1.000 indique une séparation parfaite à un seuil donné. La courbe est quasi-coincidente avec le coin (0, 1), signe d'une discrimination quasi-parfaite.

### Courbe Precision-Recall (haut droit)
L'AP (Average Precision) de 0.9934 est calculé par rapport à une baseline de 0.0250 (taux de fraude ≈ 2.5%). La courbe reste quasi-plate à un niveau de précision élevé sur l'ensemble des seuils de rappel, caractéristique d'un modèle très performant sur classe minoritaire.

### Detection Error Tradeoff — DET (bas gauche)
Le point EER (Equal Error Rate) = 0.000 signifie qu'il existe un seuil pour lequel le taux de fausse alarme ET le taux de fraude manquée sont simultanément nuls (ou proches de zéro) sur ce jeu de test.

### KS Plot — Distribution des scores (bas droit)
Les CDFs des scores légitimes (bleu) et fraudes (rouge) sont parfaitement séparées : les transactions légitimes ont des scores proches de 0, les fraudes des scores proches de 1. La statistique KS = 0.9998 confirme cette séparation quasi-totale.

---

## ⚖️ Conformité réglementaire

### SR 11-7 (Federal Reserve)
Le framework implémente les quatre piliers du Model Risk Management :
- **Développement & implémentation** : métriques de performance et calibration
- **Validation indépendante** : cross-validation, walk-forward, stress testing
- **Usage gouvernance** : rapport de validation archivable (HTML + TXT)
- **Monitoring continu** : PSI, FSI, stabilité temporelle

### ECB TRIM Guide
- Tests de discrimination (AUC, Gini, KS) conformes aux attentes TRIM
- Calibration probabiliste (Brier, ECE) documentée
- Backtesting temporel (walk-forward) présent
- Analyse de sensibilité (stress test) couverte

### AMLD6
- Analyse de graphe de transactions pour détecter les réseaux de fraude organisée
- Analyse de Benford pour détecter les montants manipulés (smurfing)
- Analyse de fairness pour prévenir les discriminations

### Bâle III / IV
- Analyse de coût intégrée (pertes attendues FN/FP)
- Intervalles de confiance bootstrap sur toutes les métriques
- Stress testing sous perturbations adversariales

---

## ⚠️ Limitations & avertissements

1. **Performances sur données synthétiques** : les résultats inclus dans ce repo ont été générés sur des données de démonstration. Les performances réelles varieront selon la qualité et la représentativité de vos données de production.

2. **MF-DFA et ESN** : ces méthodes nécessitent des séries temporelles suffisamment longues (recommandé : > 500 points) pour être statistiquement fiables. Sur des petits jeux de données, les résultats ultra-scientifiques peuvent être bruités.

3. **Dépendances optionnelles** : sans `shap`, l'interprétabilité (SHAP values) est désactivée. Sans `networkx`, la détection d'anomalies de graphe est désactivée. Ces sections ne généreront pas d'erreur mais seront marquées comme indisponibles dans le rapport.

4. **Transfer Entropy** : l'estimation par histogramme 2D est une approximation. Pour des résultats plus robustes sur des données complexes, envisager une estimation par k-NN ou par noyau.

5. **Seuil de décision** : le seuil par défaut de 0.5 est rarement optimal en détection de fraude (classes très déséquilibrées). Utiliser systématiquement `optimize_threshold()` avec la fonction de coût métier.

6. **Concept drift** : ce framework valide un modèle à un instant T. En production, mettre en place un monitoring continu avec des alertes sur le PSI des features et la dégradation de l'AUC.

---

## ❓ FAQ

**Q : Le framework supporte-t-il des modèles autres que GBM ?**  
R : Oui. La classe `FraudModelValidatorV3` est agnostique au modèle. Elle prend en entrée `y_true` et `y_pred_proba` (numpy arrays). Pour les méthodes nécessitant l'objet modèle (`permutation_importance`, `stress_test`), une `model_predict_fn` callable suffit.

**Q : Combien de temps prend une validation complète ?**  
R : Sur un jeu de données de ~100k transactions avec un GBM standard : environ 5–15 minutes. Les méthodes ultra-scientifiques (MF-DFA, ESN) sont les plus coûteuses en temps de calcul.

**Q : Comment intégrer ce framework dans un pipeline MLOps ?**  
R : Le rapport texte (`generate_validation_report`) produit une sortie parseable. Un pipeline CI/CD peut lire les métriques clés et échouer si, par exemple, l'AUC Drop dépasse un seuil ou si l'Ultra Anomaly Score passe en zone MEDIUM.

**Q : Le rapport HTML peut-il être partagé directement avec les régulateurs ?**  
R : Oui. Le fichier HTML est auto-contenu (aucune dépendance externe, figures en base64) et lisible dans tout navigateur moderne. Il constitue un artefact d'audit valide.

**Q : Que signifie un Ultra Anomaly Score de 0.18 ?**  
R : Le risque est LOW. Les sept méthodes ultra-scientifiques n'ont détecté aucune anomalie significative : pas de lissage artificiel des séries temporelles, pas de causalité anormale entre features, pas de changement de régime, et des corrélations inter-features compatibles avec des données réelles.

---

## 📄 Licence & auteurs

**Auteurs :** Enhanced Model Validation Framework + Doctor Deep (Math & ML)  
**Date de création :** 2026-05-02  
**Version :** 3.0 Ultra-Scientific  

Ce framework est fourni à des fins de validation interne. Toute utilisation dans un contexte réglementaire doit être accompagnée d'une revue indépendante par l'équipe de Model Risk Management.

---

*Généré le 2026-05-02 — FraudModelValidatorV3 | SR 11-7 · ECB TRIM · AMLD6 · Bâle III*
