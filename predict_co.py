# =============================================================================
# STP 213 — Predicting Carryover Risk Among Polytechnic Students
#            Using Supervised Machine Learning
# =============================================================================
# Supervisor : OLUMODEJI I.A.
# Session    : 2025/2026 | Department of Science Technology
# =============================================================================
# HOW TO RUN:
#   1. Install dependencies:
#      pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap openpyxl
#   2. Place STP_213_2026.xlsx in the same folder as this script.
#   3. Run:  python predict_co.py
#   All outputs (charts, tables) will be saved in an "outputs/" folder.
# =============================================================================


# ─── 0. SETUP ────────────────────────────────────────────────────────────────

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model      import LogisticRegression
from sklearn.tree              import DecisionTreeClassifier, export_text
from sklearn.ensemble          import RandomForestClassifier
from sklearn.svm               import SVC
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.preprocessing     import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics           import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from imblearn.over_sampling    import SMOTE
from xgboost                   import XGBClassifier

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

os.makedirs("outputs", exist_ok=True)
print("Setup complete. Outputs will be saved to outputs/")


# ─── 1. DATA LOADING ─────────────────────────────────────────────────────────

FILE   = "STP_213_2026.xlsx"
COLS   = ["SNO", "MATRIC", "SNAME", "INIT", "Exam", "Pract", "CA", "Total"]
SHEETS = {
    "nd2"     : 0,   # main cohort   → carryover label = 0
    "ND2 CO23": 1,   # 2023 repeaters → carryover label = 1
    "nd2 co19": 1,   # 2019 repeaters → carryover label = 1
}


def parse_sheet(path, sheet_name, carryover_label):
    """Load one sheet, keep only valid student rows, attach carryover label."""
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    df  = raw.iloc[11:].copy()          # data rows start at index 11
    df.columns = COLS
    df  = df[df["MATRIC"].astype(str).str.startswith("ST/")]
    for col in ["Exam", "Pract", "CA", "Total"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["Carryover"] = carryover_label
    return df.reset_index(drop=True)


frames = [parse_sheet(FILE, sheet, label) for sheet, label in SHEETS.items()]
df     = pd.concat(frames, ignore_index=True)

print(f"\n Total records loaded : {len(df)}")
print(f"   Carryover (label=1)  : {df['Carryover'].sum()}")
print(f"   No carryover (label=0): {(df['Carryover']==0).sum()}")
print(df[["Exam", "Pract", "CA", "Total", "Carryover"]].head(5))


# ─── 2. PREPROCESSING ────────────────────────────────────────────────────────

# 2a. Flag absent students (all scores = 0 → complete non-participation)
df["Absent"] = (
    (df["Exam"] == 0) & (df["Pract"] == 0) & (df["CA"] == 0)
).astype(int)

# Absent students are treated as highest carryover risk
df.loc[df["Absent"] == 1, "Carryover"] = 1

# 2b. Anonymise — remove all personally identifiable columns
df["Student_ID"] = ["S" + str(i + 1).zfill(3) for i in range(len(df))]
df = df.drop(columns=["SNO", "MATRIC", "SNAME", "INIT"])

print(f"\n Data anonymised.")
print(f"   Absent students flagged : {df['Absent'].sum()}")
print(f"   Updated Carryover=1 count: {df['Carryover'].sum()}")

# 2c. Basic null check
print(f"\n Null values:\n{df.isnull().sum()}")


# ─── 3. EXPLORATORY DATA ANALYSIS (EDA) ──────────────────────────────────────

print("\n Running EDA ...")

# 3a. Descriptive statistics
desc = df[["Exam", "Pract", "CA", "Total"]].describe().round(2)
print("\nDescriptive Statistics:\n", desc)
desc.to_csv("outputs/descriptive_statistics.csv")

# 3b. Class distribution bar chart
fig, ax = plt.subplots(figsize=(5, 4))
counts = df["Carryover"].value_counts()
ax.bar(["No Carryover Risk (0)", "Carryover Risk (1)"],
       [counts.get(0, 0), counts.get(1, 0)],
       color=["#2a78d6", "#e34948"], edgecolor="white", width=0.5)
ax.set_title("Class Distribution — Carryover Risk", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Students")
for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
    ax.text(i, v + 0.5, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/01_class_distribution.png", dpi=150)
plt.close()

# 3c. Score distributions (active students only — Total > 0)
active = df[df["Absent"] == 0].copy()
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, color in zip(axes, ["Exam", "Pract", "CA"],
                           ["#2a78d6", "#1baf7a", "#eda100"]):
    axes_obj = ax
    sns.histplot(active[col], ax=axes_obj, bins=20, color=color, edgecolor="white")
    axes_obj.set_title(f"{col} Score Distribution", fontweight="bold")
    axes_obj.set_xlabel("Score")
    axes_obj.axvline(active[col].mean(), color="red", linestyle="--",
                     label=f"Mean={active[col].mean():.1f}")
    axes_obj.legend(fontsize=9)
plt.suptitle("Component Score Distributions (Active Students)", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("outputs/02_score_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# 3d. Total score histogram with grade band colours
grade_bands = [(0, 39, "#e34948", "F"), (40, 44, "#eb6834", "E"),
               (45, 49, "#4a3aa7", "D"), (50, 59, "#eda100", "C"),
               (60, 69, "#2a78d6", "B"), (70, 100, "#1baf7a", "A")]
fig, ax = plt.subplots(figsize=(10, 5))
for lo, hi, color, grade in grade_bands:
    subset = active[(active["Total"] >= lo) & (active["Total"] <= hi)]["Total"]
    ax.hist(subset, bins=range(lo, hi + 2), color=color, alpha=0.85,
            edgecolor="white", label=f"Grade {grade} ({lo}–{hi})")
ax.set_title("Total Score Distribution by Grade Band", fontsize=13, fontweight="bold")
ax.set_xlabel("Total Score")
ax.set_ylabel("Number of Students")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig("outputs/03_total_score_by_grade.png", dpi=150)
plt.close()

# 3e. Box plots: carryover vs non-carryover
fig, axes = plt.subplots(1, 4, figsize=(14, 5))
for ax, col in zip(axes, ["Exam", "Pract", "CA", "Total"]):
    df.boxplot(column=col, by="Carryover", ax=ax,
               boxprops=dict(color="#2a78d6"),
               medianprops=dict(color="#e34948", linewidth=2))
    ax.set_title(col, fontweight="bold")
    ax.set_xlabel("Carryover (0=No, 1=Yes)")
    ax.set_xticklabels(["No Risk", "At Risk"])
plt.suptitle("Score Distributions: Carryover vs Non-Carryover Students",
             fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/04_boxplots_carryover.png", dpi=150)
plt.close()

# 3f. Correlation heatmap
fig, ax = plt.subplots(figsize=(6, 5))
corr = df[["Exam", "Pract", "CA", "Total", "Carryover"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues",
            ax=ax, linewidths=0.5, linecolor="white")
ax.set_title("Feature Correlation Heatmap", fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/05_correlation_heatmap.png", dpi=150)
plt.close()

print("EDA charts saved to outputs/")


# ─── 4. FEATURE ENGINEERING ──────────────────────────────────────────────────

print("\n  Engineering features ...")

# Normalised ratios
df["Exam_ratio"]   = df["Exam"]  / 40
df["Pract_ratio"]  = df["Pract"] / 40
df["CA_ratio"]     = df["CA"]    / 20

# Gap between practical and exam performance
df["Exam_Pract_gap"] = df["Pract_ratio"] - df["Exam_ratio"]

# What fraction of the total mark comes from CA?
df["CA_contribution"] = df.apply(
    lambda r: r["CA"] / r["Total"] if r["Total"] > 0 else 0, axis=1
)

# Is the student's mark driven mainly by practical?
df["Practical_dominance"] = df.apply(
    lambda r: r["Pract"] / (r["Exam"] + r["Pract"])
              if (r["Exam"] + r["Pract"]) > 0 else 0, axis=1
)

# Class rank as percentile
df["Score_percentile"] = df["Total"].rank(pct=True) * 100

# Critical low exam flag
df["Low_exam_flag"] = (df["Exam"] < 10).astype(int)

# Distance from pass mark (negative = fail)
df["Pass_margin"] = df["Total"] - 40

print("   New features added:")
eng_cols = ["Exam_ratio", "Pract_ratio", "CA_ratio", "Exam_Pract_gap",
            "CA_contribution", "Practical_dominance", "Score_percentile",
            "Low_exam_flag", "Pass_margin"]
print(df[eng_cols].describe().round(3))


# ─── 5. FEATURE SELECTION ────────────────────────────────────────────────────

print("\n Selecting features with RFE ...")

ALL_FEATURES = eng_cols + ["Absent"]
X_all = df[ALL_FEATURES]
y     = df["Carryover"]

rfe = RFE(
    estimator=RandomForestClassifier(n_estimators=100, random_state=SEED),
    n_features_to_select=7
)
rfe.fit(X_all, y)

SELECTED = [f for f, s in zip(ALL_FEATURES, rfe.support_) if s]
print(f"   Selected features ({len(SELECTED)}): {SELECTED}")

# Feature importance from the RFE estimator
importances = pd.Series(
    rfe.estimator_.feature_importances_, index=SELECTED
).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))
importances.plot(kind="barh", color="#2a78d6", ax=ax, edgecolor="white")
ax.set_title("Feature Importance (RFE — Random Forest)", fontweight="bold")
ax.set_xlabel("Importance Score")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("outputs/06_feature_importance_rfe.png", dpi=150)
plt.close()

X = df[SELECTED]


# ─── 6. TRAIN / TEST SPLIT + SMOTE ───────────────────────────────────────────

print("\n  Splitting data and applying SMOTE ...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y
)

print(f"   Train set: {len(X_train)} samples | "
      f"Class balance: {y_train.value_counts().to_dict()}")
print(f"   Test set : {len(X_test)} samples  | "
      f"Class balance: {y_test.value_counts().to_dict()}")

# Apply SMOTE only to training data
smote = SMOTE(random_state=SEED)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"   After SMOTE — Train: {pd.Series(y_train_sm).value_counts().to_dict()}")

# Scale features for distance-based models (SVM, KNN)
scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train_sm)
X_test_sc   = scaler.transform(X_test)


# ─── 7. MODEL TRAINING ───────────────────────────────────────────────────────

print("\n Training all 6 models ...")

lr  = LogisticRegression(max_iter=1000, random_state=SEED)
dt  = DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=SEED)
rf  = RandomForestClassifier(n_estimators=200, max_depth=8,
                              class_weight="balanced", random_state=SEED)
svm = SVC(kernel="rbf", probability=True, random_state=SEED)
knn = KNeighborsClassifier(n_neighbors=5)
xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                     eval_metric="logloss", random_state=SEED)

# Models using SMOTE-resampled (unscaled) data
unscaled_models = {
    "Logistic Regression": lr,
    "Decision Tree"      : dt,
    "Random Forest"      : rf,
    "XGBoost"            : xgb,
}

# Models using scaled data
scaled_models = {
    "SVM": svm,
    "KNN": knn,
}

for name, model in unscaled_models.items():
    model.fit(X_train_sm, y_train_sm)
    print(f"   {name} trained.")

for name, model in scaled_models.items():
    model.fit(X_train_sc, y_train_sm)
    print(f"   {name} trained.")


# ─── 8. EVALUATION ───────────────────────────────────────────────────────────

print("\n Evaluating all models ...")

ALL_MODELS = {**unscaled_models, **scaled_models}

def evaluate(model, X_te, y_te, name):
    pred  = model.predict(X_te)
    rep   = classification_report(y_te, pred, output_dict=True, zero_division=0)
    proba = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None
    auc   = roc_auc_score(y_te, proba) if proba is not None else float("nan")
    return {
        "Model"    : name,
        "Accuracy" : round(rep["accuracy"], 4),
        "Precision": round(rep.get("1", {}).get("precision", 0), 4),
        "Recall"   : round(rep.get("1", {}).get("recall", 0), 4),
        "F1-Score" : round(rep.get("1", {}).get("f1-score", 0), 4),
        "AUC-ROC"  : round(auc, 4),
        "_pred"    : pred,
        "_proba"   : proba,
    }

results = []
for name, model in ALL_MODELS.items():
    X_te = X_test_sc if name in scaled_models else X_test
    results.append(evaluate(model, X_te, y_test, name))

# Results table (printable)
results_df = pd.DataFrame(results).drop(columns=["_pred", "_proba"])
print("\n" + "="*70)
print(results_df.to_string(index=False))
print("="*70)
results_df.to_csv("outputs/07_model_comparison.csv", index=False)


# ─── 9. CROSS-VALIDATION ─────────────────────────────────────────────────────

print("\n Running 5-fold cross-validation ...")

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_results = []

for name, model in unscaled_models.items():
    f1_scores = cross_val_score(model, X, y, cv=kf, scoring="f1")
    cv_results.append({
        "Model"   : name,
        "CV F1 Mean": round(f1_scores.mean(), 4),
        "CV F1 Std" : round(f1_scores.std(), 4),
    })
    print(f"   {name}: F1 = {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")

cv_df = pd.DataFrame(cv_results)
cv_df.to_csv("outputs/08_cross_validation.csv", index=False)


# ─── 10. CONFUSION MATRICES ──────────────────────────────────────────────────

print("\n Generating confusion matrices ...")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, res in enumerate(results):
    cm = confusion_matrix(y_test, res["_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Risk", "At Risk"])
    disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
    axes[i].set_title(res["Model"], fontweight="bold", fontsize=11)

plt.suptitle("Confusion Matrices — All 6 Models", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("outputs/09_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Confusion matrices saved.")


# ─── 11. ROC CURVES ──────────────────────────────────────────────────────────

print("\n Plotting ROC curves ...")

fig, ax = plt.subplots(figsize=(8, 6))
colors  = ["#2a78d6", "#1baf7a", "#eda100", "#e34948", "#4a3aa7", "#eb6834"]

for res, color in zip(results, colors):
    if res["_proba"] is not None:
        fpr, tpr, _ = roc_curve(y_test, res["_proba"])
        ax.plot(fpr, tpr, label=f"{res['Model']} (AUC={res['AUC-ROC']:.3f})",
                linewidth=2, color=color)

ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — Carryover Risk Prediction", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/10_roc_curves.png", dpi=150)
plt.close()
print("ROC curves saved.")


# ─── 12. MODEL COMPARISON BAR CHART ─────────────────────────────────────────

print("\n Plotting model comparison chart ...")

metrics   = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
model_names = [r["Model"] for r in results]
x         = np.arange(len(model_names))
width     = 0.15
bar_colors = ["#2a78d6", "#1baf7a", "#eda100", "#e34948", "#4a3aa7"]

fig, ax = plt.subplots(figsize=(14, 6))
for i, (metric, color) in enumerate(zip(metrics, bar_colors)):
    vals = [r[metric] for r in results]
    bars = ax.bar(x + i * width, vals, width, label=metric,
                  color=color, alpha=0.85, edgecolor="white")

ax.set_xticks(x + width * 2)
ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=10)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.1)
ax.set_title("Model Performance Comparison — All Metrics", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.axhline(0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/11_model_comparison.png", dpi=150)
plt.close()
print("  Model comparison chart saved.")


# ─── 13. SHAP INTERPRETABILITY ───────────────────────────────────────────────

print("\n Running SHAP analysis on XGBoost ...")

explainer  = shap.TreeExplainer(xgb)
shap_vals  = explainer.shap_values(X_test)

# 13a. Beeswarm (summary) plot
plt.figure(figsize=(9, 6))
shap.summary_plot(shap_vals, X_test, feature_names=SELECTED, show=False)
plt.title("SHAP Summary — Feature Impact on Carryover Prediction",
          fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("outputs/12_shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()

# 13b. Bar chart of mean absolute SHAP values
plt.figure(figsize=(8, 5))
shap.summary_plot(shap_vals, X_test, feature_names=SELECTED,
                  plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Mean |SHAP value|)",
          fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("outputs/13_shap_importance_bar.png", dpi=150, bbox_inches="tight")
plt.close()

# 13c. Dependence plot for top feature
top_feature = SELECTED[0]
plt.figure(figsize=(7, 5))
shap.dependence_plot(top_feature, shap_vals, X_test,
                     feature_names=SELECTED, show=False)
plt.title(f"SHAP Dependence Plot — {top_feature}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/14_shap_dependence_top_feature.png", dpi=150, bbox_inches="tight")
plt.close()

# 13d. Force plot for one at-risk student (saved as HTML)
at_risk_idx = y_test[y_test == 1].index
if len(at_risk_idx) > 0:
    idx  = list(X_test.index).index(at_risk_idx[0])
    html = shap.force_plot(
        explainer.expected_value,
        shap_vals[idx],
        X_test.iloc[idx],
        feature_names=SELECTED,
        show=False
    )
    shap.save_html("outputs/15_shap_force_plot_student.html", html)
    print("Force plot saved as HTML.")

print("All SHAP plots saved.")


# ─── 14. DECISION TREE VISUALISATION ─────────────────────────────────────────

print("\nExporting Decision Tree rules ...")

tree_rules = export_text(dt, feature_names=SELECTED, max_depth=4)
with open("outputs/16_decision_tree_rules.txt", "w") as f:
    f.write("Decision Tree Rules (max depth = 4)\n")
    f.write("=" * 60 + "\n\n")
    f.write(tree_rules)
print("Decision tree rules saved to outputs/16_decision_tree_rules.txt")


# ─── 15. FINAL SUMMARY REPORT ────────────────────────────────────────────────

print("\nGenerating final summary report ...")

best_model = results_df.sort_values("F1-Score", ascending=False).iloc[0]

summary_lines = [
    "=" * 65,
    "  STP 213 — CARRYOVER RISK PREDICTION: FINAL SUMMARY",
    "=" * 65,
    f"  Dataset       : {len(df)} students ({df['Carryover'].sum()} carryover risk)",
    f"  Features used : {SELECTED}",
    f"  Train/Test    : {len(X_train_sm)} (after SMOTE) / {len(X_test)}",
    "",
    "  MODEL PERFORMANCE (on test set)",
    "-" * 65,
]
for _, row in results_df.iterrows():
    summary_lines.append(
        f"  {row['Model']:<22} | Acc={row['Accuracy']:.3f} "
        f"| Rec={row['Recall']:.3f} | F1={row['F1-Score']:.3f} "
        f"| AUC={row['AUC-ROC']:.3f}"
    )
summary_lines += [
    "-" * 65,
    f"\nBEST MODEL : {best_model['Model']}",
    f"     Recall     : {best_model['Recall']:.4f}",
    f"     F1-Score   : {best_model['F1-Score']:.4f}",
    f"     AUC-ROC    : {best_model['AUC-ROC']:.4f}",
    "",
    "  All outputs saved in: outputs/",
    "=" * 65,
]

report = "\n".join(summary_lines)
print("\n" + report)

with open("outputs/00_FINAL_SUMMARY.txt", "w") as f:
    f.write(report)

print("\nPipeline complete! All files are in the outputs/ folder.")


# ─── OUTPUTS MANIFEST ────────────────────────────────────────────────────────
# outputs/
# ├── 00_FINAL_SUMMARY.txt               ← Human-readable summary report
# ├── 01_class_distribution.png          ← Bar chart: carryover vs non-carryover
# ├── 02_score_distributions.png         ← Histograms: Exam, Pract, CA
# ├── 03_total_score_by_grade.png        ← Total scores coloured by grade band
# ├── 04_boxplots_carryover.png          ← Boxplots: carryover vs non-carryover
# ├── 05_correlation_heatmap.png         ← Feature correlation matrix
# ├── 06_feature_importance_rfe.png      ← Feature importance from RFE
# ├── 07_model_comparison.csv            ← All model metrics as CSV
# ├── 08_cross_validation.csv            ← 5-fold CV results as CSV
# ├── 09_confusion_matrices.png          ← 6 confusion matrices in one figure
# ├── 10_roc_curves.png                  ← ROC curves for all models
# ├── 11_model_comparison.png            ← Grouped bar chart of all metrics
# ├── 12_shap_summary_beeswarm.png       ← SHAP beeswarm plot
# ├── 13_shap_importance_bar.png         ← SHAP mean absolute feature importance
# ├── 14_shap_dependence_top_feature.png ← SHAP dependence plot
# ├── 15_shap_force_plot_student.html    ← SHAP force plot (one student, HTML)
# ├── 16_decision_tree_rules.txt         ← Human-readable DT rules
# └── descriptive_statistics.csv         ← Summary statistics table