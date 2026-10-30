"""Exploratory data analysis for dataset.csv (6 visualisations + summary text).

    python -m src.eda
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
NUM = ["word_count", "sentiment_score", "exclamation_count", "all_caps_pct"]
COL = {"REAL": "#1f9d62", "FAKE": "#d6453d"}

df = pd.read_csv(ROOT / "data" / "dataset.csv")

# 1 class balance
fig, ax = plt.subplots(figsize=(4, 3.4))
vc = df["label"].value_counts()
ax.bar(vc.index, vc.values, color=[COL[k] for k in vc.index])
for i, v in enumerate(vc.values):
    ax.text(i, v + 40, f"{v} ({v / len(df):.1%})", ha="center")
ax.set_title("Class balance"); ax.set_ylabel("Articles")
fig.tight_layout(); fig.savefig(OUT / "eda_1_class_balance.png", dpi=160); plt.close(fig)

# 2 feature distributions by label
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
for ax, c in zip(axes.ravel(), NUM):
    for lab in ("REAL", "FAKE"):
        ax.hist(df.loc[df.label == lab, c], bins=25, alpha=.55, label=lab, color=COL[lab], density=True)
    ax.set_title(c); ax.legend(fontsize=8)
fig.suptitle("Feature distributions by label"); fig.tight_layout()
fig.savefig(OUT / "eda_2_feature_distributions.png", dpi=160); plt.close(fig)

# 3 box plots
fig, axes = plt.subplots(1, 4, figsize=(11, 3.4))
for ax, c in zip(axes, NUM):
    ax.boxplot([df.loc[df.label == l, c] for l in ("REAL", "FAKE")], tick_labels=["REAL", "FAKE"])
    ax.set_title(c, fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "eda_3_boxplots.png", dpi=160); plt.close(fig)

# 4 correlation heatmap
d2 = df[NUM].assign(is_fake=(df.label == "FAKE").astype(int))
corr = d2.corr()
fig, ax = plt.subplots(figsize=(5.4, 4.6))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right"); ax.set_yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im); ax.set_title("Correlation matrix"); fig.tight_layout()
fig.savefig(OUT / "eda_4_correlation.png", dpi=160); plt.close(fig)

# 5 fake share by source
share = df.groupby("source")["label"].apply(lambda s: (s == "FAKE").mean()).sort_values()
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.barh(share.index, share.values, color="#2f6fed"); ax.axvline((df.label == "FAKE").mean(), color="k", ls="--", lw=1)
ax.set_xlabel("Share of articles labelled FAKE"); ax.set_title("FAKE share by source (dashed = overall)")
fig.tight_layout(); fig.savefig(OUT / "eda_5_source_fake_share.png", dpi=160); plt.close(fig)

# 6 political lean vs label
ct = pd.crosstab(df.political_lean, df.label, normalize="index")
fig, ax = plt.subplots(figsize=(5, 3.4))
ct[["REAL", "FAKE"]].plot(kind="bar", stacked=True, color=[COL["REAL"], COL["FAKE"]], ax=ax)
ax.set_ylabel("Share"); ax.set_title("Label mix by political lean"); plt.xticks(rotation=0)
fig.tight_layout(); fig.savefig(OUT / "eda_6_lean_vs_label.png", dpi=160); plt.close(fig)

summary = [f"rows={len(df)} cols={df.shape[1]} nulls={int(df.isna().sum().sum())} duplicates={int(df.duplicated().sum())}",
           df[NUM].describe().round(3).to_string(), "", "mean by label:", df.groupby("label")[NUM].mean().round(3).to_string()]
(ROOT / "reports" / "eda_summary.txt").write_text("\n".join(summary))
print("\n".join(summary))
