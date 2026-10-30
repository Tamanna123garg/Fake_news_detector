"""Extend dataset_original_500.csv to 5,000 rows (rows 501-5000 are SYNTHETIC).

Usage:  python scripts/generate_synthetic_rows.py

Design notes
------------
* Same 8 columns, dtypes and value ranges as the 500 provided rows.
* The provided 500 rows are kept verbatim (article_id 1-500).
* Rows 501-5000 are generated.  Each numeric feature is a 55/45 mixture of
  (a) a uniform draw over the full range (what the original rows look like) and
  (b) a label-conditioned truncated-normal draw that encodes well known stylistic
      tendencies of misleading articles (more "!" and ALL-CAPS, more extreme
      sentiment, shorter text).  This gives a MODEST, realistic signal - not a
      trivially separable dataset.
* political_lean is generated independently of label on purpose: it is NOT used
  as a model feature, only for the fairness audit.
* Fixed seed -> fully reproducible.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N_TOTAL = 5000
ROOT = Path(__file__).resolve().parents[1]

rng = np.random.default_rng(SEED)
orig = pd.read_csv(ROOT / "data" / "dataset_original_500.csv")
n_new = N_TOTAL - len(orig)

labels = rng.choice(["FAKE", "REAL"], size=n_new, p=[0.44, 0.56])  # original: 44.4 % FAKE
is_fake = labels == "FAKE"
mix = lambda: rng.random(n_new) < 0.45  # use label-conditioned draw for 45 % of rows


def tnorm(mu_fake, mu_real, sd, lo, hi):
    mu = np.where(is_fake, mu_fake, mu_real)
    return np.clip(rng.normal(mu, sd), lo, hi)


# word_count 200-2000 (shorter for fake)
wc_u = rng.integers(200, 2000, n_new, endpoint=True)
wc_c = tnorm(750, 1400, 350, 200, 2000).round().astype(int)
word_count = np.where(mix(), wc_c, wc_u)

# sentiment -1..1 (fake tends to be more extreme)
sent_u = rng.uniform(-1, 1, n_new)
sign = rng.choice([-1, 1], n_new)
sent_fake = sign * np.clip(rng.normal(0.72, 0.2, n_new), 0, 1)
sent_real = np.clip(rng.normal(0.0, 0.35, n_new), -1, 1)
sent_c = np.where(is_fake, sent_fake, sent_real)
sentiment = np.round(np.where(mix(), sent_c, sent_u), 3)

# exclamation_count 0-15
ex_u = rng.integers(0, 15, n_new, endpoint=True)
ex_c = tnorm(10, 4, 3, 0, 15).round().astype(int)
exclam = np.where(mix(), ex_c, ex_u)

# all_caps_pct 0-0.30
caps_u = rng.uniform(0, 0.30, n_new)
caps_c = tnorm(0.21, 0.09, 0.05, 0, 0.30)
caps = np.round(np.where(mix(), caps_c, caps_u), 3)

# source_1..source_20: a few outlets are slightly more likely to publish fake items
src_ids = np.arange(1, 21)
w_fake = np.ones(20); w_fake[[1, 4, 6, 10, 12, 14]] = 2.2   # source_2,5,7,11,13,15
w_real = np.ones(20); w_real[[1, 4, 6, 10, 12, 14]] = 0.6
source = np.where(
    is_fake,
    rng.choice(src_ids, n_new, p=w_fake / w_fake.sum()),
    rng.choice(src_ids, n_new, p=w_real / w_real.sum()),
)
source = np.array([f"source_{i}" for i in source])

# political_lean - independent of label (kept out of the model on purpose)
lean = rng.choice(["Center", "Right", "Unknown", "Left"], n_new, p=[0.27, 0.256, 0.254, 0.22])

new = pd.DataFrame({
    "article_id": np.arange(len(orig) + 1, N_TOTAL + 1),
    "source": source,
    "word_count": word_count,
    "sentiment_score": sentiment,
    "exclamation_count": exclam,
    "all_caps_pct": caps,
    "label": labels,
    "political_lean": lean,
})

full = pd.concat([orig, new], ignore_index=True)
assert list(full.columns) == list(orig.columns)
assert full.isna().sum().sum() == 0 and full.article_id.is_unique
full.to_csv(ROOT / "data" / "dataset.csv", index=False)
print(full.shape)
print(full.label.value_counts(normalize=True).round(3).to_dict())
