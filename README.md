# Fake News Detection Using NLP

HCLTech Career Shaper Programme 2025-26 - Project 31 (Digital-Data-AI / AIML)

A Flask web app and JSON API that scores text for fake-news-style wording, with word-level and feature-level explanations. **It is decision support for a human reviewer, not a fact-checker.** Read `MODEL_CARD.md` first.

* **Text model** - TF-IDF + Logistic Regression trained on the LIAR benchmark (62.1 % test accuracy).
* **Style model** - Logistic Regression + Random Forest + Gradient Boosting ensemble on `data/dataset.csv` (5,000 rows; rows 501-5000 are synthetic).
* **Explanations** - LIME-style local surrogates; the notebook uses the official `lime` library.
* **Offline experiment** - `notebooks/02_bert_ensemble_lime.ipynb` (DistilBERT + TF-IDF ensemble, Colab GPU).

## Folder layout
```
app.py                  Flask app (UI + /api/predict)
src/                    features, train, train_text, explain, eda, model_store
data/dataset.csv        5,000 rows (deliverable)         data/data_dictionary.csv
data/liar/              LIAR train/valid/test.tsv
notebooks/              BERT + ensemble + LIME notebook
reports/                metrics.json, text_metrics.json, figures/ (EDA, ROC, LIME ...)
scripts/                generate_synthetic_rows.py
tests/                  unit + API tests
```

## 1. Run locally
Requires Python 3.10+ (tested on 3.13).

```bash
# Windows (PowerShell)                      # macOS / Linux
python -m venv .venv                        python3 -m venv .venv
.venv\Scripts\activate                      source .venv/bin/activate
pip install -r requirements.txt             pip install -r requirements.txt

python -m src.train          # trains style model   (~40 s) -> models/, reports/metrics.json
python -m src.train_text     # trains LIAR text model (~5 s) -> models/, reports/text_metrics.json
python -m src.eda            # (optional) regenerates the 6 EDA charts
python -m unittest discover -s tests -v    # (optional) 9 tests
python app.py                # open http://127.0.0.1:5000
```
If you skip the two training commands the app trains automatically on first start (about a minute).

### API example
```bash
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" \
     -d '{"text": "Says the state cut school funding every year."}'
```
`GET /api/health`, `GET /api/metrics`, and `POST /api/predict` with either `{"text": "..."}` or `{"features": {"word_count":..,"sentiment_score":..,"exclamation_count":..,"all_caps_pct":..}}`.

## 2. Push to GitHub
1. Create an empty repository on https://github.com/new (name: `fake-news-detector`, Public, do **not** add a README).
2. In this folder:
```bash
git init
git add .
git commit -m "Fake news detection using NLP - HCLTech project 31"
git branch -M main
git remote add origin https://github.com/<your-username>/fake-news-detector.git
git push -u origin main
```
Your repository link is `https://github.com/<your-username>/fake-news-detector`. (If GitHub asks for a password, use a Personal Access Token: Settings > Developer settings > Tokens, or install GitHub Desktop.)

## 3. Deploy free on Render
1. Sign up at https://render.com with your GitHub account.
2. **New > Web Service** > connect the `fake-news-detector` repo.
3. Render reads `render.yaml`. If you fill the form manually use:
   * Runtime: Python 3  |  Plan: Free
   * Build command: `pip install -r requirements.txt && python -m src.train && python -m src.train_text`
   * Start command: `gunicorn app:app --workers 1 --timeout 120`
   * Environment variable: `PYTHON_VERSION` = `3.12.7`
4. Click **Create Web Service**. The first build takes ~5-8 minutes. Your deployment link looks like `https://fake-news-detector-xxxx.onrender.com`.
5. Check `https://<your-app>.onrender.com/api/health` returns `{"status":"ok",...}`.

Free instances sleep after ~15 min idle; the first request afterwards takes ~30-60 s. Open the link once before a demo or submission check.

## Ethics
Do not use this model to flag specific political content without human review. See `MODEL_CARD.md` for limitations and fairness checks.

## Data & references
* Wang, W. Y. (2017). "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection. ACL. https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
* Ribeiro, Singh, Guestrin (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier (LIME). KDD.
* Sanh et al. (2019). DistilBERT. arXiv:1910.01108.
