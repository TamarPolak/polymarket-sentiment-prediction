# Polymarket Sentiment Prediction

This project predicts short-term price movement in a Polymarket political prediction market using market data and sentiment signals from public political discourse.

## Research Question

Can sentiment and discussion volume from X/Twitter improve prediction of Benjamin Netanyahu's contract price movement on Polymarket?

## Project Goal

The goal is to compare market-only prediction with a future Market + Sentiment model. The project starts with Polymarket price history and engineered market features, then adds public discussion signals related to Netanyahu, Bennett, Bibi, Naftali Bennett, Israeli election, בחירות בישראל, נתניהו, and בנט.

## Current MVP

- Collect Polymarket price data.
- Collect public text data related to Israeli election candidates.
- Run sentiment analysis.
- Merge market and sentiment data by time windows.
- Train baseline and combined ML models.
- Display results in a dashboard.

For the current submission, the sentiment part uses a fake/sample dataset for testing only. These are not real X posts. X API access is paid, so real data collection should happen only after the local pipeline works correctly.

## Repository Structure

```text
data/
  raw/
    text_posts_sample.csv
    polymarket_price_history.csv
  processed/
    text_posts_with_sentiment.csv
    sentiment_features_by_hour.csv
docs/
  week7_technical_plan.md
src/
  collect_polymarket.py
  collect_price_history.py
  features.py
  train_binary_model.py
  train_multiclass_model.py
  sentiment.py
  build_combined_dataset.py
dashboard/
  app.py
results/
  baseline_results.md
  binary_model_results.txt
```

## Install Requirements

```bash
pip install -r requirements.txt
```

If Streamlit is not installed and you want to run the dashboard:

```bash
pip install streamlit
```

## Run Polymarket Data Collection

Use Tamar's existing Polymarket collection scripts:

```bash
python src/collect_polymarket.py
python src/collect_price_history.py
```

Do not change Tamar's collection scripts unless a later integration step requires a small compatibility update.

## Run the Binary Model

```bash
python src/train_binary_model.py
```

This is the current Market-only reference model.

## Run the Sentiment Pipeline

```bash
python src/sentiment.py
```

This reads:

```text
data/raw/text_posts_sample.csv
```

and writes:

```text
data/processed/text_posts_with_sentiment.csv
```

The output includes:

```text
timestamp,text,candidate,likes,reposts,comments,clean_text,sentiment_score,sentiment_label
```

## Build Hourly Sentiment Features

```bash
python src/build_combined_dataset.py
```

This reads:

```text
data/processed/text_posts_with_sentiment.csv
```

and writes:

```text
data/processed/sentiment_features_by_hour.csv
```

The output contains hourly sentiment features that can later be merged with Polymarket market features.

## Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard checks whether files exist before loading them, so it should still open when some market, model, or sentiment files are missing.

It can show:

- Netanyahu price over time.
- Bennett price over time.
- Price gap over time.
- Model results summary.
- Sentiment summary.

## Preliminary Results

The current baseline results are the Market-only results from Tamar's binary and multiclass models.

The sentiment pipeline currently validates the data flow only. It does not yet prove that sentiment improves prediction because the current sentiment input is sample data, not real X data.

## Next Step

The next experiment is to compare:

- Market-only model.
- Market + Sentiment model.

Before collecting real X data, the sentiment pipeline should be validated on the sample file. After that, a limited number of real public X posts can be collected under a controlled paid API budget and saved using the same schema.
