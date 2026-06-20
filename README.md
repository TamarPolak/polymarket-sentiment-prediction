# Polymarket Sentiment Prediction

This project predicts short-term price movement in a Polymarket political prediction market using market data and sentiment features from public political discourse.

Case study: `Who will be the next Prime Minister of Israel after the next election?`

## Research Question

Can sentiment and discussion volume from X/Twitter improve prediction of Polymarket price movement for the leading candidates in the Israeli Prime Minister market?

The project does not directly predict who will become Prime Minister. It predicts whether Polymarket YES contract prices for leading candidates will go up, down, or remain stable.

## Final Project Scope

Final target:

```text
target_multiclass = Up / Down / Stable
```

Main candidates:

- Benjamin Netanyahu
- Naftali Bennett

Prediction horizons:

- 10 minutes
- 1 hour
- 2 hours
- 24 hours

If the current Polymarket price resolution does not support 10-minute labels, the code skips `10m` gracefully with a clear warning.

Target definition:

```text
price_change = future_price - current_price
MOVEMENT_THRESHOLD = 0.001
```

Labels:

- `Up`: `price_change > MOVEMENT_THRESHOLD`
- `Down`: `price_change < -MOVEMENT_THRESHOLD`
- `Stable`: otherwise

## Current MVP Status

The current pipeline is working end-to-end with:

- real Polymarket price history
- sample/fake sentiment posts for pipeline validation
- market feature engineering
- sentiment feature engineering
- market + sentiment dataset creation
- multiclass model comparison
- Streamlit dashboard

The sentiment data is still sample data. These are not real X/Twitter posts. X/Twitter API access is paid, so real data should be collected only after the pipeline is stable.

## Repository Structure

```text
data/
  raw/
    polymarket_price_history.csv
    text_posts_sample.csv
  processed/
    text_posts_with_sentiment.csv
    sentiment_features_by_hour.csv
    market_features.csv
  final/
    market_sentiment_dataset.csv
dashboard/
  app.py
docs/
  final_project_plan.md
  final_report.md
  week7_technical_plan.md
results/
  final_model_comparison.md
src/
  sentiment.py
  build_combined_dataset.py
  build_market_features.py
  build_market_sentiment_dataset.py
  train_combined_model.py
  collect_polymarket.py
  collect_price_history.py
```

## Install Requirements

```bash
pip install -r requirements.txt
```

If Streamlit is not installed:

```bash
pip install streamlit
```

## Run Polymarket Collection

Use the existing Polymarket collection scripts:

```bash
python src/collect_polymarket.py
python src/collect_price_history.py
```

These scripts create or update:

```text
data/raw/polymarket_price_history.csv
```

## Run the Final Pipeline

Run the full current pipeline in this order:

```bash
python src/sentiment.py
python src/build_combined_dataset.py
python src/build_market_features.py
python src/build_market_sentiment_dataset.py
python src/train_combined_model.py
```

Expected outputs:

```text
data/processed/text_posts_with_sentiment.csv
data/processed/sentiment_features_by_hour.csv
data/processed/market_features.csv
data/final/market_sentiment_dataset.csv
results/final_model_comparison.md
```

## Run the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard shows:

- Netanyahu price over time
- Bennett price over time
- price gap over time
- Up / Down / Stable class distribution by horizon
- final model comparison metrics
- sentiment summary
- sample sentiment rows

The dashboard checks file existence before loading data, so it should not crash if an output is missing.

## Sentiment Layer

The sentiment layer is implemented in:

```text
src/sentiment.py
```

It currently supports:

- text cleaning
- rule-based sentiment score
- sentiment label: positive / negative / neutral
- political entity detection
- party and bloc detection
- political topic detection
- strong Israeli political phrase detection
- target sentiment toward the main mentioned entity
- intensity score

The Israeli political lexicon includes figures, parties, blocs, slogans, and context terms related to Israeli political discourse. Context terms such as right/left, settlers, religious/secular, Arabs, Hamas, Palestinians, and Judea and Samaria are not automatically treated as positive or negative.

## Model Comparison

Final comparison:

- Dummy Baseline
- Market-only Logistic Regression
- Market-only Random Forest
- Market + Sentiment Logistic Regression
- Market + Sentiment Random Forest

Evaluation uses a time-based train/test split, not a random split.

Metrics:

- Accuracy
- Macro-F1
- classification report
- confusion matrix

Results are saved to:

```text
results/final_model_comparison.md
```

## Current Results Summary

Current class availability from `data/processed/market_features.csv`:

```text
1h:
Stable 180
Down    77
Up      69

2h:
Stable 157
Down    91
Up      80

24h:
Down   137
Up     127
Stable 20
```

All three classes are available for 1h, 2h, and 24h.

The 10-minute horizon is not available with the current data because the Polymarket price history is mostly hourly. The code supports the horizon structurally and skips it with a warning.

## Important Interpretation Note

X/Twitter data is used as predictive features, while the labels are derived from future Polymarket price movements. Therefore, collecting real X data improves the sentiment signal, but the class distribution of Up / Down / Stable depends on the amount, resolution and volatility of Polymarket price data.

At this stage, Market + Sentiment results validate the pipeline structure but should not yet be treated as final evidence that sentiment improves prediction, because the sentiment input is still sample data.

## Next Step

The next step is to buy or access a limited amount of real X/Twitter data, save it using the same schema as `data/raw/text_posts_sample.csv`, rerun the pipeline, and compare Market-only vs Market + Sentiment performance again.

## Real X/Twitter Data Collection

Real X/Twitter collection is implemented in:

```text
src/collect_x_data.py
```

The script uses the official X API Recent Search endpoint only. It does not use Full Archive Search and does not fetch user profile details.

Safety defaults:

```text
DRY_RUN = True
MAX_POSTS_TOTAL = 500
MAX_POSTS_PER_QUERY = 100
```

The output path is:

```text
data/raw/x_posts_real_limited.csv
```

This file is intentionally under `data/raw/`, which is ignored by Git.

### Set Bearer Token in PowerShell

Do not hardcode the token in Python files. Set it only in your current PowerShell session:

```powershell
$env:X_BEARER_TOKEN="YOUR_BEARER_TOKEN_HERE"
```

To confirm that the variable exists without printing the token:

```powershell
if ($env:X_BEARER_TOKEN) { "X_BEARER_TOKEN is set" } else { "Missing token" }
```

Never commit API keys, Bearer Tokens, screenshots of tokens, or `.env` files.

### Dry Run

Before buying credits or making API calls, run:

```bash
python src/collect_x_data.py
```

With `DRY_RUN=True`, the script prints the collection plan, queries, and limits, then stops before making any API calls.

### Real Collection

After reviewing the plan and setting a strict X spending limit:

1. Set `DRY_RUN = False` in `src/collect_x_data.py`.
2. Keep `MAX_POSTS_TOTAL` limited, for example 500.
3. Make sure X auto-recharge is off.
4. Run:

```bash
python src/collect_x_data.py
```

The collector deduplicates by `post_id`, stops when `MAX_POSTS_TOTAL` is reached, and saves posts to:

```text
data/raw/x_posts_real_limited.csv
```

After collection, either copy/rename this file into the sentiment input path or update `src/sentiment.py` to read the real X file for the final run.
