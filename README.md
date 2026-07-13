# Polymarket Sentiment Prediction
🔗 **דף הנחיתה של הפרויקט:** https://tamarpolak.github.io/polymarket-sentiment-prediction/ 

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

Current implemented prediction horizons:

- 1 hour
- 2 hours
- 24 hours

The 10-minute horizon was part of the original research scope, but it was not included in the current experiments because the available Polymarket price data is hourly. The current experiments therefore focus on 1h, 2h and 24h horizons.

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

The 10-minute horizon was part of the original research scope, but it was not included in the current experiments because the available Polymarket price data is hourly. The current experiments therefore focus on 1h, 2h and 24h horizons.

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
TEST_ONE_QUERY_ONLY = True
TEST_QUERY_GROUP_NAME = "netanyahu_english"
TEST_MAX_POSTS_TOTAL = 20
TEST_MAX_POSTS_PER_QUERY = 20
MAX_POSTS_TOTAL = 500
MAX_POSTS_PER_QUERY = 100
```

When `TEST_ONE_QUERY_ONLY=True`, the script uses the smaller test limits. If `TEST_QUERY_GROUP_NAME` is set, only that query group is tested. If `TEST_QUERY_GROUP_NAME = None`, the script tries query groups in order and stops only after a query returns at least one post.

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

### 20-Post Real Test

After reviewing the dry run, setting a strict X spending limit, and turning auto-recharge off:

1. Keep:

```python
TEST_ONE_QUERY_ONLY = True
TEST_QUERY_GROUP_NAME = "netanyahu_english"
TEST_MAX_POSTS_TOTAL = 20
TEST_MAX_POSTS_PER_QUERY = 20
```

2. Set:

```python
DRY_RUN = False
```

3. Run:

```bash
python src/collect_x_data.py
```

The script will print:

```text
REAL API COLLECTION MODE — this may use X credits.
```

With the current test settings, the estimated maximum cost is 20 posts * $0.005 = $0.10.

### Full 500-Post Collection

After the 20-post test succeeds:

1. Set:

```python
DRY_RUN = False
TEST_ONE_QUERY_ONLY = False
TEST_QUERY_GROUP_NAME = None
MAX_POSTS_TOTAL = 500
MAX_POSTS_PER_QUERY = 100
```

2. Keep X auto-recharge off and keep the spending limit low.
3. Run:

```bash
python src/collect_x_data.py
```

The collector deduplicates by `post_id`, skips failed query groups instead of crashing the whole run, stops when `MAX_POSTS_TOTAL` is reached, and saves posts to:

```text
data/raw/x_posts_real_limited.csv
```

If Hebrew queries fail again, set this constant in `src/collect_x_data.py`:

```python
USE_HEBREW_LANGUAGE_FILTER = False
```

After collection, either copy/rename this file into the sentiment input path or update `src/sentiment.py` to read the real X file for the final run.




## Continue X Collection to 1000 Posts

The X collector supports append/resume mode and will not overwrite the existing raw X file when `APPEND_EXISTING=True`.

Current output file:

```text
data/raw/x_posts_real_limited.csv
```

This file is ignored by Git and should not be committed.

### Hebrew Test Collection

Default safe test settings:

```python
DRY_RUN = True
APPEND_EXISTING = True
TEST_ONE_QUERY_ONLY = True
TEST_QUERY_GROUP_NAME = "netanyahu_hebrew_broad"
TEST_MAX_POSTS_TOTAL = 30
TEST_MAX_POSTS_PER_QUERY = 30
USE_HEBREW_LANGUAGE_FILTER = False
```

Run dry run first:

```bash
python src/collect_x_data.py
```

To run a small real Hebrew test, set:

```python
DRY_RUN = False
```

Then run:

```bash
python src/collect_x_data.py
```

The script loads existing posts first, deduplicates by `post_id`, and appends only new unique posts.

### Append Mode and Target 1000

For full collection until 1000 total unique posts, use:

```python
DRY_RUN = False
TEST_ONE_QUERY_ONLY = False
TEST_QUERY_GROUP_NAME = None
APPEND_EXISTING = True
TARGET_TOTAL_UNIQUE_POSTS = 1000
MAX_POSTS_TOTAL = 1000
MAX_POSTS_PER_QUERY = 200
USE_HEBREW_LANGUAGE_FILTER = False
```

The script will:

- load existing `data/raw/x_posts_real_limited.csv`
- count existing unique posts
- estimate how many additional posts are needed
- collect Hebrew broad queries first
- keep existing rows
- deduplicate by `post_id`
- stop once total unique posts reaches 1000

Keep X auto-recharge off and keep a strict spending limit. After collection, set:

```python
DRY_RUN = True
```

This helps prevent accidental extra API calls.
