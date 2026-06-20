# Final Report: Polymarket Sentiment Prediction

## Project Title

Predicting Polymarket price movement using market data and X/Twitter sentiment.

Case study: Who will be the next Prime Minister of Israel after the next election?

## Research Question

Can sentiment and discussion volume from X/Twitter improve prediction of short-term Polymarket price movement for the leading candidates in the Israeli Prime Minister market?

The project does not try to directly predict who will become the next Prime Minister of Israel. Instead, it predicts how Polymarket prices move for candidate YES contracts. The main candidates in the current project are:

- Benjamin Netanyahu
- Naftali Bennett

## Motivation

Prediction markets such as Polymarket aggregate public expectations about future events. In political markets, prices may react to polls, speeches, news events, public discussion, and viral social media posts. X/Twitter is especially relevant because it contains fast-moving political discourse.

The core idea is to test whether public political sentiment can add predictive value beyond market-only price features.

## Data Sources

### Polymarket Data

The Polymarket side includes price history for the leading candidates. The current raw file is:

```text
data/raw/polymarket_price_history.csv
```

The current available data includes price observations for:

- Benjamin Netanyahu
- Naftali Bennett

The current price history is mostly hourly. Because of that, the 10-minute horizon is supported in the code structure but skipped gracefully when the data resolution is not frequent enough.

### X/Twitter Sentiment Data

X/Twitter API access is paid. For the current MVP, the sentiment pipeline is tested with fake/sample text data:

```text
data/raw/text_posts_sample.csv
```

These are not real X posts. They are used only to validate the sentiment pipeline before paying for real X data.

The next stage is to collect a limited number of real public X posts under a controlled budget and save them using the same schema.

## Target Definition

The final project target is multiclass classification:

```text
target_multiclass = Up / Down / Stable
```

For each candidate, timestamp, and prediction horizon:

```text
price_change = future_price - current_price
MOVEMENT_THRESHOLD = 0.001
```

Labels:

- `Up`: `price_change > MOVEMENT_THRESHOLD`
- `Down`: `price_change < -MOVEMENT_THRESHOLD`
- `Stable`: otherwise

Prediction horizons:

- 10 minutes
- 1 hour
- 2 hours
- 24 hours

If 10-minute Polymarket data is not available with the current price resolution, the pipeline skips it with a clear warning.

## Feature Engineering

### Market Features

Market features are created in:

```text
src/build_market_features.py
```

Output:

```text
data/processed/market_features.csv
```

Features include:

- candidate price
- opponent price
- price gap
- 1-hour candidate price change
- 1-hour opponent price change
- 1-hour gap change
- 2-hour candidate price change
- 2-hour opponent price change
- absolute price gap
- future price
- price change
- target_multiclass

### Sentiment Features

Sentiment features are created in two steps:

```text
src/sentiment.py
src/build_combined_dataset.py
```

Outputs:

```text
data/processed/text_posts_with_sentiment.csv
data/processed/sentiment_features_by_hour.csv
```

The sentiment layer includes:

- text cleaning
- rule-based sentiment score
- sentiment label: positive / negative / neutral
- political entity detection
- party and bloc detection
- political topic detection
- strong Israeli political phrase detection
- target sentiment toward the main political entity
- intensity score

The political lexicon includes Israeli political figures, parties, blocs, ideological terms, slogans, and election-related topics. Context terms such as right/left, settlers, religious/secular, Arabs, Hamas, Palestinians, and Judea and Samaria are treated as political context indicators, not automatically as positive or negative sentiment.

### Market + Sentiment Dataset

The combined dataset is built in:

```text
src/build_market_sentiment_dataset.py
```

Output:

```text
data/final/market_sentiment_dataset.csv
```

To avoid data leakage, sentiment features are merged only from the same timestamp or an earlier timestamp. Future sentiment posts are not used to predict future price movement.

If sentiment features are missing for a timestamp, numeric sentiment features are filled with 0.

## Models

The final comparison is implemented in:

```text
src/train_combined_model.py
```

Output:

```text
results/final_model_comparison.md
```

Models compared:

- Dummy Baseline
- Market-only Logistic Regression
- Market-only Random Forest
- Market + Sentiment Logistic Regression
- Market + Sentiment Random Forest

The models are trained and evaluated separately for each horizon using a time-based train/test split. The first 75% of rows by timestamp are used for training and the final 25% are used for testing.

Metrics:

- Accuracy
- Macro-F1
- classification report
- confusion matrix

## Current Results Summary

With the current Polymarket data, valid multiclass datasets were created for:

- 1 hour
- 2 hours
- 24 hours

The 10-minute horizon was skipped because the current price history is mostly hourly.

Class availability:

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

All three classes are available for 1h, 2h, and 24h. The 24h horizon is less balanced because Stable has fewer examples.

The full model comparison is saved in:

```text
results/final_model_comparison.md
```

At this stage, the Market + Sentiment results should be interpreted as pipeline validation, not as a final proof of sentiment value, because the sentiment data is still sample data rather than real X/Twitter data.

## Important Interpretation Note

X/Twitter data is used as predictive features, while the labels are derived from future Polymarket price movements. Therefore, collecting real X data improves the sentiment signal, but the class distribution of Up / Down / Stable depends on the amount, resolution and volatility of Polymarket price data.

This means that even a large amount of real X data will not automatically create balanced classes. Class balance depends mainly on how much the Polymarket prices actually move within each horizon.

## Dashboard

The dashboard is implemented in:

```text
dashboard/app.py
```

Run:

```bash
streamlit run dashboard/app.py
```

The dashboard shows:

- Netanyahu price over time
- Bennett price over time
- price gap over time
- multiclass class distribution by horizon
- final model comparison metrics
- sentiment summary
- sample sentiment rows

The dashboard checks whether files exist before loading them, so it should not crash if some outputs are missing.

## How to Run the Current Pipeline

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

Optional dashboard:

```bash
streamlit run dashboard/app.py
```

## Current Limitations

- Sentiment data is currently fake/sample data, not real X/Twitter data.
- The sentiment pipeline is rule-based, not a trained Hebrew political NLP model.
- The current Polymarket data is mostly hourly, so 10-minute prediction is not available yet.
- The 24-hour target is less balanced than 1h and 2h.
- More historical Polymarket data is needed for stronger model evaluation.
- Real X/Twitter data is needed before drawing conclusions about whether sentiment improves prediction.

## Next Steps

1. Collect real public X/Twitter posts under a limited paid API budget.
2. Save real posts using the same schema as the sample file.
3. Rerun the sentiment pipeline on real data.
4. Rerun the full Market + Sentiment pipeline.
5. Compare whether real sentiment features improve Macro-F1 over market-only models.
6. Improve the dashboard and prepare final presentation slides.
