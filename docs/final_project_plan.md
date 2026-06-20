# Final Project Plan

## Final Research Target

The final project implements only the multiclass target:

```text
target_multiclass = Up / Down / Stable
```

The goal is to predict the direction of future Polymarket price movement for the leading candidates:

- Benjamin Netanyahu
- Naftali Bennett

For each candidate, timestamp, and horizon:

```text
price_change = future_price - current_price
MOVEMENT_THRESHOLD = 0.001
```

Target definition:

- `Up`: `price_change > MOVEMENT_THRESHOLD`
- `Down`: `price_change < -MOVEMENT_THRESHOLD`
- `Stable`: otherwise

## Prediction Horizons

The current implemented experiments support:

- 1 hour
- 2 hours
- 24 hours

The 10-minute horizon was part of the original research scope, but it was not included in the current experiments because the available Polymarket price data is hourly. The current experiments therefore focus on 1h, 2h and 24h horizons.

## Feature Sources

Market features are derived from Polymarket price history, including candidate price, opponent price, price gap, recent price changes, future price, price change, and the multiclass target.

Sentiment features are derived from the text sentiment pipeline and aggregated by hour. The merge uses only sentiment windows from the current or previous timestamp to avoid data leakage.

X/Twitter data is used as predictive features, while the labels are derived from future Polymarket price movements. Therefore, collecting real X data improves the sentiment signal, but the class distribution of Up / Down / Stable depends on the amount, resolution and volatility of Polymarket price data.

## Final Model Comparison

The final comparison focuses on:

- Dummy Baseline
- Market-only Logistic Regression
- Market-only Random Forest
- Market + Sentiment Logistic Regression
- Market + Sentiment Random Forest

Models are evaluated separately for each horizon using a time-based train/test split.

Metrics:

- Accuracy
- Macro-F1
- Classification report
- Confusion matrix

If one of the classes Up / Down / Stable is missing for a horizon, training for that horizon is skipped and the results file includes a clear note.

## Commands

```bash
python src/sentiment.py
python src/build_combined_dataset.py
python src/build_market_features.py
python src/build_market_sentiment_dataset.py
python src/train_combined_model.py
```

## Outputs

```text
data/processed/market_features.csv
data/final/market_sentiment_dataset.csv
results/final_model_comparison.md
```

