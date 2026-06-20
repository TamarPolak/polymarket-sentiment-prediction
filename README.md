# Polymarket Sentiment Prediction

This project predicts short-term price movement in a Polymarket political prediction market using market data and sentiment signals from public political discourse.

## Research Question

Can sentiment and discussion volume from X/Twitter improve prediction of Benjamin Netanyahu's contract price movement on Polymarket?

## Project Goal

The goal is to predict the direction of future Polymarket candidate price movement using market features and sentiment features. The project starts with Polymarket price history and engineered market features, then adds public discussion signals related to Netanyahu, Bennett, Bibi, Naftali Bennett, Israeli election, בחירות בישראל, נתניהו, and בנט.

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

## Run the Legacy Binary Model

```bash
python src/train_binary_model.py
```

This is a legacy market-only reference model. The final project comparison uses the multiclass pipeline below.

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

The final project target is multiclass classification: Up / Down / Stable. Labels are derived from future Polymarket price movement for each candidate and horizon.

The sentiment pipeline currently validates the data flow only. It does not yet prove that sentiment improves prediction because the current sentiment input is sample data, not real X data.

## Next Step

The next experiment is to compare:

- Dummy Baseline.
- Market-only Logistic Regression.
- Market-only Random Forest.
- Market + Sentiment Logistic Regression.
- Market + Sentiment Random Forest.

Before collecting real X data, the sentiment pipeline should be validated on the sample file. After that, a limited number of real public X posts can be collected under a controlled paid API budget and saved using the same schema.

## Improved Sentiment Layer

The sentiment pipeline was expanded to better match Israeli political discourse. It now adds these columns to `data/processed/text_posts_with_sentiment.csv`:

```text
mentioned_entities
main_entity
mentioned_parties
political_bloc_terms
political_topics
matched_political_phrases
target_sentiment
intensity_score
```

The pipeline now supports:

- Entity detection for Netanyahu/Bibi, Bennett, Lapid, Gantz, Ben Gvir, Smotrich, and future candidates through editable dictionaries.
- Party and bloc detection for Israeli parties and political blocs.
- Topic detection for elections, leadership, government, protest, responsibility/blame, security, hostages, war, religion and state, judiciary, economy, coalition, corruption, Jewish-Arab relations, and left-right polarization.
- Strong Israeli political phrase detection, including phrases such as "אתה הראש אתה אשם", "רק ביבי", "רק לא ביבי", "מדינת הלכה", and "בחירות עכשיו".
- `intensity_score` for stronger political language.
- `target_sentiment` toward the main mentioned political figure.

Political identity terms such as left/right, Arab/Jewish, religious/secular are treated as topic or bloc indicators only. They are not positive or negative by themselves.

The next step is to collect real public X posts under a limited paid API budget, run this pipeline on real data, and then compare Market-only against Market + Sentiment model performance.

## Israeli Political Lexicon Expansion

The sentiment layer now includes a broader Israeli political lexicon for MVP testing. It detects additional entities, parties, blocs, ideological context terms, and political slogans, including Lieberman, Gantz, Lapid, Ben Gvir, Smotrich, Deri, Eisenkot, Yisrael Beiteinu, Meretz, Labor, Raam, Hadash-Taal, Judea and Samaria, איו"ש, settlers, settlements, sovereignty, annexation, hostages, Hamas, Iran, and Hezbollah.

Important: political identity and context terms are not automatically positive or negative. Terms like ימין, שמאל, מתנחלים, מתיישבים, ערבים, חרדים, דתיים, חילונים, חמאס, and פלסטינים are treated as context/topic indicators only.

Only clearly evaluative words and phrases affect sentiment, for example: מושחת, כישלון, מחדל, אלטרנטיבה, חזק, מתאים, לא מתאים, אשם, and הביתה.

## Final Modeling Scope

The final project uses only the multiclass target:

```text
target_multiclass = Up / Down / Stable
```

The goal is to predict the direction of future price movement for the leading Polymarket candidates:

- Benjamin Netanyahu
- Naftali Bennett

For each candidate, timestamp, and horizon:

```text
price_change = future_price - current_price
MOVEMENT_THRESHOLD = 0.001
```

Target labels:

- `Up`: `price_change > MOVEMENT_THRESHOLD`
- `Down`: `price_change < -MOVEMENT_THRESHOLD`
- `Stable`: otherwise

Prediction horizons:

- 10 minutes
- 1 hour
- 2 hours
- 24 hours

If 10-minute Polymarket data is not available with the current price resolution, the code keeps the horizon in the structure but skips it gracefully with a clear warning.

Final model comparison:

- Dummy Baseline
- Market-only Logistic Regression
- Market-only Random Forest
- Market + Sentiment Logistic Regression
- Market + Sentiment Random Forest

X/Twitter data is used as predictive features, while the labels are derived from future Polymarket price movements. Therefore, collecting real X data improves the sentiment signal, but the class distribution of Up / Down / Stable depends on the amount, resolution and volatility of Polymarket price data.

## Final Pipeline Commands

```bash
python src/sentiment.py
python src/build_combined_dataset.py
python src/build_market_features.py
python src/build_market_sentiment_dataset.py
python src/train_combined_model.py
```

Expected output files:

```text
data/processed/text_posts_with_sentiment.csv
data/processed/sentiment_features_by_hour.csv
data/processed/market_features.csv
data/final/market_sentiment_dataset.csv
results/final_model_comparison.md
```

## Run the Final Multiclass Pipeline

```bash
python src/sentiment.py
python src/build_combined_dataset.py
python src/build_market_features.py
python src/build_market_sentiment_dataset.py
python src/train_combined_model.py
```

This creates multiclass Up / Down / Stable labels from future Polymarket price movements and compares Market-only models with Market + Sentiment models.
