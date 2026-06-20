# Week 7 Technical Progress Plan

## Project Context

The project, `polymarket-sentiment-prediction`, predicts movement in Polymarket election-related markets using market data and, in the next stage, public text sentiment signals.

The current submission focuses on creating a working MVP around the existing market pipeline and adding the first sentiment layer. The sentiment layer is intentionally built first with sample data because X API access is paid. Real public X posts will be collected only after the pipeline is tested and the expected fields are stable.

## What Was Completed So Far

- Polymarket market data collection was implemented.
- Price history collection was implemented.
- Market feature engineering was implemented.
- Multiclass Up / Down / Stable is now the final target for the project.
- Baseline model results were generated.
- A sentiment MVP structure was added using a small sample text dataset.
- A rule-based sentiment pipeline was added.
- Hourly sentiment aggregation was added.
- A basic Streamlit dashboard was added for quick inspection of market and sentiment outputs.

## What Tamar Completed

Tamar completed the Polymarket side of the project:

- Polymarket API connection.
- Price history collection.
- Market feature engineering.
- Multiclass Up / Down / Stable modeling is now the final project direction.
- Baseline results.

These files should remain the stable market baseline and should not be rewritten unless a later integration step requires a small compatibility change.

## What Reut Completed

Reut completed the first sentiment, documentation, and dashboard layer:

- Created a fake/sample text dataset for testing only.
- Created a rule-based sentiment pipeline.
- Created hourly sentiment feature aggregation.
- Added technical documentation for the current submission.
- Updated the README with project usage instructions.
- Added a dashboard that can show available price, model, and sentiment outputs.

## What Currently Works

- The sample text dataset can be processed locally without using the paid X API.
- The sentiment script creates cleaned text, sentiment scores, and sentiment labels.
- The hourly feature script aggregates sentiment into market-friendly time windows.
- The dashboard is defensive: it checks whether files exist before loading them.
- The project now has a path toward comparing Market-only features with Market + Sentiment features.

## What Did Not Work

- Real X data collection was not added in this MVP stage because X API access is paid and should be used only after the local pipeline works.
- The current sentiment model is not a trained NLP model. It is a simple rule-based baseline for pipeline validation.
- The current combined dataset script does not yet merge sentiment features with market features. It only prepares hourly sentiment features for the future merge.

## Why We Changed From Up/Down/Stable to Move/Stable

The original multiclass target tried to predict three classes: Up, Down, and Stable. Initial results showed that this was difficult for the available dataset because short-term price direction is noisy and the class distribution can be imbalanced.

The current binary formulation, Move/Stable, is more suitable for the MVP because it asks a simpler first question: whether the market price is likely to move meaningfully or stay stable. This reduces target complexity and creates a stronger baseline before adding sentiment features.

## Current MVP Definition

The current MVP is:

1. Collect and process Polymarket price history.
2. Build market-based features.
3. Train baseline market-only models.
4. Create a sample text dataset that mimics the shape of future X data.
5. Run a simple sentiment pipeline on the sample text.
6. Aggregate sentiment features by hour.
7. Prepare for a future comparison between Market-only and Market + Sentiment models.

## Initial Results Summary

The baseline results currently come from Tamar's market-only binary models. These results are treated as the reference point.

The sentiment pipeline does not yet produce predictive model results. Its current role is to create usable sentiment features so that the next experiment can test whether adding public text sentiment improves the market-only baseline.

## Current Limitations

- The sentiment data is fake/sample data and is not real X data.
- The sentiment method is rule-based and may miss sarcasm, context, political slang, and Hebrew-English mixed phrasing.
- The hourly sentiment features are not yet merged with Polymarket market features.
- The dashboard depends on whichever output files exist locally.
- The X API budget and exact collection window still need to be finalized.

## What Changed From the Original Plan

- Real X collection was delayed until after the sentiment pipeline is validated locally.
- The first sentiment model is rule-based instead of a trained NLP model.
- The target definition was simplified from Up/Down/Stable to Move/Stable for the current MVP.
- The combined dataset step now starts with hourly sentiment aggregation before full market-sentiment merging.

## X API Plan

X API access was examined and found to be paid. For the current MVP, we first use a sample text dataset. In the next stage, we will collect a limited number of real public X posts under a controlled budget.

The planned real X data fields are:

- timestamp
- text
- candidate
- likes
- reposts
- comments
- post_id
- query

The initial goal is approximately 1,000 public posts related to Netanyahu, Bennett, Bibi, Naftali Bennett, Israeli election, בחירות בישראל, נתניהו, and בנט.

## Next Steps

- Run the sentiment pipeline on the sample dataset.
- Run hourly sentiment aggregation.
- Add a controlled X collection script after API access is confirmed.
- Replace the sample file with a real X dataset in the same schema.
- Merge hourly sentiment features with Polymarket market features.
- Compare Market-only model performance against Market + Sentiment model performance.

## Sentiment Layer Improvement

The sentiment layer was improved to better match Israeli political discourse. The updated pipeline now includes:

- Candidate and entity detection for Netanyahu/Bibi, Bennett, Lapid, Gantz, Ben Gvir, Smotrich, and a flexible dictionary for adding more figures.
- Party and bloc detection for Likud, Yesh Atid, National Unity, Yisrael Beiteinu, Religious Zionism, Otzma Yehudit, Shas, United Torah Judaism, Raam, and left/right/center discourse.
- Political topic detection for elections, leadership, government, protest, responsibility/blame, security, hostages, war, religion and state, judiciary, economy, coalition, corruption, Jewish-Arab relations, and left-right polarization.
- Phrase-based Israeli political sentiment using strong phrases such as "אתה הראש אתה אשם", "רק ביבי", "רק לא ביבי", "מדינת הלכה", "בחירות עכשיו", "כישלון", and "בגידה".
- An `intensity_score` that captures stronger political language and slogans.
- A `target_sentiment` column that estimates whether the sentiment is aimed at the main mentioned political figure.

Important: terms such as left/right, Arab/Jewish, religious/secular are treated as political topic or bloc indicators, not as positive or negative sentiment by themselves.

## Updated Next Step

The next step is to collect a limited number of real public X posts under a controlled paid API budget and apply this improved sentiment pipeline to real data. After that, the hourly sentiment features can be merged with Polymarket market features to compare Market-only vs Market + Sentiment models.

## Israeli Political Lexicon Expansion

The sentiment MVP now includes a broader Israeli political lexicon. The updated lexicon adds additional political figures such as Lieberman, Gantz, Lapid, Ben Gvir, Smotrich, Deri, and Eisenkot; additional parties such as Yisrael Beiteinu, Meretz, Labor, Raam, and Hadash-Taal; and additional ideological/context terms such as strong right, center-right, center-left, nationalist camp, democratic camp, liberal camp, settlements, settlers, residents, Judea and Samaria, West Bank, sovereignty, annexation, two-state solution, Palestinian state, terror, security, hostages, war, Gaza, Hamas, Iran, and Hezbollah.

These terms are used as contextual political features. They are not automatically treated as positive or negative sentiment. For example, terms such as ימין, שמאל, מתנחלים, מתיישבים, יהודה ושומרון, איו"ש, ערבים, חרדים, דתיים, חילונים, חמאס, and פלסטינים are topic/context indicators only.

Sentiment is affected only by words or phrases with a clearer evaluative meaning, such as מושחת, כישלון, מחדל, אלטרנטיבה, חזק, מתאים, לא מתאים, אשם, and הביתה. This keeps the MVP simple while avoiding a biased interpretation of political identity terms.

## Final Project Scope Update

The final project scope has changed from binary Move / Stable back to multiclass direction prediction. The final target is now:

```text
target_multiclass = Up / Down / Stable
```

The goal is to predict the direction of future price movement for Benjamin Netanyahu and Naftali Bennett on Polymarket.

For each candidate, timestamp and horizon:

```text
price_change = future_price - current_price
MOVEMENT_THRESHOLD = 0.001
```

Labels:

- `Up`: `price_change > MOVEMENT_THRESHOLD`
- `Down`: `price_change < -MOVEMENT_THRESHOLD`
- `Stable`: otherwise

The final horizons are 10 minutes, 1 hour, 2 hours, and 24 hours. If 10-minute Polymarket data is not available with the current price resolution, the code supports the horizon structurally but skips it gracefully with a warning.

The final model comparison is:

- Dummy Baseline
- Market-only Logistic Regression
- Market-only Random Forest
- Market + Sentiment Logistic Regression
- Market + Sentiment Random Forest

X/Twitter data is used as predictive features, while the labels are derived from future Polymarket price movements. Therefore, collecting real X data improves the sentiment signal, but the class distribution of Up / Down / Stable depends on the amount, resolution and volatility of Polymarket price data.

