# Week 7 Technical Progress Plan

## Project Context

The project, `polymarket-sentiment-prediction`, predicts movement in Polymarket election-related markets using market data and, in the next stage, public text sentiment signals.

The current submission focuses on creating a working MVP around the existing market pipeline and adding the first sentiment layer. The sentiment layer is intentionally built first with sample data because X API access is paid. Real public X posts will be collected only after the pipeline is tested and the expected fields are stable.

## What Was Completed So Far

- Polymarket market data collection was implemented.
- Price history collection was implemented.
- Market feature engineering was implemented.
- Binary and multiclass baseline ML models were implemented.
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
- Binary and multiclass ML baseline models.
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

The baseline results currently come from Tamar's market-only models. These results are treated as the reference point.

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
