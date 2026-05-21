# Baseline Model Results

## Task

The MVP currently uses a binary classification task:

- Move
- Stable

The goal is to predict whether Benjamin Netanyahu's Polymarket contract price will move or remain stable in the next hour.

## Dataset

Dataset size: 164

Target distribution:

| Class | Count | Percentage |
|---|---:|---:|
| Stable | 139 | 84.76% |
| Move | 25 | 15.24% |

## Models Compared

### Dummy Baseline

The Dummy model always predicts the most frequent class.

- Accuracy: 0.606
- Macro-F1: 0.377

### Logistic Regression

- Accuracy: 0.667
- Macro-F1: 0.665

Confusion matrix:

```text
[[10  3]
 [ 8 12]]