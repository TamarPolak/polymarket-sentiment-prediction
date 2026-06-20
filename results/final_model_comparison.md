# Final Model Comparison

Target: `target_multiclass = Up / Down / Stable`.

Time-based train/test split is used for every horizon. The first 75% of rows by timestamp are used for training and the final 25% are used for testing.

X/Twitter sentiment features are predictive features. Labels are derived from future Polymarket price movements.

## Horizon: 10m

NOTE: Horizon `10m` is not available in the final dataset. For 10m, this usually means the current Polymarket price resolution is not frequent enough, so it was skipped gracefully during market feature creation.

## Horizon: 1h

Class distribution:

| class | count |
|---|---:|
| Up | 69 |
| Down | 77 |
| Stable | 180 |

Train rows: 244
Test rows: 82

### Dummy Baseline

- Accuracy: 0.5000
- Macro-F1: 0.2222

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.00      0.00      0.00        23
        Down       0.00      0.00      0.00        18
      Stable       0.50      1.00      0.67        41

    accuracy                           0.50        82
   macro avg       0.17      0.33      0.22        82
weighted avg       0.25      0.50      0.33        82

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 0 | 0 | 23 |
| Down | 0 | 0 | 18 |
| Stable | 0 | 0 | 41 |

### Market-only Logistic Regression

- Accuracy: 0.4512
- Macro-F1: 0.3738

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.50      0.09      0.15        23
        Down       0.27      0.56      0.36        18
      Stable       0.61      0.61      0.61        41

    accuracy                           0.45        82
   macro avg       0.46      0.42      0.37        82
weighted avg       0.50      0.45      0.43        82

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 2 | 11 | 10 |
| Down | 2 | 10 | 6 |
| Stable | 0 | 16 | 25 |

### Market-only Random Forest

- Accuracy: 0.4878
- Macro-F1: 0.4595

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.54      0.30      0.39        23
        Down       0.31      0.56      0.40        18
      Stable       0.62      0.56      0.59        41

    accuracy                           0.49        82
   macro avg       0.49      0.47      0.46        82
weighted avg       0.53      0.49      0.49        82

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 7 | 8 | 8 |
| Down | 2 | 10 | 6 |
| Stable | 4 | 14 | 23 |

### Market + Sentiment Logistic Regression

- Accuracy: 0.4512
- Macro-F1: 0.3738

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.50      0.09      0.15        23
        Down       0.27      0.56      0.36        18
      Stable       0.61      0.61      0.61        41

    accuracy                           0.45        82
   macro avg       0.46      0.42      0.37        82
weighted avg       0.50      0.45      0.43        82

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 2 | 11 | 10 |
| Down | 2 | 10 | 6 |
| Stable | 0 | 16 | 25 |

### Market + Sentiment Random Forest

- Accuracy: 0.4878
- Macro-F1: 0.4606

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.58      0.30      0.40        23
        Down       0.30      0.56      0.39        18
      Stable       0.62      0.56      0.59        41

    accuracy                           0.49        82
   macro avg       0.50      0.47      0.46        82
weighted avg       0.54      0.49      0.49        82

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 7 | 8 | 8 |
| Down | 2 | 10 | 6 |
| Stable | 3 | 15 | 23 |

## Horizon: 24h

Class distribution:

| class | count |
|---|---:|
| Up | 127 |
| Down | 137 |
| Stable | 20 |

Train rows: 213
Test rows: 71

### Dummy Baseline

- Accuracy: 0.4366
- Macro-F1: 0.2026

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.00      0.00      0.00        33
        Down       0.44      1.00      0.61        31
      Stable       0.00      0.00      0.00         7

    accuracy                           0.44        71
   macro avg       0.15      0.33      0.20        71
weighted avg       0.19      0.44      0.27        71

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 0 | 33 | 0 |
| Down | 0 | 31 | 0 |
| Stable | 0 | 7 | 0 |

### Market-only Logistic Regression

- Accuracy: 0.2958
- Macro-F1: 0.2259

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.00      0.00      0.00        33
        Down       0.56      0.61      0.58        31
      Stable       0.06      0.29      0.09         7

    accuracy                           0.30        71
   macro avg       0.20      0.30      0.23        71
weighted avg       0.25      0.30      0.26        71

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 0 | 11 | 22 |
| Down | 0 | 19 | 12 |
| Stable | 1 | 4 | 2 |

### Market-only Random Forest

- Accuracy: 0.5070
- Macro-F1: 0.3496

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.58      0.88      0.70        33
        Down       0.78      0.23      0.35        31
      Stable       0.00      0.00      0.00         7

    accuracy                           0.51        71
   macro avg       0.45      0.37      0.35        71
weighted avg       0.61      0.51      0.48        71

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 29 | 0 | 4 |
| Down | 16 | 7 | 8 |
| Stable | 5 | 2 | 0 |

### Market + Sentiment Logistic Regression

- Accuracy: 0.2958
- Macro-F1: 0.2259

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.00      0.00      0.00        33
        Down       0.56      0.61      0.58        31
      Stable       0.06      0.29      0.09         7

    accuracy                           0.30        71
   macro avg       0.20      0.30      0.23        71
weighted avg       0.25      0.30      0.26        71

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 0 | 11 | 22 |
| Down | 0 | 19 | 12 |
| Stable | 1 | 4 | 2 |

### Market + Sentiment Random Forest

- Accuracy: 0.5211
- Macro-F1: 0.3659

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.59      0.88      0.71        33
        Down       0.80      0.26      0.39        31
      Stable       0.00      0.00      0.00         7

    accuracy                           0.52        71
   macro avg       0.46      0.38      0.37        71
weighted avg       0.62      0.52      0.50        71

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 29 | 0 | 4 |
| Down | 15 | 8 | 8 |
| Stable | 5 | 2 | 0 |

## Horizon: 2h

Class distribution:

| class | count |
|---|---:|
| Up | 80 |
| Down | 92 |
| Stable | 156 |

Train rows: 246
Test rows: 82

### Dummy Baseline

- Accuracy: 0.5000
- Macro-F1: 0.2222

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.00      0.00      0.00        23
        Down       0.00      0.00      0.00        18
      Stable       0.50      1.00      0.67        41

    accuracy                           0.50        82
   macro avg       0.17      0.33      0.22        82
weighted avg       0.25      0.50      0.33        82

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 0 | 0 | 23 |
| Down | 0 | 0 | 18 |
| Stable | 0 | 0 | 41 |

### Market-only Logistic Regression

- Accuracy: 0.5122
- Macro-F1: 0.4878

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.44      0.35      0.39        23
        Down       0.43      0.56      0.49        18
      Stable       0.59      0.59      0.59        41

    accuracy                           0.51        82
   macro avg       0.49      0.50      0.49        82
weighted avg       0.51      0.51      0.51        82

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 8 | 4 | 11 |
| Down | 2 | 10 | 6 |
| Stable | 8 | 9 | 24 |

### Market-only Random Forest

- Accuracy: 0.4878
- Macro-F1: 0.4689

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.35      0.52      0.42        23
        Down       0.60      0.33      0.43        18
      Stable       0.58      0.54      0.56        41

    accuracy                           0.49        82
   macro avg       0.51      0.46      0.47        82
weighted avg       0.52      0.49      0.49        82

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 12 | 1 | 10 |
| Down | 6 | 6 | 6 |
| Stable | 16 | 3 | 22 |

### Market + Sentiment Logistic Regression

- Accuracy: 0.5122
- Macro-F1: 0.4878

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.44      0.35      0.39        23
        Down       0.43      0.56      0.49        18
      Stable       0.59      0.59      0.59        41

    accuracy                           0.51        82
   macro avg       0.49      0.50      0.49        82
weighted avg       0.51      0.51      0.51        82

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 8 | 4 | 11 |
| Down | 2 | 10 | 6 |
| Stable | 8 | 9 | 24 |

### Market + Sentiment Random Forest

- Accuracy: 0.4756
- Macro-F1: 0.4451

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.35      0.52      0.42        23
        Down       0.50      0.28      0.36        18
      Stable       0.58      0.54      0.56        41

    accuracy                           0.48        82
   macro avg       0.48      0.45      0.45        82
weighted avg       0.50      0.48      0.47        82

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 12 | 1 | 10 |
| Down | 7 | 5 | 6 |
| Stable | 15 | 4 | 22 |
