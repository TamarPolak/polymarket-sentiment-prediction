# Final Model Comparison

Target: `target_multiclass = Up / Down / Stable`.

Time-based train/test split is used for every horizon. The first 75% of rows by timestamp are used for training and the final 25% are used for testing.

X/Twitter sentiment features are predictive features. Labels are derived from future Polymarket price movements.

## Horizon: 1h

Class distribution:

| class | count |
|---|---:|
| Up | 70 |
| Down | 76 |
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

- Accuracy: 0.4634
- Macro-F1: 0.3981

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.60      0.13      0.21        23
        Down       0.28      0.56      0.37        18
      Stable       0.61      0.61      0.61        41

    accuracy                           0.46        82
   macro avg       0.50      0.43      0.40        82
weighted avg       0.53      0.46      0.45        82

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 3 | 10 | 10 |
| Down | 2 | 10 | 6 |
| Stable | 0 | 16 | 25 |

### Market-only Random Forest

- Accuracy: 0.4756
- Macro-F1: 0.4416

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.50      0.26      0.34        23
        Down       0.30      0.56      0.39        18
      Stable       0.62      0.56      0.59        41

    accuracy                           0.48        82
   macro avg       0.47      0.46      0.44        82
weighted avg       0.52      0.48      0.48        82

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 6 | 9 | 8 |
| Down | 2 | 10 | 6 |
| Stable | 4 | 14 | 23 |

### Market + Sentiment Logistic Regression

- Accuracy: 0.4634
- Macro-F1: 0.3981

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.60      0.13      0.21        23
        Down       0.28      0.56      0.37        18
      Stable       0.61      0.61      0.61        41

    accuracy                           0.46        82
   macro avg       0.50      0.43      0.40        82
weighted avg       0.53      0.46      0.45        82

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 3 | 10 | 10 |
| Down | 2 | 10 | 6 |
| Stable | 0 | 16 | 25 |

### Market + Sentiment Random Forest

- Accuracy: 0.4878
- Macro-F1: 0.4502

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.50      0.26      0.34        23
        Down       0.31      0.56      0.40        18
      Stable       0.63      0.59      0.61        41

    accuracy                           0.49        82
   macro avg       0.48      0.47      0.45        82
weighted avg       0.52      0.49      0.49        82

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 6 | 9 | 8 |
| Down | 2 | 10 | 6 |
| Stable | 4 | 13 | 24 |

## Horizon: 24h

Class distribution:

| class | count |
|---|---:|
| Up | 128 |
| Down | 135 |
| Stable | 21 |

Train rows: 213
Test rows: 71

### Dummy Baseline

- Accuracy: 0.4366
- Macro-F1: 0.2026

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.00      0.00      0.00        32
        Down       0.44      1.00      0.61        31
      Stable       0.00      0.00      0.00         8

    accuracy                           0.44        71
   macro avg       0.15      0.33      0.20        71
weighted avg       0.19      0.44      0.27        71

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 0 | 32 | 0 |
| Down | 0 | 31 | 0 |
| Stable | 0 | 8 | 0 |

### Market-only Logistic Regression

- Accuracy: 0.3239
- Macro-F1: 0.2610

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.50      0.03      0.06        32
        Down       0.56      0.61      0.58        31
      Stable       0.09      0.38      0.14         8

    accuracy                           0.32        71
   macro avg       0.38      0.34      0.26        71
weighted avg       0.48      0.32      0.30        71

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 1 | 11 | 20 |
| Down | 0 | 19 | 12 |
| Stable | 1 | 4 | 3 |

### Market-only Random Forest

- Accuracy: 0.5070
- Macro-F1: 0.3496

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.57      0.91      0.70        32
        Down       0.78      0.23      0.35        31
      Stable       0.00      0.00      0.00         8

    accuracy                           0.51        71
   macro avg       0.45      0.38      0.35        71
weighted avg       0.60      0.51      0.47        71

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 29 | 0 | 3 |
| Down | 16 | 7 | 8 |
| Stable | 6 | 2 | 0 |

### Market + Sentiment Logistic Regression

- Accuracy: 0.3239
- Macro-F1: 0.2610

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.50      0.03      0.06        32
        Down       0.56      0.61      0.58        31
      Stable       0.09      0.38      0.14         8

    accuracy                           0.32        71
   macro avg       0.38      0.34      0.26        71
weighted avg       0.48      0.32      0.30        71

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 1 | 11 | 20 |
| Down | 0 | 19 | 12 |
| Stable | 1 | 4 | 3 |

### Market + Sentiment Random Forest

- Accuracy: 0.4789
- Macro-F1: 0.3152

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.55      0.91      0.68        32
        Down       0.71      0.16      0.26        31
      Stable       0.00      0.00      0.00         8

    accuracy                           0.48        71
   macro avg       0.42      0.36      0.32        71
weighted avg       0.56      0.48      0.42        71

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 29 | 0 | 3 |
| Down | 18 | 5 | 8 |
| Stable | 6 | 2 | 0 |

## Horizon: 2h

Class distribution:

| class | count |
|---|---:|
| Up | 81 |
| Down | 90 |
| Stable | 157 |

Train rows: 246
Test rows: 82

### Dummy Baseline

- Accuracy: 0.5244
- Macro-F1: 0.2293

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.00      0.00      0.00        23
        Down       0.00      0.00      0.00        16
      Stable       0.52      1.00      0.69        43

    accuracy                           0.52        82
   macro avg       0.17      0.33      0.23        82
weighted avg       0.27      0.52      0.36        82

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 0 | 0 | 23 |
| Down | 0 | 0 | 16 |
| Stable | 0 | 0 | 43 |

### Market-only Logistic Regression

- Accuracy: 0.5000
- Macro-F1: 0.4726

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.70      0.30      0.42        23
        Down       0.30      0.62      0.41        16
      Stable       0.62      0.56      0.59        43

    accuracy                           0.50        82
   macro avg       0.54      0.50      0.47        82
weighted avg       0.58      0.50      0.51        82

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 7 | 7 | 9 |
| Down | 0 | 10 | 6 |
| Stable | 3 | 16 | 24 |

### Market-only Random Forest

- Accuracy: 0.5000
- Macro-F1: 0.4675

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.39      0.57      0.46        23
        Down       0.45      0.31      0.37        16
      Stable       0.61      0.53      0.57        43

    accuracy                           0.50        82
   macro avg       0.48      0.47      0.47        82
weighted avg       0.52      0.50      0.50        82

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 13 | 1 | 9 |
| Down | 5 | 5 | 6 |
| Stable | 15 | 5 | 23 |

### Market + Sentiment Logistic Regression

- Accuracy: 0.5000
- Macro-F1: 0.4726

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.70      0.30      0.42        23
        Down       0.30      0.62      0.41        16
      Stable       0.62      0.56      0.59        43

    accuracy                           0.50        82
   macro avg       0.54      0.50      0.47        82
weighted avg       0.58      0.50      0.51        82

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 7 | 7 | 9 |
| Down | 0 | 10 | 6 |
| Stable | 3 | 16 | 24 |

### Market + Sentiment Random Forest

- Accuracy: 0.5000
- Macro-F1: 0.4644

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.39      0.52      0.44        23
        Down       0.45      0.31      0.37        16
      Stable       0.60      0.56      0.58        43

    accuracy                           0.50        82
   macro avg       0.48      0.46      0.46        82
weighted avg       0.51      0.50      0.50        82

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 12 | 1 | 10 |
| Down | 5 | 5 | 6 |
| Stable | 14 | 5 | 24 |
