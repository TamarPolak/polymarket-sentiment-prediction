# Final Model Comparison - All Candidates

Target: `target_multiclass = Up / Down / Stable`.

Time split: first 75% of rows by timestamp for training, last 25% for testing.

Number of modeled candidates: 18

X/Twitter sentiment features are predictive features. Labels are derived from future Polymarket price movements.

## Horizon: 1h

Rows: 2970
Modeled candidates: 18

Class distribution:

| class | count |
|---|---:|
| Up | 190 |
| Down | 192 |
| Stable | 2588 |

Train rows: 2227
Test rows: 743

### Dummy Baseline

- Accuracy: 0.8950
- Macro-F1: 0.3149

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.00      0.00      0.00        38
        Down       0.00      0.00      0.00        40
      Stable       0.90      1.00      0.94       665

    accuracy                           0.90       743
   macro avg       0.30      0.33      0.31       743
weighted avg       0.80      0.90      0.85       743

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 0 | 0 | 38 |
| Down | 0 | 0 | 40 |
| Stable | 0 | 0 | 665 |

### Market-only Logistic Regression

- Accuracy: 0.7793
- Macro-F1: 0.5018

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.26      0.42      0.32        38
        Down       0.19      0.68      0.29        40
      Stable       1.00      0.81      0.89       665

    accuracy                           0.78       743
   macro avg       0.48      0.63      0.50       743
weighted avg       0.92      0.78      0.83       743

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 16 | 21 | 1 |
| Down | 13 | 27 | 0 |
| Stable | 33 | 96 | 536 |

### Market-only Random Forest

- Accuracy: 0.8318
- Macro-F1: 0.5381

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.22      0.50      0.30        38
        Down       0.32      0.50      0.39        40
      Stable       0.98      0.87      0.92       665

    accuracy                           0.83       743
   macro avg       0.50      0.62      0.54       743
weighted avg       0.90      0.83      0.86       743

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 19 | 12 | 7 |
| Down | 13 | 20 | 7 |
| Stable | 56 | 30 | 579 |

### Market + Sentiment Logistic Regression

- Accuracy: 0.7685
- Macro-F1: 0.4881

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.22      0.42      0.29        38
        Down       0.19      0.65      0.29        40
      Stable       1.00      0.80      0.88       665

    accuracy                           0.77       743
   macro avg       0.47      0.62      0.49       743
weighted avg       0.91      0.77      0.82       743

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 16 | 20 | 2 |
| Down | 14 | 26 | 0 |
| Stable | 42 | 94 | 529 |

### Market + Sentiment Random Forest

- Accuracy: 0.8331
- Macro-F1: 0.5399

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.21      0.47      0.30        38
        Down       0.33      0.53      0.40        40
      Stable       0.97      0.87      0.92       665

    accuracy                           0.83       743
   macro avg       0.51      0.62      0.54       743
weighted avg       0.90      0.83      0.86       743

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 18 | 12 | 8 |
| Down | 12 | 21 | 7 |
| Stable | 54 | 31 | 580 |

### Horizon Summary

- Best model by Macro-F1: Market + Sentiment Random Forest (0.5399)
- Best market-only Macro-F1: 0.5381
- Best market + sentiment Macro-F1: 0.5399
- Sentiment improved over market-only: yes (+0.0018)

## Horizon: 2h

Rows: 2952
Modeled candidates: 18

Class distribution:

| class | count |
|---|---:|
| Up | 244 |
| Down | 239 |
| Stable | 2469 |

Train rows: 2214
Test rows: 738

### Dummy Baseline

- Accuracy: 0.8659
- Macro-F1: 0.3094

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.00      0.00      0.00        50
        Down       0.00      0.00      0.00        49
      Stable       0.87      1.00      0.93       639

    accuracy                           0.87       738
   macro avg       0.29      0.33      0.31       738
weighted avg       0.75      0.87      0.80       738

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 0 | 0 | 50 |
| Down | 0 | 0 | 49 |
| Stable | 0 | 0 | 639 |

### Market-only Logistic Regression

- Accuracy: 0.7940
- Macro-F1: 0.5452

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.31      0.54      0.39        50
        Down       0.24      0.57      0.34        49
      Stable       1.00      0.83      0.91       639

    accuracy                           0.79       738
   macro avg       0.51      0.65      0.55       738
weighted avg       0.90      0.79      0.83       738

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 27 | 21 | 2 |
| Down | 21 | 28 | 0 |
| Stable | 39 | 69 | 531 |

### Market-only Random Forest

- Accuracy: 0.8279
- Macro-F1: 0.5712

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.26      0.66      0.37        50
        Down       0.42      0.41      0.41        49
      Stable       0.99      0.87      0.93       639

    accuracy                           0.83       738
   macro avg       0.56      0.65      0.57       738
weighted avg       0.90      0.83      0.86       738

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 33 | 12 | 5 |
| Down | 29 | 20 | 0 |
| Stable | 65 | 16 | 558 |

### Market + Sentiment Logistic Regression

- Accuracy: 0.7832
- Macro-F1: 0.5244

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.29      0.44      0.35        50
        Down       0.22      0.59      0.32        49
      Stable       0.99      0.82      0.90       639

    accuracy                           0.78       738
   macro avg       0.50      0.62      0.52       738
weighted avg       0.89      0.78      0.83       738

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 22 | 24 | 4 |
| Down | 20 | 29 | 0 |
| Stable | 33 | 79 | 527 |

### Market + Sentiment Random Forest

- Accuracy: 0.8252
- Macro-F1: 0.5634

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.26      0.64      0.37        50
        Down       0.38      0.41      0.40        49
      Stable       0.99      0.87      0.93       639

    accuracy                           0.83       738
   macro avg       0.54      0.64      0.56       738
weighted avg       0.90      0.83      0.85       738

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 32 | 15 | 3 |
| Down | 28 | 20 | 1 |
| Stable | 65 | 17 | 557 |

### Horizon Summary

- Best model by Macro-F1: Market-only Random Forest (0.5712)
- Best market-only Macro-F1: 0.5712
- Best market + sentiment Macro-F1: 0.5634
- Sentiment improved over market-only: no (-0.0079)

## Horizon: 24h

Rows: 2556
Modeled candidates: 18

Class distribution:

| class | count |
|---|---:|
| Up | 462 |
| Down | 359 |
| Stable | 1735 |

Train rows: 1917
Test rows: 639

### Dummy Baseline

- Accuracy: 0.7387
- Macro-F1: 0.2832

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.00      0.00      0.00        71
        Down       0.00      0.00      0.00        96
      Stable       0.74      1.00      0.85       472

    accuracy                           0.74       639
   macro avg       0.25      0.33      0.28       639
weighted avg       0.55      0.74      0.63       639

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 0 | 0 | 71 |
| Down | 0 | 0 | 96 |
| Stable | 0 | 0 | 472 |

### Market-only Logistic Regression

- Accuracy: 0.7027
- Macro-F1: 0.5361

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.37      0.23      0.28        71
        Down       0.33      0.83      0.47        96
      Stable       1.00      0.75      0.86       472

    accuracy                           0.70       639
   macro avg       0.57      0.60      0.54       639
weighted avg       0.83      0.70      0.73       639

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 16 | 55 | 0 |
| Down | 16 | 80 | 0 |
| Stable | 11 | 108 | 353 |

### Market-only Random Forest

- Accuracy: 0.8200
- Macro-F1: 0.6335

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.36      0.70      0.48        71
        Down       0.75      0.34      0.47        96
      Stable       0.96      0.93      0.95       472

    accuracy                           0.82       639
   macro avg       0.69      0.66      0.63       639
weighted avg       0.86      0.82      0.82       639

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 50 | 5 | 16 |
| Down | 62 | 33 | 1 |
| Stable | 25 | 6 | 441 |

### Market + Sentiment Logistic Regression

- Accuracy: 0.7042
- Macro-F1: 0.5466

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.41      0.25      0.31        71
        Down       0.33      0.83      0.47        96
      Stable       1.00      0.75      0.85       472

    accuracy                           0.70       639
   macro avg       0.58      0.61      0.55       639
weighted avg       0.83      0.70      0.74       639

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 18 | 52 | 1 |
| Down | 16 | 80 | 0 |
| Stable | 10 | 110 | 352 |

### Market + Sentiment Random Forest

- Accuracy: 0.8200
- Macro-F1: 0.6287

Classification report:

```text
              precision    recall  f1-score   support

          Up       0.36      0.68      0.47        71
        Down       0.72      0.34      0.46        96
      Stable       0.96      0.94      0.95       472

    accuracy                           0.82       639
   macro avg       0.68      0.65      0.63       639
weighted avg       0.86      0.82      0.82       639

```

Confusion matrix rows=true, columns=predicted:

| true\predicted | Up | Down | Stable |
|---|---:|---:|---:|
| Up | 48 | 7 | 16 |
| Down | 62 | 33 | 1 |
| Stable | 23 | 6 | 443 |

### Horizon Summary

- Best model by Macro-F1: Market-only Random Forest (0.6335)
- Best market-only Macro-F1: 0.6335
- Best market + sentiment Macro-F1: 0.6287
- Sentiment improved over market-only: no (-0.0049)
