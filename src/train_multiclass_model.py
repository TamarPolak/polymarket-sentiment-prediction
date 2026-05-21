"""
Multiclass model: Up / Down / Stable.

This script will contain the original three-class classification experiment.

Initial findings:
- The dataset was highly imbalanced.
- Stable was the dominant class.
- The Up class had too few samples.
- Logistic Regression and Random Forest did not predict the Up class.
- Therefore, the MVP currently uses train_binary_model.py.
"""


def main():
    print("Multiclass experiment: Up / Down / Stable")
    print("Current MVP uses binary classification instead: Move / Stable")
    print("Run: python src/train_binary_model.py")


if __name__ == "__main__":
    main()