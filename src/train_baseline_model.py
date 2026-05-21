"""
This file explains the baseline model structure.

We currently use two model scripts:

1. train_multiclass_model.py
   Predicts: Up / Down / Stable
   This was the first ML formulation, but it suffered from strong class imbalance.

2. train_binary_model.py
   Predicts: Move / Stable
   This is the current MVP baseline model because it performs better on the available data.

Recommended command for the current MVP:

    python src/train_binary_model.py
"""


def main():
    print("For the current MVP, run:")
    print("python src/train_binary_model.py")
    print()
    print("The binary model predicts whether Netanyahu's Polymarket price will Move or remain Stable.")


if __name__ == "__main__":
    main()