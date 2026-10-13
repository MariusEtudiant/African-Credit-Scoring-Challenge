# Loan default prediction: African Credit Scoring Challenge (Zindi)

Predict which loans will default, for lenders in African markets. The training data covers
68,654 loans in Kenya, of which only 1.8% default; the competition test set also contains
loans from Ghana, a country absent from training.

## Results

Stratified validation split (13,731 loans, 252 defaults):

| | Precision | Recall | F1 |
|---|---:|---:|---:|
| Defaults (LightGBM, best model) | 0.91 | 0.87 | **0.89** |

An early submission scored **0.71** on the public leaderboard. The gap with validation is the
main lesson of the project: a model validated on Kenyan loans only does not transfer as well to
another country. Validation on held-out countries, or features that travel across markets
(macroeconomic indicators rather than country-specific ones), is where the remaining work lies.

## Approach

- **Features:** repayment ratio, log of the loan amount, and macroeconomic indicators of each
  country joined to the loans (interest rates and spread, inflation, exchange rate,
  unemployment) and interactions between them, selected with mutual information.
- **Imbalance:** six options compared: no resampling, SMOTE, BorderlineSMOTE, ADASYN,
  SMOTEENN and SMOTETomek.
- **Model search:** one Optuna study over the resampling method, the algorithm (XGBoost,
  LightGBM, CatBoost) and their hyperparameters, maximising the F1 score on defaults.
- **Finding:** the best configuration used **no resampling**. LightGBM handled the 1.8% minority
  class better on its own than with synthetic samples: the best resampled trial (SMOTE) reached
  an F1 of 0.87, against 0.89 without.

## Run

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost imbalanced-learn optuna matplotlib seaborn
jupyter notebook credit_scoring.ipynb
```

Download `Train.csv`, `Test.csv` and `economic_indicators.csv` from the
[competition page](https://zindi.africa/competitions/african-credit-scoring-challenge) and put
them next to the notebook. The competition data is not redistributed here.
