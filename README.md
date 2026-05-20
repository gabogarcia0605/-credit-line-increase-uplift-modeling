# Credit Line Increase (CLI) Optimization using Uplift Modeling (X-Learner)

This repository features an end-to-end Machine Learning and Causal Inference solution designed to optimize Credit Line Increase (CLI) strategies in the Fintech/Banking sector. By leveraging **Uplift Modeling**, the model identifies "Persuadables"—customers who will respond positively to the offer—while avoiding cost inefficiencies from organic spend or triggering negative behaviors ("Sleeping Dogs").

## 🚀 Business Problem
Standard propensity models predict who is likely to accept a credit line increase. However, they fail to distinguish between organic behavior (users who would convert anyway) and the incremental effect caused by marketing campaigns or target triggers. This project aims to maximize profitability and risk allocation by predicting the **Individual Treatment Effect (ITE)**.

## 🛠️ Methodology & Technical Stack
- **Feature Engineering & Store:** Built user-level aggregated metrics from behavioral, transaction, and communication channels (WhatsApp engagement trends 7d vs 30d, recent credit limit utilization, ticket growth trends).
- **Two-Stage Causal Inference (X-Learner):** Implemented an X-Learner architecture to handle severe class imbalance and treatment-control disparities:
  - **Stage 1:** Propensity scores $e(X)$ using a Scaled Logistic Regression and outcome models $\mu_1(X), \mu_0(X)$ using Stratified 5-Fold Cross-Fitting via `XGBClassifier` to generate Out-of-Fold (OOF) counterfactuals.
  - **Stage 2:** Implemented pseudo-outcome effect mapping $\tau_1(X), \tau_0(X)$ trained via `XGBRegressor` to calculate conditional average treatment effects (CATE).
- **Evaluation Metrics:** Evaluated using Uplift Curves and Qini Curves, achieving an **AUQC (Area Under Qini Curve)** of ~56.43 on the test set.

## 📁 Repository Structure
- `x_learner_synthetic_data.ipynb`: Python notebook containing full pipeline from feature extraction to model evaluation.
- `generate_mock_data.ipynb`: Python script/notebook to generate production-safe synthetic datasets.
- `requirements.txt`: Environment dependencies required to reproduce the execution pipeline.

## 📊 Evaluation & Production Alignment
The model exhibits strong ranking and separation capabilities across uplift deciles, effectively isolating positive responders from groups with negative incremental lift (Sleeping Dogs). 

*Note: Datasets used during production tracking have been completely removed from this repository to comply with privacy and confidentiality requirements. Data structures shown in the notebooks utilize fully randomized synthetic schemas (UUID generation).*