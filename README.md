Credit Risk Modeling: Loan Default Prediction and Strategic Lending Simulation

Project Overview
This project focuses on building a machine learning pipeline to predict the likelihood of loan default using real-world credit data. The goal is not only to classify applicants as high or low risk, but also to support data-driven decisions around loan approvals using expected loss calculations and a strategy table simulation.

Two classification models were developed:

Logistic Regression (baseline)

XGBoost Classifier (for better performance and handling of non-linearity)

We go beyond model accuracy by analyzing:

Default risk across different applicant segments

Business implications of adjusting loan approval thresholds

Estimation of portfolio value under various decision strategies

Dataset Features
Column	Description
person_age	Applicant’s age
person_income	Annual income of the applicant
person_home_ownership	Home ownership type (RENT, OWN, MORTGAGE)
person_emp_length	Years of employment
loan_intent	Purpose of the loan (e.g. education, medical)
loan_grade	Credit grade assigned
loan_amnt	Amount of loan requested
loan_int_rate	Interest rate of the loan
loan_status	Target variable (0 = repaid, 1 = defaulted)
cb_person_default_on_file	Prior credit default flag
cb_person_cred_hist_length	Length of credit history in years

Key Concepts Used
Probability of Default (PD): Likelihood of a borrower defaulting on the loan

Exposure at Default (EAD): Amount at risk, i.e., the loan amount

Loss Given Default (LGD): Percentage of loan likely to be lost if default occurs

Expected Loss (EL) is calculated as:

java
Copy
Edit
Expected Loss = PD × EAD × LGD
Modeling Pipeline
Data Preprocessing

Missing value imputation

One-hot encoding of categorical variables

Feature engineering (e.g., loan-to-income ratio)

Exploratory Data Analysis

Crosstabs and visualizations of default rate by intent, grade, home ownership

Identification of high-risk borrower profiles

Model Training

Logistic Regression as interpretable baseline

XGBoost to capture complex patterns and improve ROC AUC

Model Evaluation

Confusion matrix

Classification report

ROC AUC score and ROC curve plotting

Strategy Table Simulation
To evaluate how different decision thresholds affect portfolio risk and return, we built a strategy table by simulating various acceptance rates.

At each acceptance threshold, we compute:

Acceptance Rate: Proportion of applicants approved

Threshold: Probability of default cutoff used for approval

Bad Rate: Proportion of approved loans that still defaulted

Estimated Value: Total value of approved loans minus their expected losses

Estimated Value = Total Accepted Loan Amount – Total Expected Loss

Example logic:

python
Copy
Edit
expected_loss = PD * EAD * LGD
estimated_value = sum(loan_amnt) - sum(expected_loss)
Strategy Table Output (Sample)
Acceptance Rate	Threshold	Bad Rate	Estimated Value ($)
1.00	0.000	0.224	5,230,000
0.75	0.185	0.162	6,150,000
0.50	0.310	0.111	6,800,000
0.30	0.420	0.069	6,250,000

By adjusting thresholds, lenders can make informed trade-offs between volume of loans issued and risk of default. This strategy simulation turns predictive modeling into a decision-making tool.

Visualizations
Bad Rate by Probability Threshold
This chart illustrates how the bad rate increases as the probability threshold for acceptance decreases (i.e., more applicants are approved, including riskier ones):


Results Summary
XGBoost consistently outperformed Logistic Regression in predictive performance

Default rates were significantly higher for loans intended for medical and debt consolidation purposes

The strategy table highlighted optimal thresholds for maximizing portfolio value while minimizing risk

Business-driven simulation using expected loss demonstrated the value of combining machine learning with financial decision-making

