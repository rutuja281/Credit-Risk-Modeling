# Credit Risk Modeling: Loan Default Prediction and Strategic Lending Simulation

## Project Overview

This project focuses on building a machine learning pipeline to predict the likelihood of loan default using real-world credit data. The goal is not only to classify applicants as high or low risk, but also to support data-driven decisions around **loan approvals** using **expected loss calculations** and a **strategy table** simulation.

Two classification models were developed:
- Logistic Regression (baseline)
- XGBoost Classifier (for better performance and handling of non-linearity)

We go beyond model accuracy by analyzing:
- Default risk across different applicant segments
- Business implications of adjusting loan approval thresholds
- Estimation of portfolio value under various decision strategies

---

## Dataset Features

| Column | Description |
|--------|-------------|
| `person_age` | Applicant’s age |
| `person_income` | Annual income of the applicant |
| `person_home_ownership` | Home ownership type (RENT, OWN, MORTGAGE) |
| `person_emp_length` | Years of employment |
| `loan_intent` | Purpose of the loan (e.g. education, medical) |
| `loan_grade` | Credit grade assigned |
| `loan_amnt` | Amount of loan requested |
| `loan_int_rate` | Interest rate of the loan |
| `loan_status` | Target variable (0 = repaid, 1 = defaulted) |
| `cb_person_default_on_file` | Prior credit default flag |
| `cb_person_cred_hist_length` | Length of credit history in years |

---

## Key Concepts Used

- **Probability of Default (PD)**: Likelihood of a borrower defaulting on the loan
- **Exposure at Default (EAD)**: Amount at risk, i.e., the loan amount
- **Loss Given Default (LGD)**: Percentage of loan likely to be lost if default occurs

**Expected Loss (EL)** is calculated as:


---

## Modeling Pipeline

1. **Data Preprocessing**
   - Missing value imputation
   - One-hot encoding of categorical variables
   - Feature engineering (e.g., loan-to-income ratio)

2. **Exploratory Data Analysis**
   - Crosstabs and visualizations of default rate by intent, grade, home ownership
   - Identification of high-risk borrower profiles

3. **Model Training**
   - Logistic Regression as interpretable baseline
   - XGBoost to capture complex patterns and improve ROC AUC

4. **Model Evaluation**
   - Confusion matrix
   - Classification report
   - ROC AUC score and ROC curve plotting

---

## Strategy Table Simulation

To evaluate how different decision thresholds affect portfolio risk and return, we built a **strategy table** by simulating various **acceptance rates**.

At each acceptance threshold, we compute:

- **Acceptance Rate**: Proportion of applicants approved
- **Threshold**: Probability of default cutoff used for approval
- **Bad Rate**: Proportion of approved loans that still defaulted
- **Estimated Value**: Total value of approved loans minus their expected losses

**Estimated Value = Total Accepted Loan Amount – Total Expected Loss**


## Bad Rate Analysis

After generating predicted probabilities of default using the trained model, we analyzed how the **bad rate** changes with different **probability thresholds** for loan approval.

- **Bad Rate** refers to the percentage of approved loans that ultimately default.
- As we lower the threshold (i.e., approve more applicants), we also increase the risk of default — resulting in a higher bad rate.

This analysis helps identify the **optimal balance between accepting more applicants and minimizing losses**.

### Visualization: Bad Rate by Probability Threshold

The following plot shows how the bad rate changes as the loan approval threshold varies. This is essential for understanding risk exposure at different business decision levels.

![Bad Rate vs Threshold](images/bad_rate.png)

---

## Results Summary

- **XGBoost** consistently outperformed **Logistic Regression** in predictive performance, especially in ROC AUC.
- **Default rates** were significantly higher for loans taken for **medical** and **debt consolidation** purposes.
- **Renters** and applicants with **short credit histories** showed higher risk profiles.
- The **strategy table** revealed how business teams can adjust thresholds to:
  - Lower the default rate (bad rate)
  - Increase expected portfolio value
  - Make data-driven trade-offs between risk and volume
- This project demonstrated how machine learning can be extended beyond classification to guide **financial strategy and decision-making** in lending.


Example logic:

```python
expected_loss = PD * EAD * LGD
estimated_value = sum(loan_amnt) - sum(expected_loss)
