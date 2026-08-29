# Credit Risk Analytics: From Descriptive Insights to Risk Scoring | Python

> **Turning 1,000 customer credit records into actionable risk intelligence — using statistical analysis, segmentation, feature engineering, visualization, and a custom 0–10 Credit Risk Score.**

---

## 📌 Project Overview

This project is a **Python-based Credit Risk Analytics and Exploratory Data Analysis (EDA) project** built using the German Credit Risk dataset.

The objective is **not to build a Machine Learning model**, but to demonstrate how raw customer-level credit data can be transformed into meaningful **business insights, customer segments, risk indicators, statistical relationships, and a customized credit risk scoring framework**.

The project progresses through:

**Descriptive Analytics → Segmentation → Diagnostic Analytics → Feature Engineering → Risk Scoring → Risk Classification → Advanced Visualization → Business Insights**

The central business question is:

> **How can customer-level credit characteristics be transformed into interpretable risk indicators that support better understanding of credit risk?**

---

## 🎯 Business Objective

Financial institutions need to understand the characteristics and behavior of customers across different credit profiles.

This project explores relationships between:

- Credit Amount
- Age
- Loan Duration
- Employment Status
- Loan Purpose
- Housing
- Checking Account
- Saving Account
- Monthly EMI
- Customer Segments

The analysis then creates customized analytical features to develop a structured **Credit Risk Score (0–10)** and corresponding **Credit Risk Categories**.

### Key Business Questions

- Which customer segments dominate the portfolio?
- How is credit amount distributed across customer groups?
- How does loan duration relate to credit exposure?
- How does employment status differ across credit amounts?
- How does age relate to credit characteristics?
- How does Monthly EMI vary across customer segments?
- Which credit tiers contain higher risk-factor values?
- How do individual risk factors contribute to the overall Credit Risk Score?
- Is there a meaningful relationship between credit amount and risk score?
- How do different customer characteristics interact with the calculated risk profile?
- Can a customized risk-scoring framework summarize customer-level risk?

---

# 🧭 Analytical Framework

```text
Raw Credit Data
       ↓
Data Understanding & Quality Checks
       ↓
Descriptive Statistics
       ↓
Distribution & Outlier Analysis
       ↓
Customer Segmentation
       ↓
Diagnostic Analysis
       ↓
Feature Engineering
       ↓
Risk Factor Normalization
       ↓
Credit Risk Score
       ↓
Risk Classification
       ↓
Advanced Statistical & Visual Analysis
       ↓
Business Insights & Recommendations
```

---

# 📊 Dataset

The project uses a cleaned version of the **German Credit Risk dataset** containing customer-level credit information.

The dataset contains **1,000 customer records**.

Key variables include:

| Variable | Description |
|---|---|
| `Customer.Id` | Unique customer identifier |
| `Age` | Customer age |
| `Sex` | Customer gender |
| `Job` | Job/skill classification |
| `Housing` | Housing status |
| `Saving accounts` | Savings account status |
| `Checking account` | Checking account status |
| `Credit amount` | Credit/loan amount |
| `Duration` | Loan duration |
| `Loan_Purpose` | Purpose of the loan |
| `Employment_Status` | Derived employment classification |

Additional analytical variables are created during the project.

---

# 🛠️ Technologies & Libraries

### Core Python Stack

```python
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind
from statsmodels.stats.weightstats import ztest

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
```

### Statistical Analysis

```python
from statsmodels.tsa.stattools import adfuller
```

### Time-Series / Forecasting Techniques Explored

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
```

### Additional Analytical Modeling

Selected `scikit-learn` functionality is used for analytical trend estimation, including:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
```

> **Important:** This repository is classified as a **Data Analytics / Statistical Analysis project**, not a Machine Learning project. Regression functionality is used for analytical trend estimation rather than building a production predictive model.

---

# 🔍 Project Methodology

## 1. Data Loading & Initial Inspection

The cleaned dataset is loaded and inspected for:

- Dataset structure
- Data types
- Number of records
- Missing values
- Variable characteristics
- Customer-level observations

---

# 2. Descriptive Statistics

Detailed statistical analysis is performed on credit amount and other numerical variables.

Metrics include:

- Mean
- Median
- Standard deviation
- Variance
- Minimum
- Maximum
- Range
- Quartiles
- Distribution
- Skewness

This establishes the statistical foundation before deeper risk analysis.

---

# 3. Outlier Analysis

Credit amount is analyzed using the **Interquartile Range (IQR)** method.

```text
IQR = Q3 − Q1

Upper Bound = Q3 + 1.5 × IQR
Lower Bound = Q1 − 1.5 × IQR
```

The analysis identifies:

- Number of outliers
- Percentage of outliers
- Outlier thresholds
- Impact of extreme credit amounts

---

# 4. Distribution & Skewness Analysis

The distribution of `Credit amount` is examined using:

- Histograms
- Mean vs. median comparison
- Quartiles
- Standard deviation
- Skewness
- Outlier visualization

This helps determine whether the credit portfolio is evenly distributed or concentrated toward particular credit ranges.

---

# 5. Customer Segmentation

One of the central components of the project is **custom customer segmentation**.

Additional analytical segments are created to make the dataset easier to interpret.

### Age Status

Customers are segmented into:

- Young Age
- Middle Age
- Early Old Age
- Very Old Age

### Tenure Status

Loan duration is segmented into:

- Small Tenure
- Medium Tenure
- Long Tenure
- Extremely Long Tenure

### Credit Tier

Credit amount is segmented into:

- Modest
- Lower Mediate
- Intermediate
- Upper Mediate
- Very High
- Extremely High

These segments support deeper comparison across customer groups.

---

# 6. Initial Descriptive Insights

The project examines the composition of the customer portfolio.

Examples include:

- Employment-status distribution
- Age-group distribution
- Loan-purpose distribution
- Credit-tier distribution
- Tenure distribution

Selected observations from the analysis include:

- **Skilled customers:** 630 out of 1,000
- **Middle Age:** 464 customers
- **Young Age:** 411 customers
- **Vehicle loans:** 337
- **Electronic Appliances:** 280
- **Household Furnishings:** 181
- **Medium Tenure:** 590

These findings establish the baseline customer structure before deeper diagnostic analysis.

---

# 7. Diagnostic Analysis

The project progresses beyond:

> **"What happened?"**

toward:

> **"What relationships and patterns may explain what we are seeing?"**

The analysis investigates credit amount and EMI behavior across:

- Employment Status
- Age Status
- Loan Purpose
- Tenure Status
- Credit Tier
- Checking Account
- Customer Age

Visualization techniques include:

- Box plots
- Scatter plots
- Line charts
- Interactive Plotly visualizations
- Trend lines
- Distribution comparisons

---

# 8. Monthly EMI Analysis

A customized `Monthly_EMI` field is created:

```python
df['Monthly_EMI'] = df['Credit amount'] / df['Duration']
```

This allows the project to investigate the relationship between:

**Credit Amount → Loan Duration → Monthly EMI**

Monthly EMI is then analyzed across:

- Age
- Age Status
- Checking Account
- Employment Status

This provides an additional dimension for understanding customer-level financial burden.

---

# 9. Statistical Relationship Analysis

The project explores statistical relationships between important numerical variables.

### Pearson Correlation

The relationship between:

```text
Age ↔ Duration
```

is evaluated using Pearson correlation and its associated p-value.

The analysis also explores standardized values and customer-level statistical relationships to understand how observations behave relative to the overall distributions.

---

# 10. Risk Factor Engineering

A major component of the project is the creation of customized numerical risk factors.

Three major risk dimensions are created.

### 1. Amount Risk Factor

Credit amount is normalized to a **0–10 scale**.

```python
Amount_Risk_Factor =
((Credit amount - minimum) /
(maximum - minimum)) × 10
```

---

### 2. Age Risk Factor

Age is normalized to a **0–10 scale**.

```python
Age_Risk_Factor =
((Age - minimum) /
(maximum - minimum)) × 10
```

---

### 3. Tenure Risk Factor

Loan duration is normalized to a **0–10 scale**.

```python
Tenure_Risk_Factor =
((Duration - minimum) /
(maximum - minimum)) × 10
```

These engineered variables place three different numerical dimensions onto a common scale for analytical comparison.

---

# 🧮 Credit Risk Score

The three risk factors are combined into a customized:

## `Credit_Risk_Score`

The project uses the **median** of the three risk factors:

```python
df['Credit_Risk_Score'] = (
    df[
        [
            'Amount_Risk_Factor',
            'Age_Risk_Factor',
            'Tenure_Risk_Factor'
        ]
    ]
    .median(axis=1)
    .round(1)
)
```

### Why Median?

The median provides a robust central measure of the three risk dimensions and reduces the influence of an unusually high or low individual factor.

The resulting score is represented on a:

```text
0 ───────────────────── 10
Low                    Higher
Risk                    Risk
```

scale.

---

# 🏷️ Credit Risk Classification

The numerical risk score is translated into categorical risk groups.

The project creates:

- Low Risk
- Medium Risk
- High Risk
- Very High Risk

The classification also considers the `Mean_Risk_Factor` alongside the overall `Credit_Risk_Score`.

This provides a more interpretable customer-level risk profile.

---

# 📈 Advanced Visualization

The project uses static and interactive visualization techniques.

### Matplotlib

Used for:

- Statistical visualization
- Category counts
- Distribution analysis

### Seaborn

Used for:

- Statistical plots
- Scatter analysis
- Distribution visualization
- Relationship exploration

### Plotly

Used for:

- Interactive scatter plots
- Box plots
- Line charts
- Risk-score analysis
- Trend visualization
- Customer-level exploration
- Hover-based analysis
- Multi-dimensional visualization

---

# 📊 Key Visual Analyses

### Credit Amount Analysis

- Credit Amount Distribution
- Credit Amount vs Duration
- Outlier Identification
- Credit Amount by Employment Status
- Credit Amount by Age Status
- Credit Amount by Loan Purpose
- Credit Amount by Tenure Status

### EMI Analysis

- Monthly EMI by Age
- Monthly EMI by Age Status
- Monthly EMI by Checking Account
- Monthly EMI by Employment Status

### Risk Analysis

- Amount Risk Factor by Credit Tier
- Age Risk Factor by Credit Tier
- Tenure Risk Factor by Credit Tier
- Credit Risk Score vs Credit Amount
- Credit Risk Score vs Risk Factors
- Credit Risk Score vs Mean Risk Factor
- Risk Category Visualization
- Trend-line Analysis
- Polynomial Trend Analysis
- Customer-level Risk Visualization

---

# 🔬 Trend & Relationship Analysis

The project uses multiple approaches to examine relationships between engineered risk variables and the final Credit Risk Score.

These include:

- Linear trend analysis
- Polynomial trend analysis
- OLS-based trend estimation
- R² evaluation
- Smoothed trends
- Reference-line comparisons
- Aggregated and non-aggregated visual analysis

This provides multiple analytical perspectives rather than relying on a single statistical measure.

---

# 💡 Key Analytical Insights

### Credit Amount

Credit amounts are predominantly concentrated within lower-to-intermediate credit tiers.

### Employment

Skilled customers form the largest employment group in the dataset.

### Age

Young and Middle Age customers represent the dominant age groups.

### Loan Purpose

Vehicle loans and Electronic Appliance loans represent major loan-purpose categories.

### Tenure

Medium Tenure is the dominant duration segment.

### EMI

The analysis investigates how Monthly EMI varies across age, employment, account status, and other customer segments.

### Risk Factors

Credit Amount, Age, and Tenure behave differently across customer segments and therefore provide multiple dimensions for constructing the customized risk framework.

### Statistical Relationships

The project demonstrates that a single correlation or statistical metric does not necessarily capture the complete structure of customer-level risk behavior.

---

# 🧠 Business Interpretation

The project demonstrates how a business analyst can transform raw customer data into a structured decision-support framework.

Instead of looking at:

```text
Age
Credit Amount
Duration
Employment
Checking Account
Loan Purpose
```

independently, the project combines multiple dimensions into:

```text
Customer Segmentation
        ↓
Risk Factors
        ↓
Credit Risk Score
        ↓
Risk Category
```

This creates a more compact framework for interpreting customer-level credit exposure.

---

# ⚠️ Important Analytical Limitation

The customized `Credit_Risk_Score` developed in this project is an **analytical scoring framework created specifically for this project**.

It is **not an officially validated banking credit score** and should not be interpreted as a real-world lending decision model.

Similarly:

- Normalization does not establish causal risk.
- Correlation does not imply causation.
- The engineered risk factors reflect analytical assumptions.
- The dataset may not contain all variables required for real-world credit underwriting.
- The scoring framework has not been validated against actual default outcomes.

This distinction is especially important when applying analytics to financial services.

---

# 🚀 What This Project Demonstrates

## Data Analysis

- Data inspection
- Data profiling
- Statistical summaries
- Distribution analysis
- Outlier detection
- Skewness analysis

## Data Manipulation

- Feature creation
- Segmentation
- Binning
- Grouping
- Aggregation
- Normalization

## Statistics

- Mean
- Median
- Variance
- Standard deviation
- Quartiles
- IQR
- Skewness
- Pearson correlation
- Hypothesis-testing techniques
- Regression-based trend analysis

## Data Visualization

- Matplotlib
- Seaborn
- Plotly
- Interactive visualization
- Scatter plots
- Box plots
- Histograms
- Line charts
- Trend lines
- Multi-dimensional visualization

## Business Analytics

- Customer segmentation
- Risk-factor analysis
- Credit exposure analysis
- EMI analysis
- Risk classification
- Business interpretation
- Analytical storytelling

---

# 🗂️ Project Structure

```text
German-Credit-Risk-Analytics/
│
├── German_Credit_Risk_Analysis_Report.ipynb
│
├── german_credit_risk_data_cleaned.csv
│
├── README.md
│
└── Visuals/
    └── Generated analytical visualizations
```

---

# 📓 Notebook Structure

```text
01. Data Loading
02. Dataset Inspection
03. Statistical Analysis
04. Distribution Analysis
05. Outlier Analysis
06. Skewness Analysis
07. Customer Segmentation
08. Initial Insights
09. Diagnostic Analysis
10. Monthly EMI Analysis
11. Correlation Analysis
12. Risk Factor Engineering
13. Credit Risk Score
14. Risk Classification
15. Advanced Visualization
16. Trend Analysis
17. Final Insights
```

---

# 🏦 BFSI Relevance

This project was designed with a **banking and credit-risk analytics perspective**.

It demonstrates how customer-level data can be transformed into analytical indicators that could support areas such as:

- Credit Risk Analytics
- Retail Banking Analytics
- Customer Segmentation
- Portfolio Analysis
- Loan Analytics
- Credit Exposure Analysis
- Risk Monitoring
- Business Intelligence
- Management Reporting

The project is an **analytical prototype**, not a production lending system.

---

# 🎯 Project Outcome

The final analytical dataset contains original customer attributes alongside engineered fields such as:

```text
Age_Status
Tenure_Status
Credit_Tier
Monthly_EMI
Amount_Risk_Factor
Age_Risk_Factor
Tenure_Risk_Factor
Mean_Risk_Factor
Credit_Risk_Score
Credit_Risk_Factor
```

This transforms the original dataset into a richer analytical framework for exploring customer credit profiles.

---

# 🔮 Future Scope

A separate Machine Learning project can build on this work using a different predictive objective.

Possible future directions include:

- Default prediction
- Probability of Default (PD)
- Classification modeling
- CatBoost
- LightGBM
- XGBoost
- Random Forest
- Model evaluation
- Feature importance
- Explainable AI
- Model comparison

The distinction would be:

```text
THIS PROJECT
Data Analytics + Statistics + Visualization
                     ↓
              Risk Scoring


FUTURE ML PROJECT
Features + Historical Outcome
                     ↓
             Machine Learning
                     ↓
          Predictive Risk Model
```

---

# 👨‍💻 Skills Demonstrated

**Python | Pandas | NumPy | SciPy | Statsmodels | Matplotlib | Seaborn | Plotly | Exploratory Data Analysis | Statistical Analysis | Data Manipulation | Feature Engineering | Customer Segmentation | Risk Analytics | Data Visualization | Business Analytics | Credit Risk Analytics**

---

# ⭐ Why This Project Matters

The objective was not simply to create charts.

It was to demonstrate the complete analytical thought process:

> **Understand the data → identify patterns → segment customers → investigate relationships → engineer meaningful features → construct an interpretable risk score → classify risk → communicate business insights.**

This progression demonstrates the difference between simply **working with data** and using data to **support structured business decision-making**.

---

## 📌 Disclaimer

This project is created for educational, analytical, and portfolio purposes.

The customized Credit Risk Score and Risk Categories are project-specific analytical constructs and are **not intended to represent an official credit score, banking underwriting policy, or regulated credit-risk model**.

---

## 👤 Author

**Somnath Roy**

Data & Business Analytics | Python | SQL | Power BI | Tableau | Statistical Analysis | Machine Learning

---

### 🔗 Project Repository

**GitHub:**  
`[https://drive.google.com/file/d/1nhgOM0lrPP-o_0ZOnHes_xy9FLIWaJ8Q/view?usp=sharing]`
