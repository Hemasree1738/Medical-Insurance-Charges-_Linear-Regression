# Medical-Insurance-Charges-_Linear-Regression
Linear regression on medical insurance data, from a single-feature model to multi-feature regression with categorical and one-hot encoding, built with pandas, scikit-learn, and Plotly.
Overview

This project walks through the full linear regression workflow on a medical insurance dataset (1,338 records): exploratory data analysis, correlation analysis, a simple one-variable model, a from-scratch gradient-descent intuition exercise, multi-feature regression, and categorical feature encoding (label + one-hot).

Dataset

medical_data.csv — 1,338 rows, 7 columns:

Column	Description
age	Age of the primary beneficiary
sex	male / female
bmi	Body mass index
children	Number of dependents covered
smoker	yes / no
region	Residential area in the US (northeast, northwest, southeast, southwest)
charges	Annual medical insurance charges (target variable)
Exploratory Data Analysis
Distribution plots for age, bmi, and charges using Plotly histograms.
Scatter plots of age vs charges and bmi vs charges, colored by smoker status.
Violin plots comparing charges across smokers/non-smokers.
Correlation heatmap across numeric features.

Key finding: smoker status is by far the strongest predictor of charges (correlation ≈ 0.79), well ahead of age (≈ 0.30) and bmi (≈ 0.20). children has almost no linear relationship with charges (≈ 0.07).

Modeling

The notebook builds up in stages:

Manual parameter search — a hand-written try_parameters(w, b) function to build intuition for how slope and intercept affect model fit, before introducing sklearn.
Simple linear regression — age → charges using LinearRegression. RMSE ≈ 11,552.
SGDRegressor comparison — same single feature, fit with stochastic gradient descent (with StandardScaler) to compare against the closed-form solution. RMSE ≈ 11,552, confirming both approaches converge to essentially the same fit.
Multiple linear regression — adding bmi and children alongside age. RMSE improves to ≈ 11,355.
Categorical encoding — smoker and sex mapped to binary 0/1, combined with age, bmi, children. RMSE drops sharply to ≈ 6,056, confirming smoker is the single most important predictor.
One-hot encoding — region encoded into four binary columns (northeast, northwest, southeast, southwest) using sklearn.preprocessing.OneHotEncoder and added to the feature set. RMSE improves slightly further to ≈ 6,042.
Tech Stack
Python, pandas, NumPy
scikit-learn (LinearRegression, SGDRegressor, OneHotEncoder, StandardScaler)
Matplotlib, Seaborn, Plotly (visualization)
How to Run
bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly jupyter
jupyter notebook linear_reg.ipynb

Make sure medical_data.csv is in the same directory as the notebook.

Results Summary
Model	Features	RMSE
Simple linear regression	age	≈ 11,552
SGDRegressor (scaled)	age	≈ 11,552
Multiple linear regression	age, bmi, children	≈ 11,355
+ categorical encoding	age, bmi, children, smoker, sex	≈ 6,056
+ one-hot encoding	above + region (one-hot)	≈ 6,042
Key Learnings
Smoking status dominates insurance charge prediction — a strong reminder that correlation analysis before modeling saves time. Adding smoker to the feature set nearly halved the RMSE, far outweighing the effect of any other single feature.
A single-feature model captures only part of the picture; adding relevant numeric features (bmi, children) gives a modest improvement, but the real gains come from properly encoding categorical variables (smoker, region) rather than ignoring them.
Closed-form (LinearRegression) and iterative (SGDRegressor) solvers converge to nearly identical results on this dataset, which is a useful sanity check when learning gradient descent.
Next Steps
Cross-validation to get a more robust estimate of model performance.
Regularization (Ridge/Lasso) to guard against overfitting as more features are added.
Compare against a non-linear baseline (e.g., Random Forest or Gradient Boosting).
