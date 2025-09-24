# Get residuals

Here, we adjust the accelerometry features for covariates (sex, age, BMI) by training a linear model on the unaffected control group and subtracting their effect from the raw data. The notebook uses features in residuals.py.