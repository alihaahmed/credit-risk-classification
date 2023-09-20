# credit-risk-classification

## Overview of the Analysis

In this report, various machine learning techniques have been used to build and train two models with the predictive capacity to categorize borrower creditworthiness with varying accuracy, precision and recall rates. Input and label data was sourced from a dataset of historical lending activity from a peer-to-peer lending services company. This includes borrower profile data such as loan size, rate of interest, income level, debt-to-income ratio, number of accounts, derogatory marks and total debt. After reading this CSV file into a Pandas DataFrame, the data was separated into labels and features with the above features used as inputs into the model and the loan status column separated and stored as the y-variable to be predicted by the model. The data was further split into training and testing datasets using the train_test_split module from scikit-learn, and a logistic regression model (Logistic Regression Model 1) was initialized and fitted using the training data. The test feature data was used to generate and store predictions from the fitted model, which were evaluated against the test label data using a confusion matrix and classification report.

A review of the dataset using the value_counts function revealed a disproportionate number of low-risk data points (75,036) to high-risk data points (2,500). As such, the data was resampled using the RandomOverSampler module from the imbalanced-learn library to obtain an equal number of data points for both low-risk and high-risk labels. The original training data was fitted to the random oversampler model and a second logistic regression model (Logistic Regression Model 2) was initialized and fitted using the resampled training data to make predictions. The results and performance metrics of both models are provided below.

## Results

**Logistic Regression Model 1:**
  * Balanced Accuracy: 94.43%
  * Precision (Healthy Loans): 100%
  * Recall (Healthy Loans): 100%
  * Precision (High-Risk Loans): 87%
  * Recall (High-Risk Loans): 89%

The logistic regression model predicts healthy loans from the test data with a 100% precision rate, while high-risk loans from the test data are predicted with an 87% precision rate. Likewise, the recall rates for healthy loans and high-risk loans are 100% and 89%, respectively. The balanced accuracy score of the model is 94.43%. Overall, precision and recall in the prediction of healthy loans is significantly stronger than that of high-risk loans, which indicates that false positives and false negative rates were more effectively minimized by the model in the case of healthy loans. Nevertheless, the model presents a near-perfect accuracy rate for both labels.

**Logistic Regression Model 2:**
  * Balanced Accuracy: 99.60%
  * Precision (Healthy Loans): 100%
  * Recall (Healthy Loans): 100%
  * Precision (High-Risk Loans): 87%
  * Recall (High-Risk Loans): 100%

The resampled logistic regression model predicts healthy loans from the test data with a 100% precision rate, while high-risk loans from the test data are predicted with an 87% precision rate. Recall rates for healthy loans and high-risk loans are both 100%. The balanced accuracy score of the model is nearly 100%, at 99.60%. Overall, precision in the prediction of healthy loans is significantly stronger than that of high-risk loans, indicating that false positives were more effectively minimized by the model in the case of healthy loans. However, fitting the logistic regression model with oversampled data has increased the recall rate for high-risk loans to 100% and the balanced accuracy score to nearly 100% as compared to the original logistic regression model. Model performance/prediction has therefore improved considerably as a result of fitting the model with oversampled data. 

## Summary

Logistic Regression Model 2 appears to be the stronger choice in predicting and categorizing credit risk. While both models maintained 100% precision and recall rates in the prediction of healthy loans, Model 2 demonstrated a substantial improvement in the recall rate for high-risk loans, which increased from 89% to 100%. Moreover, Model 2 is able to categorize credit risk with a balanced accuracy score of 100%, which is a considerable improvement from Model 1's balanced accuracy score of 94.43%. In the categorization of credit risk, the potential default loss associated with incorrectly categorizing a true high-risk loan and erroneously sanctioning a loan to a borrower with poor credit (i.e., a false positive) must be weighed against the potential loss of opportunity associated with incorrectly categorizing a true healthy loan and foregoing lending to a creditworthy borrower (i.e., a false negative). As such, selecting a model with high precision to minimize the prevalence of false positives, as well as high recall to minimize the prevalence of false negatives, is of great importance to effectively categorize credit risk. Therefore, Model 2 is the better choice between both models.
