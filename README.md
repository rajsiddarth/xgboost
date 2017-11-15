# Gradient Boosting in R using h20 and xgboost library
The data set considered has the following variables.
ID: Customer ID

Age: Customer Age

Experience: #years of Professional experience

Income: Annual Income of the customer in $000

ZIP Code:Home address Zip Code

Family: Family size of the customer

CCAvg: Avg spending on credit cards per month in $000

Education: Education level 1:Undergrad 2: Graduate 3:Advanced/Professional

Mortgage: Value of mortgage if any $000

Securities Account: Does the customer have a securities account with the bank?

CD Account:Does the customer have a certificate of deposit account with the bank?

Online : Does the customer use internet banking facilities?

CreditCard: Does the customer use a credit card issued by the respective bank?

Personal Loan : Did the customer default on the loan or not ?

## gradient_boost_h20.R

H2O is an open source, in-memory, distributed, fast, and scalable machine learning and predictive analytics platform that allows you to build machine learning models on big data and provides easy productionalization of those models in an enterprise environment.
https://www.h2o.ai/
I used the availability of h2O in R.More information can be found at the following link.
http://docs.h2o.ai/h2o/latest-stable/h2o-docs/quick-start-videos.html#h2o-quick-start-with-r
The objective is to predict whether the customer is going to default on his loan or not using Gradient boosting ensemble techniques implemented with h2O frame work in R.

## xgboost.R

This algorithm follows the similar approach as above i.e., uses gradient boosting to predict the customer loan default. It uses **xgboost** library available in R.
