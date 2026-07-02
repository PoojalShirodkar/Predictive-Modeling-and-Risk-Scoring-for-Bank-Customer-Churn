1. Introduction

Customer retention is an essential aspect of banking operations. Acquiring new customers is often more expensive than retaining existing ones. Therefore, predicting customer churn has become a significant application of machine learning in the banking industry.

This project focuses on building a predictive model capable of identifying customers who are likely to discontinue banking services. Along with prediction, an interactive dashboard was developed to help business users understand customer behavior and analyze churn risk.

2. Problem Statement

Banks lose customers every year due to various reasons such as poor customer engagement, low satisfaction, high competition, and changing financial needs. Without predictive analytics, banks are unable to identify customers who are at risk of leaving.

The objective is to develop a machine learning solution that accurately predicts customer churn and provides actionable insights through an interactive dashboard.

3. Objectives

The primary objectives of this project are:

Predict customer churn using machine learning algorithms.
Calculate churn probability and assign a risk score.
Identify important factors contributing to customer churn.
Develop an interactive Streamlit dashboard for visualization.
Enable business users to perform what-if scenario analysis.
Support customer retention strategies using predictive analytics.
4. Dataset Description

The project uses the European Bank Customer Churn Dataset, which contains customer information collected from a European bank.

Dataset Features
Customer ID
Credit Score
Geography
Gender
Age
Tenure
Balance
Number of Products
Has Credit Card
Is Active Member
Estimated Salary
Year
Exited (Target Variable)

The target variable Exited indicates whether a customer has left the bank.

0 → Customer Retained
1 → Customer Churned
5. Data Preprocessing

The following preprocessing steps were performed:

Removed unnecessary columns such as Customer ID and Surname.
Checked for missing values.
Verified duplicate records.
Converted categorical variables into numerical format using one-hot encoding.
Standardized numerical features using StandardScaler.
Split the dataset into training and testing datasets.
6. Exploratory Data Analysis (EDA)

Several visualizations were created to understand customer behavior and identify churn patterns.

The dashboard includes:

Customer Churn Distribution
Age Distribution
Balance Distribution
Salary Distribution
Credit Score Distribution
Geography-wise Churn
Gender-wise Churn
Products Owned by Customers
Active Members vs Churn
Credit Card Holders vs Churn
Correlation Heatmap
Balance vs Salary Scatter Plot

These visualizations helped identify the factors influencing customer churn.

7. Feature Engineering

Additional features were created to improve model performance.

These include:

Balance-to-Salary Ratio
Product Density
Engagement Product
Age × Tenure Interaction
Encoded Geography Variables
Encoded Gender Variable

Feature engineering improves the ability of machine learning models to capture customer behavior.

8. Machine Learning Models

Different machine learning algorithms were evaluated for predicting customer churn.

Models considered include:

Logistic Regression
Random Forest Classifier
Gradient Boosting Classifier
Decision Tree Classifier

The best-performing model was selected based on evaluation metrics such as:

Accuracy
Precision
Recall
F1 Score
ROC-AUC Score

The trained model was saved using Joblib for deployment.

9. Risk Scoring

Instead of providing only a binary prediction, the system calculates the probability of customer churn.

Customers are categorized into three risk levels:

Low Risk: Probability below 30%
Medium Risk: Probability between 30% and 70%
High Risk: Probability above 70%

This enables business users to prioritize customer retention efforts effectively.

10. Dashboard Development Using Streamlit

A user-friendly dashboard was developed using Streamlit.

Dashboard Features
Interactive KPI Cards
Customer Churn Prediction
Churn Probability Indicator
Risk Score Gauge
Dynamic Charts
Feature Importance Analysis
What-if Scenario Simulator
Interactive Filters
Responsive Dashboard Design

The dashboard provides business users with an intuitive interface for analyzing customer churn.

11. Results

The developed system successfully predicts customer churn based on customer information.

Key outcomes include:

Accurate churn prediction
Real-time risk scoring
Interactive visualization of customer insights
Identification of important churn factors
Easy deployment through Streamlit

The dashboard simplifies data interpretation for non-technical users and supports informed business decisions.

12. Conclusion

This project demonstrates the application of machine learning in solving customer churn problems in the banking sector. By combining predictive modeling with an interactive dashboard, the solution enables banks to identify high-risk customers and implement proactive retention strategies.

The project also highlights the importance of data preprocessing, feature engineering, and visualization in building effective machine learning applications.

13. Future Scope

The project can be enhanced in several ways:

Integrate real-time customer data.
Deploy the application on a cloud platform.
Include SHAP values for explainable AI.
Add customer segmentation using clustering techniques.
Develop personalized retention recommendations.
Improve prediction accuracy using deep learning models.
Connect the dashboard with live banking databases.
14. References
Aurélien Géron, Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow, O'Reilly Media.
Ian Goodfellow, Yoshua Bengio, Aaron Courville, Deep Learning, MIT Press.
Scikit-learn Documentation – https://scikit-learn.org
Streamlit Documentation – https://streamlit.io
Plotly Documentation – https://plotly.com/python/
Pandas Documentation – https://pandas.pydata.org/
NumPy Documentation – https://numpy.org/
European Bank Customer Churn Dataset (used for academic purposes).
