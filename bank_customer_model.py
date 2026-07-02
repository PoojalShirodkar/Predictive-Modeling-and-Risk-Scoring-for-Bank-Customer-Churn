# Predictive Modeling and Risk Scoring for Bank Customer Churn
# bank_customer_model.py
# Part 1.1 - Imports, Loading Dataset, Cleaning, EDA

# Import Libraries

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier

# Create folders if not present

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

print("=" * 60)
print("BANK CUSTOMER CHURN PREDICTION PROJECT")
print("=" * 60)

# Load Dataset

try:
    df = pd.read_csv("data/European_Bank.csv")
except:
    df = pd.read_csv("European_Bank.csv")

print("\nDataset Loaded Successfully\n")

# Basic Information

print("First Five Rows\n")
print(df.head())

print("\nShape of Dataset")
print(df.shape)

print("\nColumns")
print(df.columns)

print("\nInformation")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

# Missing Values

print("\nMissing Values")

print(df.isnull().sum())

# Fill numerical missing values
df.fillna(df.mean(numeric_only=True), inplace=True)


# Duplicate Values

print("\nDuplicate Rows :", df.duplicated().sum())

df.drop_duplicates(inplace=True)


# Remove Unnecessary Columns

columns_to_drop = ["CustomerId", "Surname"]

if "Year" in df.columns:
    columns_to_drop.append("Year")

df.drop(columns_to_drop, axis=1, inplace=True)

print("\nRemaining Columns")
print(df.columns)


# Exploratory Data Analysis (EDA)


print("\nGenerating EDA Charts...")


# Churn Count

plt.figure(figsize=(6,5))
sns.countplot(data=df, x="Exited")
plt.title("Customer Churn Count")
plt.savefig("plots/churn_count.png")
plt.show()


# Gender vs Churn

plt.figure(figsize=(6,5))
sns.countplot(data=df, x="Gender", hue="Exited")
plt.title("Gender vs Churn")
plt.savefig("plots/gender_vs_churn.png")
plt.show()


# Geography vs Churn

plt.figure(figsize=(8,5))
sns.countplot(data=df, x="Geography", hue="Exited")
plt.title("Geography vs Churn")
plt.savefig("plots/geography_vs_churn.png")
plt.show()


# Age Distribution

plt.figure(figsize=(7,5))
sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.savefig("plots/age_distribution.png")
plt.show()


# Credit Score Distribution

plt.figure(figsize=(7,5))
sns.histplot(df["CreditScore"], bins=30, kde=True)
plt.title("Credit Score Distribution")
plt.savefig("plots/credit_score_distribution.png")
plt.show()


# Balance Distribution

plt.figure(figsize=(7,5))
sns.histplot(df["Balance"], bins=30, kde=True)
plt.title("Balance Distribution")
plt.savefig("plots/balance_distribution.png")
plt.show()


# Salary Distribution

plt.figure(figsize=(7,5))
sns.histplot(df["EstimatedSalary"], bins=30, kde=True)
plt.title("Estimated Salary Distribution")
plt.savefig("plots/salary_distribution.png")
plt.show()


# Products

plt.figure(figsize=(6,5))
sns.countplot(data=df, x="NumOfProducts")
plt.title("Number of Products")
plt.savefig("plots/products.png")
plt.show()


# Active Members

plt.figure(figsize=(6,5))
sns.countplot(data=df, x="IsActiveMember")
plt.title("Active Members")
plt.savefig("plots/active_members.png")
plt.show()

# Correlation Heatmap

plt.figure(figsize=(14,10))

corr = df.select_dtypes(include=np.number).corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.savefig("plots/correlation_heatmap.png")

plt.show()

print("\nEDA Completed Successfully.")


# End of Part 1.1


# Part 1.2 - Feature Engineering

print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

# 1. Balance to Salary Ratio
df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)

# 2. Product Density
df["ProductDensity"] = df["NumOfProducts"] / (df["Age"] + 1)

# 3. Engagement Product Interaction
df["EngagementProduct"] = (
    df["IsActiveMember"] * df["NumOfProducts"]
)

# 4. Age-Tenure Interaction
df["AgeTenure"] = (
    df["Age"] * df["Tenure"]
)

print("\nFeature Engineering Completed Successfully.")


# One Hot Encoding


print("\n" + "="*60)
print("ENCODING CATEGORICAL VARIABLES")
print("="*60)

df = pd.get_dummies(
    df,
    columns=["Gender", "Geography"],
    drop_first=True
)

print(df.head())


# Feature Selection


print("\n" + "="*60)
print("FEATURES & TARGET")
print("="*60)

X = df.drop("Exited", axis=1)
y = df["Exited"]

print("\nFeatures Shape :", X.shape)
print("Target Shape :", y.shape)

# Save feature names
feature_names = X.columns


# Train Test Split

print("\n" + "="*60)
print("TRAIN TEST SPLIT")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Samples :", len(X_train))
print("Testing Samples :", len(X_test))


# Feature Scaling

print("\n" + "="*60)
print("FEATURE SCALING")
print("="*60)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("\nScaler Saved Successfully!")


# Build Models

print("\n" + "="*60)
print("INITIALIZING MODELS")
print("="*60)

models = {

    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )
}

print("Models Ready for Training.")


# End of Part 1.2


# Part 1.3 - Model Training & Evaluation

print("\n" + "="*60)
print("MODEL TRAINING")
print("="*60)

results = []

best_model = None
best_model_name = ""
best_auc = 0

for name, model in models.items():

    print("\n" + "="*50)
    print(f"Training {name}")
    print("="*50)

    # Logistic Regression requires scaled data
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:,1]

    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC AUC   : {roc:.4f}")

    print("\nClassification Report\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap="Blues"
    )

    plt.title(f"{name} Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.savefig(f"plots/{name}_confusion_matrix.png")

    plt.show()

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1,
        roc
    ])

    if roc > best_auc:
        best_auc = roc
        best_model = model
        best_model_name = name

print("\n")
print("="*60)
print("MODEL COMPARISON")
print("="*60)

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC"
    ]
)

print(results_df)

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

print("\nBest Model :", best_model_name)
print("Best ROC AUC :", round(best_auc,4))


# Part 1.4 - Save Model, Feature Importance & SHAP


print("\n" + "="*60)
print("SAVING BEST MODEL")
print("="*60)

# Save the best model
joblib.dump(best_model, "models/random_forest.pkl")

print(f"\nBest Model ({best_model_name}) saved successfully!")
print("Model Location : models/random_forest.pkl")


# Feature Importance


print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

# Only Tree-based models support feature importance
if hasattr(best_model, "feature_importances_"):

    importance = best_model.feature_importances_

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    print(feature_importance)

    feature_importance.to_csv(
        "models/feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(10,8))

    sns.barplot(
        data=feature_importance.head(15),
        x="Importance",
        y="Feature"
    )

    plt.title("Top 15 Important Features")

    plt.tight_layout()

    plt.savefig("plots/feature_importance.png")

    plt.show()

else:

    print("Feature Importance not available for this model.")


# SHAP Explainability


print("\n" + "="*60)
print("SHAP EXPLAINABILITY")
print("="*60)

try:

    if hasattr(best_model, "feature_importances_"):

        explainer = shap.TreeExplainer(best_model)

        shap_values = explainer.shap_values(X_test)

        shap.summary_plot(
            shap_values,
            X_test,
            show=False
        )

        plt.tight_layout()

        plt.savefig("plots/shap_summary.png")

        plt.close()

        print("SHAP Summary Plot Saved Successfully!")

except Exception as e:

    print("SHAP skipped due to:", e)


# Sample Prediction


print("\n" + "="*60)
print("SAMPLE CHURN PROBABILITY")
print("="*60)

sample_customer = X_test.iloc[[0]]

prediction = best_model.predict(sample_customer)[0]

probability = best_model.predict_proba(sample_customer)[0][1]

print("Prediction :", prediction)

print("Churn Probability :", round(probability*100,2), "%")


# Final Message


print("\n" + "="*60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("="*60)

print("""
Files Generated

✔ models/random_forest.pkl

✔ models/scaler.pkl

✔ models/model_comparison.csv

✔ models/feature_importance.csv

✔ plots/churn_count.png

✔ plots/feature_importance.png

✔ plots/shap_summary.png

✔ Confusion Matrix Images

Machine Learning Pipeline Completed Successfully.
""")