# Customer Purchase Prediction

This project aims to predict whether a customer will make a purchase based on various features using several Machine Learning (Random Forest, SVM, Logistic Regression) and Deep Learning (Artificial Neural Network) models. The dataset is preprocessed using StandardScaler. Additionally, a Streamlit app has been created to allow users to interactively predict customer purchases.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Features](#features)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Evaluation](#evaluation)
- [Streamlit App](#streamlit-app)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Introduction
The goal of this project is to build predictive models that classify whether a customer will make a purchase. Four different models are used: Random Forest, Support Vector Machine (SVM), Logistic Regression, and an Artificial Neural Network (ANN). A Streamlit app is also developed to facilitate user interaction with the predictive models.

## Dataset
The dataset includes the following features:
- `Age`: Customer's age
- `Gender`: Customer's gender (0: Male, 1: Female)
- `AnnualIncome`: Annual income of the customer in dollars
- `NumberOfPurchases`: Total number of purchases made by the customer
- `ProductCategory`: Category of the purchased product (0: Electronics, 1: Clothing, 2: Home Goods, 3: Beauty, 4: Sports)
- `TimeSpentOnWebsite`: Time spent by the customer on the website in minutes
- `LoyaltyProgram`: Whether the customer is a member of the loyalty program (0: No, 1: Yes)
- `DiscountsAvailed`: Number of discounts availed by the customer (range: 0-5)
- `PurchaseStatus`: Likelihood of the customer making a purchase (0: No, 1: Yes) - Target Variable

## Features
- `Age`
- `Gender`
- `AnnualIncome`
- `NumberOfPurchases`
- `ProductCategory`
- `TimeSpentOnWebsite`
- `LoyaltyProgram`
- `DiscountsAvailed`
- `PurchaseStatus` (Target Variable)

## Preprocessing
The dataset is preprocessed using StandardScaler to ensure that all features have a mean of 0 and a standard deviation of 1. This helps in improving the performance and training stability of the models.

## Models
Four models are built to predict the purchase status:
1. **Random Forest Classifier**: A robust and interpretable machine learning model.
2. **Support Vector Machine (SVM)**: A powerful classifier for high-dimensional spaces.
3. **Logistic Regression**: A simple yet effective linear model.
4. **Artificial Neural Network (ANN)**: A deep learning model capable of capturing complex patterns in the data.

## Evaluation
The models are evaluated based on their accuracy on the testing dataset. The evaluation metrics include:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

## Streamlit App
A Streamlit app has been created to allow users to interactively predict whether a customer will make a purchase based on the input features. The app uses the trained models to make predictions and displays the results to the user.

### Running the Streamlit App
1. Ensure you have Streamlit installed:
   ```bash
   pip install streamlit
