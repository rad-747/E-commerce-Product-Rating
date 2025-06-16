# 🛒 E-Commerce Product Rating Prediction (MachineHack)

> **🥉 3rd Place Winner** in the MachineHack E-Commerce Product Rating Prediction Hackathon  
> [Competition Link](https://machinehack.com/hackathons/ecommerce_product_rating_prediction/overview)

## 📌 Overview

This repository contains my solution to the MachineHack challenge — **predicting e-commerce product ratings** based on product metadata, descriptions, and seller information. The goal was to build a robust regression model that generalizes well on unseen product listings.

## 🏆 Highlights

- 🥉 Secured **3rd Rank**  
- 📉 Achieved **Train RMSE: 0.7512** and **Validation RMSE: 0.7541**
- 🧠 Engineered custom NLP + metadata features and trained ensemble models
- 📊 Performed full EDA with store trends, metadata parsing, and TF-IDF

---

## 📂 Dataset

The dataset consists of two main CSV files:

- `Train.csv`: Includes product title, store name, details (JSON + text), and target product rating
- `Test.csv`: Same format without the rating, used for predictions

> Note: The dataset was provided by MachineHack exclusively for competition use.

---

## 🔍 Exploratory Data Analysis (EDA)

Some of the key insights and visualizations include:

- 📈 Distribution of product ratings (centered around 3.5–4.5)
- 🏪 Top stores by product count and average ratings
- ✍️ Text length (title/details) and correlation with rating
- 🔑 Most frequent metadata keys in the product JSON
- 🔤 Most common keywords in product titles (e.g., "case", "cover")

---

## 🛠️ Feature Engineering

The following features were extracted and used for training:

- **TF-IDF vectors** from `Title` and `Details` (bi-grams, top 300 tokens each)
- **Average product dimensions** and **weight** parsed from `Details` JSON
- **Store name** encoded using `LabelEncoder`
- **Text length**, log-transformed
- **Top metadata keys** binarized for presence/absence

---

## 🧪 Model Training

A lightweight ensemble model was trained:

```python
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

model = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=5),
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)
