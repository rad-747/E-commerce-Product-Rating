# 🛒 E-Commerce Product Rating Prediction (MachineHack)

> **🥉 3rd Place Winner** in the MachineHack E-Commerce Product Rating Prediction Hackathon  
> [Competition Link](https://machinehack.com/hackathons/ecommerce_product_rating_prediction/overview)

## 📌 Overview

This repository contains my solution to the MachineHack challenge — **predicting e-commerce product ratings** based on product metadata, descriptions, and seller information. The goal was to build a robust regression model that generalizes well on unseen product listings.

## 🏆 Highlights

- 🧠 Implemented advanced **feature engineering**, **TF-IDF** pipelines, and **ensemble modeling**
- 📈 Achieved high performance on public and private leaderboard using **AdaBoost + Decision Trees**

---

## 📂 Dataset

The dataset consists of two main CSV files:

- `Train.csv`: Product title, store name, product details (text + JSON), and target rating
- `Test.csv`: Same format (without target), for prediction submission

> Note: Dataset was provided exclusively via MachineHack and is not open-source.

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights explored:
- 📊 **Target distribution**: Ratings concentrated around 3.5–4.5
- 🏪 **Store quality metrics**: Avg. rating per store
- ✍️ **Title and details length**: Correlated with product rating
- 🧩 **JSON metadata**: Extracted useful features like dimensions, weight
- 🧠 **Top keyword trends** in product titles (e.g. “case”, “cover”, “slim”)

---

## 🛠️ Features Engineered

- **TF-IDF features** from Title and Details (bi-grams)
- **Product dimensions** and **weight** extracted from Details JSON
- **Store label encoding**
- **Text length (log-transformed)**
- **Top keys presence heatmap** from product metadata

---

## 🧪 Model Training

Used `AdaBoostRegressor` with a `DecisionTreeRegressor` base:

```python
model = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=5),
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)
