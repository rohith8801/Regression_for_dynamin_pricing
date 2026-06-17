# Regression_for_dynamin_pricing
# Dynamic Pricing Prediction System

## Overview

The Dynamic Pricing Prediction System is a Machine Learning application designed to estimate optimal property prices by analyzing demand, supply, location, and property-specific characteristics. The project simulates real-world pricing strategies used by hospitality and rental platforms by leveraging advanced feature engineering and predictive modeling techniques.

The system transforms raw listing attributes into business-relevant signals and predicts prices using Gradient Boosting Regression, enabling data-driven pricing decisions with high predictive accuracy.

---

## Problem Statement

Pricing decisions in hospitality and rental businesses are influenced by multiple interconnected factors such as location, property quality, customer demand, and availability. Traditional fixed-price approaches often fail to capture these dynamic relationships, resulting in suboptimal pricing strategies.

This project aims to develop an intelligent pricing system capable of learning these relationships and generating accurate price recommendations through machine learning.

---

## Objectives

* Build an end-to-end machine learning pipeline for price prediction.
* Perform data preprocessing and feature engineering.
* Compare multiple regression algorithms.
* Evaluate model performance using industry-standard metrics.
* Select the best-performing model for deployment.
* Provide explainable predictions through SHAP-based feature interpretation.

---

## Dataset

The project utilizes Airbnb-style property listing data containing:

* Property Information
* Room Type
* Availability
* Review Statistics
* Location Attributes
* Demand Indicators

Additional business-driven features were engineered to simulate real-world pricing behavior.

---

## Machine Learning Workflow

### 1. Data Collection

* Dataset loading and inspection
* Data quality assessment

### 2. Data Preprocessing

* Missing value handling
* Feature selection
* Categorical encoding

### 3. Feature Engineering

Created domain-specific features including:

* Demand Score
* Amenities Score
* Quality Factor
* Supply Factor
* Location Factor
* Dynamic Pricing Components

These engineered features significantly improved model learning capability.

### 4. Model Development

Implemented and compared:

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor

### 5. Model Evaluation

Evaluation metrics:

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score
* K-Fold Cross Validation

### 6. Explainability

Integrated SHAP (SHapley Additive Explanations) to provide:

* Feature importance analysis
* Prediction transparency
* Business interpretability

---

## Results

| Model             | Test R² |
| ----------------- | ------- |
| Linear Regression | ~0.17   |
| Random Forest     | ~0.91   |
| Gradient Boosting | ~0.92   |

### Best Model

Gradient Boosting Regressor

### Final Performance

* Test R²: ~0.92
* Strong generalization capability
* Low prediction error
* Balanced bias-variance tradeoff

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-Learn
* SHAP
* Matplotlib
* Seaborn

### Deployment

* Flask
* HTML
* CSS

---

## Project Structure

```text
Dynamic-Pricing-System/
│
├── model/
│   ├── price_model.pkl
│   └── columns.pkl
│
├── templates/
│   └── index.html
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

---

## Key Learnings

* Feature engineering often contributes more to performance than algorithm selection.
* Proper evaluation requires cross-validation and multiple metrics.
* Tree-based models effectively capture non-linear relationships.
* Explainable AI improves model transparency and trustworthiness.
* End-to-end ML systems require both modeling and deployment considerations.

---

## Future Enhancements

* Real-time market demand integration
* Seasonal and surge pricing strategies
* Recommendation-based pricing optimization
* Cloud deployment and monitoring
* Automated retraining pipeline

---

## Author

Rohith Visarapu

Machine Learning | Software Engineering | AI & Data Science
