# How to Run the Dynamic Pricing Prediction System

## Prerequisites

Ensure the following software is installed:

* Python 3.10+
* pip
* Git (optional)

---

## Clone the Repository

```bash
git clone <repository-url>
cd Dynamic-Pricing-System
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is unavailable:

```bash
pip install pandas numpy scikit-learn flask shap matplotlib seaborn
```

---

## Train the Model

Run the training script to generate model files:

```bash
python train_model.py
```

This will create:

```text
model/
├── price_model.pkl
└── columns.pkl
```

---

## Verify Project Structure

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
├── README.md
└── HOW_TO_RUN.md
```

---

## Start the Flask Application

```bash
python app.py
```

Expected output:

```text
* Running on http://127.0.0.1:5000
```

---

## Open the Application

Open your browser and navigate to:

```text
http://127.0.0.1:5000
```

---

## Sample Input

```text
Room Size          : 30
Furnishing Level   : 3
AC                 : 1
Availability       : 200
Reviews            : 50
Reviews Per Month  : 2
Location           : Manhattan
```

---

## Output

The application will display:

* Predicted Property Price
* Pricing Status
* SHAP-Based Feature Explanation
* Top Factors Influencing Prediction

---

## Troubleshooting

### Model File Not Found

Error:

```text
No such file or directory: model/price_model.pkl
```

Solution:

```bash
python train_model.py
```

---

### Missing Library

Error:

```text
ModuleNotFoundError
```

Solution:

```bash
pip install -r requirements.txt
```

---

### Port Already In Use

Error:

```text
Address already in use
```

Solution:

Terminate the running Flask process or use another port.

---

## Project Workflow

```text
Dataset
   ↓
Preprocessing
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Model Evaluation
   ↓
SHAP Explainability
   ↓
Flask Deployment
   ↓
Real-Time Prediction
```
