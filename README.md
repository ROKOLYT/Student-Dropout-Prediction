# Higher Education Student Dropout Prediction

**Machine Learning Zoomcamp — Midterm Project**

- **Problem:** Predicting student dropout in higher education to enable proactive early interventions.
- **Dataset:** [Predict Students' Dropout and Academic Success Dataset (UCI Machine Learning Repository)](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- **Model:** Tuned XGBoost Classifier (ROC-AUC: 0.9335, F1-Score: 0.8053)
- **Deployment:** Containerized REST API using FastAPI, Uvicorn, and Docker

---

## Table of Contents
1. [Problem Description](#problem-description)
2. [Dataset Overview](#dataset-overview)
3. [Exploratory Data Analysis (EDA) Summary](#exploratory-data-analysis-eda-summary)
4. [Model Training & Tuning Summary](#model-training--tuning-summary)
5. [Project Structure](#project-structure)
6. [How to Run](#how-to-run)
   - [1. Environment Setup](#1-environment-setup)
   - [2. Exploratory Notebook](#2-exploratory-notebook)
   - [3. Training the Model (`train.py`)](#3-training-the-model-trainpy)
   - [4. Running the Web Service (`predict.py`)](#4-running-the-web-service-predictpy)
   - [5. Testing the API](#5-testing-the-api)
   - [6. Docker Containerization](#6-docker-containerization)
   - [7. Cloud Deployment Guide](#7-cloud-deployment-guide)

---

## Problem Description

Higher education attrition is a critical challenge. When students drop out of university, there are severe consequences:
- **Students** incur debt and lost time without earning a credential.
- **Universities** lose tuition revenue and state funding.
- **Society** suffers from skill shortages and reduced social mobility.

### Early Warning System (EWS)
Traditional retention measures are often reactive: universities only find out a student has dropped out when they fail to register for the next academic year. By then, it is too late to intervene.

This project builds a machine learning **Early Warning System (EWS)** that evaluates student records (academic progress, tuition status, and demographics) after the 1st or 2nd semester. When the model detects high dropout probability, academic advisors and student support services can intervene with:
1. **Academic tutoring & study counseling** for students failing exams.
2. **Emergency financial aid & payment plans** for students behind on tuition fees.
3. **Workload and time-management advising** for mature students balancing work and study.

### Machine Learning Formulation
- **Target Variable:** The original dataset contains three classes: `Graduate`, `Dropout`, and `Enrolled`.
- **Formulation:** Binary classification:
  - `Dropout` = **1** (At-Risk — Requires Intervention)
  - `Graduate` & `Enrolled` = **0** (Retained / On-Track)
- **Class Distribution:** 32.1% dropouts vs 67.9% non-dropouts. Because of this imbalance, evaluation relies on **ROC-AUC** and **F1-Score** rather than plain accuracy.

---

## Dataset Overview

The dataset comes from the Polytechnic Institute of Portalegre in Portugal, published on the UCI Machine Learning Repository:
- **Instances:** 4,424 student records
- **Features:** 36 input variables spanning:
  - **Academic performance:** Curricular units credited, enrolled, evaluated, approved, and grades (for 1st and 2nd semesters).
  - **Financial health:** Tuition fees up to date, debtor status, scholarship holder.
  - **Demographics & Background:** Age at enrollment, gender, marital status, nationality, parents' qualifications and occupations.
  - **Macroeconomic indicators:** Unemployment rate, inflation rate, GDP.
- **Data File:** Included in repository as [`data.csv`].

---

## Exploratory Data Analysis (EDA) Summary

Key findings from [`project.ipynb`]:
1. **Tuition Payment Status is the strongest financial predictor:**
   - Students behind on tuition (`tuition_fees_up_to_date == 0`) have an **86.6% dropout rate**.
   - Students current on tuition have only a **24.7% dropout rate**.
2. **Debtor Status:** Students flagged as debtors experience a **62.0% dropout rate** vs 28.3% for non-debtors.
3. **Scholarships provide a strong buffer:** Only **12.2%** of scholarship holders drop out, compared to **38.7%** of non-scholarship students.
4. **2nd Semester Unit Completion:**
   - Passing 0 units in semester 2 $\rightarrow$ **83.6% dropout rate**.
   - Passing 6+ units $\rightarrow$ **7.4% dropout rate**.
5. **Age & Gender:** Older students (mature entrants over 23) and male students (45.1% dropout rate vs 25.1% for females) face higher attrition.
6. **Macroeconomic Indicators:** Unemployment rate, inflation, and GDP show near-zero correlation ($|r| < 0.05$) with individual dropout.

---

## Model Training & Tuning Summary

Four model families were trained and tuned on a 60% Train / 20% Validation / 20% Test stratified split:

| Model | Hyperparameters | Val ROC-AUC | Val F1 | Val Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| **XGBoost** | `eta=0.1, max_depth=3, min_child_weight=3` | **0.9332** | **0.7856** | **0.8723** |
| **Logistic Regression** | `solver='liblinear', C=1.0` | 0.9238 | 0.7933 | 0.8746 |
| **Random Forest** | `n_estimators=200, max_depth=8, min_samples_leaf=5` | 0.9217 | 0.7743 | 0.8689 |
| **Decision Tree** | `max_depth=4, min_samples_leaf=20` | 0.9013 | 0.7490 | 0.8599 |

### Final Evaluation on Held-Out Test Set (20% of data):
- **Test ROC-AUC:** **0.9335**
- **Test F1-Score:** **0.8053**

---

## Project Structure

```
07-midterm/
├── data.csv            # UCI student dataset (semicolon-separated)
├── project.ipynb       # Exploratory analysis, modeling, and parameter tuning
├── notebook.ipynb      # Symlink to project.ipynb
├── train.py            # Script to train final XGBoost model and save model.bin
├── predict.py          # FastAPI web service serving predictions
├── test_predict.py     # Script to test the FastAPI endpoint
├── model.bin           # Serialized DictVectorizer and XGBoost model
├── requirements.txt    # Project dependencies
├── Dockerfile          # Docker container definition
└── README.md           # Project documentation
```

---

## How to Run

### 1. Environment Setup

Clone this repository and create a virtual environment:

```bash
# Clone the repository
git clone https://github.com/ROKOLYT/Student-Dropout-Prediction
cd Student-Dropout-Prediction

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux / macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Exploratory Notebook

To review or run the notebook:

```bash
jupyter notebook project.ipynb
```

### 3. Training the Model (`train.py`)

To train the final model and export `model.bin`:

```bash
python train.py
```

Expected output:
```
Loading data from data.csv...
Dataset loaded: 4424 rows, 37 columns. Base dropout rate: 0.321
Training XGBoost model...
Validation Results:
  ROC-AUC:  0.9335
  F1-Score: 0.8053

Retraining final model on the entire dataset...
Final model successfully exported to 'model.bin'!
```

### 4. Running the Web Service (`predict.py`)

Start the FastAPI application with Uvicorn:

```bash
uvicorn predict:app --host 0.0.0.0 --port 8000 --reload
```

Interactive Swagger documentation will be available at:
`http://localhost:8000/docs`

### 5. Testing the API

In a separate terminal, test the running service:

```bash
python test_predict.py
```

Or send a direct request via `curl`:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age_at_enrollment": 32,
       "tuition_fees_up_to_date": 0,
       "debtor": 1,
       "scholarship_holder": 0,
       "gender": 1,
       "curricular_units_1st_sem_approved": 1,
       "curricular_units_1st_sem_grade": 10.0,
       "curricular_units_2nd_sem_approved": 0,
       "curricular_units_2nd_sem_grade": 0.0
     }'
```

Response:
```json
{"dropout_probability":0.8661,
"dropout_prediction":true,
"risk_level":"High Risk"}
```

---

## 6. Docker Containerization

### Build the Docker Image:

```bash
docker build -t student-dropout-api .
```

### Run the Docker Container:

```bash
docker run -it --rm -p 8000:8000 student-dropout-api
```

The service will now listen on `http://localhost:8000`. Test it with `python test_predict.py` or curl.

---

## 7. Cloud Deployment Guide

The containerized service can be deployed to any major cloud provider:

### Render / Fly.io
1. Push the repository to GitHub.
2. Link the repository to [Render](https://render.com) or [Fly.io](https://fly.io).
3. Select **Docker** environment. Render will automatically detect the `Dockerfile`, build the image, and assign a public URL (e.g. `https://student-dropout-api.onrender.com`) there is no code required.
