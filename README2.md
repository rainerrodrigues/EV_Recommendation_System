# EV Efficiency Prediction System 🚗⚡
![Python](https://img.shields.io/badge/Python-3.9-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-lightblue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-yellowgreen)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-9932CC)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)

## Project Overview
We built an Electric Vehicle (EV) efficiency prediction system that predicts whether a given EV is High Efficiency or Low Efficiency based on specifications such as battery size, range, charging speed, acceleration, top speed, and safety ratings.

This project was designed as a portfolio-grade machine learning application with a full-stack web interface and Dockerized deployment, making it ready for real-world use.

## How It Started
- We wanted to combine EV data analysis with web deployment for a portfolio project.
- Initial goal: predict efficiency class from EV specifications using machine learning models.
- We collected datasets of EV specifications and cleaned the data for training.
- Iteratively built and tested ML models, integrated them with a FastAPI backend, and created an interactive frontend.

## What We Built
### 1. Machine Learning Model
- Model trained with Scikit-learn
- Predicts High Efficiency (1) or Low Efficiency (0)
- Features used: battery, range, charging time, speed, safety rating, etc.
- Achieved accuracy: 90.34% on test data 

### 2. Backend
- FastAPI application with Swagger documentation
- REST API endpoint: `POST /predict`
- Containerized with Docker

### 3. Frontend
- Interactive web UI built with HTML, CSS, JavaScript
- Users can input EV specs and get instant predictions

### 4. Deployment
- Fully Dockerized for easy deployment

## Model Comparison
| Model               | Test Accuracy | Test ROC-AUC | Test Precision | Test Recall | Test F1 | Overfitting Gap |
| ------------------- | ------------- | ------------ | -------------- | ----------- | ------- | --------------- |
| Logistic Regression | 0.803         | 0.884        | 0.805          | 0.801       | 0.803   | 0.001           |
| Random Forest       | 0.804         | 0.881        | 0.812          | 0.792       | 0.801   | 0.005           |
| XGBoost             | 0.804         | 0.884        | 0.808          | 0.797       | 0.802   | 0.006           |
| LightGBM            | 0.803         | 0.884        | 0.808          | 0.796       | 0.802   | 0.003           |

## Team Members & Contributions
| Name | Contribution |
|------|----------------|
| Aarushi Jain | ML model development: training multiple models, evaluating metrics, saving best model, documentation|
| Sowmya Reddy | |
| AL Musheer | |
| Sai Chaithanya Reddy Yudururi| |

## Tech Stack
- Machine Learning: Scikit-learn, Pandas, Joblib
- Frontend: HTML, CSS, JavaScript
- Backend: FastAPI, Uvicorn
- Containerization: Docker
- Python: 3.11

## Repo Links
- GitHub: https://github.com/rainerrodrigues/EV_Recommendation_System
- Docker Hub: https://hub.docker.com/r/mohdmusheer/ev-efficiency-api-g1