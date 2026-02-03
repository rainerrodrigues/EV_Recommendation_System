# 🚗 EV Efficiency Recommendation System

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0+-05998b?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed?logo=docker&logoColor=white)](https://hub.docker.com/r/mohdmusheer/ev-efficiency-api-g1)
[![Render](https://img.shields.io/badge/Render-Deployed-success?style=flat&logo=render)](https://ev-recommendation-system.onrender.com)

An end-to-end Machine Learning solution designed to predict Electric Vehicle (EV) efficiency. This project integrates a **Scikit-learn** classifier with a **FastAPI** backend and a responsive frontend, all encapsulated within a **Docker** container for seamless deployment.





## 🚀 Live Demo
**🌐 Web App:** [View Live Deployment on Render](https://ev-recommendation-system.onrender.com)  
**📑 API Documentation:** [Swagger UI (Local)](http://localhost:8000/docs)

---

## ✨ Key Features
* **🔮 Predictive Intelligence:** Classifies EV efficiency as **High** or **Low**.
* **⚡ Comprehensive Data Inputs:** Processes battery, range, charging, and safety ratings.
* **🌐 Interactive UI:** Clean interface for manual data entry and instant results.
* **🚀 Production-Ready API:** Powered by FastAPI for high-performance handling.
* **🐳 Containerized Workflow:** Fully Dockerized to ensure cross-platform consistency.

---

## 🛠 Tech Stack

| Category | Tools |
| :--- | :--- |
| **Language** | Python 3.11 |
| **Machine Learning** | Scikit-learn, Pandas, Joblib |
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | HTML5, CSS3, JavaScript |
| **DevOps** | Docker, Docker Hub, Render |

---

## 📦 Getting Started (Docker)

To run this project locally without installing Python dependencies:

### 1. Pull the Image
```bash
docker pull mohdmusheer/ev-efficiency-api-g1
```
### 2. Run the container
```bash
docker run -d -p 8000:8000 mohdmusheer/ev-efficiency-api-g1
```
### 3. Access the app
Web UI: http://localhost:8000
API Docs: http://localhost:8000/docs

## 📂 Project Structure
```text
.
├── api/
│   └── api.py          # FastAPI backend logic
├── model/
│   └── ev_efficiency_classifier.pkl  # Trained ML Model
├── UI/
│   └── index.html      # Frontend interface
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container configuration
└── README.md           # Documentation
```

## 🔗 API Usage
```
POST /predict
```
Send a JSON payload to receive an efficiency prediction. 
```JSON
{
  "manufacturer": "Tesla",
  "model": "Model 3",
  "battery_kwh": 60,
  "range_km": 450,
  "charging_time_hr": 1.2,
  "top_speed_kmph": 225,
  "safety_rating": 5
}
```
Response:
1 → High Efficiency
0 → Low Efficiency

## 🎯 Target Use Cases
1. Academic Portfolios: Demonstrating end-to-end ML deployment.
2. Engineering Interviews: Highlights full-stack and hardware-adjacent software skills.
3. DevOps Practice: Template for CI/CD and containerization workflows.
