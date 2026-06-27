# Venom Triage Production API 🐍🐳

This repository contains the production-ready inference microservice for the Venom Triage clinical support system. The core LightGBM model (trained on 1,000,000 rows using Polars) is exposed via a high-performance REST API and fully containerized.

## 🛠️ Tech Stack
* **Backend Framework:** FastAPI (Asynchronous, high-performance)
* **ML Engine:** LightGBM & Scikit-Learn Pipeline
* **Containerization:** Docker (`python:3.12-slim`)
* **Deployment:** Hugging Face Spaces (Docker-backed infrastructure)

## 🌐 Live Demo & API Documentation
The API is live and publicly accessible. You can test the endpoints in real-time using the interactive Swagger UI:
👉 **[Live Swagger UI Endpoint](https://jacklittleiron-venom-triage-api.hf.space/docs)**

## 🚀 How to Run Locally (Docker)

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/TuoUsername/venom-triage-api.git](https://github.com/TuoUsername/venom-triage-api.git)
   cd venom-triage-api
