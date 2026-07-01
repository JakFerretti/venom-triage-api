# Venom Triage Production API

This repository contains the production-ready inference microservice for the Venom Triage clinical support system. The core LightGBM model (trained on 1,000,000 rows using Polars) is exposed via a high-performance REST API and fully containerized for seamless deployment.

## Project Ecosystem & Live Demos
To demonstrate a complete MLOps workflow, this project is split into three core components:
* 📓 **Model Training & Research:** [Kaggle Notebook - Polars & LightGBM (99.7% Accuracy)](https://www.kaggle.com/code/jacopoferretti/polars-lightgbm-99-7-venom-triage-api-docker) — *Includes large-scale data engineering with Polars, model training, and TensorFlow benchmarks.*
* 🌐 **Live Interactive API Endpoint:** [Hugging Face Spaces (Swagger UI)](https://jacklittleiron-venom-triage-api.hf.space/docs) — *Test the containerized FastAPI application live in production.*

---

## Tech Stack
* **Backend Framework:** FastAPI (Asynchronous, high-performance)
* **ML Engine:** LightGBM & Scikit-Learn Pipeline
* **Data Engineering (Training):** Polars (Rust-powered lazy evaluation)
* **Containerization:** Docker (`python:3.12-slim` base image)
* **Infrastructure:** Deployed on Hugging Face Spaces (Docker-backed CPU architecture)

---

## Architecture Overview

The pipeline decouples model training from real-time inference, encapsulating the production artifacts into a portable container:

[ Kaggle Notebook (Polars + LightGBM) ]
──> (Exports serialized artifacts)
venom_triage_pipeline.pkl & label_encoder_target.pkl
──> (Packaged inside)
[ Docker Container (FastAPI) ] ──> Live API (/predict)
