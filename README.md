# Final Report Submission

## Overview

This repository contains a hybrid recommendation system built as part of the Saras Lab final report submission. The project includes:

- Data exploration and feature engineering labs
- Content-based and collaborative filtering recommendation labs
- ALS and FAISS-based personalization labs
- Hybrid routing and A/B testing experiments
- MLflow and FastAPI deployment for production-ready inference

The main API entrypoint is `app.py`, which loads recommendation artifacts and serves hybrid recommendations via FastAPI.

## Repository structure

- `app.py` - FastAPI application and recommender inference service
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container image definition
- `docker-compose.yml` - Local compose setup for app and optional Redis
- `data/` - Data files required to run the experiments and API
- `output/` - Generated outputs, reports, and lab artifacts
- `scripts/` - Lab-specific modules and notebooks organized by module and lab

## Environment

This project uses environment variables defined in a `.env` file.
The `.env` file is not committed to source control; add it locally before running the application.

Example `.env` contents:

```env
# Redis configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_TTL=3600

# Artifact and data paths
ARTIFACTS_DIR=/path/to/data
DATA_DIR=/path/to/data

# Recommendation configuration
TOP_K=10
INTERACTION_THRESHOLD=3
LOG_LEVEL=INFO

# FastAPI / deployment
API_URL=http://localhost:8000
```

> Note: Update `ARTIFACTS_DIR` and `DATA_DIR` to point to the local path where the `data/` folder is stored.

## Setup

1. Create and activate a Python virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the repository root using the example values above.

4. Download the `data/` folder from the Gmail Drive share and place it in the repository root.

https://drive.google.com/drive/folders/1Pg2QkE-IHmfTHlm8vwesL7GoDh6Le15a?usp=sharing

## Data folder instructions

The `data/` folder is required for training, evaluation, and model artifact loading. It is intentionally excluded from version control.

- Download the `data/` folder from the Gmail Drive shared location: https://drive.google.com/drive/folders/1Pg2QkE-IHmfTHlm8vwesL7GoDh6Le15a?usp=sharing
- Place the downloaded folder here:

```text
Final-Report-Submission/
```

The project expects the following files at minimum:

- `data/events.csv`
- `data/category_tree.csv`
- `data/item_properties_part1.csv`
- `data/item_properties_part2.csv`
- `data/user_features_baseline.csv`
- `data/user_features_engineered.csv`

If any additional data files are required, they will be referenced in the lab scripts under `scripts/module*`.

## Running the API

Start the FastAPI service locally:

```bash
uvicorn app:app --reload
```

Then visit:

```text
http://127.0.0.1:8000/docs
```

## Docker

To build and run the container:

```bash
docker build -t final-report-app .
docker run --env-file .env -p 8000:8000 final-report-app
```

If you use `docker-compose`, ensure the local `.env` file is available and data paths are mounted correctly.

## Notes for enterprise deployment

- Keep the `.env` file private and do not commit it.
- The `data/` directory and serialized artifacts should be provisioned through secure enterprise storage.
- The application currently loads pickle artifacts and CSV data from `ARTIFACTS_DIR`.
- Redis is optional, but recommended for caching if available.

## Important

- Do not commit large binary data or serialized artifacts to Git.
- The repository already ignores `data/`, `*.csv`, `*.pkl`, and `*.parquet` in `.gitignore`.
- Use the enterprise Gmail Drive share to obtain the full dataset and place it in the local `data/` folder.
