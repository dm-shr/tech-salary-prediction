# Tech Salary Prediction Service

With the upcoming EU Pay Transparency Directive in 2026, understanding and predicting fair salary ranges is becoming increasingly crucial for businesses. This project offers a valuable tool for ensuring equitable compensation practices by providing data-driven salary predictions based on job roles and descriptions.

The service predicts salaries for IT job positions based on job descriptions and related metadata.

The architecture involves a blended model of CatBoost and Transformer models for prediction that achieves state-of-the-art prediction accuracy.

The stack overview:
- **Python**, **FastAPI**, **Docker** for backend;
- **Airflow** for orchestration;
- **MLflow** and **DVC** for tracking and data versioning;
- **AWS S3** for storage;
- **Next.js**, **TypeScript**, and **Vercel** for the frontend.

The project demo is now live! Check it out at this [**link.**](https://tech-salary-prediction.vercel.app/)


## Project Structure

```
dags/                 # Airflow DAGs for pipeline automation

.github/workflows     # GitHub Actions CI/CD workflows

backend/              # Python backend service
├── configs/          # Configuration files
├── src/              # Source code
│   ├── feature_building   # Feature engineering
│   ├── monitoring         # Monitoring tools
│   ├── preprocessing      # Data preprocessing scripts
│   ├── scraping           # Web scraping scripts
│   ├── training           # Model training scripts
│   │   ├── catboost       # CatBoost model training
│   │   └── transformer  # Transformer model training
│   └── utils              # Utility functions
├── tests/            # Unit tests
└── docker/           # Docker configuration

frontend/             # Next.js frontend application
├── src/
│   ├── app/          # Next.js app directory
│   ├── components/
│   └── utils/        # Frontend helper functions

notebooks/            # Jupyter notebooks for experiments
```

## Directory Trees

### Notebooks

```
notebooks
├── baselines              # Baseline models for comparison
│   ├── average.ipynb      # Simple average salary prediction
│   ├── bi_gru_cnn.ipynb   # Bi-directional GRU-CNN model
│   ├── catboost.ipynb     # CatBoost model
│   └── transformer.ipynb  # Transformer model
├── experiments            # Experimentation with different models and configurations
│   ├── double-bert-huber-loss-tsdae.ipynb       # pre-tuning the model with TSDAE
│   ├── double-bert-mse-loss.ipynb                # two separate encoders for textual features
│   ├── huber-loss-cross-attention-e5-mean-pooling.ipynb  # cross-attention for textual features, e5 model
│   ├── huber-loss-cross-attention.ipynb          # cross-attention for textual features, smaller model
│   ├── huber-loss-e5-mean-pooling.ipynb         # mean pooling instead of [CLS] pooling
│   ├── huber-loss-mask-pooling.ipynb             # pooling via [MASK] token embedding
│   └── huber-loss.ipynb                          # use of Huber loss instead of MSE
```


## Stack

### Backend
- ML:
  - CatBoost
  - PyTorch
  - Transformers
  - MLflow

- Data:
  - AWS S3
  - DVC for data tracking

- Orchestration:
  - docker-compose
  - Airflow

- Deploy:
  - Docker
  - GitHub Actions CI/CD
  - Python 3.10+


### Frontend

- Next.js
- TypeScript
- Vercel deployment
- Tailwind CSS
- Radix UI

## Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker & Docker Compose
- AWS credentials (configured with `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` in `.env`) for S3 storage
- Google Gemini API key (configured with `GEMINI_API_KEY` in `.env`) for job description translation
- Git access configured for CI/CD (set `GIT_EMAIL`, `GIT_NAME`, `GITHUB_TOKEN`, and `GIT_REPO_URL` in `.env`)

### Backend Setup - Dependency Management

Before deploying the backend, choose one of the following dependency setups:

- **Blended Model (CatBoost & Transformer)**:
  1.  Enable the transformer model in `backend/configs/params.yaml` by setting `models.transformer.enabled` to `true`.
  2.  Replace the base `requirements.txt` file with the blended model requirements:
      ```bash
      cp backend/requirements_blended.txt backend/requirements.txt
      ```

- **CatBoost Only**:
  1.  Disable the transformer model in `backend/configs/params.yaml` by setting `models.transformer.enabled` to `false`.
  2.  Use the CatBoost-only requirements:
      ```bash
      cp backend/requirements_catboost.txt backend/requirements.txt
      ```

- **No ML Debug (Minimal Dependencies)**:
  1.  Use the minimal dependencies for a lighter setup (no ML models):
      ```bash
      cp backend/requirements_no_ml.txt backend/requirements.txt
      ```

### Backend Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd tech-salary-prediction
```

2. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your AWS credentials and other configurations
```

3. Start all services using Docker Compose:

```bash
docker-compose up -d
```

### Frontend Setup

1.  Install dependencies:

```bash
cd frontend
npm install
```

2.  Create frontend environment variables:

```bash
cp frontend/.env.example frontend/.env.local
# Edit .env.local with your configurations
```

3. Start the development server:

```bash
npm run dev
```

### Vercel Deployment

The frontend is deployed using Vercel. You can deploy your own version by following the instructions on the [Vercel website](https://vercel.com/docs).

1.  Install Vercel CLI:

```bash
npm install -g vercel
```

2.  Deploy the frontend:

```bash
cd frontend
vercel
```

### DAG and ML Training Pipeline

1.  After starting all services with Docker Compose, access the Airflow web UI at http://localhost:8080.
2.  Log in with the default credentials (admin/admin).
3.  Find the `tech_salary_prediction` DAG and unpause it.
4.  Trigger the DAG to start the data processing and model training pipeline.
5.  Monitor the DAG run in the Airflow UI.
6.  Track the experiments and model performance using MLflow at http://mlflow:5000 (default, can be set with `MLFLOW_TRACKING_URI`, `MLFLOW_S3_ENDPOINT_URL`, and `MLFLOW_EXPERIMENT_NAME` in `.env`)


### Frontend Usage (locally)
1. Access the web interface at http://localhost:3000
2. Enter job details:
    - Click "Predict Salary" to get the estimated salary



## Results Summary

The solution architecture involves a blended model that combines CatBoost and Transformer architectures. The data processing and model training pipeline is orchestrated by Airflow, and the experiments are tracked using MLflow.

Metrics are reported as a mean value ± 95% confidence intervals across three random seeds. Overall state-of-the-art results are in **bold**, while the best results for a solo transformer model are in *italics*.

| Experiment | R² score | MAE |
|------------|----------|-----|
| Baselines | | |
| By average | 0.000 ± 0.000 | 0.513 ± 0.002 |
| Bi-GRU-CNN | 0.652 ± 0.012 | 0.288 ± 0.007 |
| CatBoost | 0.734 ± 0.005 | 0.248 ± 0.004 |
| rubert-tiny-turbo (29M) | 0.645 ± 0.027 | 0.289 ± 0.012 |
| Modifications | | |
|------------|----------|-----|
| Double rubert-tiny-turbo | 0.643 ± 0.024 | 0.291 ± 0.013 |
| + Huber loss + TSDAE | 0.657 ± 0.056 | 0.285 ± 0.024 |
|------------|----------|-----|
| rubert-tiny-turbo + Huber loss | 0.655 ± 0.035 | 0.286 ± 0.016 |
| + extra [MASK] pooling | 0.599 ± 0.034 | 0.313 ± 0.015 |
| *+ cross-attention* | *0.671 ± 0.027* | *0.279 ± 0.014* |
|------------|----------|-----|
| multilingual-e5-small (118M) + Huber loss | 0.723 ± 0.024 | 0.254 ± 0.013 |
| *+ cross-attention* | *0.729 ± 0.017* | *0.251 ± 0.009* |
| **+ CatBoost** | **0.770 ± 0.001** | **0.229 ± 0.003** |
| + cross-attention + CatBoost | 0.769 ± 0.014 | 0.229 ± 0.01 |


## License
This project is licensed under the MIT License - see the LICENSE file for details.
