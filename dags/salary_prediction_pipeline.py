import os
from datetime import timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from dotenv import load_dotenv

import docker


# Add debug logging
def print_config():
    print("DAG Directory:", os.getenv("AIRFLOW__CORE__DAGS_FOLDER"))
    print("Current Directory:", os.getcwd())
    print("Repository Path:", os.getenv("REPO_PATH"))
    print("Files in DAG folder:", os.listdir(os.getenv("AIRFLOW__CORE__DAGS_FOLDER", ".")))


# Load environment variables
load_dotenv()

default_args = {
    "owner": os.getenv("AIRFLOW_OWNER", "airflow"),
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email": [os.getenv("NOTIFICATION_EMAIL", "your-email@example.com")],
    "email_on_failure": os.getenv("EMAIL_ON_FAILURE", "True").lower() == "true",
    "retries": int(os.getenv("DAG_RETRIES", 2)),
    "retry_delay": timedelta(minutes=int(os.getenv("RETRY_DELAY_MINUTES", 5))),
}


# Configuration
REPO_PATH = os.getenv("REPO_PATH", "/opt/airflow/repo")  # Keep REPO_PATH at the root


with DAG(
    "salary_prediction_pipeline",
    default_args=default_args,
    description="Weekly salary prediction pipeline with DVC",
    schedule_interval="0 0 * * MON",
    catchup=False,
    tags=["ml", "production", "dvc"],
) as dag:

    # Add debug task
    debug_config = PythonOperator(task_id="debug_config", python_callable=print_config, dag=dag)

    # Initialize Git and DVC - Modified for both scenarios
    init_git_dvc = BashOperator(
        task_id="init_git_dvc",
        bash_command="""
            echo "Initializing Git and DVC..."
            set -e
            export HOME=/home/airflow
            export PATH="/home/airflow/.local/bin:$PATH"

            cd ${REPO_PATH}

            if [ -d ".git" ]; then
                # Configure git
                git config --global credential.helper store
                git config --global url."https://oauth2:${GITHUB_TOKEN}@github.com".insteadOf "https://github.com"
                git config pull.rebase false

                # Clean up any existing merge conflicts or leftover state
                git reset --hard
                git clean -fd

                # OPTION 1: Use current state (for development)
                if [ "${USE_CURRENT_STATE}" = "true" ]; then
                    echo "Using current state..."
                    git checkout ${GIT_BRANCH} || git checkout -b ${GIT_BRANCH}
                # OPTION 2: Start from base state (for production)
                else
                    echo "Using base state..."
                    if git fetch origin && git checkout base-state-test 2>/dev/null; then
                        echo "Using base-state-test tag"
                    else
                        echo "Base tag not found, using main branch"
                        git checkout main
                    fi
                    git checkout -B ${GIT_BRANCH}
                fi

                # Clean data directory but keep DVC cache
                find backend/data/preprocessed/merged -type f -name "*.csv" -delete
                find backend/data/preprocessed/merged -type f -name "*.dvc" -delete
            else
                # Fresh clone
                git clone ${GIT_REPO_URL} .
                if [ "${USE_CURRENT_STATE}" != "true" ]; then
                    if git fetch origin && git checkout base-state-test 2>/dev/null; then
                        echo "Using base-state-test tag"
                    else
                        echo "Base tag not found, using main branch"
                        git checkout main
                    fi
                fi
                git checkout -b ${GIT_BRANCH}

                # Initialize DVC
                dvc init
                dvc remote add -d storage "${DVC_REMOTE_URL}"
                dvc remote modify storage endpointurl "${MLFLOW_S3_ENDPOINT_URL}"
            fi

            # Debug info
            echo "Current branch: $(git branch --show-current)"
            git status
            dvc status
            dvc remote list
        """,
        cwd=REPO_PATH,
        env={
            "REPO_PATH": REPO_PATH,
            "DVC_REMOTE_URL": os.getenv("DVC_REMOTE_URL"),
            "GITHUB_TOKEN": os.getenv("GITHUB_TOKEN"),
            "GIT_REPO_URL": os.getenv("GIT_REPO_URL"),
            "GIT_BRANCH": os.getenv("GIT_BRANCH"),
            "HOME": "/home/airflow",
            "PATH": f"/home/airflow/.local/bin:{os.environ.get('PATH', '')}",
            "MLFLOW_S3_ENDPOINT_URL": os.getenv("MLFLOW_S3_ENDPOINT_URL"),
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "USE_CURRENT_STATE": os.getenv("USE_CURRENT_STATE", "false"),  # Add this line
        },
    )

    # Run scraping with DVC
    scrape_data = BashOperator(
        task_id="scrape_data",
        bash_command="""
            # set -e
            # export HOME=/home/airflow
            # export PATH="/home/airflow/.local/bin:$PATH"
            cd ${REPO_PATH}
            cd backend
            python -m src.scraping.main
        """,
        cwd=REPO_PATH,
        env={
            "WEEK": '{{ execution_date.strftime("%V") }}',
            "YEAR": '{{ execution_date.strftime("%Y") }}',
        },
    )

    # Preprocess with DVC tracking
    preprocess_data = BashOperator(
        task_id="preprocess_data",
        bash_command="""
            # Run preprocessing and capture output path
            cd ${REPO_PATH}
            cd backend
            OUTPUT=$(python -m src.preprocessing.main)
            CSV_PATH=$(echo "$OUTPUT" | grep "MERGED_CSV_PATH=" | cut -d'=' -f2)

            # Save path for next task
            echo "CSV_PATH=${CSV_PATH}" > /tmp/merged_csv_path

            echo "Generated CSV file: ${CSV_PATH}"
        """,
        cwd=REPO_PATH,
    )

    dvc_add_merged = BashOperator(
        task_id="dvc_add_merged",
        bash_command="""
            set -e
            export HOME=/home/airflow
            export PATH="/home/airflow/.local/bin:$PATH"
            cd ${REPO_PATH}
            cd backend

            # Get CSV path and ensure it's relative to repo root
            source /tmp/merged_csv_path
            echo "Working with file: ${CSV_PATH}"

            # Configure git
            git config user.email "${GIT_EMAIL}" || true
            git config user.name "${GIT_NAME}" || true
            git config --global core.hooksPath /dev/null

            echo "Adding file to DVC..."
            python -m dvc add "${CSV_PATH}" -v

            echo "DVC status after add:"
            python -m dvc status
        """,
        cwd=REPO_PATH,
        env={
            "REPO_PATH": REPO_PATH,
            "GIT_EMAIL": os.getenv("GIT_EMAIL"),
            "GIT_NAME": os.getenv("GIT_NAME"),
            "DVC_REMOTE_URL": os.getenv("DVC_REMOTE_URL"),
            "MLFLOW_S3_ENDPOINT_URL": os.getenv("MLFLOW_S3_ENDPOINT_URL"),
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "AWS_ALLOW_HTTP": "true",
            "DVC_CACHE_UMASK": "002",
            "PATH": f"/home/airflow/.local/bin:{os.environ.get('PATH', '')}",
        },
    )

    dvc_push_merged = BashOperator(
        task_id="dvc_push_merged",
        bash_command="""
            set -e
            export HOME=/home/airflow
            export PATH="/home/airflow/.local/bin:$PATH"
            cd ${REPO_PATH}
            cd backend

            # Get CSV path from previous task
            source /tmp/merged_csv_path
            echo "Processing file: ${CSV_PATH}"

            # Configure git
            git config user.email "${GIT_EMAIL}" || true
            git config user.name "${GIT_NAME}" || true
            git config --global core.hooksPath /dev/null

            # Push to DVC first
            python -m dvc push -v

            # Git operations
            if [ -f "${CSV_PATH}.dvc" ]; then
                TAG_NAME=$(basename ${CSV_PATH} .csv)_$(date +%H%M%S)

                # Add only the specific DVC file and create tag
                git add -f "${CSV_PATH}.dvc"

                if ! git diff --staged --quiet; then
                    # Create commit (needed for tag) but don't push it
                    git commit -m "Add ${TAG_NAME}"

                    # Create and push tag only
                    git tag -a "${TAG_NAME}" -m "Data version: ${TAG_NAME}"
                    git push origin "${TAG_NAME}"

                    echo "Successfully created and pushed tag ${TAG_NAME}"

                    # Reset the commit to keep the branch clean
                    git reset HEAD~1 --hard
                fi
            else
                echo "Error: DVC file ${CSV_PATH}.dvc not found"
                exit 1
            fi
        """,
        cwd=REPO_PATH,
        env={
            "REPO_PATH": REPO_PATH,
            "GIT_BRANCH": os.getenv("GIT_BRANCH"),
            "GIT_EMAIL": os.getenv("GIT_EMAIL"),
            "GIT_NAME": os.getenv("GIT_NAME"),
            "GITHUB_TOKEN": os.getenv("GITHUB_TOKEN"),
            "DVC_REMOTE_URL": os.getenv("DVC_REMOTE_URL"),
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION"),
        },
    )

    # Build features
    build_features = BashOperator(
        task_id="build_features",
        bash_command="""
            cd ${REPO_PATH}
            cd backend
            python -m src.feature_building.main
        """,
        cwd=REPO_PATH,
    )

    # Train models
    train_models = BashOperator(
        task_id="train_models",
        bash_command="""
            cd ${REPO_PATH}
            cd backend
            python -m src.training.main
        """,
        cwd=REPO_PATH,
        env={
            # Add MLflow environment variables
            "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI"),
            "MLFLOW_S3_ENDPOINT_URL": os.getenv("MLFLOW_S3_ENDPOINT_URL"),
            "MLFLOW_BUCKET_NAME": os.getenv("MLFLOW_BUCKET_NAME"),
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "MLFLOW_S3_IGNORE_TLS": "true",
        },
    )

    # After successful run, trigger model update in inference service
    notify_streamlit = PythonOperator(
        task_id="notify_streamlit",
        python_callable=lambda: restart_docker_container("streamlit"),
        dag=dag,
    )

    notify_fastapi = PythonOperator(
        task_id="notify_fastapi",
        python_callable=lambda: restart_docker_container("fastapi"),
        dag=dag,
    )

    def restart_docker_container(container_name):
        """Restarts a Docker container using the Docker API."""
        try:
            client = docker.from_env()
            container = client.containers.get(container_name)
            container.restart()
            print(f"Container '{container_name}' restarted successfully.")
        except docker.errors.NotFound:
            print(f"Container '{container_name}' not found.")
        except docker.errors.APIError as e:
            print(f"Error restarting container '{container_name}': {e}")

    # Cleanup after successful inference trigger
    cleanup_files = BashOperator(
        task_id="cleanup_files",
        bash_command=r"""
            find . -type f \( -name "*.csv" -o -name "*.pt" -o -name "*.pkl" -o -name "*.cbm" -o -name "*.npy" \) -delete
            rm -rf backend/data/preprocessed/merged/*
            rm -rf .dvc/cache
            rm -rf .dvc/tmp
        """,
        cwd=REPO_PATH,
    )
    (
        debug_config
        >> init_git_dvc
        >> scrape_data
        >> preprocess_data
        >> dvc_add_merged
        >> dvc_push_merged
        >> build_features
        >> train_models
        >> notify_streamlit
        >> notify_fastapi
        >> cleanup_files
    )
