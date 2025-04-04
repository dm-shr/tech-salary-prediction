git checkout main
git pull origin main
docker-compose down
docker-compose up -d --build fastapi prometheus grafana
