FROM alpine:latest

WORKDIR /app

# Copy the project files, respecting .dockerignore
COPY . /app/

# Install rsync for more efficient copying with pattern exclusion
RUN apk add --no-cache rsync

# Script to copy files with proper patterns
COPY backend/docker/repo-init/copy-repo.sh /copy-repo.sh
RUN chmod +x /copy-repo.sh

CMD ["/copy-repo.sh"]
