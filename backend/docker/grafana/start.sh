#!/bin/bash

# Debug information
echo "Checking provisioning directories..."
mkdir -p /etc/grafana/provisioning/datasources
mkdir -p /etc/grafana/provisioning/dashboards

# Datasource setup
echo "Creating datasource config..."
cat > /etc/grafana/provisioning/datasources/prometheus.yml << EOL
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
    uid: prometheus
EOL

# Dashboard provider setup
echo "Creating dashboard provider config..."
cat > /etc/grafana/provisioning/dashboards/provider.yml << EOL
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
      foldersFromFilesStructure: false
EOL

# Debug logs
echo "Listing provisioning directories:"
ls -la /etc/grafana/provisioning/
ls -la /etc/grafana/provisioning/datasources/
ls -la /etc/grafana/provisioning/dashboards/

echo "Starting Grafana..."
/run.sh
