FROM grafana/grafana:latest

# Allow unsigned plugins - include all problematic plugins
ENV GF_PLUGINS_ALLOW_LOADING_UNSIGNED_PLUGINS=grafana-pyroscope-app,grafana-lokiexplore-app

# Add environment variables to address datasource issues
ENV GF_LOG_LEVEL=debug

# Use a different approach to create and set up our startup script
USER root
COPY ./backend/docker/grafana/start.sh /start.sh
RUN chmod +x /start.sh
USER grafana

# Override entrypoint to use our script
ENTRYPOINT ["/start.sh"]
