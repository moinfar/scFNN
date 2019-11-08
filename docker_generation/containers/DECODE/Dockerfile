FROM moinfar/bio-r-base:3.5.1

COPY installer.sh /tmp/installer.sh
RUN sh /tmp/installer.sh && \
    rm /tmp/installer.sh

COPY runner.sh /app/runner.sh
COPY extract_cols.sh /app/extract_cols.sh

WORKDIR /app
ENTRYPOINT ["bash", "runner.sh"]
