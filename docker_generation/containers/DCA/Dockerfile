FROM moinfar/bio-python:3.6

COPY installer.sh /tmp/installer.sh
RUN sh /tmp/installer.sh && \
    rm /tmp/installer.sh && \
    mkdir /app/

COPY runner.sh /app/runner.sh
COPY tsv_to_csv.py /app/tsv_to_csv.py

WORKDIR /app
ENTRYPOINT ["bash", "runner.sh"]
