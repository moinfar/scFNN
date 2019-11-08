FROM moinfar/bio-python:3.6

COPY installer.sh /tmp/installer.sh
RUN sh /tmp/installer.sh && \
    rm /tmp/installer.sh

COPY runner.sh /app/runner.sh
COPY transpose_it.py /app/transpose_it.py

WORKDIR /app
ENTRYPOINT ["bash", "runner.sh"]
