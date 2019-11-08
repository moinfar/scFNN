FROM moinfar/bio-python:2.7

COPY installer.sh /tmp/installer.sh

RUN  sh /tmp/installer.sh && \
     rm /tmp/installer.sh && \
     mkdir /app/

COPY runner.py /app/runner.py
COPY runner.sh /app/runner.sh

WORKDIR /app
ENTRYPOINT ["bash", "runner.sh"]
