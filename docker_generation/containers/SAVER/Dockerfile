FROM moinfar/bio-r-base:3.5.1

COPY installer.R /tmp/installer.R
RUN Rscript /tmp/installer.R && \
    rm /tmp/installer.R && \
    mkdir /app/

COPY runner.R /app/runner.R
COPY runner.sh /app/runner.sh

WORKDIR /app
ENTRYPOINT ["bash", "runner.sh"]
