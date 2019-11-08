ARG pyver
FROM python:$pyver

# install
COPY installer.sh /tmp/installer.sh
RUN sh /tmp/installer.sh && \
    rm /tmp/installer.sh

COPY time_it.sh /time_it.sh
