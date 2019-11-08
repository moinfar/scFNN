FROM ufoym/deepo:tensorflow-py36-cu90

# install
COPY installer.sh /tmp/installer.sh
RUN sh /tmp/installer.sh && \
    rm /tmp/installer.sh

COPY time_it.sh /time_it.sh
