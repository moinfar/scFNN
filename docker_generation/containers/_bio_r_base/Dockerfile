ARG rver
# rocker/tidyverse has bioconductor and devtools installed
FROM rocker/tidyverse:$rver

# install
COPY installer.sh /tmp/installer.sh
RUN sh /tmp/installer.sh && \
    rm /tmp/installer.sh

COPY time_it.sh /time_it.sh
