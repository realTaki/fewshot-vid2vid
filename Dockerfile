# Isolated migration candidate, not a validated GPU/training image.
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG PYTHON_VERSION=3.11.17
ARG PYTHON_SHA256=bfb74ad39efae27cda510f134ab408e00f9992c56851cfc0b1cdb5646da11599
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:/opt/python311/bin:$PATH \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX" \
    MAX_JOBS=2

# Do not replace Ubuntu's system Python or add a third-party Python PPA.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc-11 g++-11 ca-certificates curl xz-utils git cmake \
      ninja-build ffmpeg libx264-dev imagemagick libnss3-dev \
      libbz2-dev libffi-dev libgdbm-dev liblzma-dev libncursesw5-dev \
      libreadline-dev libsqlite3-dev libssl-dev uuid-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && curl --fail --show-error --location \
       "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz" \
       --output /tmp/python.tar.xz \
    && echo "${PYTHON_SHA256}  /tmp/python.tar.xz" | sha256sum --check --strict \
    && mkdir /tmp/python-src \
    && tar -xJf /tmp/python.tar.xz -C /tmp/python-src --strip-components=1 \
    && cd /tmp/python-src \
    && ./configure --prefix=/opt/python311 --enable-shared --with-ensurepip=install \
    && make -j2 && make altinstall \
    && echo /opt/python311/lib > /etc/ld.so.conf.d/python311.conf && ldconfig \
    && /opt/python311/bin/python3.11 -c "import ssl, sqlite3, bz2, lzma, ctypes, venv" \
    && /opt/python311/bin/python3.11 -m venv /opt/venv \
    && rm -rf /tmp/python-src /tmp/python.tar.xz

ENV CC=gcc-11 CXX=g++-11
ENTRYPOINT []
WORKDIR /work
COPY requirements.txt requirements-python311-cu118.lock /work/
COPY scripts /work/scripts
COPY src /work/src
RUN ./scripts/install.sh
WORKDIR /work/src
