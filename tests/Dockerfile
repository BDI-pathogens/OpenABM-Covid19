ARG PYTHON_VERSION=3.8.17
FROM python:${PYTHON_VERSION}-buster

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -qqy && apt-get install -qqy \
  make \
  clang \
  curl \
  gcc \
  g++ \
  libgsl23 \
  libgsl-dev \
  swig
RUN python3 -m pip install numpy pandas scipy pytest

COPY tests/requirements.txt /requirements.txt
RUN python3 -m pip install -r /requirements.txt

COPY . /OpenABM-Covid19

WORKDIR /OpenABM-Covid19
RUN make clean && make install

WORKDIR /OpenABM-Covid19
ENTRYPOINT ["/usr/local/bin/pytest", "-vv"]
