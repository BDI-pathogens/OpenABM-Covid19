FROM python:3.7-buster

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
RUN python3.7 -m pip install numpy pandas scipy pytest

COPY tests/requirements.txt /requirements.txt
RUN python3 -m pip install -r /requirements.txt

COPY . /OpenABM-Covid19

WORKDIR /OpenABM-Covid19
RUN make clean && make install

WORKDIR /OpenABM-Covid19
ENTRYPOINT ["/usr/local/bin/pytest", "-s"]
