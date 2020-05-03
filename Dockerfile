FROM ubuntu:eoan
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -qqy && apt-get install -qqy \
  curl \
  python3.7-dev \
  python3-distutils \
  make \
  clang \
  gcc \
  g++ \
  libgsl23 \
  libgsl-dev \
  swig \
  valgrind
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.7
RUN pip3 install pytest numpy pandas scipy
WORKDIR /COVID19-IBM
ENTRYPOINT ["/bin/bash"]