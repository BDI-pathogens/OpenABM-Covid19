FROM python:3.7-buster

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -qqy && apt-get install -qqy \
  make \
  clang \
  wget \
  gcc \
  g++ \
  libgsl23 \
  libgsl-dev \
  swig

COPY tests/requirements.txt /requirements.txt
RUN python3 -m pip install -r /requirements.txt

COPY . /OpenABM-Covid19

# Install latest version of Swig
RUN wget https://sourceforge.net/projects/swig/files/swig/swig-4.0.1/swig-4.0.1.tar.gz && \
    tar xvfz swig-4.0.1.tar.gz && \
    cd swig-4.0.1 && \
    ./autogen.sh && ./configure && make && make install

WORKDIR /OpenABM-Covid19
RUN cd src && make clean && make install

ENTRYPOINT ["/usr/local/bin/pytest", "-s"]
