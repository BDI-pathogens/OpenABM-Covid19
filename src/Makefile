# For use of the icc compiler
ifeq ($(compiler),icc)
    C = icc
else
    C = gcc
endif

ifeq (, $(shell which python3))
	PYTHON = python
else
	PYTHON = python3
endif

PIP_FLAGS := --upgrade

OBJS = utilities.o constant.o demographics.o params.o model.o individual.o main.o input.o network.o disease.o interventions.o
GSLFLAGS= -lgsl -lgslcblas -lm -O3
LFLAGS = $(GSLFLAGS)

# Name of executable
_EXE = covid19ibm.exe
EXE = $(_EXE)

INC = /usr/local/include
LIB = /usr/local/lib

# Compilation options and libraries to be used
CFLAGS = -g -Wall -fmessage-length=0 -I$(INC) -O0

# Swig's output
SWIG_OUTPUT = openabm_wrap.o openabm_wrap.c openabm.py _openabm.cpython-37m-darwin.so build

# To compile
swig-all:
	swig -python openabm.i
	$(PYTHON) -m pip install $(PIP_FLAGS) .

all: $(OBJS)
	$(C) -L$(LIB) -o $(EXE) $(OBJS) $(LFLAGS)

clean:
	rm -rf $(OBJS) $(EXE) $(SWIG_OUTPUT)
	$(PYTHON) -m pip uninstall -y openabm

.c.o:
	$(C) $(CFLAGS) -c $< -o $@

.PHONY: swig-all all clean
