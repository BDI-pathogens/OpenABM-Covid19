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


OBJS = utilities.o constant.o demographics.o params.o model.o individual.o main.o input.o network.o disease.o interventions.o hospital.o doctor.o nurse.o ward.o list.o strain.o

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
SWIG_OUTPUT = covid19_wrap.o covid19_wrap.c covid19.py _covid19.cpython-37m-darwin.so build covid19.egg-info

# To compile
install: $(OBJS)
install: all;
	swig -python covid19.i
	$(PYTHON) -m pip install $(PIP_FLAGS) .

dev: PIP_FLAGS += -e
dev: install;

all: $(OBJS)
	$(C) -L$(LIB) -o $(EXE) $(OBJS) $(LFLAGS)

clean:
	$(PYTHON) -m pip uninstall -y covid19
	rm -rf $(OBJS) $(EXE) $(SWIG_OUTPUT)

.c.o:
	$(C) $(CFLAGS) -c $< -o $@

.PHONY: install dev all clean
