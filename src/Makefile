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


OBJS = utilities.o constant.o demographics.o params.o model.o individual.o main.o input.o network.o disease.o interventions.o hospital.o doctor.o nurse.o ward.o list.o

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
	rm -rf $(OBJS) $(EXE) $(SWIG_OUTPUT) $(R_PKG)

.c.o:
	$(C) $(CFLAGS) -c $< -o $@


# Generated source file (for R source package)
update_R_files:
	swig -r -o covid19_wrap_R.c covid19.i
	mv covid19.R ../R/covid19.R

  # Update NAMESPACE & DESCRIPTION files
	grep '`.*` = function(.*)' ../R/covid19.R | sed 's|`\(.*\)`.*|export( \1 )|' > ../NAMESPACE
  # 'Version' must be only numbers with '.' or '-': edit git-tag to match this
	sed \
	  -e "s/@VERSION@/$$(git describe --tags --abbrev=0 | cut -c 2-)/" \
	  -e "s/@GIT_TAG@/$$(git describe --tags)/" \
	  -e "s/@DATE@/$$(date +%Y-%m-%d)/" \
	  DESCRIPTION.in > ../DESCRIPTION


.PHONY: install dev all clean update_R_files
