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
SWIG_OUTPUT_PY = covid19_wrap.o covid19_wrap.c covid19.py _covid19.cpython-37m-darwin.so build covid19.egg-info
SWIG_OUTPUT_R = covid19_wrap_R.c covid19.R
SWIG_OUTPUT = $(SWIG_OUTPUT_PY) $(SWIG_OUTPUT_R)

# R source package tarball
R_PKG = covid19.tar.gz
# Define the tar's file structure using --xform expressions.
R_TAR_FLAGS  = --xform='s|\(.*\).R$$|R/\1.R|'# Put R files in 'R/' subdir
R_TAR_FLAGS += --xform='s|\(.*\).\([ch]\)$$|src/\1.\2|'# Put C files in 'src/' subdir
R_TAR_FLAGS += --xform='s|^Makevars$$|src/Makevars|'# Put Makevars in 'src/' subdir
R_TAR_FLAGS += --xform='s|\(.*\)$$|covid19/\1|'# Put all files in covid19/ subdir
R_PKG_CONTENT = DESCRIPTION NAMESPACE Makevars $(filter-out main.h,$(OBJS:%.o=%.c) $(OBJS:%.o=%.h)) structure.h $(SWIG_OUTPUT_R)


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


# Generating an R source package
covid19_wrap_R.c:
	swig -r -o covid19_wrap_R.c covid19.i
covid19.R: covid19_wrap_R.c

NAMESPACE: covid19.R
  # Search for generated wrapper functions in .R file, and add them to NAMESPACE
	grep '`.*` = function(.*)' covid19.R | sed 's|`\(.*\)`.*|export( \1 )|' > NAMESPACE
DESCRIPTION: DESCRIPTION.in
  # 'Version' must be only numbers with '.' or '-': edit git-tag to match this
	sed \
	  -e "s/@VERSION@/$$(git describe --tags --abbrev=0 | cut -c 2-)/" \
	  -e "s/@GIT_TAG@/$$(git describe --tags)/" \
	  -e "s/@DATE@/$$(date +%Y-%m-%d)/" \
	  DESCRIPTION.in > DESCRIPTION

$(R_PKG): $(R_PKG_CONTENT)
	tar -czf $(R_PKG) $(R_TAR_FLAGS) $(R_PKG_CONTENT)


.PHONY: install dev all clean
