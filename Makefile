# For use of the icc compiler
ifeq ($(compiler),icc)
    C = icc
else
	ifndef USE_STATS
    C = gcc
	else
		C = g++
	endif
endif

ifeq (Windows_NT, $(OS))
	WHICH=where
else
	WHICH=which
endif


ifeq (, $(shell $(WHICH) python3))
	PYTHON = python
else
	PYTHON = python3
endif

PIP_FLAGS := --upgrade

D=$(shell pwd)

# The 'in-place' flag is different on macOS (BSD) and Linux/minGW.
# https://linux.die.net/man/1/sed
# https://www.freebsd.org/cgi/man.cgi?query=sed&sektion=&n=1
# Note on Windows: install Rtools for 'sed'
ifeq ($(shell uname),Darwin)
	SED_I=sed -i ''
#  PYTHONLIB="-undefined dynamic_lookup"
	ifndef USE_STATS
    C = clang
	else
		C = clang++
	endif
else
	SED_I=sed -i
endif

ifndef USE_STATS
	SRC_SCILIB = 
	OBJS_SCILIB = src/random_gsl.o
	LFLAGS_SCILIB = -lgsl -lgslcblas
	LDFLAGS_SCILIB = $(shell gsl-config --libs)
	CFLAGS_SCILIB = $(shell gsl-config --cflags) 
	CPPFLAGS_SCILIB = 
else
	ifndef GSL_COMPAT
		COMPAT = 
	else
		COMPAT = -DGSL_COMPAT
	endif
	SRC_SCILIB = src/random_stats.h
	OBJS_SCILIB = src/random_stats.o
	LFLAGS_SCILIB = 
	LDFLAGS_SCILIB = 
	CFLAGS_SCILIB = -DSTATS_ENABLE_STDVEC_WRAPPERS -DSTATS_GO_INLINE -DUSE_STATS $(COMPAT) -Istats/include -Igcem/include -std=c++17
	CPPFLAGS_SCILIB = -DSTATS_ENABLE_STDVEC_WRAPPERS -DSTATS_GO_INLINE -DUSE_STATS $(COMPAT) -Istats/include -Igcem/include -std=c++17
endif

OBJS = $(OBJS_SCILIB) src/utilities.o src/constant.o src/demographics.o src/params.o src/model.o src/individual.o src/main.o src/input.o src/network.o src/disease.o src/interventions.o src/hospital.o src/doctor.o src/nurse.o src/ward.o src/list.o src/strain.o

LFLAGS = $(LFLAGS_SCILIB) -lm -O2

# Name of executable
_EXE = src/covid19ibm.exe
EXE = $(_EXE)

INC = /usr/local/include
LIB = /usr/local/lib

# Compilation options and libraries to be used
CFLAGS = -Wall -fmessage-length=0 -I$(INC) $(CFLAGS_SCILIB) -O2
CPPFLAGS = -Wall -fmessage-length=0 -I$(INC) $(CPPFLAGS_SCILIB) -O2
LDFLAGS = -L$(LIB) $(LDFLAGS_SCILIB)

# Swig's input
SWIG_INPUT = src/disease.h src/ward.h src/nurse.h src/network_utils.i src/vaccine_utils.i src/strain_utils.i src/input.h src/individual.h src/hospital.h src/params.h src/structure.h src/constant.h src/doctor.h src/utilities.h src/model_utils.i src/covid19.i src/list.h src/network.h src/model.h src/interventions.h src/params_utils.i src/demographics.h src/strain.h src/random.h $(SRC_SCILIB)

# Swig's output
SWIG_OUTPUT_PY = src/covid19_wrap.o src/covid19_wrap.c src/covid19.py src/_covid19.cpython-37m-darwin.so src/build src/covid19.egg-info
SWIG_OUTPUT_R = src/covid19_wrap_R.c src/covid19_wrap_R.o R/OpenABMCovid19.R src/OpenABMCovid19.so
SWIG_OUTPUT = $(SWIG_OUTPUT_PY) $(SWIG_OUTPUT_R)

ifndef SWIG3
	SWIG3 = swig
endif

# Roxygen generated files
ROXYGEN_OUTPUT= man/SAFE_UPDATE_PARAMS.Rd man/Parameters.Rd man/Environment.Rd man/Network.Rd man/Agent.Rd man/VaccineSchedule.Rd man/VACCINE_STATUS.Rd man/Model.Rd man/AgeGroupEnum.Rd man/NETWORK_CONSTRUCTIONS.Rd man/COVID19IBM.Rd man/VACCINE_TYPES.Rd man/Simulation.Rd

# To compile
install: $(OBJS)
install: all;
	cd src && swig -python covid19.i
	cd src && CC=$(C) CXX=$(C) LDSHARED=$(C) D=$(D) $(PYTHON) -m pip install -v $(PIP_FLAGS) .

dev: PIP_FLAGS += -e
dev: install;

all: $(OBJS)
	$(C) $(LDFLAGS) -o $(EXE) $(OBJS) $(LFLAGS)

clean:
	cd src && $(PYTHON) -m pip uninstall -y covid19
	rm -rf $(OBJS) $(EXE) $(SWIG_OUTPUT) $(ROXYGEN_OUTPUT)

# TODO add check if [ ! -d "$(PWD)/stats" ]; then echo "Please check out the stats library with: git clone --depth 1 https://github.com/kthohr/stats"; exit 1; fi
%.o : %.cpp
	$(C) $(CPPFLAGS) -c $< -o $@

.c.o:
	$(C) $(CFLAGS) -c $< -o $@

# Generating swig3 source for R bindings (and post-processing)
R/OpenABMCovid19.R: $(SWIG_INPUT)
	$(SWIG3) -r -package OpenABMCovid19 -Isrc -o src/covid19_wrap_R.c -outdir R src/covid19.i
# edit generated C source to mute R check note.
	$(SED_I) 's/R_registerRoutines/R_useDynamicSymbols(dll,0);R_registerRoutines/' src/covid19_wrap_R.c
# edit generated src lines are cause R check warnings.
	$(SED_I) 's/.Call("R_SWIG_debug_getCallbackFunctionData"/.Call("R_SWIG_debug_getCallbackFunctionData", PACKAGE="OpenABMCovid19"/' R/OpenABMCovid19.R
	$(SED_I) 's/.Call("R_SWIG_R_pushCallbackFunctionData"/.Call("R_SWIG_R_pushCallbackFunctionData", PACKAGE="OpenABMCovid19"/' R/OpenABMCovid19.R
# edit generated src that causes errors like: "p_char" is not a defined class
	$(SED_I) 's/ans <- new("_p_char"/#ans <- new("_p_char"/' R/OpenABMCovid19.R
src/covid19_wrap_R.c: R/OpenABMCovid19.R
Rswig: R/OpenABMCovid19.R

.PHONY: install dev all clean Rswig
