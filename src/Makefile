# For use of the icc compiler
ifeq ($(compiler),icc)
    C = icc
else
    C = gcc
endif

ifeq (, $(shell which pip3))
	PIP = pip
else
	PIP = pip3
endif

OBJS = utilities.o constant.o demographics.o params.o model.o individual.o main.o input.o network.o disease.o interventions.o hospital.o doctor.o nurse.o ward.o

GSLFLAGS= -lgsl -lgslcblas -lm -O3
LFLAGS = ${GSLFLAGS}

# Name of executable
_EXE = covid19ibm.exe
EXE = $(_EXE)

INC = /usr/local/include
LIB = /usr/local/lib

# Compilation options and libraries to be used
CFLAGS = -g -Wall -fmessage-length=0 -I$(INC) -O0

# Swig's output
SWIG_OUTPUT = covid19_wrap.o covid19_wrap.c covid19.py _covid19.cpython-37m-darwin.so build

# To compile
swig-all:
	# swig -outdir COVID19 -python covid19.i
	swig -python covid19.i
	# python3 setup.py build_ext --inplace
	${PIP} install --user --upgrade .

all: ${OBJS}
	${C} -L${LIB} -o ${EXE} ${OBJS} ${LFLAGS}

clean:
	rm -rf $(OBJS) $(EXE) $(SWIG_OUTPUT)
	${PIP} uninstall -y covid19

.c.o:
	${C} ${CFLAGS} -c $< -o $@

