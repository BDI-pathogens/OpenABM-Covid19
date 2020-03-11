# For use of the icc compiler
ifeq ($(compiler),icc)
    C = icc
else
    C = gcc
endif

OBJS = utilities.o params.o model.o individual.o main.o input.o
GSLFLAGS= -lgsl -lgslcblas -lm -O3
LFLAGS = ${GSLFLAGS}

# Name of executable
_EXE = covid19ibm.exe
EXE = $(_EXE)

# For compilation on Rescomp
ifeq ($(location),rescomp)
	INC = /apps/well/gsl/2.2.1-gcc4.9.3/include
	LIB = /apps/well//gsl/2.2.1-gcc4.9.3/lib
else
    INC = /usr/local/include
	LIB = /usr/local/lib
endif

# Compilation options and libraries to be used
CFLAGS = -g -Wall -fmessage-length=0 -I$(INC) -O0

# To compile
all: ${OBJS}
	${C} -L${LIB} -o ${EXE} ${OBJS} ${LFLAGS}

clean:
	rm -f $(OBJS) $(EXE)

.c.o:
	${C} ${CFLAGS} -c $< -o $@	

