CC=gcc
CFLAGS=-c -Wall -I../../src -pg
LDFLAGS= -pg
LIBS= -lgsl
SOURCES=main.c ../../src/random_gsl.c
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=randomdatagsl

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -O2 $(LIBS) -o $@

.c.o:
	$(CC) $(CFLAGS) -O2 $< -o $@
