CC=gcc
CFLAGS=-c -Wall 
LDFLAGS= 
LIBS= -lgsl
SOURCES=main.c
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=invgamma

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -O3 $(LIBS) -o $@

.c.o:
	$(CC) $(CFLAGS) -O3 $< -o $@
