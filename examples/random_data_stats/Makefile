CC=g++
CFLAGS=-c -Wall -I../../src -I../../stats/include -I../../gcem/include -fopenmp -pg -g -std=c++17
LDFLAGS= -g -pg -fopenmp
CPPFLAGS= 
LIBS= 
SOURCES=main.cpp ../../src/random_stats.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=randomdatastats

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -O2 $(LIBS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $(CPPFLAGS) -O2 -DUSE_STATS $< -o $@
