CC=g++
CFLAGS=-c -Wall -I../../src -I../../stats/include -I../../gcem/include -fopenmp
LDFLAGS= -fopenmp
CPPFLAGS= -std=c++17
LIBS= 
SOURCES=main.cpp ../../src/random_stats.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=statsparalleltest

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -O2 $(LIBS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $(CPPFLAGS) -O2 -DSTATS_GO_INLINE -DSTATS_ENABLE_STDVEC_WRAPPERS -DUSE_STATS $< -o $@
