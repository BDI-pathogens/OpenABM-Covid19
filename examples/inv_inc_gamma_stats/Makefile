CC=g++
CFLAGS=-c -Wall -I../../stats/include -I../../gcem/include
LDFLAGS= 
CPPFLAGS= -I../../stats/include -I../../gcem/include
LIBS= 
SOURCES=main.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=invgamma

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -O3 $(LIBS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $(CPPFLAGS) -O3 $< -o $@
