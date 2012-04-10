CC=g++
MYFLAGS= -O3 -fopenmp

SOURCES=expectation_maximization.cpp main.cpp malevich_classifier.cpp util.cpp
OBJECTS=$(SOURCES:.c=.o)

all: $(SOURCES) classifier

classifier: $(OBJECTS)
	$(CC) $(MYFLAGS) $(OBJECTS) -o $@

.c.o:
	$(CC) -c $(MYFLAGS) $< -o $@

clean:
	rm -rf *.o classifier run.e* run.o*
