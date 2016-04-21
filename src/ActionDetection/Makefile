TOPDIR = ../

include ../definitions.make

OBJECTS = LengthModelling.o \
          Scorer.o \
          LinearSearch.o \
          SlidingWindowSearch.o

OBJ = $(patsubst %, objects/%, $(OBJECTS))

LIB = ../Core/libCore.a \
      ../Math/libMath.a \
      ../Features/libFeatures.a \
      ../Nn/libNeuralNetwork.a \
      ../Lm/libLm.a

.PHONY: all prepare clean Application libActionDetection.a

all: prepare $(OBJ) libActionDetection.a Application

prepare:
	@mkdir -p objects

$(OBJ): objects/%.o : %.cc %.hh
	$(CC) $(COPTS) -c $< -lcppunit -o $@

libActionDetector.a:
	$(MAKELIB) $(ARFLAGS) $@ $(OBJ)

Application: $(OBJ)
	$(CC) $(COPTS) $@.cc $(OBJ) -Wl,--start-group $(LIB) -Wl,--end-group $(CLIB) -o action-detector

clean:
	rm -rf objects/ libActionDetection.a action-detector
