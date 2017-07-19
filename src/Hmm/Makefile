TOPDIR = ../

include ../definitions.make

OBJECTS = HiddenMarkovModel.o \
          Grammar.o \
          GrammarGenerator.o \
          Scorer.o \
          LengthModel.o \
          ViterbiDecoding.o

OBJ = $(patsubst %, objects/%, $(OBJECTS))

LIB = ../Core/libCore.a \
      ../Math/libMath.a \
      ../Features/libFeatures.a \
      ../Nn/libNeuralNetwork.a

.PHONY: all prepare clean Application libHmm.a

all: prepare $(OBJ) libHmm.a Application

prepare:
	@mkdir -p objects

$(OBJ): objects/%.o : %.cc %.hh
	$(CC) $(COPTS) -c $< -lcppunit -o $@

libHmm.a:
	$(MAKELIB) $(ARFLAGS) $@ $(OBJ)

Application: $(OBJ)
	$(CC) $(COPTS) $@.cc $(OBJ) -Wl,--start-group $(LIB) -Wl,--end-group $(CLIB) -o hmm-tool

clean:
	rm -rf objects/ libHmm.a hmm-tool
