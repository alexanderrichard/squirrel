TOPDIR = ../

include ../definitions.make

OBJECTS = FeatureQuantization.o \
          Kernel.o \
          MultiChannelRbfChiSquareKernel.o \
          Normalizations.o \
          PrincipalComponentAnalysis.o \
          SequenceIntegrator.o \
          FramePadding.o

OBJ = $(patsubst %, objects/%, $(OBJECTS))

LIB = ../Core/libCore.a \
      ../Math/libMath.a \
      ../Features/libFeatures.a

.PHONY: all prepare clean libFeatureTransformation.a Application

all: prepare $(OBJ) libFeatureTransformation.a Application

prepare:
	@mkdir -p objects

$(OBJ): objects/%.o : %.cc %.hh
	$(CC) $(COPTS) -c $< -lcppunit -o $@

libFeatureTransformation.a:
	$(MAKELIB) $(ARFLAGS) $@ $(OBJ)

Application: $(OBJ)
	$(CC) $(COPTS) $@.cc $(OBJ) -Wl,--start-group $(LIB) -Wl,--end-group $(CLIB) -o feature-transformation

clean:
	rm -rf objects/ libFeatureTransformation.a feature-transformation
