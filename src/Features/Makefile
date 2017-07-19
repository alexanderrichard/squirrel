TOPDIR = ../

include ../definitions.make

OBJECTS = FeatureCache.o \
          FeatureReader.o \
          AlignedFeatureReader.o \
          FeatureWriter.o \
          Preprocessor.o \
          FeatureCacheManager.o

OBJ = $(patsubst %, objects/%, $(OBJECTS))

LIB = ../Core/libCore.a \
      ../Math/libMath.a

.PHONY: all prepare clean libFeatures.a Application

all: prepare $(OBJ) libFeatures.a Application

prepare:
	@mkdir -p objects

$(OBJ): objects/%.o : %.cc %.hh
	$(CC) $(COPTS) -c $< -lcppunit -o $@

libFeatures.a:
	$(MAKELIB) $(ARFLAGS) $@ $(OBJ)

Application: $(OBJ)
	$(CC) $(COPTS) $@.cc $(OBJ) -Wl,--start-group $(LIB) -Wl,--end-group $(CLIB) -o feature-cache-manager

clean:
	rm -rf objects/ libFeatures.a feature-cache-manager
