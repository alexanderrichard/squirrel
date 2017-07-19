TOPDIR = ../

include ../definitions.make

OBJECTS = KMeans.o \
	  Gmm.o

OBJ = $(patsubst %, objects/%, $(OBJECTS))

LIB = ../Core/libCore.a \
      ../Math/libMath.a \
      ../Features/libFeatures.a

.PHONY: all prepare clean Application libClustering.a

all: prepare $(OBJ) libClustering.a Application

prepare:
	@mkdir -p objects

$(OBJ): objects/%.o : %.cc %.hh
	$(CC) $(COPTS) -c $< -lcppunit -o $@

libClustering.a:
	$(MAKELIB) $(ARFLAGS) $@ $(OBJ)

Application: $(OBJ)
	$(CC) $(COPTS) $@.cc $(OBJ) -Wl,--start-group $(LIB) -Wl,--end-group $(CLIB) -o clustering-tool

clean:
	rm -rf objects/ libClustering.a clustering-tool
