TOPDIR = ../

include ../definitions.make

OBJECTS = DenseTrajectoriesConversion.o \
          LabelConversion.o \
          LibSvmConversion.o \
          MatrixToCacheConversion.o \
          AsciiCacheToBinaryCache.o

OBJ = $(patsubst %, objects/%, $(OBJECTS))

LIB = ../Core/libCore.a \
      ../Math/libMath.a \
      ../Features/libFeatures.a \

.PHONY: all prepare clean libConverter.a Application

all: prepare $(OBJ) libConverter.a Application

prepare:
	@mkdir -p objects

$(OBJ): objects/%.o : %.cc %.hh
	$(CC) $(COPTS) -c $< -lcppunit -o $@

libConverter.a:
	$(MAKELIB) $(ARFLAGS) $@ $(OBJ)

Application: $(OBJ)
	$(CC) $(COPTS) $@.cc $(OBJ) -Wl,--start-group $(LIB) -Wl,--end-group $(CLIB) -o converter

clean:
	rm -rf objects/ libConverter.a converter
