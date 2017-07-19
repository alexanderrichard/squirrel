TOPDIR = ../

include ../definitions.make

OBJECTS = Types.o \
          Parameter.o \
          Configuration.o \
          Utils.o \
          Log.o \
          Error.o \
          IOStream.o \
          Application.o

OBJ = $(patsubst %, objects/%, $(OBJECTS))

.PHONY: all prepare clean libCore

all: prepare $(OBJ) libCore

prepare:
	@mkdir -p objects

$(OBJ): objects/%.o : %.cc %.hh
	$(CC) $(COPTS) -c $< -o $@

libCore:
	$(MAKELIB) $(ARFLAGS) libCore.a $(OBJ)

clean:
	rm -rf objects/ libCore.a
