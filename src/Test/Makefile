TOPDIR = ../

include ../definitions.make

OBJECTS = Registry.o \
          Core_Tree.o \
          Core_HashMap.o \
          Features_Preprocessor.o \
          Features_AlignedFeatureReader.o \
          Math_Matrix.o \
          Math_Vector.o \
          Math_CudaMatrix.o \
          Math_CudaVector.o \
          Math_FastVectorOperations.o \
          Math_MultithreadingHelper.o \
          Nn_NeuralNetwork.o \
          Nn_MinibatchGenerator.o \
          Nn_MatrixContainer.o \
          Nn_Trainer.o \
          Nn_TestHelpers.o

OBJ = $(patsubst %, objects/%, $(OBJECTS))

LIB = ../Math/libMath.a \
      ../Core/libCore.a \
      ../Features/libFeatures.a \
      ../Nn/libNeuralNetwork.a

.PHONY: all prepare clean UnitTester

all: prepare $(OBJ) UnitTester

prepare:
	@mkdir -p objects

$(OBJ): objects/%.o : %.cc
	$(CC) $(COPTS) -c $< -o $@

# compile and link test.cc

UnitTester: $(OBJ)
	$(CC) $(COPTS) $@.cc $(OBJ) -Wl,--start-group $(LIB) -Wl,--end-group $(CLIB) -lcppunit -o unit-test

clean:
	rm -rf objects/ unit-test
