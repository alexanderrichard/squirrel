TOPDIR = ../

include ../definitions.make

OBJECTS = Random.o \
          CudaDataStructure.o \
          CudnnDataStructure.o

ifdef MODULE_CUDA
CUDA_OBJECTS += CudaMatrixKernels.o
endif

OBJ = $(patsubst %, objects/%, $(OBJECTS))
CUDA_OBJ = $(patsubst %, objects/%, $(CUDA_OBJECTS))

LIB = ../Core/libCore.a

.PHONY: all prepare clean libMath.a

all: prepare $(CUDA_OBJ) $(OBJ) libMath.a

prepare:
	@mkdir -p objects

ifdef MODULE_CUDA
objects/CudaMatrixKernels.o: CudaMatrixKernels.cu
	$(NVCC) $(NVCCFLAGS) -o objects/CudaMatrixKernels.o -c CudaMatrixKernels.cu
endif

$(OBJ): objects/%.o : %.cc %.hh
	$(CC) $(COPTS) -c $< -lcppunit -o $@

libMath.a:
	$(MAKELIB) $(ARFLAGS) $@ $(OBJ) $(CUDA_OBJ)

clean:
	rm -rf objects/ libMath.a
