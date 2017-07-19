### Modules (check for consistency with Modules.hh) ###
# comment out or assign empty value for module deactivation
MODULE_CUDA := 1
MODULE_OPENMP := 1
MODULE_OPENCV := 1
MODULE_CUDNN := 1

# TODO change to your paths. If you don't want to use Intel MKL, give the path to another blas/lapack library
MKLROOT := /opt/intel/mkl
CUDAROOT := /usr/local/cuda
OPENCVROOT := /usr/local
CUDNNROOT := /usr/local

### G++ ###
CC := g++
CFLAGS := -Wall -m64 -g -rdynamic
ifdef MODULE_OPENMP
CFLAGS := $(CFLAGS) -fopenmp
endif

### INCLUDES ###
CINC := -I$(TOPDIR)
# Intel MKL
CINC := $(CINC) -I$(MKLROOT)/include
# CUDA
ifdef MODULE_CUDA
CINC := $(CINC) -I$(CUDAROOT)/include
endif
# openCV
ifdef MODULE_OPENCV
CINC := $(CINC) -I$(OPENCVROOT)/include
endif
# cudnn
ifdef MODULE_CUDNN  
CINC := $(CINC) -I$(CUDNNROOT)/include
endif

### LIBRARIES ###
CLIB :=
# Intel MKL TODO change this line if you want to use another Blas/Lapack library than Intel MKL
CLIB :=  $(CLIB) -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -ldl -lpthread -lm
# zlib
CLIB := $(CLIB) -lz
# CUDA
ifdef MODULE_CUDA
CLIB := $(CLIB) -L$(CUDAROOT)/lib64/ -lcublas -lcudart -lcurand
endif
# openCV
ifdef MODULE_OPENCV
CLIB := $(CLIB) -L$(OPENCVROOT)/lib -lopencv_core -lopencv_highgui -lopencv_imgproc
endif
# cudnn
ifdef MODULE_CUDNN
CLIB := $(CLIB) -L$(CUDNNROOT)/lib64 -lcudnn
endif
### OPTIONS ###
COPTS := $(CFLAGS) $(CINC)

### STATIC LIBRARIES ###
MAKELIB := ar
ARFLAGS := rucs

### NVCC ###
ifdef MODULE_CUDA
NVCC := $(CUDAROOT)/bin/nvcc
NVCCFLAGS := -arch sm_52
endif
