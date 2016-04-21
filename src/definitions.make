### Modules (check for consistency with Modules.hh) ###
# comment out or assign empty value for module deactivation
MODULE_CUDA := 1
MODULE_OPENMP := 1

MKLROOT := /opt/intel/mkl
CUDAROOT := /usr/local/cuda-7.5

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

### LIBRARIES ###
CLIB :=
# Intel MKL
CLIB :=  $(CLIB) -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -ldl -lpthread -lm
# zlib
CLIB := $(CLIB) -lz
# CUDA
ifdef MODULE_CUDA
CLIB := $(CLIB) -L$(CUDAROOT)/lib64/ -lcublas -lcudart -lcurand
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
