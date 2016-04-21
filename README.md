# Squirrel

Squirrel is a framework for action recognition with a focus on neural networks that runs on Linux. So far, feed forward networks as well as recurrent neural networks are supported. Convolution is to be added soon.

The neural network module can be used with any kind of data, as long as the data is in the squirrel input format. The neural-network example shows how to convert ascii data to Squirrel caches.

A more detailed documentation going beyond what is presented in the examples will be added later.

The framework also provides an implementation of the methods from the following two papers:

    A. Richard, J. Gall:
    A BoW-equivalent Recurrent Neural Network for Action Recognition,
    in British Machine Vision Conference, 2015

and

    A. Richard, J. Gall:
    A BoW-equivalent Recurrent Neural Network for Action Recognition,
    in IEEE Int. Conf. on Computer Vision and Patter Recognition, 2016

**Please cite this paper if you use the framework.**

Special thanks go to the Human Language Technology and Pattern Recognition Group from RWTH University for the permission to use their matrix and vector classes from

    S. Wiesler, A. Richard, P. Golik, R. Schuter, H. Ney: RASR/NN:
    The RWTH neural network toolkit for speech recognition,
    in IEEE Int. Conf. on Acoustics, Speech and Signal Processing, 2014

### Installation

The software uses NVidia Cuda and Intel MKL for parallelization. Modify the  ```definitions.make ``` in  ```src``` to have the correct paths to the MKL and Cuda libraries. If you do not want to use Cuda or OpenMP, comment out these lines:

    MODULE_CUDA := 1
    MODULE_OPENMP := 1

and the corresponding lines in ```Modules.hh ```.

If you do not want to use MKL but some other BLAS implementation, you might want to modify the includes in  ```src/Math/Blas.hh ``` and  ```src/Math/Lapack.hh ```. In this case, do not forget to also modify the library paths in  ```definitions.make ```.

To compile the code, go to the  ```src ``` directory and invoke  ```make ```.
Run the ```copyExecutables.sh``` script to create an  ```executables ``` directory and copy the latest build to this directory. Now, you are ready to run the examples.

### Contact

In case of questions, please contact Alexander Richard (richard [at] iai.uni-bonn.de).
