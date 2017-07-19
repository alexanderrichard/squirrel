# Squirrel

Squirrel is a framework for action recognition and temporal action detection with a focus on neural networks that runs on Linux. So far, feed forward networks, CNNS, and recurrent neural networks are supported.

Please inspect the examples or see the manual for information on how to use the framework and to determine the correct data format.

The framework also provides an implementation of the methods from the following papers:

**(Please cite one of these papers if you use the framework)**

    A. Richard, H. Kuehne, J. Gall:
    Weakly Supervised Action Learning with RNN based Fine-to-coarse Modeling
    in IEEE Int. Conf. on Computer Vision and Pattern Recognition, 2017

    A. Richard, J. Gall:
    A Bag-of-words Equivalent Recurrent Neural Network for Action Recognition,
    in Computer Vison and Image Understanding, 2017

    A. Richard, J. Gall:
    Temporal Action Detection using a Statistical Language Model,
    in IEEE Int. Conf. on Computer Vision and Pattern Recognition, 2016

    A. Richard, J. Gall:
    A BoW-equivalent Recurrent Neural Network for Action Recognition,
    in British Machine Vision Conference, 2015

Special thanks go to the Human Language Technology and Pattern Recognition Group from RWTH University for the permission to use their matrix and vector classes from

    S. Wiesler, A. Richard, P. Golik, R. Schuter, H. Ney: RASR/NN:
    The RWTH neural network toolkit for speech recognition,
    in IEEE Int. Conf. on Acoustics, Speech and Signal Processing, 2014

### Installation

The software uses NVidia Cuda and Intel MKL for parallelization. Modify the  ```definitions.make ``` in  ```src``` to have the correct paths to the MKL and Cuda libraries. If you do not want to use Cuda, OpenCV, or OpenMP, comment out these lines:

    MODULE_CUDA := 1
    MODULE_OPENMP := 1
    MODULE_OPENCV := 1

Note that CUDA-8.0 is required for the framework.

and the corresponding lines in ```Modules.hh ```.

If you do not want to use MKL but some other BLAS implementation, you might want to modify the includes in  ```src/Math/Blas.hh ``` and  ```src/Math/Lapack.hh ```. In this case, do not forget to also modify the library paths in  ```definitions.make ```.

To compile the code, go to the  ```src ``` directory and invoke  ```make ```.

### Contact

In case of questions, please contact Alexander Richard (richard [at] iai.uni-bonn.de).
