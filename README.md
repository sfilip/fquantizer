# fquantizer
=========================================

## Description ##
Routines for linear-phase FIR filter design with fixed-point coefficients, based on
Euclidean lattice basis reduction algorithms. For a detailed presentation, see
[1, Ch. 4] and [2], with [2] being more up to date.

## Installation instructions ##
This code has been tested **only** on Linux machines. In order to compile and use it, a recent version of g++ with
C++11 support is necessary (a version >= 4.8 should work nicely). Some external utilities and libraries must also be
installed and available on your system search paths:
* CMake version 2.8 or newer
* Eigen version 3 or newer
* GMP version 5 or newer
* MPFR version 3 or newer
* fplll version 4 or newer
* (optional) Google gtest framework for generating the test executables


Assuming these prerequisites are taken care of and you are located at the base folder containing the source code and
test files, building the library can be done using the following commands (with the appropriate user privileges):

        mkdir build
        cd build
        cmake ..
        make all
        make install

This series of steps will install the library on your system. Usually the default location is /usr/local, but this
behavior can be changed when calling CMake by specifying an explicit install prefix:

        cmake -DCMAKE_INSTALL_PREFIX=/your/install/path/here ..

If gtest is installed on your system, the *make all* command should have generated the test executable
* quantization_test : should contain code to generate quantization examples


We must make the precision that both the gtest library header files and the static and shared versions of the library
*must* be installed in order to generate the test files. In the case of an Ubuntu installation of gtest from their
official repositories, the static and shared versions of gtest are not installed. Ways of solving this problem are
described at the following link: http://askubuntu.com/questions/145887/why-no-library-files-installed-for-google-test


The make all target also generates the documentation if Doxygen was found on your system when running cmake. It can also
be generated individually by running the command

        make doc

after cmake was called.

## References
[1] S.-I. Filip, Robust tools for weighted Chebyshev approximation and
applications to digital filter design, Ph.D. dissertation, ENS de Lyon, France, 2016.
[2] N. Brisebarre, S.-I. Filip, G. Hanrot, A lattice basis reduction approach
for the design of finite wordlength FIR filters, IEEE Transactions on Signal Processing 66,
10 (May 2018), 2673 - 2684
