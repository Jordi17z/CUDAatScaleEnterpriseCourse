# Image Filtering using NVIDIA NPP with CUDA: Gaussian and Laplace Filter

## Overview
This project exhibits the use of NVIDIA Performance Primitives (NPP) library with CUDA to perform two types of linear filters. One of them it's the gaussian, which is a operation that smooths an image by averaging the pixel values in a neighborhood (a blurred effect). The other filter it's the laplace filter which is commonly used to detect edges in images. The goal is to utilize GPU acceleration to efficiently apply this operations in images using the computational power of modern GPUs. This project is a part of the CUDA at Scale for the Enterprise course.

## Code Organization
<mark> bin/  </mark> This folder should hold all binary/executable code that is built automatically or manually. 
The executable code should have use the .exe extension or programming language-specific extension.

<mark> data/ </mark> This folder holds the example data in any format (usually .pgm and .png).

<mark> Common/ </mark> Any library needed.

<mark> src/ </mark> The source code.

## Key Concepts
Performance Strategies, Image Processing, NPP Library

## Supported SM Architectures 
SM 3.5 SM 3.7 SM 5.0 SM 5.2 SM 6.0 SM 6.1 SM 7.0 SM 7.2 SM 7.5 SM 8.0 SM 8.

## Supported OS
Linux, Windows

## Supported CPU Architecture
x86_64, ppc64le, armv7l

## Dependencies needed to build/run
FreeImage, NPP

## Prerequisites
Download and install the CUDA Toolkit 11.4 for your corresponding platform. Make sure the dependencies mentioned in Dependencies section above are installed.

## Build and Run
### Windows
The Windows samples are built using the Visual Studio IDE. Solution files (.sln) are provided for each supported version of Visual Studio, using the format:

`*_vs<version>.sln - for Visual Studio <version>`

Each individual sample has its own set of solution files in its directory:

To build/examine all the samples at once, the complete solution files should be used. To build/examine a single sample, the individual sample solution files should be used.

### Linux
The Linux samples are built using makefiles. To use the makefiles, change the current directory to the directory you wish to build, and run make:

`$ cd <sample_dir>`

`$ make clean build` 

or just 

`make build`

The samples makefiles can take advantage of certain options:

TARGET_ARCH= - cross-compile targeting a specific architecture. Allowed architectures are x86_64, ppc64le, armv7l. By default, TARGET_ARCH is set to HOST_ARCH. On a x86_64 machine, not setting TARGET_ARCH is the equivalent of setting TARGET_ARCH=x86_64.
$ make TARGET_ARCH=x86_64
$ make TARGET_ARCH=ppc64le
$ make TARGET_ARCH=armv7l
See here for more details.

dbg=1 - build with debug symbols

`$ make dbg=1`

SMS="A B ..." - override the SM architectures for which the sample will be built, where "A B ..." is a space-delimited list of SM architectures. For example, to generate SASS for SM 50 and SM 60, use SMS="50 60".

`$ make SMS="50 60"`

HOST_COMPILER=<host_compiler> - override the default g++ host compiler. See the Linux Installation Guide for a list of supported host compilers.

`$ make HOST_COMPILER=g++`

## Running the program
After building the project, you can run the program using the following command:

`./boxFilterNPP -input "Name of your file placed in 'data' folder" -f "laplace/gaussian"`

* It is necessary to set an input file (the program will not run if not), the -f flag is not necessary, by default would be "gaussian".
