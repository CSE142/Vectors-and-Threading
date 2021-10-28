This project applies vectors and threading to optimize a larger CNN model.  There are two new kinds of layers, more loops that were optimized. 

Altogether, the following optimizations were done:

1.  Loop reordering
2.  Loop tiling
3.  Multithreading
4.  Vectorization


### Learn a Little About the Model 

The model we are using for this is a simplified version of AlexNet
(https://en.wikipedia.org/wiki/AlexNet), the deep neural network that kickstarted the current mania around deep learning.

AlexNet was built to compete in the ImageNet competion
(http://image-net.org/challenges/LSVRC/2016/index) which measure
accuracy in classifying 256x256-pixel RGB images.  AlexNet was
enormous (for the time):

1. It had 60 million internal weights
2. It took two weeks to train it using two big, fast GPUs.
3. If expressed using the layers that Canela provides, it has about 20 layers.

Our version is a bit smaller.  It works on 128x128 images and is only 18 layers.  It also does not include a few sophisticated optimizations to the learning process.

It is built in `model.cpp`.

The model contains multiple convolution, pooling, and fully-connected layers and the whole model occupies 590MB of DRAM, so make efficient use of caches is very important.

### Command Line Options

The executable takes a few useful command line options:

* `--scale <n>` sets the execution time for each tested functions to `n` seconds (approximately).  For very short functions, it runs them 10,000 times.
* `--function <function_name>` Collect measurements for a particular function.   Use `./cnn.exe --help` (after you've done `make cnn.exe`) to get a list of possible values.

### Read the Source


## Measuring Performance of Particular Functions

There are quite a few functions to tune and optimize.  To make this easier (and less time consuming) the `cnn.exe` executable has built-in support for running and timing specific functions and for getting information about the model.

The option `--describe-model`.  This will list some information about the model and exit. 


**Note** the `--scale` option sets (very approximately) the number of seconds the test will run for.  3 is good number -- it should give pretty consistent results without taking too long.  However, if you want to run lots of tests quickly, you could run at a smaller scale and then confirm with a bigger scale.  `--scale 0` will run the function exactly once (which is useful with `mtrace`).

## Parameterizing and Iterating Your Tests

`cnn.cpp` has some simple support for running multiple tests in quick succession.  It's meant to handle tasks like comparing the performance of multiple implementations of a function or varying tile sizes.
