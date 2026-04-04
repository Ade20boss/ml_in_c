#!/bin/bash
set -xe # Print commands and exit on failure

gcc -Wall -Wextra -g -o neuron zero-bias_linear_regressor.c 

./neuron
