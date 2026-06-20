#!/bin/bash
set -xe # Print commands and exit on failure

gcc -Wall -Wextra -g -o neuron neuron.c 

./neuron
