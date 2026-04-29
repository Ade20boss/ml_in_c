#!/bin/bash

set -xe 

gcc -Wall -Wextra -g full_gates.c -o full_gates -lm
./full_gates