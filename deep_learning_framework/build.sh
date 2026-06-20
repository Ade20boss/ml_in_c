#!/bin/bash
# Build script for Kestrel deep learning framework
#
set -xe
gcc -Wall -Wextra -Werror -o kestrel kestrel.c -lm
./kestrel
