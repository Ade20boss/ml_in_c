#!/bin/bash

set -xe

gcc -Wall -Wextra -g or_gate.c -o orgate -lm

./orgate
