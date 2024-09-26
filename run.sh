#!/usr/bin/env bash
make clean build

make run ARGS="-input=./data/input/grey-sloth.pgm -f gaussian"