#!/bin/bash

cd bin

g++ -c -I../include -I../src ../src/main.cc ../src/ray.cc

cd ../

g++ -o raytrace bin/main.o bin/ray.o -Llib -lpalette
