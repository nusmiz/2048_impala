# Solve sokoban game with IMPALA

## Requirements

* gcc (>=7.3)
* cmake (>=3.10)
* boost (>=1.65)
* python (>=3.6)
* pytorch (>=0.4)

## How to compile

    $ git clone https://github.com/nusmiz/sokoban_impala
    $ cd sokoban_impala
    $ git submodule update --init
    $ mkdir build
    $ cd build
    $ cmake .. -DCMAKE_BUILD_TYPE=Release
    $ make -j 4
    $ cd ..

## How to run

    $ ./build/train2048

## Disable CUDA

    $ cmake .. -DUSE_CUDA=OFF

## GUI Viewer

    $ cmake .. -DGUI_VIEWER=ON

Requirements
* libglfw3-dev
* libpng-dev

