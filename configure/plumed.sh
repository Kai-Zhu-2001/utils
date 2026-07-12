#!/usr/bin/env bash

SOURCE_DIR=/leonardo/home/userexternal/kzhu0000/opt/plumed-2.10.0
INSTALL_DIR=/leonardo/home/userexternal/kzhu0000/opt/plumed2.10.0

cd "$SOURCE_DIR"

CPPFLAGS="$(python src/metatomic/flags-from-python.py --cppflags)"
LDFLAGS="$(python src/metatomic/flags-from-python.py --ldflags)"

./configure \
  CXX=mpicxx \
  CXXFLAGS="-O3 -std=c++17" \
  CPPFLAGS="$CPPFLAGS" \
  LDFLAGS="$LDFLAGS" \
  --prefix="$INSTALL_DIR" \
  --enable-mpi \
  --enable-openmp \
  --enable-libtorch \
  --enable-libmetatomic \
  --enable-modules=all \
  --disable-external-blas \
  --disable-external-lapack

make -j 2
make install
