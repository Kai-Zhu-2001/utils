# MODULES
module load gcc
module load gsl/2.7.1--gcc--12.2.0
module load openmpi
module load fftw
module load openblas/0.3.24--gcc--12.2.0
module load cuda/12.6       
module load intel-oneapi-mkl/2023.2.0
 
# remove check on plumed version
sed "s/api_version *> *[0-9][0-9]*/false/" src/PLUMED/fix_plumed.cpp > src/PLUMED/fix_plumed.cpp.fix
mv src/PLUMED/fix_plumed.cpp.fix src/PLUMED/fix_plumed.cpp
 
mkdir build 
cd build
 
export PKG_CONFIG_PATH=$HOME/opt/plumed2.9.4-gnn/lib/pkgconfig:$PKG_CONFIG_PATH

cmake ../cmake \
-D CMAKE_BUILD_TYPE=Release \
-D CMAKE_INSTALL_PREFIX=$HOME/opt/lammps \
-D BUILD_MPI=ON \
-D BUILD_SHARED_LIBS=ON \
-D BUILD_PYTHON=ON \
-D CMAKE_CXX_COMPILER=/leonardo/home/userexternal/kzhu0000/opt/lammps_stable/lib/kokkos/bin/nvcc_wrapper \
-D PKG_KOKKOS=ON \
-D Kokkos_ENABLE_CUDA=ON \
-D Kokkos_ARCH_AMPERE100=ON \
-D Kokkos_ENABLE_CUDA_UVM=ON \
-D CMAKE_CUDA_ARCHITECTURES=80 \
-D PKG_OMP=yes \
-D PKG_PLUMED=ON \
-D DOWNLOAD_PLUMED=OFF \
-D PKG_ML-MACE=ON \
-D PKG_ML-IAP=ON \
-D CMAKE_PREFIX_PATH=$CONDA_PREFIX \
-D CMAKE_INSTALL_RPATH="/usr/lib64;$HOME/opt/plumed2.9.4-gnn/lib;$CONDA_PREFIX/lib" \
-D CMAKE_LIBRARY_PATH="$CUDA_HOME/lib64/stubs;$CUDA_HOME/lib64" \
-D CMAKE_EXE_LINKER_FLAGS="-L$CUDA_HOME/lib64/stubs -lcuda" \
-D PKG_MANYBODY=ON
