
rm -rf build

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/leonardo/home/userexternal/kzhu0000/opt/gromacs2025.4 \
    -DGMX_MPI=ON \
    -DGMX_THREAD_MPI=OFF \
    -DGMX_BUILD_OWN_FFTW=ON \
    -DGMX_GPU=CUDA \
    -DGMX_USE_PLUMED=ON

cmake --build build --parallel 8
cmake --install build
