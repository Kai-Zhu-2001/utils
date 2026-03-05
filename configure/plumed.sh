# PyTorch path
export TORCH_DIR=/work/kzhu/anaconda3/envs/mlcolvar-gnn/lib/python3.11/site-packages/torch

# Library paths
export LD_LIBRARY_PATH=$TORCH_DIR/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/kzhu/opt/plumed2.9.4-gnn/lib:$LD_LIBRARY_PATH

# Use C++17 for libtorch
sed -i "s/c++11/c++17/g" conf*
sed -i "s/c++14/c++17/g" conf*
sed -i "s/c++11/c++17/g" Makefile*
sed -i "s/c++14/c++17/g" Makefile*

# Torch / NCCL linking
export LDFLAGS="
-L$TORCH_DIR/lib -Wl,-rpath,$TORCH_DIR/lib
-L$EBROOTNCCL/lib -Wl,-rpath,$EBROOTNCCL/lib
-lc10 -ltorch_cpu -lc10_cuda -ltorch_cuda -lnccl
"

# Configure PLUMED
./configure \
CXX=mpicxx \
CXXFLAGS="-O3 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1" \
CPPFLAGS="-I$TORCH_DIR/include -I$TORCH_DIR/include/torch/csrc/api/include" \
LDFLAGS="-L$TORCH_DIR/lib -Wl,-rpath,$TORCH_DIR/lib" \
--prefix=/work/kzhu/opt/plumed2.9.4-gnn \
--enable-mpi \
--enable-openmp \
--enable-libtorch \
--enable-modules=all \

make -j 8 install
