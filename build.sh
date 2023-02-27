function build_dipu_py() {
    export CMAKE_BUILD_TYPE=debug
    export _GLIBCXX_USE_CXX11_ABI=1
    export MAX_JOBS=12
    # PYTORCH_INSTALL_DIR is /mnt/lustre/fandaoyi/torch20/pytorch/torch
    # python  setup.py build_clib 2>&1 | tee ./build1.log
    python setup.py build_ext 2>&1 | tee ./build1.log
    cp build/python_ext/torch_dipu/_C.cpython-37m-x86_64-linux-gnu.so torch_dipu
}


function config_dipu_cmake() {
    cd ./build && rm -rf ./*
    PYTORCH_DIR="/mnt/lustre/fandaoyi/torch20/pytorch"
    PYTHON_INCLUDE_DIR="/mnt/lustre/fandaoyi/.conda/envs/torch20/include/python3.7m"
    cmake ../  -DCMAKE_BUILD_TYPE=Debug \
     -DCAMB=ON -DPYTORCH_DIR=${PYTORCH_DIR} \
     -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR}
      # -DCMAKE_C_FLAGS_DEBUG="-g -O0" \
      # -DCMAKE_CXX_FLAGS_DEBUG="-g -O0"
    cd ../
}

function build_dipu_lib() {
    export DIOPI_ROOT=/mnt/lustre/fandaoyi/camb241_torch19/DIOPI-IMPL/camb/build
    export LIBRARY_PATH=$DIOPI_ROOT:$LIBRARY_PATH;

    config_dipu_cmake
    #  2>&1 | tee ./build1.log
    cd build && make -j8  2>&1 | tee ./build1.log &&  cd ..
    cp ./build/torch_dipu/csrc_dipu/libtorch_dipu.so   ./torch_dipu
    cp ./build/torch_dipu/csrc_dipu/libtorch_dipu_python.so   ./torch_dipu
}

if [[ "$1" == "builddl" ]]; then
    build_dipu_lib
elif [[ "$1" == "builddp" ]]; then
    build_dipu_py