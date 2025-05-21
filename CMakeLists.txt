# Require CMake 3.15+ (matching scikit-build-core) Use new versions of all
# policies up to CMake 3.27
cmake_minimum_required(VERSION 3.15...3.27)
set(CMAKE_CXX_STANDARD 20)

# Scikit-build-core sets these values for you, or you can just hard-code the
# name and version.
project(
        ${SKBUILD_PROJECT_NAME}
        VERSION ${SKBUILD_PROJECT_VERSION}
        LANGUAGES CXX)

# Find the module development requirements (requires FindPython from 3.17 or
# scikit-build-core's built-in backport)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

set(deepCABAC_DIR nncodec-fork/extensions/deepCABAC/source/Lib)

# Add a library using FindPython's tooling (pybind11 also provides a helper like
# this)
python_add_library(_core
        MODULE
        src/bindings.cpp
        src/cpp/fast_quant.cpp
        src/cpp/entropy_model.cpp
        src/cpp/gptq.cpp
        ${deepCABAC_DIR}/CommonLib/Quant.cpp
        ${deepCABAC_DIR}/CommonLib/ContextModeler.cpp
        ${deepCABAC_DIR}/CommonLib/ContextModel.cpp
        ${deepCABAC_DIR}/EncLib/BinEncoder.cpp
        ${deepCABAC_DIR}/EncLib/CABACEncoder.cpp
        WITH_SOABI)
target_link_libraries(_core PRIVATE pybind11::headers)


# This is passing in the version as a define just as an example
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_include_directories(_core PRIVATE src/cpp ${deepCABAC_DIR})

# The install directory is the output (wheel) directory
install(TARGETS _core DESTINATION nn_compression)