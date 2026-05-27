cmake_minimum_required(VERSION 3.15)
project(Octopus CXX)

set(CMAKE_CXX_STANDARD 20)    # 强制开启C++20 标准
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
if(Eigen3_FOUND)
    message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Eigen3 NOT FOUND! Please install it first.")
endif()


#add_executable(itensor_test tests/test_all.cpp)
add_executable(itensor_test ./test_all.cpp)


target_compile_definitions(itensor_test PRIVATE TEST_IT)
target_link_libraries(itensor_test PRIVATE Eigen3::Eigen)


#target_compile_options(itensor_test PRIVATE
#    $<$<CXX_COMPILER_ID:GNU,Clang>:
#        -O3                             # 最高级性能优化
#        -march=native                   # 原地激活本机的 AVX2 和 FMA3
#        -ffast-math                     # 牺牲部分 IEEE 精度换取浮点极致吞吐
#        -Wall                           # 严格审查代码 Bug
#        -Wno-deprecated-enum-enum-conversion # 强行让 Eigen 的枚举警告闭嘴
#    >
#)
#
#

#enable_testing()
#add_test(NAME test_tensor COMMAND test_tensor_exe)
#add_test(NAME test_quantum COMMAND test_quantum_exe)


## 声明张量测试程序（它拥有自己独立的独立编译单元和独立的 main）
#add_executable(test_tensor_exe tests/test_tensor.cpp)
#target_link_libraries(test_tensor_exe PRIVATE Eigen3::Eigen)
#
## 声明量子数测试程序（同样物理隔离）
#add_executable(test_quantum_exe tests/test_quantum.cpp)
#target_link_libraries(test_quantum_exe PRIVATE Eigen3::Eigen)

## 目标 1：只测试张量 (编译巨快，只重编张量逻辑)
#add_executable(test_tensor_exe tests/test_all.cpp)
#target_compile_definitions(test_tensor_exe PRIVATE ITENSOR_TENSOR_TEST)
#target_link_libraries(test_tensor_exe PRIVATE Eigen3::Eigen)
#
## 目标 2：只测量子数 (编译巨快，不带任何张量包袱)
#add_executable(test_quantum_exe tests/test_all.cpp)
#target_compile_definitions(test_quantum_exe PRIVATE ITENSOR_QUANTUM_NUMBER_TEST)
#target_link_libraries(test_quantum_exe PRIVATE Eigen3::Eigen)
#
## 目标 3：全量大盘测试 (全量构建，CI/CD流水线常用)
#add_executable(test_all_exe tests/test_all.cpp)
#target_compile_definitions(test_all_exe PRIVATE ITENSOR_TENSOR_TEST ITENSOR_QUANTUM_NUMBER_TEST)
#target_link_libraries(test_all_exe PRIVATE Eigen3::Eigen)
#
#cmake --build build --target test_all_exe

#" 仅增量编译并运行量子数测试，其余几百个文件动都不会动
#:!cmake --build build --target test_quantum_exe && ./build/test_quantum_exe

# example of compiling
    # cmake --build build --parallel 24  # using 24 cores 
    # cmake --build build --target test_A    # only compile test_A.cpp 


# automatic
    ## 1. 自动抓取 test 目录下所有带 main 的源文件
    #file(GLOB TEST_SOURCES "test/*.cpp")
    ## 2. 核心黑魔法：用一个循环，把它们彼此孤立开
    #foreach(test_file ${TEST_SOURCES})
    #    # 从路径中提取出文件名（比如把 "test/cholesky.cpp" 提取成 "cholesky"）
    #    get_filename_component(test_name ${test_file} NAME_WE)
    #    # 【最关键的隔离动作】：每一个 .cpp 文件，都被单独声明为一个“独立的可执行程序目标”！
    #    add_executable(${test_name} ${test_file})
    #    # 让这几百个独立的小程序分别去挂载 Eigen 军火库
    #    target_link_libraries(${test_name} PRIVATE Eigen3::Eigen)
    #endforeach()
    #
