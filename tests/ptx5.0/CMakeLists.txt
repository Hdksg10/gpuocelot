cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# PTX JIT compilation require CUDA Toolkit
find_package(CUDAToolkit REQUIRED)
find_package(LLVM REQUIRED CONFIG)
message(STATUS ${${PROJECT_NAME}_INCLUDE_DIRS})
set(${PROJECT_NAME}_INCLUDE_DIRS
    ${${PROJECT_NAME}_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${CUDAToolkit_INCLUDE_DIRS}
    )


file(GLOB TESTS_NEW_PTX_SRC "./src/*.cpp")
# file(GLOB TESTS_NEW_PTX_HEADER "./*.h")

add_library(${PROJECT_NAME}_tests_new_ptx SHARED ${TESTS_NEW_PTX_SRC})
target_include_directories(${PROJECT_NAME}_tests_new_ptx PRIVATE ${${PROJECT_NAME}_INCLUDE_DIRS})
set_property(TARGET ${PROJECT_NAME}_tests_new_ptx PROPERTY CXX_STANDARD 17)
# target_include_directories(${PROJECT_NAME}_tests_new_ptx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/${name})

# target_link_directories(${PROJECT_NAME}_tests_new_ptx PUBLIC ${CUDAToolkit_LIBRARY_DIR})
target_link_libraries(${PROJECT_NAME}_tests_new_ptx PUBLIC CUDA::cuda_driver ${PROJECT_NAME} ${${PROJECT_NAME}_LINK_LIBRARIES})


function(ptx_add_tests name)
    file(GLOB TESTS "instruction/${name}/Test*.cpp")
    foreach(TEST ${TESTS})
        get_filename_component(TEST_WE ${TEST} NAME_WE)
        set(TEST_NAME ${PROJECT_NAME}_${name}_${TEST_WE})
        add_executable(${TEST_NAME} ${TEST})
        set_property(TARGET ${TEST_NAME} PROPERTY CXX_STANDARD 17)
        target_compile_definitions(${TEST_NAME} PRIVATE ${${PROJECT_NAME}_DEFINITIONS})
        target_compile_options(${TEST_NAME} PRIVATE PRIVATE ${${PROJECT_NAME}_CXXFLAGS})
        target_include_directories(${TEST_NAME} PRIVATE ${${PROJECT_NAME}_INCLUDE_DIRS})
        target_link_libraries(${TEST_NAME} PRIVATE ${PROJECT_NAME}_tests_new_ptx)
        

        add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${TEST_NAME}> -v)
        set_property(TEST ${TEST_NAME} PROPERTY ENVIRONMENT LD_PRELOAD=$<TARGET_FILE:${PROJECT_NAME}>)

    endforeach()
endfunction()

ptx_add_tests(lop3)
ptx_add_tests(shf)
