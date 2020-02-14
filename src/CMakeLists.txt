cmake_minimum_required(VERSION 3.12)
set(PROJECT_NAME_STR fquantizerlib-build)
project(${PROJECT_NAME_STR})

set(CMAKE_BUILD_TYPE Release)
add_definitions("-std=c++11")

file(GLOB PROJECT_SRC_FILES ${PROJECT_SOURCE_DIR}/*.cpp)
set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../include)
include_directories(${PROJECT_INCLUDE_DIR})
add_definitions     (${EIGEN3_DEFINITIONS})
include_directories (${PROJECT_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIRS} ${MPFR_INCLUDES})

add_library(fquantizer SHARED ${PROJECT_SRC_FILES})


target_link_libraries(fquantizer PRIVATE ${GMP_LIBRARIES} ${MPFR_LIBRARIES} ${FPLLL_LIBRARY} OpenMP::OpenMP_CXX)

install(TARGETS fquantizer DESTINATION lib)
install(DIRECTORY ${PROJECT_INCLUDE_DIR}/fquantizer DESTINATION include)
