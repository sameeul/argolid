cmake_minimum_required(VERSION 3.24)
project(OmePyrGen)
#==== Compiler Options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# this is a workaround for GitHub Action for wheelbuiling
if(DEFINED ENV{OMEPYRGEN_DEP_DIR})
	set(CMAKE_PREFIX_PATH $ENV{OMEPYRGEN_DEP_DIR})
	link_directories($ENV{OMEPYRGEN_DEP_DIR}/lib)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
  endif()
  
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(SOURCE 
           src/ts_driver/tiled_tiff/omexml.cc
           src/ts_driver/tiled_tiff/tiled_tiff_key_value_store.cc
           src/ts_driver/ometiff/metadata.cc
           src/ts_driver/ometiff/driver.cc
           src/ome_tiff_to_zarr_converter.cpp
           src/zarr_pyramid_assembler.cpp
           src/zarr_base_to_pyr_gen.cpp
           src/ome_tiff_to_zarr_pyramid.cpp
           src/utilities.cpp
)
add_executable(test_pyrgen ${SOURCE} tests/main.cpp)

include(FetchContent)

FetchContent_Declare(
  tensorstore
  URL "https://github.com/google/tensorstore/archive/refs/tags/v0.1.36.tar.gz"
  URL_HASH SHA256=5857582b1b68e5a3f470d2ee8e9a7fa1ad6e6c7e0867c7c69e5a523ae00bc002
)

# Additional FetchContent_Declare calls as needed...

FetchContent_MakeAvailable(tensorstore)

# Define a target that depends on TensorStore...

include_directories(lib/pugixml)
include_directories(lib/plog/include)
include_directories(lib/bs_threadpool)
target_link_libraries(
  test_pyrgen
  PRIVATE
    tensorstore::tensorstore tensorstore::all_drivers
)


find_package(filepattern REQUIRED)
target_link_libraries(test_pyrgen PRIVATE filepattern::filepattern)

find_package(Threads QUIET)
if (Threads_FOUND)
    if (CMAKE_USE_PTHREADS_INIT)
		SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    endif (CMAKE_USE_PTHREADS_INIT)
    list(APPEND Build_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
else ()
	message(STATUS "Unable to find threads. OmePyrGen must have a threading library i.e. pthreads.")
endif ()

# Python bindings.
# PyBind
add_subdirectory(lib/pybind11)

pybind11_add_module(libomepyrgen 
  ${SOURCE}
  src/python/pyramid_python_interface.cpp
)

#target_compile_definitions(libomepyrgen PRIVATE WITH_PYTHON_H)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC")
  target_link_libraries(libomepyrgen PRIVATE stdc++fs)
endif()

target_link_libraries(libomepyrgen PRIVATE tensorstore::tensorstore tensorstore::all_drivers filepattern::filepattern)
target_link_libraries(libomepyrgen PRIVATE ${Build_LIBRARIES})  

target_link_libraries(
  test_pyrgen
  PRIVATE
    ${Build_LIBRARIES}
)