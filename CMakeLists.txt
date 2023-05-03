cmake_minimum_required(VERSION 3.5)
project(lane_detection)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add source files
set(SOURCES
    src/poly.cpp
    src/ImageProcessor.cpp
    main.cpp
    src/lane.cpp
)

# Add header files
set(HEADERS
    header/poly.hpp
    header/ImageProcessor.hpp
    header/lane.hpp
)

# Add executable target
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link against OpenCV libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
