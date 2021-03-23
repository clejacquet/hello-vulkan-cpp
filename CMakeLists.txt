cmake_minimum_required (VERSION 3.10)
project (HelloVulkan)


cmake_policy(SET CMP0074 NEW)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(glm REQUIRED)
find_package(GLFW3 3.3 REQUIRED)
find_package(Vulkan REQUIRED)

if (MSVC)
    add_compile_options(/std:c++latest)
endif()

add_executable(HelloVulkanExe 
    src/main.cpp
    src/vertex.cpp
    )

set_property(TARGET HelloVulkanExe PROPERTY CXX_STANDARD 20)
set_property(TARGET HelloVulkanExe PROPERTY CXX_STANDARD_REQUIRED ON)

target_include_directories(HelloVulkanExe PRIVATE ${GLFW3_INCLUDE_DIR} ${Vulkan_INCLUDE_DIRS} ${GLM_INCLUDE_DIRS})
target_link_libraries(HelloVulkanExe ${GLFW3_LIBRARY} Vulkan::Vulkan)