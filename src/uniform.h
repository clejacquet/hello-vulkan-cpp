#ifndef UNIFORM_H
#define UNIFORM_H

#include <glm/mat4x4.hpp>

struct TransformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
};



#endif