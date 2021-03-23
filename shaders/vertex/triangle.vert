#version 450

layout(binding = 0) uniform TransformObject {
    mat4 model;
    mat4 view;
    mat4 projection;
} transform;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoords;

layout(location = 0) out vec3 outColor;
layout(location = 1) out vec2 outTexCoords;

void main() {
    gl_Position = transform.projection * transform.view * transform.model * vec4(inPosition, 1.0);
    outColor = inColor;
    outTexCoords = inTexCoords;
}