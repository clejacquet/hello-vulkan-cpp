#ifndef DATA_H
#define DATA_H

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <vector>
#include <array>
#include <vulkan/vulkan.hpp>


#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>




struct Vertex {
    glm::vec3 position;
    glm::vec3 color; 
    glm::vec2 tex_coords;

    static vk::VertexInputBindingDescription getBindingDescription() {
        return vk::VertexInputBindingDescription {
            0,
            sizeof(Vertex),
            vk::VertexInputRate::eVertex
        };
    }

    static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() {
        return std::array<vk::VertexInputAttributeDescription, 3> {
            vk::VertexInputAttributeDescription {
                0,
                0,
                vk::Format::eR32G32B32Sfloat,
                offsetof(Vertex, position)
            },
            vk::VertexInputAttributeDescription {
                1,
                0,
                vk::Format::eR32G32B32Sfloat,
                offsetof(Vertex, color)
            },
            vk::VertexInputAttributeDescription {
                2,
                0,
                vk::Format::eR32G32Sfloat,
                offsetof(Vertex, tex_coords)
            }
        };
    }

    bool operator==(const Vertex& other) const {
        return position == other.position && color == other.color && tex_coords == other.tex_coords;
    }
};



namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<glm::vec3>()(vertex.position) ^
                   (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
                   (hash<glm::vec2>()(vertex.tex_coords) << 1);
        }
    };
}

std::vector<Vertex> loadVertices();
std::vector<uint16_t> loadIndices();

#endif