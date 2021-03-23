#version 450

layout(binding = 1) uniform sampler2D tex_sampler;

layout(location = 0) in vec3 in_color;
layout(location = 1) in vec2 in_tex_coords;

layout(location = 0) out vec4 out_color;

void main() {
    vec4 tex_color = texture(tex_sampler, in_tex_coords); 
    out_color = vec4(tex_color.rgb, 1.0);
}