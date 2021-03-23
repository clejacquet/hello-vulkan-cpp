#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <vulkan/vulkan.hpp>

#include <GLFW/glfw3.h>
#include <glm/vec3.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>
#include <fstream>
#include <set>
#include <optional>
#include <iostream>
#include <unordered_map>
#include <stdexcept>
#include <functional>
#include <algorithm>
#include <cstdlib>

#include "vertex.h"
#include "uniform.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static constexpr int MAX_FRAMES_IN_FLIGHT = 6;

static std::string GetSeverityString(VkDebugUtilsMessageSeverityFlagBitsEXT severity_flags) {
    if (severity_flags & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        return "ERROR";
    } else if (severity_flags & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        return "WARNING";
    } else if (severity_flags & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
        return "INFO";
    } else if (severity_flags & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
        return "VERBOSE";
    } else {
        return "<undefined>";
    }
}

static std::vector<uint32_t> readFile(const std::string& filename) {
    auto file = std::ifstream { filename, std::ios::ate | std::ios::binary };

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file \"" + filename + "\"");
    }

    auto file_size = std::size_t(file.tellg());
    auto file_contents = std::vector<uint32_t>(int(std::ceil(float(file_size) / float(sizeof(uint32_t)))));

    file.seekg(0);
    file.read((char*) file_contents.data(), file_size);

    file.close();

    return file_contents;
}

struct QueueFamilyIndices {
    std::optional<uint32_t> graphics_family;
    std::optional<uint32_t> present_family;

    bool isComplete() {
        return graphics_family.has_value() && present_family.has_value();
    }
};

struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

class HelloTriangleApplication {
private:
    const uint32_t WIDTH = 800;
    const uint32_t HEIGHT = 600;
    const std::string MODEL_PATH = "assets/viking_room.obj";
    const std::string TEXTURE_PATH = "assets/viking_room.png";
    
    GLFWwindow* _window;
    vk::Instance _instance;
    vk::PhysicalDevice _physical_device;
    vk::Device _logical_device;
    vk::Queue _graphics_queue;
    vk::Queue _present_queue;
    vk::SurfaceKHR _surface;
    vk::SwapchainKHR _swap_chain;
    std::vector<vk::Image> _swap_chain_images;
    std::vector<vk::ImageView> _swap_chain_image_views;
    vk::Format _swap_chain_format;
    vk::Extent2D _swap_chain_extent;
    std::vector<vk::Framebuffer> _swap_chain_framebuffers;
    vk::DescriptorSetLayout _descriptor_set_layout;
    vk::PipelineLayout _pipeline_layout;
    vk::RenderPass _render_pass;
    vk::Pipeline _graphics_pipeline;
    vk::CommandPool _command_pool;
    std::vector<vk::CommandBuffer> _command_buffers;
    std::vector<vk::Semaphore> _image_available_semaphores;
    std::vector<vk::Semaphore> _render_finished_semaphores;
    std::vector<vk::Fence> _in_flight_fences;
    std::vector<vk::Fence> _images_in_flight;
    int _current_frame = 0;
    bool _framebuffer_resized = false;
    std::vector<Vertex> _vertices;
    std::vector<uint32_t> _indices;
    vk::Buffer _vertex_indice_buffer;
    vk::DeviceMemory _vertex_indice_buffer_memory;
    std::vector<vk::Buffer> _transform_buffers;
    std::vector<vk::DeviceMemory> _transform_buffers_memory;
    vk::DescriptorPool _descriptor_pool;
    std::vector<vk::DescriptorSet> _descriptor_sets;
    uint32_t _mip_levels;
    vk::Image _texture_image;
    vk::DeviceMemory _texture_image_memory;
    vk::ImageView _texture_image_view;
    vk::Sampler _texture_sampler;
    vk::Image _depth_image;
    vk::ImageView _depth_image_view;
    vk::DeviceMemory _depth_image_memory;
    vk::SampleCountFlagBits _msaa_sample_count;
    vk::Image _color_image;
    vk::DeviceMemory _color_image_memory;
    vk::ImageView _color_image_view;
    
    #ifndef NDEBUG
    VkDebugUtilsMessengerEXT _debug_callback;

    std::vector<const char*> _validation_layers = {
        "VK_LAYER_KHRONOS_validation"
    };
    #endif

    std::vector<const char*> _device_extensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };


    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData) 
    {
        std::cerr << "validation layer: [" << GetSeverityString(messageSeverity) << "] " << pCallbackData->pMessage << std::endl;

        return VK_FALSE;
    }

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        _window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(_window, this);
        glfwSetFramebufferSizeCallback(_window, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app_ptr = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app_ptr->_framebuffer_resized = true;
    }

    bool checkValidationLayerSupport(const std::vector<const char*>& validation_layers) const {
        auto available_layers = vk::enumerateInstanceLayerProperties();

        // Returns true if all layers has been found
        return validation_layers.end() == std::find_if_not(validation_layers.begin(), validation_layers.end(), [&available_layers](auto& layer) {
            // Returns true if layer has been found among the available layers
            return available_layers.end() != std::find_if(available_layers.begin(), available_layers.end(), [&layer](auto& available_layer) {
                return std::string(available_layer.layerName.data()) == std::string(layer);
            });
        });
    }

    std::vector<const char*> getUsedExtensions() const {
        uint32_t glfw_extension_count = 0;
        const char** glfw_extensions;
        glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

        auto extensions = std::vector<const char*> { glfw_extensions, glfw_extensions + glfw_extension_count };

        #ifndef NDEBUG 
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        #endif

        return extensions;
    }


#ifndef NDEBUG
    VkResult createDebugMessenger(VkDebugUtilsMessengerCreateInfoEXT& create_info) {
        auto func = PFN_vkCreateDebugUtilsMessengerEXT(vkGetInstanceProcAddr(_instance, "vkCreateDebugUtilsMessengerEXT"));
        if (func != nullptr) {
            return func(_instance, &create_info, nullptr, &_debug_callback);
        } else {
            return VK_ERROR_EXTENSION_NOT_PRESENT;
        }
    }

    void destroyDebugMessenger() {
        auto func = PFN_vkDestroyDebugUtilsMessengerEXT(vkGetInstanceProcAddr(_instance, "vkDestroyDebugUtilsMessengerEXT"));

        if (func != nullptr) {
            func(_instance, _debug_callback, nullptr);
        }
    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& create_info) {
        create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        create_info.messageSeverity = 
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        create_info.messageType = 
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | 
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | 
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        create_info.pfnUserCallback = debugCallback;
        create_info.pUserData = nullptr;
    }

    void setupDebugMessenger() {
        VkDebugUtilsMessengerCreateInfoEXT create_info;
        populateDebugMessengerCreateInfo(create_info);

        if (createDebugMessenger(create_info) != VK_SUCCESS) {
            throw std::runtime_error("Debugging Messenger could not be created");
        } else {
            std::cout << "Debugging Messenger successfully created" << std::endl;
        }
    }
#endif

    SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice device) {
        SwapChainSupportDetails details;

        details.capabilities = device.getSurfaceCapabilitiesKHR(_surface);
        details.formats = device.getSurfaceFormatsKHR(_surface);
        details.presentModes = device.getSurfacePresentModesKHR(_surface);

        return details;
    }

    QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device) {
        QueueFamilyIndices indices;

        auto queue_family_properties = device.getQueueFamilyProperties();

        int i = 0;
        for (auto& queue_family : queue_family_properties) {
            auto presentation_support = device.getSurfaceSupportKHR(i, _surface);

            if (presentation_support) {
                indices.present_family = i;
            }

            if (queue_family.queueFlags & vk::QueueFlagBits::eGraphics) {
                indices.graphics_family = i;
            }

            if (indices.isComplete()) {
                break;
            }

            ++i;
        }

        return indices;
    }

    bool checkDeviceExtensionSupport(vk::PhysicalDevice device) {
        auto device_extensions = device.enumerateDeviceExtensionProperties();

        auto required_extension_names = std::set<std::string>(_device_extensions.begin(), _device_extensions.end());

        for (auto& extension : device_extensions) {
            required_extension_names.erase(std::string(extension.extensionName.data()));
        }

        return required_extension_names.empty();
    }

    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& available_formats) {
        for (auto& available_format : available_formats) {
            if (available_format.format == vk::Format::eB8G8R8A8Srgb && available_format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                return available_format;
            }
        }

        return available_formats.front();
    }

    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& available_modes) {
        for (auto& available_mode : available_modes) {
            if (available_mode == vk::PresentModeKHR::eMailbox) {
                return available_mode;
            }
        }

        return vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != UINT32_MAX) {
            return capabilities.currentExtent;
        }

        int width, height;
        glfwGetFramebufferSize(_window, &width, &height);

        return {
            std::clamp(uint32_t(width), capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            std::clamp(uint32_t(height), capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
        };
    }

    bool isDeviceSuitable(vk::PhysicalDevice device) {
        // VkPhysicalDeviceProperties device_properties;
        // vkGetPhysicalDeviceProperties(device, &device_properties);

        // std::cout << device_properties.deviceName << std::endl;

        auto queue_family = findQueueFamilies(device);

        bool are_extension_supported = checkDeviceExtensionSupport(device);

        auto supported_features = device.getFeatures();

        bool swap_chain_compatible = false;
        if (are_extension_supported) {
            auto swap_chain_support_details = querySwapChainSupport(device);

            swap_chain_compatible = !swap_chain_support_details.formats.empty() && !swap_chain_support_details.presentModes.empty();
        }

        return queue_family.isComplete() && are_extension_supported && swap_chain_compatible && supported_features.samplerAnisotropy;
    }

    vk::SampleCountFlagBits getMaxUsableSampleCount() {
        
        auto physical_properties = _physical_device.getProperties();

        auto max_sample_count =
            physical_properties.limits.framebufferColorSampleCounts &
            physical_properties.limits.framebufferDepthSampleCounts;

        if (max_sample_count & vk::SampleCountFlagBits::e64) { std::cout << "vk::SampleCountFlagBits::e64" << std::endl; }
        else if (max_sample_count & vk::SampleCountFlagBits::e32) { std::cout << "vk::SampleCountFlagBits::e32" << std::endl; }
        else if (max_sample_count & vk::SampleCountFlagBits::e16) { std::cout << "vk::SampleCountFlagBits::e16" << std::endl; }
        else if (max_sample_count & vk::SampleCountFlagBits::e8) { std::cout << "vk::SampleCountFlagBits::e8" << std::endl; }
        else if (max_sample_count & vk::SampleCountFlagBits::e4) { std::cout << "vk::SampleCountFlagBits::e4" << std::endl; }
        else if (max_sample_count & vk::SampleCountFlagBits::e2) { std::cout << "vk::SampleCountFlagBits::e2" << std::endl; }


        if (max_sample_count & vk::SampleCountFlagBits::e64) { return vk::SampleCountFlagBits::e64; }
        if (max_sample_count & vk::SampleCountFlagBits::e32) { return vk::SampleCountFlagBits::e32; }
        if (max_sample_count & vk::SampleCountFlagBits::e16) { return vk::SampleCountFlagBits::e16; }
        if (max_sample_count & vk::SampleCountFlagBits::e8) { return vk::SampleCountFlagBits::e8; }
        if (max_sample_count & vk::SampleCountFlagBits::e4) { return vk::SampleCountFlagBits::e4; }
        if (max_sample_count & vk::SampleCountFlagBits::e2) { return vk::SampleCountFlagBits::e2; }

        return vk::SampleCountFlagBits::e1;
    }

    void pickPhysicalDevice() {
        auto physical_devices = _instance.enumeratePhysicalDevices();

        if (physical_devices.empty()) {
            throw std::runtime_error("No GPU available with Vulkan support");
        }

        for (auto& device : physical_devices) {
            if (isDeviceSuitable(device)) {
                _physical_device = device;
                _msaa_sample_count = getMaxUsableSampleCount();
                break;
            }
        }

        if (!_physical_device) {
            throw std::runtime_error("No suitable GPU found");
        }
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(_window)) {
            glfwPollEvents();
            drawFrame();
        }

        _logical_device.waitIdle();
    }

    void drawFrame() {
        auto wait_result = _logical_device.waitForFences(_in_flight_fences[_current_frame], true, UINT64_MAX);

        if (wait_result != vk::Result::eSuccess && wait_result != vk::Result::eTimeout) {
            throw std::runtime_error("Could not wait on fences");
        }

        auto [ acquire_result, image_index ] = _logical_device.acquireNextImageKHR(_swap_chain, UINT64_MAX, _image_available_semaphores[_current_frame], {});
        
        if (acquire_result == vk::Result::eErrorOutOfDateKHR) {
            recreateSwapChain();
            return;
        } else if (acquire_result != vk::Result::eSuccess && acquire_result != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error("Could not acquire swap chain image");
        }

        // Check if a previous frame is using this image (i.e. there is its fence to wait on)
        if (_images_in_flight[image_index]) {
            auto wait_result = _logical_device.waitForFences(_images_in_flight[image_index], true, UINT64_MAX);

            if (wait_result != vk::Result::eSuccess && wait_result != vk::Result::eTimeout) {
                throw std::runtime_error("Could not wait on fences");
            }
        }

        _images_in_flight[image_index] = _in_flight_fences[_current_frame];

        updateUniformBuffer(image_index);

        vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;

        auto submit_info = vk::SubmitInfo {
            _image_available_semaphores[_current_frame],
            wait_stage,
            _command_buffers[image_index],
            _render_finished_semaphores[_current_frame]
        };

        _logical_device.resetFences(_in_flight_fences[_current_frame]);

        _graphics_queue.submit(submit_info, _in_flight_fences[_current_frame]);

        auto present_info = vk::PresentInfoKHR {
            _render_finished_semaphores[_current_frame],
            _swap_chain,
            image_index
        };

        auto present_result = _present_queue.presentKHR(&present_info);

        if (present_result == vk::Result::eErrorOutOfDateKHR || present_result == vk::Result::eSuboptimalKHR || _framebuffer_resized) {
            recreateSwapChain();
            _framebuffer_resized = false;
        } else if (present_result != vk::Result::eSuccess) {
            throw std::runtime_error("Could not present swap chain image");
        }

        _current_frame = (_current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void updateUniformBuffer(int image_index) {
        static auto start_time = std::chrono::steady_clock::now();

        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(now - start_time).count();
        start_time = now;

        static float angle = 0.0f; 
        float speed = 0.05f * (2.0f * glm::pi<float>() / 1000.0f);

        angle += speed * dt;

        auto transform = TransformBufferObject {};
        transform.model = glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0.0f, 0.0f, 1.0f));
        transform.view = glm::lookAt(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        transform.projection = glm::perspective(glm::radians(70.0f), float(_swap_chain_extent.width) / float(_swap_chain_extent.height), 0.1f, 100.0f);

        // Because of Vulkan Y coordinate inverted compared to OpenGL Y coordinate
        transform.projection[1][1] *= -1.0f;
        
        void* data = _logical_device.mapMemory(_transform_buffers_memory[image_index], 0, sizeof(TransformBufferObject));

        memcpy(data, &transform, sizeof(TransformBufferObject));

        _logical_device.unmapMemory(_transform_buffers_memory[image_index]);
    }

    void createInstance() {
        auto app_info = vk::ApplicationInfo {
            "Hello Triangle",
            VK_MAKE_VERSION(1, 0, 0),
            "Custom Engine",
            VK_MAKE_VERSION(1, 0, 0),
            VK_API_VERSION_1_0
        };

        auto used_extensions = getUsedExtensions();
        
        auto create_info = vk::InstanceCreateInfo {
            {},
            &app_info,
            {},
            used_extensions,
        };

        #ifndef NDEBUG
            VkDebugUtilsMessengerCreateInfoEXT debug_create_info;
            populateDebugMessengerCreateInfo(debug_create_info);

            create_info.enabledLayerCount = uint32_t(_validation_layers.size());
            create_info.ppEnabledLayerNames = _validation_layers.data();
            create_info.pNext = &debug_create_info;
        #else
            create_info.enabledLayerCount = 0;
            create_info.pNext = nullptr;
        #endif
        
        _instance = vk::createInstance(create_info);
    }


    void createSurface() {
        auto raw_surface = VkSurfaceKHR(_surface);
        
        if (glfwCreateWindowSurface(_instance, _window, nullptr, &raw_surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface");
        }

        _surface = raw_surface;
    }

    void createLogicalDevice() {
        auto indices = findQueueFamilies(_physical_device);

        auto queue_priority = std::vector<float> { 1.0f };

        std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;
        std::set<uint32_t> unique_queue_families = { indices.graphics_family.value(), indices.present_family.value() };

        for (auto& queue_family : unique_queue_families) {
            
            auto queue_create_info = vk::DeviceQueueCreateInfo {
                {},
                queue_family,
                queue_priority
            };

            queue_create_infos.push_back(queue_create_info);
        }

        auto device_features = vk::PhysicalDeviceFeatures {};
        device_features.samplerAnisotropy = true;
        device_features.sampleRateShading = true;

        auto device_create_info = vk::DeviceCreateInfo {
            {},
            queue_create_infos,
            {},
            _device_extensions,
            &device_features
        };

        _logical_device = _physical_device.createDevice(device_create_info);
        _graphics_queue = _logical_device.getQueue(indices.graphics_family.value(), 0);
        _present_queue = _logical_device.getQueue(indices.present_family.value(), 0);
    }

    void createSwapChain() {
        auto support_details = querySwapChainSupport(_physical_device);

        auto surface_format = chooseSwapSurfaceFormat(support_details.formats);
        auto present_mode = chooseSwapPresentMode(support_details.presentModes);
        auto extent = chooseSwapExtent(support_details.capabilities);

        auto min_image_count = support_details.capabilities.minImageCount + 1;

        if (support_details.capabilities.maxImageCount > 0 && min_image_count > support_details.capabilities.maxImageCount) {
            min_image_count = support_details.capabilities.maxImageCount;
        }

        auto indices = findQueueFamilies(_physical_device);

        auto create_info = vk::SwapchainCreateInfoKHR {
            {},
            _surface,
            min_image_count,
            surface_format.format,
            surface_format.colorSpace,
            extent,
            1,
            vk::ImageUsageFlagBits::eColorAttachment,
            (indices.graphics_family != indices.present_family) ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive,
            (indices.graphics_family != indices.present_family) ? std::vector<uint32_t> { indices.graphics_family.value(), indices.present_family.value() } : std::vector<uint32_t> {},
            support_details.capabilities.currentTransform,
            vk::CompositeAlphaFlagBitsKHR::eOpaque,
            present_mode,
            true
        };

        _swap_chain = _logical_device.createSwapchainKHR(create_info);
        _swap_chain_images = _logical_device.getSwapchainImagesKHR(_swap_chain);

        _swap_chain_format = surface_format.format;
        _swap_chain_extent = extent;
    }

    vk::ImageView createImageView2D(vk::Image image, vk::Format format, vk::ImageAspectFlags aspect_flags, uint32_t mip_levels) {
        vk::ImageView image_view;

        auto create_info = vk::ImageViewCreateInfo {
            {},
            image,
            vk::ImageViewType::e2D,
            format,
            {},
            vk::ImageSubresourceRange {
                aspect_flags,
                0,
                mip_levels,
                0,
                1
            }
        };

        return _logical_device.createImageView(create_info);
    }

    void createImageViews() {
        std::cout << "Swap chain images: " << _swap_chain_images.size() << std::endl;
        _swap_chain_image_views.resize(_swap_chain_images.size());

        std::transform(_swap_chain_images.begin(), _swap_chain_images.end(), _swap_chain_image_views.begin(), [&](const auto& image) {
            return createImageView2D(image, _swap_chain_format, vk::ImageAspectFlagBits::eColor, 1);
        });
    }

    vk::ShaderModule createShaderModule(const std::vector<uint32_t>& code) {
        auto create_info = vk::ShaderModuleCreateInfo {
            {},
            code
        };

        return _logical_device.createShaderModule(create_info);
    }

    void createRenderPass() {
        auto color_attachment = vk::AttachmentDescription {
            {},
            _swap_chain_format,
            _msaa_sample_count,
            vk::AttachmentLoadOp::eClear,
            vk::AttachmentStoreOp::eStore,
            vk::AttachmentLoadOp::eDontCare,
            vk::AttachmentStoreOp::eDontCare,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal
        };

        auto depth_attachment = vk::AttachmentDescription {
            {},
            findDepthFormat(),
            _msaa_sample_count,
            vk::AttachmentLoadOp::eClear,
            vk::AttachmentStoreOp::eDontCare,
            vk::AttachmentLoadOp::eDontCare,
            vk::AttachmentStoreOp::eDontCare,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eDepthStencilAttachmentOptimal
        };

        auto color_attachment_resolve = vk::AttachmentDescription {
            {},
            _swap_chain_format,
            vk::SampleCountFlagBits::e1,
            vk::AttachmentLoadOp::eClear,
            vk::AttachmentStoreOp::eStore,
            vk::AttachmentLoadOp::eDontCare,
            vk::AttachmentStoreOp::eDontCare,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::ePresentSrcKHR
        };

        auto color_attachment_reference = vk::AttachmentReference {
            0,
            vk::ImageLayout::eColorAttachmentOptimal
        };

        auto depth_attachment_reference = vk::AttachmentReference {
            1,
            vk::ImageLayout::eDepthStencilAttachmentOptimal
        };

        auto color_attachment_resolve_reference = vk::AttachmentReference {
            2,
            vk::ImageLayout::eColorAttachmentOptimal
        };

        auto subpass = vk::SubpassDescription {
            {},
            vk::PipelineBindPoint::eGraphics,
            {},
            color_attachment_reference,
            color_attachment_resolve_reference,
            &depth_attachment_reference,
            {}
        };

        auto subpass_dependency = vk::SubpassDependency {
            VK_SUBPASS_EXTERNAL,
            0,
            vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
            vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
            {},
            vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite
        };

        auto attachments = std::vector<vk::AttachmentDescription> { color_attachment, depth_attachment, color_attachment_resolve };

        auto render_pass_info = vk::RenderPassCreateInfo {
            {},
            attachments,
            subpass,
            subpass_dependency
        };

        _render_pass = _logical_device.createRenderPass(render_pass_info);
    }

    void createDescriptorSetLayouts() {
        auto ubo_layout_binding = vk::DescriptorSetLayoutBinding {
            0,
            vk::DescriptorType::eUniformBuffer,
            1,
            vk::ShaderStageFlagBits::eVertex,
            {}
        };

        auto sampler_layout_binding = vk::DescriptorSetLayoutBinding {
            1,
            vk::DescriptorType::eCombinedImageSampler,
            1,
            vk::ShaderStageFlagBits::eFragment,
            nullptr
        };

        auto bindings = std::vector<vk::DescriptorSetLayoutBinding> { ubo_layout_binding, sampler_layout_binding };

        auto descriptor_layout_info = vk::DescriptorSetLayoutCreateInfo {
            {},
            bindings
        };

        _descriptor_set_layout = _logical_device.createDescriptorSetLayout(descriptor_layout_info);
    }


    void createGraphicsPipeline() {
        auto vert_code = readFile("shaders/spirv/triangle.vert.spv");
        auto frag_code = readFile("shaders/spirv/color.frag.spv");

        auto vert_module = createShaderModule(vert_code);
        auto frag_module = createShaderModule(frag_code);

        auto vert_stage_create_info = vk::PipelineShaderStageCreateInfo {
            {},
            vk::ShaderStageFlagBits::eVertex,
            vert_module,
            "main"
        };

        auto frag_stage_create_info = vk::PipelineShaderStageCreateInfo {
            {},
            vk::ShaderStageFlagBits::eFragment,
            frag_module,
            "main"
        };

        auto shader_stages = std::vector<vk::PipelineShaderStageCreateInfo> { vert_stage_create_info, frag_stage_create_info };

        auto binding_desc = Vertex::getBindingDescription();
        auto attributes_desc = Vertex::getAttributeDescriptions();

        auto vertex_input_info = vk::PipelineVertexInputStateCreateInfo {
            {},
            binding_desc,
            attributes_desc,
        };

        auto input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo {
            {},
            vk::PrimitiveTopology::eTriangleList,
            false
        };

        auto viewport = vk::Viewport {
            0.0f,
            0.0f,
            float(_swap_chain_extent.width),
            float(_swap_chain_extent.height),
            0.0f,
            1.0f
        };

        auto scissor = vk::Rect2D {
            { 0, 0 },
            _swap_chain_extent
        };

        auto viewport_info = vk::PipelineViewportStateCreateInfo {
            {},
            viewport,
            scissor
        };

        auto rasterizer_info = vk::PipelineRasterizationStateCreateInfo {
            {},
            false,
            false,
            vk::PolygonMode::eFill,
            vk::CullModeFlagBits::eFront,
            vk::FrontFace::eClockwise,
            false,
            0.0f,
            0.0f,
            0.0f,
            1.0f
        };

        auto multisampling_info = vk::PipelineMultisampleStateCreateInfo {
            {},
            _msaa_sample_count,
            true,
            1.0f
        };

        auto color_blend_attachment = vk::PipelineColorBlendAttachmentState {
            false,
            vk::BlendFactor::eOne,
            vk::BlendFactor::eZero,
            vk::BlendOp::eAdd,
            vk::BlendFactor::eOne,
            vk::BlendFactor::eZero,
            vk::BlendOp::eAdd,
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        };

        auto color_blend_info = vk::PipelineColorBlendStateCreateInfo {
            {},
            false,
            vk::LogicOp::eCopy,
            color_blend_attachment,
            {
                0.0f,
                0.0f,
                0.0f,
                0.0f
            }
        };

        auto depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo {
            {},
            true,
            true,
            vk::CompareOp::eLess,
            false,
            false,
            {},
            {},
            0.0f,
            1.0f
        };

        // VkDynamicState dynamic_states[] = {
        //     VK_DYNAMIC_STATE_VIEWPORT,
        //     VK_DYNAMIC_STATE_LINE_WIDTH
        // };

        // auto dynamic_state_info = VkPipelineDynamicStateCreateInfo {};
        // dynamic_state_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        // dynamic_state_info.dynamicStateCount = 2;
        // dynamic_state_info.pDynamicStates = dynamic_states;


        auto pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            {},
            _descriptor_set_layout
        };

        _pipeline_layout = _logical_device.createPipelineLayout(pipeline_layout_info);

        auto pipeline_info = vk::GraphicsPipelineCreateInfo {
            {},
            shader_stages,
            &vertex_input_info,
            &input_assembly_info,
            nullptr,
            &viewport_info,
            &rasterizer_info,
            &multisampling_info,
            &depth_stencil_info,
            &color_blend_info,
            nullptr,
            _pipeline_layout,
            _render_pass
        };

        auto result = _logical_device.createGraphicsPipelines({}, pipeline_info);

        if (result.result != vk::Result::eSuccess) {
            throw std::runtime_error("Could not create the graphic pipeline");
        }

        _graphics_pipeline = result.value[0];

        _logical_device.destroyShaderModule(vert_module);
        _logical_device.destroyShaderModule(frag_module);
    }

    void createFramebuffers() {
        _swap_chain_framebuffers.resize(_swap_chain_image_views.size());

        for (int i = 0; i < int(_swap_chain_image_views.size()); ++i) {
            auto attachments = { _color_image_view, _depth_image_view, vk::ImageView(_swap_chain_image_views[i]) };

            auto framebuffer_info = vk::FramebufferCreateInfo {
                {},
                _render_pass,
                attachments,
                _swap_chain_extent.width,
                _swap_chain_extent.height,
                1
            };

            _swap_chain_framebuffers[i] = _logical_device.createFramebuffer(framebuffer_info);
        }
    }

    void createCommandPool() {
        auto queue_family_indices = findQueueFamilies(_physical_device);

        auto command_pool_info = vk::CommandPoolCreateInfo {
            {},
            queue_family_indices.graphics_family.value()
        };

        _command_pool = _logical_device.createCommandPool(command_pool_info);
    }

    void createImage(
        uint32_t width, 
        uint32_t height,
        uint32_t mip_levels,
        vk::SampleCountFlagBits sample_count,
        vk::Format format,
        vk::ImageTiling tiling,
        vk::ImageUsageFlags usage,
        vk::MemoryPropertyFlags properties,
        vk::Image& image,
        vk::DeviceMemory& image_memory)
    {
        auto image_info = vk::ImageCreateInfo {
            {},
            vk::ImageType::e2D,
            format,
            { width, height, 1 },
            mip_levels,
            1,
            sample_count,
            tiling,
            usage,
            vk::SharingMode::eExclusive,
            {},
            vk::ImageLayout::eUndefined
        };

        image = _logical_device.createImage(image_info);

        auto memory_requirements = _logical_device.getImageMemoryRequirements(image);

        auto alloc_info = vk::MemoryAllocateInfo {
            memory_requirements.size,
            findMemoryType(memory_requirements.memoryTypeBits, properties)
        };

        image_memory = _logical_device.allocateMemory(alloc_info);
        _logical_device.bindImageMemory(image, image_memory, 0);
    }

    void generateMipmaps(vk::Image image, vk::Format image_format, int32_t tex_width, int32_t tex_height, uint32_t mip_levels) {
        // Check if image format supports linear blitting
        auto format_properties = _physical_device.getFormatProperties(image_format);

        if (!(format_properties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
            throw std::runtime_error("texture image format does not support linear blitting!");
        }

        auto command_buffer = beginSingleTimeCommands();

        auto barrier = vk::ImageMemoryBarrier {};

        barrier.image = image;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.subresourceRange.levelCount = 1;

        int32_t mip_width = tex_width;
        int32_t mip_height = tex_height;

        for (uint32_t i = 1; i < _mip_levels; ++i) {
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
            barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

            command_buffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eTransfer,
                {},
                {},
                {},
                barrier
            );

            int32_t next_mip_width = (mip_width > 1) ? mip_width / 2 : 1;
            int32_t next_mip_height = (mip_height > 1) ? mip_height / 2 : 1;

            auto blit = vk::ImageBlit {
                vk::ImageSubresourceLayers {
                    vk::ImageAspectFlagBits::eColor,
                    i - 1,
                    0,
                    1
                },
                std::array<vk::Offset3D, 2> {
                    vk::Offset3D { 0, 0, 0 },
                    vk::Offset3D { mip_width, mip_height, 1 }
                },
                vk::ImageSubresourceLayers {
                    vk::ImageAspectFlagBits::eColor,
                    i,
                    0,
                    1
                },
                std::array<vk::Offset3D, 2> {
                    vk::Offset3D { 0, 0, 0 },
                    vk::Offset3D { next_mip_width, next_mip_height, 1 }
                }
            };

            command_buffer.blitImage(
                image, 
                vk::ImageLayout::eTransferSrcOptimal,
                image, 
                vk::ImageLayout::eTransferDstOptimal,
                blit,
                vk::Filter::eLinear
            );

            barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
            barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            command_buffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eFragmentShader,
                {},
                {},
                {},
                barrier
            );

            mip_width = next_mip_width;
            mip_height = next_mip_height;
        }

        barrier.subresourceRange.baseMipLevel = mip_levels - 1;
        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        command_buffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eFragmentShader,
            {},
            {},
            {},
            barrier
        );

        endSingleTimeCommands(command_buffer);
    }

    void createTextureImage() {
        int tex_width, tex_height, tex_channels;

        auto pixels = stbi_load(TEXTURE_PATH.c_str(), &tex_width, &tex_height, &tex_channels, STBI_rgb_alpha);

        if (pixels == nullptr) {
            throw std::runtime_error("Could not load texture file");
        }

        _mip_levels = static_cast<uint32_t>(std::floor(std::log2(std::max(tex_width, tex_height)))) + 1;

        auto image_size = vk::DeviceSize { uint32_t(tex_width * tex_height * 4) };

        auto staging_buffer = vk::Buffer {};
        auto staging_buffer_memory = vk::DeviceMemory {};

        createBuffer(
            image_size, 
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            staging_buffer,
            staging_buffer_memory);

        void* data = _logical_device.mapMemory(staging_buffer_memory, 0, image_size);
        memcpy(data, pixels, std::size_t(image_size));
        _logical_device.unmapMemory(staging_buffer_memory);

        stbi_image_free(pixels);

        createImage(
            uint32_t(tex_width), 
            uint32_t(tex_height), 
            _mip_levels,
            vk::SampleCountFlagBits::e1,
            vk::Format::eR8G8B8A8Srgb, 
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            _texture_image,
            _texture_image_memory);

        transitionImageLayout(_texture_image, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, _mip_levels);
        copyBufferToImage(staging_buffer, _texture_image, tex_width, tex_height);

        generateMipmaps(_texture_image, vk::Format::eR8G8B8A8Srgb, tex_width, tex_height, _mip_levels);

        _logical_device.destroyBuffer(staging_buffer);
        _logical_device.freeMemory(staging_buffer_memory);
    }

    void createTextureImageView() {
        _texture_image_view = createImageView2D(_texture_image, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, _mip_levels);
    }

    void createColorResources() {
        auto format = _swap_chain_format;

        createImage(
            _swap_chain_extent.width, 
            _swap_chain_extent.height,
            1,
            _msaa_sample_count,
            format,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            _color_image,
            _color_image_memory
        );

        _color_image_view = createImageView2D(_color_image, format, vk::ImageAspectFlagBits::eColor, 1);
    }

    vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features) {
        for (auto& format : candidates) {
            auto props = _physical_device.getFormatProperties(format);

            if (
                (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) ||
                (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features)
            ) {
                return format;
            }
        }

        throw std::runtime_error("Could not find a supported format for the depth image");
    }


    vk::Format findDepthFormat() {
        return findSupportedFormat(
            {
                vk::Format::eD32Sfloat,
                vk::Format::eD32SfloatS8Uint,
                vk::Format::eD24UnormS8Uint,
            },
            vk::ImageTiling::eOptimal,
            vk::FormatFeatureFlagBits::eDepthStencilAttachment
        );
    }


    bool hasStencilComponent(vk::Format format) {
        return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
    }

    void createDepthResources() {
        auto format = findDepthFormat();

        createImage(
            _swap_chain_extent.width, 
            _swap_chain_extent.height,
            1,
            _msaa_sample_count,
            format,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eDepthStencilAttachment,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            _depth_image,
            _depth_image_memory
        );

        _depth_image_view = createImageView2D(_depth_image, format, vk::ImageAspectFlagBits::eDepth, 1);

        transitionImageLayout(_depth_image, format, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal, 1);
    }

    void createTextureSampler() {
        auto properties = _physical_device.getProperties();

        auto sampler_info = vk::SamplerCreateInfo {
            {},
            vk::Filter::eLinear,
            vk::Filter::eLinear,
            vk::SamplerMipmapMode::eLinear,
            vk::SamplerAddressMode::eRepeat,
            vk::SamplerAddressMode::eRepeat,
            vk::SamplerAddressMode::eRepeat,
            0.0f,
            true,
            properties.limits.maxSamplerAnisotropy,
            false,
            vk::CompareOp::eAlways,
            0.0f,
            float(_mip_levels),
            vk::BorderColor::eIntOpaqueBlack
        };

        _texture_sampler = _logical_device.createSampler(sampler_info);
    }

    void loadModel() {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str())) {
            throw std::runtime_error(warn + err);
        }

        auto vertex_map = std::unordered_map<Vertex, uint32_t> {};

        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                auto vertex = Vertex {};

                vertex.position = {
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2]
                };

                vertex.tex_coords = {
                    attrib.texcoords[2 * index.texcoord_index + 0],
                    1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                };

                vertex.color = {1.0f, 1.0f, 1.0f};

                if (vertex_map.find(vertex) == vertex_map.end()) {
                    vertex_map[vertex] = uint32_t(_vertices.size());
                    _vertices.push_back(vertex);
                }
                
                _indices.push_back(vertex_map.at(vertex));
            }
        }
    }


    void createIndexVertexBuffer() {
        vk::DeviceSize vertex_size = sizeof(Vertex) * uint32_t(_vertices.size());
        vk::DeviceSize indice_size = sizeof(uint32_t) * uint32_t(_indices.size());
        vk::DeviceSize total_size = vertex_size + indice_size;

        vk::Buffer staging_buffer;
        vk::DeviceMemory staging_buffer_memory;

        createBuffer(
            total_size, 
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            staging_buffer,
            staging_buffer_memory);

        uint8_t* data = (uint8_t*) _logical_device.mapMemory(staging_buffer_memory, 0, total_size, {});
        memcpy(data, _vertices.data(), std::size_t(vertex_size));
        memcpy(data + std::size_t(vertex_size), _indices.data(), std::size_t(indice_size));
        _logical_device.unmapMemory(staging_buffer_memory);

        createBuffer(
            total_size, 
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            _vertex_indice_buffer,
            _vertex_indice_buffer_memory);

        copyBuffer(staging_buffer, _vertex_indice_buffer, total_size);

        vkDestroyBuffer(_logical_device, staging_buffer, nullptr);
        vkFreeMemory(_logical_device, staging_buffer_memory, nullptr);
    }

    void createUniformBuffers() {
        auto buffer_size = sizeof(TransformBufferObject);

        _transform_buffers.resize(_swap_chain_images.size());
        _transform_buffers_memory.resize(_swap_chain_images.size());

        for (int i = 0; i < int(_swap_chain_images.size()); ++i) {
            createBuffer(
                buffer_size,
                vk::BufferUsageFlagBits::eUniformBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                _transform_buffers[i],
                _transform_buffers_memory[i]);
        }
    }

    void createDescriptorPool() {
        auto pool_size = std::array<vk::DescriptorPoolSize, 2> {
            vk::DescriptorPoolSize {
                vk::DescriptorType::eUniformBuffer,
                uint32_t(_swap_chain_images.size())
            },
            vk::DescriptorPoolSize {
                vk::DescriptorType::eCombinedImageSampler,
                uint32_t(_swap_chain_images.size())
            }
        };

        auto pool_create_info = vk::DescriptorPoolCreateInfo {
            {},
            uint32_t(_swap_chain_images.size()),
            pool_size
        };

        _descriptor_pool = _logical_device.createDescriptorPool(pool_create_info);
    }

    void createDescriptorSets() {
        auto layouts = std::vector<vk::DescriptorSetLayout> { _swap_chain_images.size(), _descriptor_set_layout };

        auto alloc_info = vk::DescriptorSetAllocateInfo {
            _descriptor_pool,
            layouts
        };

        _descriptor_sets = _logical_device.allocateDescriptorSets(alloc_info);

        for (int i = 0; i < int(_swap_chain_images.size()); ++i) {
            auto descriptor_buffer_info = vk::DescriptorBufferInfo {
                _transform_buffers[i],
                0,
                sizeof(TransformBufferObject)
            };

            auto descriptor_image_info = vk::DescriptorImageInfo {
                _texture_sampler,
                _texture_image_view,
                vk::ImageLayout::eShaderReadOnlyOptimal
            };

            auto descriptor_write = std::array<vk::WriteDescriptorSet, 2> {
                vk::WriteDescriptorSet {
                    _descriptor_sets[i],
                    0,
                    0,
                    vk::DescriptorType::eUniformBuffer,
                    {},
                    descriptor_buffer_info,
                    {}   
                },

                vk::WriteDescriptorSet {
                    _descriptor_sets[i],
                    1,
                    0,
                    vk::DescriptorType::eCombinedImageSampler,
                    descriptor_image_info,
                    {},
                    {}  
                }
            };

            _logical_device.updateDescriptorSets(descriptor_write, {});
        }
    }

    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Buffer& buffer, vk::DeviceMemory& buffer_memory) {
        auto buffer_info = vk::BufferCreateInfo {
            {},
            size,
            usage,
            vk::SharingMode::eExclusive,
            {}
        };

        buffer = _logical_device.createBuffer(buffer_info);

        auto memory_requirements = _logical_device.getBufferMemoryRequirements(buffer);

        auto allocate_info = vk::MemoryAllocateInfo {
            memory_requirements.size,
            findMemoryType(memory_requirements.memoryTypeBits, properties)
        };

        buffer_memory = _logical_device.allocateMemory(allocate_info);

        _logical_device.bindBufferMemory(buffer, buffer_memory, 0);
    }
    
    vk::CommandBuffer beginSingleTimeCommands() {
        auto alloc_info = vk::CommandBufferAllocateInfo {
            _command_pool,
            vk::CommandBufferLevel::ePrimary,
            1
        };

        auto command_buffer = _logical_device.allocateCommandBuffers(alloc_info).front();

        auto begin_info = vk::CommandBufferBeginInfo {
            vk::CommandBufferUsageFlagBits::eOneTimeSubmit
        };

        command_buffer.begin(begin_info);

        return command_buffer;
    }

    void endSingleTimeCommands(vk::CommandBuffer command_buffer) {
        command_buffer.end();

        auto submit_info = vk::SubmitInfo {
            {},
            {},
            command_buffer
        };

        _graphics_queue.submit(submit_info, {});
        _graphics_queue.waitIdle();
        
        _logical_device.freeCommandBuffers(_command_pool, command_buffer);
    }

    void copyBuffer(vk::Buffer src_buffer, vk::Buffer dst_buffer, vk::DeviceSize size) {
        auto command_buffer = beginSingleTimeCommands();

        auto copy = vk::BufferCopy {
            0,
            0,
            size
        };

        command_buffer.copyBuffer(src_buffer, dst_buffer, 1, &copy);

        endSingleTimeCommands(command_buffer);
    }

    void copyBufferToImage(vk::Buffer src_buffer, vk::Image dst_image, uint32_t width, uint32_t height) {
        auto command_buffer = beginSingleTimeCommands();

        auto copy = vk::BufferImageCopy {
            0,
            0,
            0,
            vk::ImageSubresourceLayers {
                vk::ImageAspectFlagBits::eColor,
                0,
                0,
                1
            },
            { 0, 0, 0 },
            { width, height, 1 }
        };

        command_buffer.copyBufferToImage(src_buffer, dst_image, vk::ImageLayout::eTransferDstOptimal, 1, &copy);

        endSingleTimeCommands(command_buffer);
    }

    void transitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout old_layout, vk::ImageLayout new_layout, uint32_t mip_levels) {
        auto command_buffer = beginSingleTimeCommands();

        auto barrier = vk::ImageMemoryBarrier {};

        barrier.oldLayout = old_layout;
        barrier.newLayout = new_layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;

        if (new_layout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
            barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;

            if (hasStencilComponent(format)) {
                barrier.subresourceRange.aspectMask |= vk::ImageAspectFlagBits::eStencil;
            }
        } else {
            barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        }
        
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = mip_levels;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        vk::PipelineStageFlags src_stage;
        vk::PipelineStageFlags dst_stage;

        if (old_layout == vk::ImageLayout::eUndefined && new_layout == vk::ImageLayout::eTransferDstOptimal) {
            barrier.srcAccessMask = {};
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

            src_stage = vk::PipelineStageFlagBits::eTopOfPipe;
            dst_stage = vk::PipelineStageFlagBits::eTransfer;
        } else if (old_layout == vk::ImageLayout::eTransferDstOptimal && new_layout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            src_stage = vk::PipelineStageFlagBits::eTransfer;
            dst_stage = vk::PipelineStageFlagBits::eFragmentShader;
        } else if (old_layout == vk::ImageLayout::eUndefined && new_layout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
            barrier.srcAccessMask = {};
            barrier.dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite;

            src_stage = vk::PipelineStageFlagBits::eTopOfPipe;
            dst_stage = vk::PipelineStageFlagBits::eEarlyFragmentTests;
        } else {
            throw std::runtime_error("Unsupported layout transition");
        }

        command_buffer.pipelineBarrier(
            src_stage,
            dst_stage,
            {},
            {},
            {},
            { barrier }
        );

        endSingleTimeCommands(command_buffer);
    }

    uint32_t findMemoryType(uint32_t type_filter, vk::MemoryPropertyFlags properties) {
        auto memory_properties = _physical_device.getMemoryProperties();

        for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
            if (type_filter & (1 << i) && (memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("Could not find suitable memory type");
    }


    void createCommandBuffers() {
        _command_buffers.resize(_swap_chain_framebuffers.size());

        auto alloc_info = vk::CommandBufferAllocateInfo {
            _command_pool,
            vk::CommandBufferLevel::ePrimary,
            uint32_t(_command_buffers.size())
        };

        _command_buffers = _logical_device.allocateCommandBuffers(alloc_info);

        for (auto i = 0; i < int(_command_buffers.size()); ++i) {
            auto begin_info = vk::CommandBufferBeginInfo {};
            _command_buffers[i].begin(begin_info);

            auto clear_values = {
                vk::ClearValue { std::array<float, 4> { 0.0f, 0.0f, 0.0f, 1.0f } }, 
                vk::ClearValue { vk::ClearDepthStencilValue({ 1.0f, 0 }) },
                vk::ClearValue { std::array<float, 4> { 0.0f, 0.0f, 0.0f, 1.0f } }
            };

            auto render_pass_info = vk::RenderPassBeginInfo {
                _render_pass,
                _swap_chain_framebuffers[i],
                {
                    { 0, 0 },
                    _swap_chain_extent
                },
                clear_values
            };

            auto vertex_offset = vk::DeviceSize { 0 };
            auto indice_offset = vk::DeviceSize { sizeof(Vertex) * _vertices.size() };

            _command_buffers[i].beginRenderPass(render_pass_info, vk::SubpassContents::eInline);
            _command_buffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, _graphics_pipeline);

            _command_buffers[i].bindVertexBuffers(0, _vertex_indice_buffer, vertex_offset);

            _command_buffers[i].bindIndexBuffer(_vertex_indice_buffer, indice_offset, vk::IndexType::eUint32);

            _command_buffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, _pipeline_layout, 0, _descriptor_sets[i], {});


            _command_buffers[i].drawIndexed(uint32_t(_indices.size()), 1, 0, 0, 0);
            _command_buffers[i].endRenderPass();

            _command_buffers[i].end();
        }
    }

    void createSyncObjects() {
        _image_available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
        _render_finished_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
        _in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);
        _images_in_flight.resize(_swap_chain_images.size());

        auto semaphore_info = vk::SemaphoreCreateInfo {};

        auto fence_info = vk::FenceCreateInfo {
            vk::FenceCreateFlagBits::eSignaled
        };

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            _image_available_semaphores[i] = _logical_device.createSemaphore(semaphore_info);
            _render_finished_semaphores[i] = _logical_device.createSemaphore(semaphore_info);
            _in_flight_fences[i] = _logical_device.createFence(fence_info);
        }
    }

    void cleanup() {
        #ifndef NDEBUG
            destroyDebugMessenger();
        #endif

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            _logical_device.destroySemaphore(_image_available_semaphores[i]);
            _logical_device.destroySemaphore(_render_finished_semaphores[i]);
            _logical_device.destroyFence(_in_flight_fences[i]);
        }

        cleanupSwapChain();

        _logical_device.destroySampler(_texture_sampler);
        _logical_device.destroyImageView(_texture_image_view);

        _logical_device.destroyImage(_texture_image);
        _logical_device.freeMemory(_texture_image_memory);

        _logical_device.destroyDescriptorSetLayout(_descriptor_set_layout);

        _logical_device.destroyBuffer(_vertex_indice_buffer);
        _logical_device.freeMemory(_vertex_indice_buffer_memory);

        _logical_device.destroyCommandPool(_command_pool);

        _logical_device.destroy();

        _instance.destroySurfaceKHR(_surface);
        _instance.destroy();

        glfwDestroyWindow(_window);
        glfwTerminate();
    }

    void cleanupSwapChain() {
        _logical_device.destroyImageView(_color_image_view);
        _logical_device.destroyImage(_color_image);
        _logical_device.freeMemory(_color_image_memory);

        _logical_device.destroyImageView(_depth_image_view);
        _logical_device.destroyImage(_depth_image);
        _logical_device.freeMemory(_depth_image_memory);

        for (auto& buffer_memory : _transform_buffers_memory) {
            _logical_device.freeMemory(buffer_memory);
        }

        for (auto& buffer : _transform_buffers) {
            _logical_device.destroyBuffer(buffer);
        }

        _logical_device.destroyDescriptorPool(_descriptor_pool);

        for (auto& framebuffer : _swap_chain_framebuffers) {
            _logical_device.destroyFramebuffer(framebuffer);
        }

        for (auto& image_view : _swap_chain_image_views) {
            _logical_device.destroyImageView(image_view);
        }

        _logical_device.freeCommandBuffers(_command_pool, _command_buffers);

        _logical_device.destroyPipeline(_graphics_pipeline);
        _logical_device.destroyPipelineLayout(_pipeline_layout);
        _logical_device.destroyRenderPass(_render_pass);

        _logical_device.destroySwapchainKHR(_swap_chain);
    }

    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(_window, &width, &height);

        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(_window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(_logical_device);

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createColorResources();
        createDepthResources();
        createFramebuffers();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
    }

    void initVulkan() {
        createInstance();

        #ifndef NDEBUG
        if (!checkValidationLayerSupport(_validation_layers)) {
            throw std::runtime_error("Validation layers enabled but none of them are available");
        } else {
            std::cout << "Validation Layers enabled and available!" << std::endl;
        }

        setupDebugMessenger();
        #endif
        
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayouts();
        createGraphicsPipeline();
        createCommandPool();
        createColorResources();
        createDepthResources();
        createFramebuffers();
        createTextureImage();
        createTextureImageView();
        createTextureSampler();
        loadModel();
        createIndexVertexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }
};

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}