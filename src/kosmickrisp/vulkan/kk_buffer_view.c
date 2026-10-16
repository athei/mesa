/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_buffer_view.h"

#include "kk_buffer.h"
#include "kk_device.h"
#include "kk_entrypoints.h"
#include "kk_format.h"
#include "kk_image_layout.h"
#include "kk_nir_lower_vbo.h"
#include "kk_physical_device.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/mtl_format.h"

#include "vk_format.h"

static bool
kk_is_rgb32_format(enum pipe_format format)
{
   return format == PIPE_FORMAT_R32G32B32_FLOAT ||
          format == PIPE_FORMAT_R32G32B32_UINT ||
          format == PIPE_FORMAT_R32G32B32_SINT;
}

VkFormatFeatureFlags2
kk_get_buffer_format_features(struct kk_physical_device *pdev,
                              VkFormat vk_format)
{
   VkFormatFeatureFlags2 features = 0;
   enum pipe_format p_format = vk_format_to_pipe_format(vk_format);

   if (p_format == PIPE_FORMAT_NONE)
      return 0;

   const struct kk_va_format *format = kk_get_va_format(p_format);
   if (format) {
      if (format->texel_buffer.read)
         features |= VK_FORMAT_FEATURE_2_UNIFORM_TEXEL_BUFFER_BIT;

      if (format->texel_buffer.write) {
         features |= VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_BIT;
         features |= VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT;
         features |= VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT;
      }

      /* Only these formats allow atomics for texel buffers */
      if (vk_format == VK_FORMAT_R32_UINT || vk_format == VK_FORMAT_R32_SINT)
         features |= VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_ATOMIC_BIT;
   }

   if (kk_is_rgb32_format(p_format))
      features |= VK_FORMAT_FEATURE_2_UNIFORM_TEXEL_BUFFER_BIT;

   if (kk_vbo_supports_format(p_format))
      features |= VK_FORMAT_FEATURE_2_VERTEX_BUFFER_BIT;

   return features;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CreateBufferView(VkDevice _device, const VkBufferViewCreateInfo *pCreateInfo,
                    const VkAllocationCallbacks *pAllocator,
                    VkBufferView *pBufferView)
{
   VK_FROM_HANDLE(kk_device, dev, _device);
   struct kk_buffer_view *view =
      vk_buffer_view_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*view));
   if (!view)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   enum pipe_format p_format = vk_format_to_pipe_format(view->vk.format);
   struct kk_buffer *buffer =
      container_of(view->vk.buffer, struct kk_buffer, vk);
   uint32_t el_size_B = util_format_get_blocksize(p_format);

   if (kk_is_rgb32_format(p_format)) {
      view->rgb32_addr = vk_buffer_address(&buffer->vk, view->vk.offset);
      goto residency;
   }

   const struct kk_va_format *supported_format = kk_get_va_format(p_format);

   /* If we reached here, we support reading at least */
   enum mtl_texture_usage usage = MTL_TEXTURE_USAGE_SHADER_READ;
   if (supported_format->texel_buffer.write)
      usage |= MTL_TEXTURE_USAGE_SHADER_WRITE;

   /* Only these formats allow atomics for texel buffers */
   if (view->vk.format == VK_FORMAT_R32_UINT ||
       view->vk.format == VK_FORMAT_R32_SINT)
      usage |= MTL_TEXTURE_USAGE_SHADER_ATOMIC;

   uint64_t aligned_offset_B =
      view->vk.offset & ~(uint64_t)(KK_MIN_TEXEL_BUFFER_ALIGNMENT - 1);
   uint32_t lead_B = view->vk.offset - aligned_offset_B;
   assert(lead_B % el_size_B == 0);
   view->offset_el = lead_B / el_size_B;

   struct kk_image_layout layout = {
      .width_px = view->vk.elements + view->offset_el,
      .height_px = 1u,
      .depth_px = 1u,
      .layers = 1u,
      .type = MTL_TEXTURE_TYPE_TEXTURE_BUFFER,
      .sample_count_sa = 1u,
      .levels = 1u,
      .linear = true,
      .optimized_layout = false,
      .usage = usage,
      .format = {.pipe = p_format, .mtl = supported_format->mtl_pixel_format},
      .swizzle =
         {
            .red = supported_format->unswizzle.red,
            .green = supported_format->unswizzle.green,
            .blue = supported_format->unswizzle.blue,
            .alpha = supported_format->unswizzle.alpha,
         },
      .linear_stride_B = view->vk.range + lead_B,
   };
   view->mtl_texel_buffer_handle = mtl_new_texture_with_descriptor_linear(
      buffer->metal.handle, &layout,
      kk_buffer_mtl_offset(buffer, aligned_offset_B));
   if (!view->mtl_texel_buffer_handle) {
      vk_buffer_view_destroy(&dev->vk, pAllocator, &view->vk);
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }
   view->texel_buffer_gpu_id =
      mtl_texture_get_gpu_resource_id(view->mtl_texel_buffer_handle);

residency:
   if (buffer->residency.pages) {
      simple_mtx_lock(&dev->sparse_residency.lock);
      uint32_t id = util_idalloc_alloc(&dev->sparse_residency.views);
      if (id >= KK_SPARSE_RESIDENCY_VIEWS) {
         util_idalloc_free(&dev->sparse_residency.views, id);
         id = 0u;
      }
      simple_mtx_unlock(&dev->sparse_residency.lock);

      if (id == 0u) {
         kk_DestroyBufferView(_device, kk_buffer_view_to_handle(view),
                              pAllocator);
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }

      uint64_t entry =
         (uint64_t)kk_buffer_residency(buffer, view->vk.offset) |
         ((view->vk.offset % KK_SPARSE_PAGE_SIZE) << 32) |
         ((uint64_t)el_size_B << 48);
      ((uint64_t *)dev->sparse_residency.bo->cpu)[id] = entry;
      view->residency_view = id;
   }

   *pBufferView = kk_buffer_view_to_handle(view);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
kk_DestroyBufferView(VkDevice _device, VkBufferView bufferView,
                     const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(kk_device, dev, _device);
   VK_FROM_HANDLE(kk_buffer_view, view, bufferView);

   if (!view)
      return;

   if (view->residency_view) {
      simple_mtx_lock(&dev->sparse_residency.lock);
      util_idalloc_free(&dev->sparse_residency.views, view->residency_view);
      simple_mtx_unlock(&dev->sparse_residency.lock);
   }

   mtl_release(view->mtl_texel_buffer_handle);
   vk_buffer_view_destroy(&dev->vk, pAllocator, &view->vk);
}

void
kk_buffer_view_set_label(struct kk_buffer_view *bview, const char *label)
{
   if (bview->mtl_texel_buffer_handle)
      mtl_resource_set_label(bview->mtl_texel_buffer_handle, label);
}
