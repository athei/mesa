/*
 * Copyright 2024 Autodesk, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef WSI_COMMON_METAL_LAYER_H
#define WSI_COMMON_METAL_LAYER_H

#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_metal.h>

#include <stdint.h>
#include <stdbool.h>

typedef void CAMetalDrawable;

/* The application owns the CAMetalLayer it passes in
 * VkMetalSurfaceCreateInfoEXT and may release it as soon as it no longer
 * needs it itself. Surfaces and swapchains hold their own reference. */
const CAMetalLayer *
wsi_metal_layer_retain(const CAMetalLayer *metal_layer);

void
wsi_metal_layer_release(const CAMetalLayer *metal_layer);

void
wsi_metal_layer_size(const CAMetalLayer *metal_layer,
   uint32_t *width, uint32_t *height);

/* Calls refresh_rate_handler with the highest refresh rate in Hz of the
 * screen showing the layer, or 0 when no screen reports one. AppKit's views,
 * windows and screens belong to the main thread: called on the main thread,
 * the handler runs before this returns; called anywhere else, the query is
 * queued on the main dispatch queue and the handler runs there later, so in
 * a process whose main thread never services that queue the swapchain keeps
 * its initial rate. */
void
wsi_metal_layer_query_refresh_rate(const CAMetalLayer *metal_layer,
                                   void (*refresh_rate_handler)(void *, uint32_t),
                                   void *data);

VkResult
wsi_metal_layer_configure(const CAMetalLayer *metal_layer,
   uint32_t width, uint32_t height, uint32_t image_count,
   VkFormat format, VkColorSpaceKHR color_space,
   bool enable_opaque, bool framebuffer_only);

void
wsi_metal_layer_set_hdr_metadata(const CAMetalLayer *metal_layer,
                                 const VkHdrMetadataEXT *metadata);

CAMetalDrawable *
wsi_metal_layer_acquire_drawable(const CAMetalLayer *metal_layer);

void
wsi_metal_release_drawable(CAMetalDrawable *drawable_ptr);

/* Calls presented_handler(data, present_id, paced, presented_time) once
 * Metal has shown the drawable, with the time it was shown, or has let it go
 * without showing it, with 0. present_id and paced are passed through. */
void
wsi_metal_layer_add_presented_handler(CAMetalDrawable *drawable_ptr,
                                      void (*presented_handler)(void *, uint64_t,
                                                                bool, double),
                                      void *data, uint64_t present_id,
                                      bool paced);

struct wsi_metal_layer_blit_context;

struct wsi_metal_layer_blit_context *
wsi_create_metal_layer_blit_context();

void
wsi_destroy_metal_layer_blit_context(struct wsi_metal_layer_blit_context *context);

void
wsi_metal_layer_blit_and_present(struct wsi_metal_layer_blit_context *context,
   CAMetalDrawable **drawable_ptr, void *buffer,
   uint32_t width, uint32_t height, uint32_t row_pitch,
   double min_duration);

/* Presents the drawable no sooner than min_duration seconds after the
 * previous drawable of the layer was shown, or as soon as it can when
 * min_duration is 0. */
void
wsi_metal_layer_present(CAMetalDrawable **drawable_ptr, double min_duration);

void
wsi_metal_layer_make_queue_resident(const CAMetalLayer *metal_layer,
   void *mtl4_command_queue);

void
wsi_metal_layer_remove_queue_resident(const CAMetalLayer *metal_layer,
   void *mtl4_command_queue);

#endif // WSI_COMMON_METAL_LAYER_H
