/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "mtl_bridge.h"

// kk_image_layout.h should also includes "vulkan/vulkan.h", but just to be safe
#include "vulkan/vulkan.h"
#include "kk_image_layout.h"

#include "util/macros.h"

#include <Metal/MTL4CommandBuffer.h>
#include <Metal/MTLCommandBuffer.h>
#include <Metal/MTLCommandQueue.h>
#include <Metal/MTLDevice.h>
#include <Metal/MTLHeap.h>
#include <Metal/MTLEvent.h>
#include <Metal/MTLResidencySet.h>

#include <QuartzCore/CAMetalLayer.h>

static_assert(sizeof(MTLResourceID) == sizeof(uint64_t), "Must match, otherwise descriptors are broken");

mtl_texture *
mtl_drawable_get_texture(void *drawable_ptr)
{
   @autoreleasepool {
      id<CAMetalDrawable> drawable = (id<CAMetalDrawable>)drawable_ptr;
      return drawable.texture;
   }
}

void
mtl_command_buffer_use_drawable_residency(mtl_command_buffer *cmd_buf,
                                          void *drawable_ptr)
{
   @autoreleasepool {
      id<MTL4CommandBuffer> buffer = (id<MTL4CommandBuffer>)cmd_buf;
      id<CAMetalDrawable> drawable = (id<CAMetalDrawable>)drawable_ptr;
      id<MTLResidencySet> residency_set = drawable.layer.residencySet;
      if (residency_set)
         [buffer useResidencySet:residency_set];
   }
}

void *
mtl_retain(void *handle)
{
   @autoreleasepool {
      NSObject *obj = (NSObject *)handle;
      return [obj retain];
   }
}

void
mtl_release(void *handle)
{
   @autoreleasepool {
      NSObject *obj = (NSObject *)handle;
      [obj release];
   }
}
