/*
 * Copyright 2026 David Andrews
 * SPDX-License-Identifier: MIT
 */

#include "kk_buffer.h"
#include "kk_cmd_buffer.h"
#include "kk_entrypoints.h"

#include "kosmickrisp/libkk/kk_query.h"

VKAPI_ATTR void VKAPI_CALL
kk_CmdBindTransformFeedbackBuffersEXT(VkCommandBuffer commandBuffer,
                                      uint32_t firstBinding,
                                      uint32_t bindingCount,
                                      const VkBuffer *pBuffers,
                                      const VkDeviceSize *pOffsets,
                                      const VkDeviceSize *pSizes)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   struct kk_graphics_state *gfx = &cmd->state.gfx;

   for (uint32_t i = 0; i < bindingCount; i++) {
      VK_FROM_HANDLE(kk_buffer, buffer, pBuffers[i]);
      uint64_t size = pSizes ? pSizes[i] : VK_WHOLE_SIZE;
      struct kk_addr_range range =
         kk_buffer_addr_range(buffer, pOffsets[i], size);
      range.range = MIN2(range.range, UINT32_MAX);
      gfx->xfb.buffers[firstBinding + i] = range;
   }
}

static uint64_t
kk_xfb_counter_addr(uint32_t i, uint32_t firstCounterBuffer,
                    uint32_t counterBufferCount,
                    const VkBuffer *pCounterBuffers,
                    const VkDeviceSize *pCounterBufferOffsets)
{
   if (pCounterBuffers == NULL || i < firstCounterBuffer ||
       i - firstCounterBuffer >= counterBufferCount)
      return 0;

   uint32_t c = i - firstCounterBuffer;
   VK_FROM_HANDLE(kk_buffer, buffer, pCounterBuffers[c]);
   if (buffer == NULL)
      return 0;

   uint64_t offset = pCounterBufferOffsets ? pCounterBufferOffsets[c] : 0;
   return vk_buffer_address(&buffer->vk, offset);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdBeginTransformFeedbackEXT(VkCommandBuffer commandBuffer,
                                uint32_t firstCounterBuffer,
                                uint32_t counterBufferCount,
                                const VkBuffer *pCounterBuffers,
                                const VkDeviceSize *pCounterBufferOffsets)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   struct kk_graphics_state *gfx = &cmd->state.gfx;

   struct kk_ptr offsets =
      kk_pool_alloc(cmd, sizeof(uint32_t) * KK_MAX_XFB_BUFFERS, 4);
   if (!offsets.gpu)
      return;

   memset(offsets.cpu, 0, sizeof(uint32_t) * KK_MAX_XFB_BUFFERS);

   for (uint32_t i = 0; i < KK_MAX_XFB_BUFFERS; i++) {
      uint64_t counter =
         kk_xfb_counter_addr(i, firstCounterBuffer, counterBufferCount,
                             pCounterBuffers, pCounterBufferOffsets);
      if (counter) {
         libkk_copy_u32(cmd, kk_grid_1d(1), true,
                        offsets.gpu + i * sizeof(uint32_t), counter);
      }
   }

   gfx->xfb.offsets = offsets.gpu;
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdEndTransformFeedbackEXT(VkCommandBuffer commandBuffer,
                              uint32_t firstCounterBuffer,
                              uint32_t counterBufferCount,
                              const VkBuffer *pCounterBuffers,
                              const VkDeviceSize *pCounterBufferOffsets)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   struct kk_graphics_state *gfx = &cmd->state.gfx;

   if (!gfx->xfb.offsets)
      return;

   for (uint32_t i = 0; i < KK_MAX_XFB_BUFFERS; i++) {
      uint64_t counter =
         kk_xfb_counter_addr(i, firstCounterBuffer, counterBufferCount,
                             pCounterBuffers, pCounterBufferOffsets);
      if (counter) {
         libkk_copy_u32(cmd, kk_grid_1d(1), true, counter,
                        gfx->xfb.offsets + i * sizeof(uint32_t));
      }
   }

   gfx->xfb.offsets = 0;
}
