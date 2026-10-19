/*
 * Copyright 2026 David Andrews
 * SPDX-License-Identifier: MIT
 */
#include "kk_cmd_dgc.h"

#include "kk_cmd_buffer.h"
#include "kk_device.h"
#include "kk_entrypoints.h"
#include "kk_physical_device.h"

#include "libkk.h"

#include "util/macros.h"

static bool
kk_dgc_needs_roots(const struct vk_indirect_command_layout *layout)
{
   return layout->dgc_info &
          (BITFIELD_BIT(MESA_VK_DGC_PC) | BITFIELD_BIT(MESA_VK_DGC_SI) |
           BITFIELD_BIT(MESA_VK_DGC_VB));
}

static uint32_t
kk_dgc_cmd_size_B(const struct vk_indirect_command_layout *layout)
{
   if (layout->dgc_info & BITFIELD_BIT(MESA_VK_DGC_DRAW_INDEXED))
      return sizeof(VkDrawIndexedIndirectCommand);
   if (layout->dgc_info & BITFIELD_BIT(MESA_VK_DGC_DRAW))
      return sizeof(VkDrawIndirectCommand);
   return sizeof(VkDispatchIndirectCommand);
}

static uint64_t
kk_dgc_layout_buffers(const struct vk_indirect_command_layout *layout,
                      uint32_t max_seq, struct kk_dgc_buffers *bufs)
{
   uint64_t offset = 0;

   *bufs = (struct kk_dgc_buffers){
      .root_stride_B = align(sizeof(struct kk_root_descriptor_table), 16),
      .cmd_stride_B = kk_dgc_cmd_size_B(layout),
   };

   if (kk_dgc_needs_roots(layout)) {
      bufs->roots = offset;
      offset += (uint64_t)max_seq * bufs->root_stride_B;
   }

   offset = align64(offset, 16);
   bufs->cmds = offset;
   offset += (uint64_t)max_seq * bufs->cmd_stride_B;

   if (layout->dgc_info & BITFIELD_BIT(MESA_VK_DGC_IB)) {
      offset = align64(offset, 16);
      bufs->ib_info = offset;
      offset += (uint64_t)max_seq * 4 * sizeof(uint32_t);

      offset = align64(offset, 16);
      bufs->unrolled = offset;
      offset += (uint64_t)max_seq * sizeof(VkDrawIndexedIndirectCommand);
   }

   return align64(MAX2(offset, 16), 16);
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CreateIndirectCommandsLayoutEXT(
   VkDevice _device, const VkIndirectCommandsLayoutCreateInfoEXT *pCreateInfo,
   const VkAllocationCallbacks *pAllocator,
   VkIndirectCommandsLayoutEXT *pIndirectCommandsLayout)
{
   VK_FROM_HANDLE(kk_device, dev, _device);

   struct kk_indirect_commands_layout *layout = vk_indirect_command_layout_create(
      &dev->vk, pCreateInfo, pAllocator, sizeof(*layout));
   if (!layout)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   *pIndirectCommandsLayout = kk_indirect_commands_layout_to_handle(layout);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
kk_DestroyIndirectCommandsLayoutEXT(
   VkDevice _device, VkIndirectCommandsLayoutEXT indirectCommandsLayout,
   const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(kk_device, dev, _device);
   VK_FROM_HANDLE(kk_indirect_commands_layout, layout, indirectCommandsLayout);

   if (!layout)
      return;

   vk_indirect_command_layout_destroy(&dev->vk, pAllocator, &layout->vk);
}

VKAPI_ATTR void VKAPI_CALL
kk_GetGeneratedCommandsMemoryRequirementsEXT(
   VkDevice _device, const VkGeneratedCommandsMemoryRequirementsInfoEXT *pInfo,
   VkMemoryRequirements2 *pMemoryRequirements)
{
   VK_FROM_HANDLE(kk_device, dev, _device);
   VK_FROM_HANDLE(kk_indirect_commands_layout, layout,
                  pInfo->indirectCommandsLayout);
   struct kk_physical_device *pdev = kk_device_physical(dev);
   struct kk_dgc_buffers bufs;

   pMemoryRequirements->memoryRequirements = (VkMemoryRequirements){
      .size = kk_dgc_layout_buffers(&layout->vk, pInfo->maxSequenceCount, &bufs),
      .alignment = 16,
      .memoryTypeBits = BITFIELD_MASK(pdev->mem_type_count),
   };
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdPreprocessGeneratedCommandsEXT(
   VkCommandBuffer commandBuffer,
   const VkGeneratedCommandsInfoEXT *pGeneratedCommandsInfo,
   VkCommandBuffer stateCommandBuffer)
{
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdExecuteGeneratedCommandsEXT(
   VkCommandBuffer commandBuffer, VkBool32 isPreprocessed,
   const VkGeneratedCommandsInfoEXT *pGeneratedCommandsInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(kk_indirect_commands_layout, layout,
                  pGeneratedCommandsInfo->indirectCommandsLayout);

   if (pGeneratedCommandsInfo->maxSequenceCount == 0)
      return;

   if (layout->vk.dgc_info & BITFIELD_BIT(MESA_VK_DGC_DISPATCH))
      kk_cmd_dispatch_generated(cmd, layout, pGeneratedCommandsInfo);
   else
      kk_cmd_draw_generated(cmd, layout, pGeneratedCommandsInfo);
}

bool
kk_dgc_prepare(struct kk_cmd_buffer *cmd,
               const struct kk_indirect_commands_layout *layout,
               const VkGeneratedCommandsInfoEXT *info,
               struct libkk_dgc_params *params, struct kk_dgc_buffers *bufs)
{
   const struct vk_indirect_command_layout *vk = &layout->vk;
   uint64_t base = info->preprocessAddress & KK_SPARSE_ADDRESS_MASK;
   uint64_t size = kk_dgc_layout_buffers(vk, info->maxSequenceCount, bufs);

   if (!base || info->preprocessSize < size)
      return false;

   if (kk_dgc_needs_roots(vk))
      bufs->roots += base;
   else
      bufs->roots = 0;
   bufs->cmds += base;
   if (bufs->ib_info) {
      bufs->ib_info += base;
      bufs->unrolled += base;
   }

   bool dispatch = vk->dgc_info & BITFIELD_BIT(MESA_VK_DGC_DISPATCH);

   *params = (struct libkk_dgc_params){
      .stream = info->indirectAddress & KK_SPARSE_ADDRESS_MASK,
      .count = info->sequenceCountAddress & KK_SPARSE_ADDRESS_MASK,
      .predicate =
         cmd->state.cond_render.enabled ? cmd->state.cond_render.address : 0,
      .roots = bufs->roots,
      .draws = bufs->cmds,
      .ib_info = bufs->ib_info,
      .predicate_inverted = cmd->state.cond_render.inverted,
      .max_seq = info->maxSequenceCount,
      .stream_stride_dw = vk->stride / 4,
      .root_size_dw = sizeof(struct kk_root_descriptor_table) / 4,
      .root_stride_dw = bufs->root_stride_B / 4,
      .cmd_src_dw =
         (dispatch ? vk->dispatch_src_offset_B : vk->draw_src_offset_B) / 4,
      .cmd_size_dw = bufs->cmd_stride_B / 4,
      .cmd_stride_dw = bufs->cmd_stride_B / 4,
      .has_ib = !!(vk->dgc_info & BITFIELD_BIT(MESA_VK_DGC_IB)),
      .ib_src_dw = vk->index_src_offset_B / 4,
      .ib_dx = vk->index_mode_is_dx,
      .push_dw = offsetof(struct kk_root_descriptor_table, push) / 4,
      .buffer_strides_dw =
         offsetof(struct kk_root_descriptor_table, draw.buffer_strides) / 4,
      .attrib_base_dw =
         offsetof(struct kk_root_descriptor_table, draw.attrib_base) / 4,
      .attrib_clamps_dw =
         offsetof(struct kk_root_descriptor_table, draw.attrib_clamps) / 4,
      .has_si = !!(vk->dgc_info & BITFIELD_BIT(MESA_VK_DGC_SI)),
      .si_dst_dw = vk->si_layout.dst_offset_B / 4,
   };

   for (uint32_t l = 0; l < vk->n_pc_layouts; l++) {
      const struct vk_indirect_command_push_constant_layout *pc =
         &vk->pc_layouts[l];
      for (uint32_t off = 0; off < pc->size_B; off += 4) {
         if (params->pc_count == LIBKK_DGC_MAX_PC_DWORDS)
            return false;
         params->pc_dst_dw[params->pc_count] = (pc->dst_offset_B + off) / 4;
         params->pc_src_dw[params->pc_count] = (pc->src_offset_B + off) / 4;
         params->pc_count++;
      }
   }

   for (uint32_t v = 0; v < vk->n_vb_layouts && v < LIBKK_DGC_MAX_VBS; v++) {
      params->vb_binding[v] = vk->vb_layouts[v].binding;
      params->vb_src_dw[v] = vk->vb_layouts[v].src_offset_B / 4;
      params->vb_count++;
   }

   return true;
}

void
kk_dgc_process(struct kk_cmd_buffer *cmd,
               const struct libkk_dgc_params *params)
{
   struct kk_ptr gpu = kk_pool_upload(cmd, params, sizeof(*params), 8u);
   if (unlikely(!gpu.gpu))
      return;

   libkk_dgc_process(cmd, kk_grid_1d(params->max_seq), true, gpu.gpu);
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CreateIndirectExecutionSetEXT(
   VkDevice _device, const VkIndirectExecutionSetCreateInfoEXT *pCreateInfo,
   const VkAllocationCallbacks *pAllocator,
   VkIndirectExecutionSetEXT *pIndirectExecutionSet)
{
   VK_FROM_HANDLE(kk_device, dev, _device);
   return vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
}

VKAPI_ATTR void VKAPI_CALL
kk_DestroyIndirectExecutionSetEXT(VkDevice _device,
                                  VkIndirectExecutionSetEXT indirectExecutionSet,
                                  const VkAllocationCallbacks *pAllocator)
{
}

VKAPI_ATTR void VKAPI_CALL
kk_UpdateIndirectExecutionSetPipelineEXT(
   VkDevice _device, VkIndirectExecutionSetEXT indirectExecutionSet,
   uint32_t executionSetWriteCount,
   const VkWriteIndirectExecutionSetPipelineEXT *pExecutionSetWrites)
{
}

VKAPI_ATTR void VKAPI_CALL
kk_UpdateIndirectExecutionSetShaderEXT(
   VkDevice _device, VkIndirectExecutionSetEXT indirectExecutionSet,
   uint32_t executionSetWriteCount,
   const VkWriteIndirectExecutionSetShaderEXT *pExecutionSetWrites)
{
}
