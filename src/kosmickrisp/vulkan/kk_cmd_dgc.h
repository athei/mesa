/*
 * Copyright 2026 David Andrews
 * SPDX-License-Identifier: MIT
 */
#ifndef KK_CMD_DGC_H
#define KK_CMD_DGC_H 1

#include "kk_private.h"

#include "kosmickrisp/libkk/kk_dgc.h"

#include "vulkan/runtime/vk_device_generated_commands.h"

struct kk_cmd_buffer;

struct kk_indirect_commands_layout {
   struct vk_indirect_command_layout vk;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(kk_indirect_commands_layout, vk.base,
                               VkIndirectCommandsLayoutEXT,
                               VK_OBJECT_TYPE_INDIRECT_COMMANDS_LAYOUT_EXT)

struct kk_dgc_buffers {
   uint64_t roots;
   uint32_t root_stride_B;
   uint64_t cmds;
   uint32_t cmd_stride_B;
   uint64_t ib_info;
   uint64_t unrolled;
};

bool kk_dgc_prepare(struct kk_cmd_buffer *cmd,
                    const struct kk_indirect_commands_layout *layout,
                    const VkGeneratedCommandsInfoEXT *info,
                    struct libkk_dgc_params *params,
                    struct kk_dgc_buffers *bufs);

void kk_dgc_process(struct kk_cmd_buffer *cmd,
                    const struct libkk_dgc_params *params);

void kk_cmd_draw_generated(struct kk_cmd_buffer *cmd,
                           const struct kk_indirect_commands_layout *layout,
                           const VkGeneratedCommandsInfoEXT *info);

void kk_cmd_dispatch_generated(struct kk_cmd_buffer *cmd,
                               const struct kk_indirect_commands_layout *layout,
                               const VkGeneratedCommandsInfoEXT *info);

#endif
