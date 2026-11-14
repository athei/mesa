/*
 * Copyright 2026 LunarG, Inc.
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_ICB_H
#define KK_ICB_H 1

#include "kk_private.h"

#include "kosmickrisp/bridge/mtl_types.h"

#include "util/u_dynarray.h"

struct kk_cmd_buffer;
struct kk_device;

/* A Metal indirect command buffer of draws, which the fill kernel writes on
 * the GPU (kk_cmd_draw.c) */
struct kk_icb {
   mtl_indirect_command_buffer *handle;
   uint64_t resource_id;
   uint32_t capacity;
   /* When its command buffer returned it (os_time_get_nano) */
   int64_t idle_since;
};

/* The arguments of the fill kernel, laid out as in kk_icb.c's MSL */
struct kk_icb_fill_args {
   uint64_t icb;
   uint64_t draws;
   uint64_t count;
   uint64_t cond;
   uint64_t range;
   uint64_t index_buffer;
   uint64_t root;
   uint64_t samplers;
   uint64_t per_draw;
   uint64_t per_draw_out;
   uint64_t heap;
   uint32_t max_draws;
   uint32_t stride_el;
   uint32_t index_range_el;
   uint32_t index_size_B;
   uint32_t prim;
   uint32_t cond_inverted;
   uint32_t per_draw_size_B;
};

/* Threads of the fill kernel per draw */
#define KK_ICB_FILL_LANES 32u

void kk_device_init_icbs(struct kk_device *dev);
void kk_device_finish_icbs(struct kk_device *dev);

struct kk_icb *kk_cmd_get_icb(struct kk_cmd_buffer *cmd, uint32_t count);
void kk_icbs_release(struct kk_device *dev, struct util_dynarray *icbs);

#endif /* KK_ICB_H */
