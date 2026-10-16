/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_QUERY_POOL_H
#define KK_QUERY_POOL_H 1

#include "kk_private.h"

#include "vulkan/runtime/vk_query_pool.h"

struct kk_device;
struct util_dynarray;

struct kk_oq_state {
   unsigned queries;
};

struct kk_query_pool {
   struct vk_query_pool vk;

   struct kk_bo *bo;

   uint32_t index_start;
   uint32_t query_start;
   uint32_t query_stride;

   struct kk_oq_state oq;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(kk_query_pool, vk.base, VkQueryPool,
                               VK_OBJECT_TYPE_QUERY_POOL)

uint16_t *kk_pool_index_ptr(const struct kk_query_pool *pool);

void kk_device_init_timestamps(struct kk_device *dev);
void kk_device_finish_timestamps(struct kk_device *dev);
void kk_timestamp_entries_release(struct kk_device *dev,
                                  struct util_dynarray *entries);

#endif /* KK_QUERY_POOL_H */
