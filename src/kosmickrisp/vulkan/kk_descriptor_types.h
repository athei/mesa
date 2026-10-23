/*
 * Copyright © 2024 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */
#ifndef KK_DESCRIPTOR_TYPES
#define KK_DESCRIPTOR_TYPES 1

#include "kk_private.h"

/* TODO_KOSMICKRISP Reduce size to 32 bytes by moving border to a heap. */
struct kk_sampled_image_descriptor {
   uint64_t image_gpu_resource_id;
   uint16_t sampler_index;

   /* Negative if there is no border color, else the clamp=0 sampler index used
    * for custom border color emulation.
    */
   int16_t clamp_0_sampler_index_or_negative;

   uint16_t pad_lod_bias;
   uint16_t sampler_lod_min_fp16;
   uint16_t sampler_lod_max_fp16;

   uint16_t image_min_lod_fp16;
   uint16_t image_min_lod_uint16;

   uint16_t sampler_minmax_flags;
   uint32_t border[4];
   uint64_t residency_gpu_resource_id;
   uint16_t image_flags;
   uint16_t pad16;
   float sampler_lod_bias;
   /* The faces of a cube view as a 2D array, see
    * kk_nir_lower_non_seamless_cube */
   uint64_t cube_array_gpu_resource_id;
};

static_assert(sizeof(struct kk_sampled_image_descriptor) == 64,
              "kk_sampled_image_descriptor has no holes");

struct kk_storage_image_descriptor {
   uint64_t image_gpu_resource_id;
   uint32_t buffer_offset_el;
   uint32_t buffer_size_el;
   uint64_t residency_gpu_resource_id;
   uint64_t buffer_rgb32_addr;
};

static_assert(sizeof(struct kk_storage_image_descriptor) == 32,
              "kk_storage_image_descriptor has no holes");

/* This has to match nir_address_format_64bit_bounded_global */
struct kk_buffer_address {
   uint64_t base_addr;
   uint32_t size;
   uint32_t residency;
};

static_assert(sizeof(struct kk_buffer_address) == 16,
              "kk_buffer_address has no holes");

#endif /* KK_DESCRIPTOR_TYPES */
