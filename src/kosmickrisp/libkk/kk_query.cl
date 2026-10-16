/*
 * Copyright 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2024 Valve Corporation
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */
#include "compiler/libcl/libcl_vk.h"

#include "kk_query.h"
#include "poly/geometry.h"

static inline global uint64_t *
query_report(global uint64_t *results, global uint16_t *oq_index,
             uint reports_per_query, uint query)
{
   /* For occlusion queries, results[] points to the device global heap. We
    * need to remap indices according to the query pool's allocation.
    */
   uint result_index = oq_index ? oq_index[query] : query;

   return results + (result_index * reports_per_query);
}

/**
 * Goes through a series of consecutive query indices in the given pool,
 * setting all element values to 0 and emitting them as available.
 */
KERNEL(1)
libkk_reset_query(global uint32_t *availability, global uint64_t *results,
                  global uint16_t *oq_index, uint32_t first_query,
                  uint16_t reports_per_query, int set_available)
{
   uint32_t query = first_query + cl_global_id.x;

   uint64_t value = 0;
   if (availability) {
      availability[query] = set_available;
   } else {
      value = set_available ? 0 : UINT64_MAX;
   }

   global uint64_t *report =
      query_report(results, oq_index, reports_per_query, query);

   /* XXX: is this supposed to happen on the begin? */
   for (unsigned j = 0; j < reports_per_query; ++j) {
      report[j] = value;
   }
}

KERNEL(1)
libkk_write_u32_array(global struct libkk_imm_write *write_array)
{
   uint id = cl_global_id.x;
   *(write_array[id].address) = write_array[id].value;
}

KERNEL(1)
libkk_write_u32(global uint32_t *address, uint32_t value)
{
   *address = value;
}

KERNEL(1)
libkk_copy_u32(global uint32_t *dst, global uint32_t *src)
{
   *dst = *src;
}

KERNEL(1)
libkk_copy_queries(global uint32_t *availability, global uint64_t *results,
                   global uint16_t *oq_index, uint64_t dst_addr,
                   uint64_t dst_stride, uint32_t first_query,
                   VkQueryResultFlagBits flags, uint16_t reports_per_query)
{
   uint index = cl_group_id.x;
   uint64_t dst = dst_addr + (((uint64_t)index) * dst_stride);
   uint32_t query = first_query + index;

   bool available;
   if (availability)
      available = availability[query];
   else
      available = (results[query] != LIBKK_QUERY_UNAVAILABLE);

   if (available || (flags & VK_QUERY_RESULT_PARTIAL_BIT)) {
      /* For occlusion queries, results[] points to the device global heap. We
       * need to remap indices according to the query pool's allocation.
       */
      uint result_index = oq_index ? oq_index[query] : query;
      uint idx = result_index * reports_per_query;

      for (unsigned i = 0; i < reports_per_query; ++i) {
         vk_write_query(dst, i, flags, results[idx + i]);
      }
   }

   if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) {
      vk_write_query(dst, reports_per_query, flags, available);
   }
}

static void
libkk_stats_add(global uint64_t *counter, uint64_t value)
{
   if (counter)
      *counter += value;
}

KERNEL(1)
libkk_add_u64(global uint64_t *counter, uint64_t value)
{
   *counter += value;
}

KERNEL(1024)
libkk_stats_draws(constant struct libkk_stats_draw *draws, uint32_t count,
                  global uint64_t *ia_vertices, global uint64_t *ia_primitives,
                  global uint64_t *vs_invocations,
                  global uint64_t *c_invocations,
                  global uint64_t *c_primitives, global uint64_t *tcs_patches)
{
   local uint64_t vertices[1024];
   local uint64_t primitives[1024];
   uint tid = cl_local_id.x;

   uint64_t v = 0, p = 0;
   for (uint32_t i = tid; i < count; i += 1024) {
      constant struct libkk_stats_draw *d = &draws[i];
      uint64_t instances = d->draw[1];
      v += (uint64_t)d->draw[0] * instances;
      p += (uint64_t)poly_decomposed_prims_for_vertices_with_tess(
              (enum mesa_prim)d->prim, d->draw[0], d->patch_size) *
           instances;
   }

   vertices[tid] = v;
   primitives[tid] = p;
   barrier(CLK_LOCAL_MEM_FENCE);

   for (uint s = 512; s > 0; s >>= 1) {
      if (tid < s) {
         vertices[tid] += vertices[tid + s];
         primitives[tid] += primitives[tid + s];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if (tid == 0) {
      libkk_stats_add(ia_vertices, vertices[0]);
      libkk_stats_add(ia_primitives, primitives[0]);
      libkk_stats_add(vs_invocations, vertices[0]);
      libkk_stats_add(c_invocations, primitives[0]);
      libkk_stats_add(c_primitives, primitives[0]);
      libkk_stats_add(tcs_patches, primitives[0]);
   }
}

KERNEL(1024)
libkk_stats_restart_draws(constant uint32_t *draws, uint32_t stride_el,
                          uint32_t count, uint64_t index_buffer,
                          uint32_t index_buffer_size_el, uint32_t index_size_B,
                          uint32_t restart_index, uint32_t prim,
                          global uint64_t *ia_vertices,
                          global uint64_t *ia_primitives,
                          global uint64_t *vs_invocations,
                          global uint64_t *c_invocations,
                          global uint64_t *c_primitives)
{
   local uint64_t vertices[1024];
   local uint64_t primitives[1024];
   uint tid = cl_local_id.x;

   uint64_t v = 0, p = 0;
   for (uint32_t i = tid; i < count; i += 1024) {
      constant uint32_t *d = draws + i * stride_el;
      uintptr_t in_ptr = (uintptr_t)poly_index_buffer(
         index_buffer, index_buffer_size_el, d[2], index_size_B);
      uint in_range_el =
         poly_index_buffer_range_el(index_buffer_size_el, d[2]);

      uint32_t prims = 0, verts = 0, run = 0;
      for (uint32_t k = 0; k < d[0]; k++) {
         if (poly_load_index(in_ptr, in_range_el, k, index_size_B) ==
             restart_index) {
            prims += u_decomposed_prims_for_vertices((enum mesa_prim)prim, run);
            run = 0;
         } else {
            run++;
            verts++;
         }
      }
      prims += u_decomposed_prims_for_vertices((enum mesa_prim)prim, run);

      v += (uint64_t)verts * d[1];
      p += (uint64_t)prims * d[1];
   }

   vertices[tid] = v;
   primitives[tid] = p;
   barrier(CLK_LOCAL_MEM_FENCE);

   for (uint s = 512; s > 0; s >>= 1) {
      if (tid < s) {
         vertices[tid] += vertices[tid + s];
         primitives[tid] += primitives[tid + s];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if (tid == 0) {
      libkk_stats_add(ia_vertices, vertices[0]);
      libkk_stats_add(ia_primitives, primitives[0]);
      libkk_stats_add(vs_invocations, vertices[0]);
      libkk_stats_add(c_invocations, primitives[0]);
      libkk_stats_add(c_primitives, primitives[0]);
   }
}

KERNEL(1)
libkk_tess_stats(constant uint32_t *out_draw, global uint32_t *tes_count,
                 uint32_t verts_per_prim, global uint64_t *tes_invocations,
                 global uint64_t *c_invocations, global uint64_t *c_primitives)
{
   if (tes_count) {
      libkk_stats_add(tes_invocations, *tes_count);
      *tes_count = 0;
   }

   uint64_t primitives = (uint64_t)(out_draw[0] / verts_per_prim) * out_draw[1];
   libkk_stats_add(c_invocations, primitives);
   libkk_stats_add(c_primitives, primitives);
}

static uint
libkk_gs_stream_prims(global struct poly_geometry_params *p, int static_count,
                      int count_index, uint count_words, uint prefix_sum)
{
   uint in_prims = p->input_primitives;
   if (static_count >= 0)
      return in_prims * static_count;
   if (count_index < 0 || in_prims == 0 || !p->count_buffer)
      return 0;

   uint index = count_index;
   if (prefix_sum)
      index += (in_prims - 1) * count_words;

   uint prims = p->count_buffer[index];
   if (!prefix_sum)
      p->count_buffer[index] = 0;
   return prims;
}

KERNEL(1)
libkk_gs_stats(global struct poly_geometry_params *p, uint32_t invocations,
               uint32_t count_words, uint32_t prefix_sum, uint32_t raster_stream,
               int32_t static_count0, int32_t static_count1,
               int32_t static_count2, int32_t static_count3,
               int32_t count_index0, int32_t count_index1,
               int32_t count_index2, int32_t count_index3,
               global uint64_t *gs_invocations,
               global uint64_t *gs_primitives,
               global uint64_t *c_invocations, global uint64_t *c_primitives)
{
   int static_count[4] = {static_count0, static_count1, static_count2,
                          static_count3};
   int count_index[4] = {count_index0, count_index1, count_index2,
                         count_index3};

   uint64_t emitted = 0, rasterized = 0;
   for (uint i = 0; i < 4; i++) {
      uint prims = libkk_gs_stream_prims(p, static_count[i], count_index[i],
                                         count_words, prefix_sum);
      emitted += prims;
      if (i == raster_stream)
         rasterized = prims;
   }

   libkk_stats_add(gs_invocations,
                   (uint64_t)p->input_primitives * invocations);
   libkk_stats_add(gs_primitives, emitted);
   libkk_stats_add(c_invocations, rasterized);
   libkk_stats_add(c_primitives, rasterized);
}

KERNEL(1)
libkk_dispatch_stats(constant uint32_t *groups, uint32_t stride_el,
                     uint32_t count, uint32_t local_size,
                     global uint64_t *cs_invocations)
{
   uint64_t invocations = 0;
   for (uint32_t i = 0; i < count; i++) {
      constant uint32_t *g = groups + i * stride_el;
      invocations += (uint64_t)g[0] * g[1] * g[2];
   }
   *cs_invocations += invocations * local_size;
}
