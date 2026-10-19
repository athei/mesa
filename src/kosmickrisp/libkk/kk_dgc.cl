/*
 * Copyright 2026 David Andrews
 * SPDX-License-Identifier: MIT
 */
#include "compiler/libcl/libcl_vk.h"

#include "poly/cl/restart.h"

#include "kk_dgc.h"

static uint32_t
libkk_dgc_index_size_B(uint32_t format, bool dx)
{
   if (dx)
      return format == 57 ? 2 : format == 42 ? 4 : 0;

   switch (format) {
   case VK_INDEX_TYPE_UINT16:
      return 2;
   case VK_INDEX_TYPE_UINT32:
      return 4;
   case VK_INDEX_TYPE_UINT8_KHR:
      return 1;
   default:
      return 0;
   }
}

KERNEL(64)
libkk_dgc_process(constant struct libkk_dgc_params *p)
{
   uint i = cl_global_id.x;
   if (i >= p->max_seq)
      return;

   uint count = p->max_seq;
   if (p->count)
      count = min(*p->count, p->max_seq);

   if (p->predicate) {
      bool zero = *p->predicate == 0;
      if (zero != (bool)p->predicate_inverted)
         count = 0;
   }

   bool active = i < count;
   global uint32_t *seq = p->stream + (uint64_t)i * p->stream_stride_dw;
   global uint32_t *cmd = p->draws + (uint64_t)i * p->cmd_stride_dw;

   for (uint d = 0; d < p->cmd_size_dw; d++)
      cmd[d] = active ? seq[p->cmd_src_dw + d] : 0;

   if (p->has_ib) {
      global uint32_t *info = p->ib_info + (uint64_t)i * 4;
      uint32_t size_B = active ? seq[p->ib_src_dw + 2] : 0;
      uint32_t el_B = libkk_dgc_index_size_B(seq[p->ib_src_dw + 3], p->ib_dx);

      if (!active || el_B == 0) {
         el_B = 4;
         size_B = 0;
      }

      info[0] = active ? seq[p->ib_src_dw + 0] : 0;
      info[1] = active ? seq[p->ib_src_dw + 1] & 0xffffu : 0;
      info[2] = size_B / el_B;
      info[3] = el_B;
   }

   if (!active || !p->roots)
      return;

   global uint32_t *root = p->roots + (uint64_t)i * p->root_stride_dw;
   for (uint d = 0; d < p->root_size_dw; d++)
      root[d] = p->template_root[d];

   for (uint k = 0; k < p->pc_count; k++)
      root[p->push_dw + p->pc_dst_dw[k]] = seq[p->pc_src_dw[k]];

   if (p->has_si)
      root[p->push_dw + p->si_dst_dw] = i;

   for (uint v = 0; v < p->vb_count; v++) {
      uint32_t binding = p->vb_binding[v];
      global uint32_t *vb = seq + p->vb_src_dw[v];
      uint64_t va = vb[0] | ((uint64_t)(vb[1] & 0xffffu) << 32);
      uint32_t size_B = vb[2];
      uint32_t stride_B = vb[3];

      root[p->buffer_strides_dw + binding] = stride_B;

      for (uint a = 0; a < p->attrib_count; a++) {
         constant struct libkk_dgc_attrib *attr = &p->attribs[a];
         if (!attr->valid || attr->binding != binding)
            continue;

         uint64_t base = 0;
         uint32_t clamp = 0;
         uint32_t needed_B = attr->offset_B + attr->size_B;
         if (size_B >= needed_B) {
            base = va + attr->offset_B;
            clamp = stride_B ? (size_B - needed_B) / stride_B : UINT32_MAX;
         }

         root[p->attrib_base_dw + a * 2 + 0] = (uint32_t)base;
         root[p->attrib_base_dw + a * 2 + 1] = (uint32_t)(base >> 32);
         root[p->attrib_clamps_dw + a] = clamp;
      }
   }
}

KERNEL(1024)
libkk_dgc_unroll(global struct poly_heap *heap, constant uint32_t *in_draws,
                 global uint32_t *out_draws, constant uint32_t *ib_info,
                 uint32_t in_draw_stride_el, uint32_t restart,
                 uint32_t flatshade_first, uint32_t mode)
{
   uint gid = cl_group_id.x;
   constant uint32_t *info = ib_info + gid * 4;
   uint64_t index_buffer = info[0] | ((uint64_t)info[1] << 32);
   uint32_t el_B = info[3];
   uint32_t restart_index = UINT32_MAX;

   if (restart)
      restart_index = el_B == 1 ? 0xffu : el_B == 2 ? 0xffffu : UINT32_MAX;

   POLY_DECL_UNROLL_RESTART_SCRATCH(scratch, 1024);
   poly_unroll_geometry(out_draws + gid * 5, heap,
                        in_draws + gid * in_draw_stride_el, index_buffer,
                        info[2], el_B, 4u, restart_index, flatshade_first, true,
                        mode, scratch);
}
