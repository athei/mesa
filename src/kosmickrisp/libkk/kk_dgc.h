/*
 * Copyright 2026 David Andrews
 * SPDX-License-Identifier: MIT
 */
#ifndef KK_DGC_H
#define KK_DGC_H

#include "compiler/libcl/libcl.h"

#define LIBKK_DGC_MAX_PC_DWORDS 64
#define LIBKK_DGC_MAX_VBS       32
#define LIBKK_DGC_MAX_ATTRIBS   32

struct libkk_dgc_attrib {
   uint32_t binding;
   uint32_t offset_B;
   uint32_t size_B;
   uint32_t valid;
};

struct libkk_dgc_params {
   DEVICE(uint32_t) stream;
   DEVICE(uint32_t) count;
   DEVICE(uint32_t) predicate;
   DEVICE(uint32_t) template_root;
   DEVICE(uint32_t) roots;
   DEVICE(uint32_t) draws;
   DEVICE(uint32_t) ib_info;

   uint32_t predicate_inverted;
   uint32_t max_seq;
   uint32_t stream_stride_dw;
   uint32_t root_size_dw;
   uint32_t root_stride_dw;

   uint32_t cmd_src_dw;
   uint32_t cmd_size_dw;
   uint32_t cmd_stride_dw;

   uint32_t has_ib;
   uint32_t ib_src_dw;
   uint32_t ib_dx;

   uint32_t push_dw;
   uint32_t buffer_strides_dw;
   uint32_t attrib_base_dw;
   uint32_t attrib_clamps_dw;

   uint32_t has_si;
   uint32_t si_dst_dw;

   uint32_t pc_count;
   uint32_t pc_dst_dw[LIBKK_DGC_MAX_PC_DWORDS];
   uint32_t pc_src_dw[LIBKK_DGC_MAX_PC_DWORDS];

   uint32_t vb_count;
   uint32_t vb_binding[LIBKK_DGC_MAX_VBS];
   uint32_t vb_src_dw[LIBKK_DGC_MAX_VBS];

   uint32_t attrib_count;
   struct libkk_dgc_attrib attribs[LIBKK_DGC_MAX_ATTRIBS];
};

#endif /* KK_DGC_H */
