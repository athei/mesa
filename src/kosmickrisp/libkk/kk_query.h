/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright © 2024 Alyssa Rosenzweig
 * Copyright © 2024 Valve Corporation
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */
#ifndef KK_QUERY_H
#define KK_QUERY_H

#include "compiler/libcl/libcl.h"

struct libkk_imm_write {
   DEVICE(uint32_t) address;
   uint32_t value;
};

#define LIBKK_QUERY_UNAVAILABLE (uint64_t)((int64_t)-1)

struct libkk_stats_draw {
   DEVICE(uint32_t) draw;
   uint32_t prim;
   uint32_t patch_size;
};

#endif /* KK_QUERY_H */
