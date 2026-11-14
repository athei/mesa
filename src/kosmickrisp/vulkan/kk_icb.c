/*
 * Copyright 2026 LunarG, Inc.
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_icb.h"

#include "kk_cmd_buffer.h"
#include "kk_device.h"
#include "kk_physical_device.h"

#include "kosmickrisp/bridge/mtl_bridge.h"

#include "util/os_time.h"
#include "util/u_math.h"

/* The fewest commands of an indirect command buffer; one command takes about
 * 750 B */
#define KK_ICB_MIN_COMMANDS 4096u

/* How long the device keeps an indirect command buffer no command buffer
 * takes, beyond one of each size */
#define KK_ICB_IDLE_NS 1000000000ll

/* Writes the commands of one indirect-count call: one threadgroup of
 * KK_ICB_FILL_LANES threads per maxDrawCount slot, encoding the draws below
 * the count, none when conditional rendering discards the call. ICB draws
 * have no index buffer length, so a draw that reads past the bound index
 * range draws a copy of its indices from the geometry heap instead, with 0
 * past the range as the encoder's bounds check would read. */
static const char kk_icb_fill_msl[] =
   "#include <metal_stdlib>\n"
   "using namespace metal;\n"
   "\n"
   "struct args {\n"
   "   command_buffer icb;\n"
   "   ulong draws, count, cond, range, index_buffer, root, samplers;\n"
   "   ulong per_draw, per_draw_out, heap;\n"
   "   uint max_draws, stride_el, index_range_el, index_size_B, prim;\n"
   "   uint cond_inverted, per_draw_size_B;\n"
   "};\n"
   "\n"
   "struct heap {\n"
   "   ulong base;\n"
   "   atomic_uint bottom;\n"
   "   uint size;\n"
   "};\n"
   "\n"
   "kernel void\n"
   "kk_icb_fill(constant args &a [[buffer(0)]],\n"
   "            uint slot [[threadgroup_position_in_grid]],\n"
   "            uint lane [[thread_index_in_threadgroup]],\n"
   "            uint lanes [[threads_per_threadgroup]])\n"
   "{\n"
   "   threadgroup ulong copy;\n"
   "\n"
   "   uint count = min(*(device const uint *)a.count, a.max_draws);\n"
   "   if (a.cond && (*(device const uint *)a.cond == 0u) != (a.cond_inverted != 0u))\n"
   "      count = 0u;\n"
   "   if (slot == 0u && lane == 0u) {\n"
   "      device uint *range = (device uint *)a.range;\n"
   "      range[0] = 0u;\n"
   "      range[1] = count;\n"
   "   }\n"
   "   if (slot >= count)\n"
   "      return;\n"
   "\n"
   "   device const uint *draw = (device const uint *)a.draws + slot * a.stride_el;\n"
   "   uint vertices = draw[0], instances = draw[1];\n"
   "   bool empty = vertices == 0u || instances == 0u;\n"
   "\n"
   "   ulong per_draw = a.per_draw;\n"
   "   if (a.per_draw_out && !empty) {\n"
   "      per_draw = a.per_draw_out + (ulong)slot * a.per_draw_size_B;\n"
   "      device uint *out = (device uint *)per_draw;\n"
   "      constant uint *in = (constant uint *)a.per_draw;\n"
   "      /* draw_id comes first */\n"
   "      for (uint i = lane; i < a.per_draw_size_B / 4u; i += lanes)\n"
   "         out[i] = i == 0u ? slot : in[i];\n"
   "   }\n"
   "\n"
   "   ulong index_addr = 0u;\n"
   "   if (a.index_size_B && !empty) {\n"
   "      ulong first = draw[2];\n"
   "      index_addr = a.index_buffer + first * a.index_size_B;\n"
   "      if (first + vertices > a.index_range_el) {\n"
   "         device heap *h = (device heap *)a.heap;\n"
   "         if (lane == 0u) {\n"
   "            ulong size = (ulong)vertices * a.index_size_B;\n"
   "            copy = 0u;\n"
   "            if (size <= h->size) {\n"
   "               uint offs = atomic_fetch_add_explicit(&h->bottom, uint(size) + 16u,\n"
   "                                                     memory_order_relaxed);\n"
   "               offs = (offs + 15u) & ~15u;\n"
   "               if ((ulong)offs + size <= h->size)\n"
   "                  copy = h->base + offs;\n"
   "            }\n"
   "         }\n"
   "         threadgroup_barrier(mem_flags::mem_threadgroup);\n"
   "         index_addr = copy;\n"
   "         for (uint i = lane; index_addr && i < vertices; i += lanes) {\n"
   "            bool in_range = first + i < a.index_range_el;\n"
   "            if (a.index_size_B == 2u) {\n"
   "               device const ushort *src = (device const ushort *)a.index_buffer;\n"
   "               ((device ushort *)index_addr)[i] = in_range ? src[first + i] : 0u;\n"
   "            } else {\n"
   "               device const uint *src = (device const uint *)a.index_buffer;\n"
   "               ((device uint *)index_addr)[i] = in_range ? src[first + i] : 0u;\n"
   "            }\n"
   "         }\n"
   "      }\n"
   "   }\n"
   "\n"
   "   if (lane != 0u)\n"
   "      return;\n"
   "\n"
   "   render_command cmd(a.icb, slot);\n"
   "   if (empty || (a.index_size_B && !index_addr)) {\n"
   "      cmd.reset();\n"
   "      return;\n"
   "   }\n"
   "\n"
   "   cmd.set_vertex_buffer((constant uchar *)a.root, 0u);\n"
   "   cmd.set_vertex_buffer((constant uchar *)a.samplers, 1u);\n"
   "   cmd.set_vertex_buffer((constant uchar *)per_draw, 2u);\n"
   "   cmd.set_fragment_buffer((constant uchar *)a.root, 0u);\n"
   "   cmd.set_fragment_buffer((constant uchar *)a.samplers, 1u);\n"
   "   cmd.set_fragment_buffer((constant uchar *)per_draw, 2u);\n"
   "\n"
   "   primitive_type prim = primitive_type(a.prim);\n"
   "   if (a.index_size_B == 2u)\n"
   "      cmd.draw_indexed_primitives(prim, vertices, (device const ushort *)index_addr,\n"
   "                                  instances, draw[3], draw[4]);\n"
   "   else if (a.index_size_B == 4u)\n"
   "      cmd.draw_indexed_primitives(prim, vertices, (device const uint *)index_addr,\n"
   "                                  instances, draw[3], draw[4]);\n"
   "   else\n"
   "      cmd.draw_primitives(prim, draw[2], vertices, instances, draw[3]);\n"
   "}\n";

static mtl_compute_pipeline_state *
kk_icb_fill_pipeline(struct kk_device *dev)
{
   struct kk_physical_device *pdev = kk_device_physical(dev);

   mtl_library *library = mtl_new_library(
      dev->mtl_compiler_handle, kk_icb_fill_msl, pdev->info.msl_version,
      MTL_MATH_MODE_FAST, MTL_MATH_FLOATING_POINT_FUNCTIONS_FAST);
   if (library == NULL)
      return NULL;

   mtl_function_descriptor *function =
      mtl_new_library_function_descriptor(library, "kk_icb_fill");
   mtl_compute_pipeline_state *pipeline = mtl_new_compute_pipeline_state(
      dev->mtl_compiler_handle, function, KK_ICB_FILL_LANES);
   mtl_release(function);
   mtl_release(library);
   return pipeline;
}

/* Like the timestamp entries (kk_query_pool.c), an indirect command buffer
 * is either in the device's free list or in the icbs of exactly one command
 * buffer, which returns it in kk_cmd_release_resources, when it cannot be
 * pending. Without the fill kernel, draws do not use the ICB path. */
void
kk_device_init_icbs(struct kk_device *dev)
{
   simple_mtx_init(&dev->icbs.lock, mtx_plain);
   util_dynarray_init(&dev->icbs.free, NULL);
   dev->icbs.fill = kk_icb_fill_pipeline(dev);
   if (dev->icbs.fill == NULL)
      mesa_logw("Cannot compile the indirect command buffer kernel; "
                "indirect-count draws issue one Metal draw per slot");
}

static void
kk_icb_destroy(struct kk_device *dev, struct kk_icb *icb)
{
   kk_device_remove_buffer_from_residency_set(dev, icb->handle);
   mtl_release(icb->handle);
   free(icb);
}

void
kk_device_finish_icbs(struct kk_device *dev)
{
   util_dynarray_foreach(&dev->icbs.free, struct kk_icb *, icb)
      kk_icb_destroy(dev, *icb);
   util_dynarray_fini(&dev->icbs.free);
   if (dev->icbs.fill)
      mtl_release(dev->icbs.fill);
   simple_mtx_destroy(&dev->icbs.lock);
}

static struct kk_icb *
kk_icb_create(struct kk_device *dev, uint32_t capacity)
{
   struct kk_icb *icb = calloc(1, sizeof(*icb));
   if (icb == NULL)
      return NULL;

   /* Root, samplers and per-draw data */
   icb->handle = mtl_new_indirect_command_buffer(dev->mtl_handle, capacity, 3u);
   if (icb->handle == NULL) {
      free(icb);
      return NULL;
   }
   icb->resource_id = mtl_indirect_command_buffer_get_resource_id(icb->handle);
   icb->capacity = capacity;
   kk_device_add_buffer_to_residency_set(dev, icb->handle);
   return icb;
}

/* An indirect command buffer of at least count commands for one draw of
 * `cmd`. Every draw of a command buffer writes its commands from the first
 * one on, as the unroll kernels reuse the geometry heap: the kernel of a draw
 * runs after the render encoder that executed the draw before it. So a
 * command buffer takes another indirect command buffer only for a draw larger
 * than those it holds. Their sizes are powers of two, and the device may
 * hand out a larger one than the draw needs, so each is at least twice the
 * one before: a command buffer holds less than twice the commands of the
 * largest one it holds, at most 2 * KK_ICB_MAX_DRAWS - 1. The smaller ones
 * stay with it until it is reset, since its earlier render encoders execute
 * them. */
struct kk_icb *
kk_cmd_get_icb(struct kk_cmd_buffer *cmd, uint32_t count)
{
   struct kk_device *dev = kk_cmd_buffer_device(cmd);
   struct util_dynarray *held = &cmd->icbs;

   util_dynarray_foreach(held, struct kk_icb *, held_icb) {
      if ((*held_icb)->capacity >= count)
         return *held_icb;
   }

   /* The smallest free one that fits */
   struct kk_icb *icb = NULL;
   simple_mtx_lock(&dev->icbs.lock);
   util_dynarray_foreach(&dev->icbs.free, struct kk_icb *, free_icb) {
      if ((*free_icb)->capacity >= count &&
          (icb == NULL || (*free_icb)->capacity < icb->capacity))
         icb = *free_icb;
   }
   if (icb != NULL)
      util_dynarray_delete_unordered(&dev->icbs.free, struct kk_icb *, icb);
   simple_mtx_unlock(&dev->icbs.lock);

   if (icb == NULL) {
      icb = kk_icb_create(
         dev, MAX2(KK_ICB_MIN_COMMANDS, util_next_power_of_two(count)));
      if (icb == NULL)
         return NULL;
   }

   util_dynarray_append(held, icb);
   return icb;
}

/* Returns the indirect command buffers of a command buffer that is reset or
 * destroyed, and destroys the free ones no command buffer took for
 * KK_ICB_IDLE_NS, except the last one returned of each size: the device keeps
 * what its command buffers reuse and drops what a burst of large draws left.
 * Creating one takes about 0.5 ms for 4096 commands and 8 ms for 65536, in
 * the draw call, so one of each size stays for a draw that comes less often
 * than the age. */
void
kk_icbs_release(struct kk_device *dev, struct util_dynarray *icbs)
{
   uint32_t count = util_dynarray_num_elements(icbs, struct kk_icb *);

   simple_mtx_lock(&dev->icbs.lock);
   if (count == 0 && !util_dynarray_contains(&dev->icbs.free, struct kk_icb *)) {
      simple_mtx_unlock(&dev->icbs.lock);
      return;
   }

   int64_t now = os_time_get_nano();
   util_dynarray_foreach(icbs, struct kk_icb *, icb)
      (*icb)->idle_since = now;
   if (count != 0)
      util_dynarray_append_array(&dev->icbs.free, struct kk_icb *, icbs->data,
                                 count);

   struct kk_icb **free_icbs = util_dynarray_begin(&dev->icbs.free);
   uint32_t free_count =
      util_dynarray_num_elements(&dev->icbs.free, struct kk_icb *);

   /* The last one returned of each size stays whatever its age */
   struct kk_icb *newest[32] = {0};
   for (uint32_t i = 0u; i < free_count; i++) {
      struct kk_icb **slot = &newest[util_logbase2(free_icbs[i]->capacity)];
      if (*slot == NULL || free_icbs[i]->idle_since > (*slot)->idle_since)
         *slot = free_icbs[i];
   }

   uint32_t kept = 0u;
   for (uint32_t i = 0u; i < free_count; i++) {
      struct kk_icb *icb = free_icbs[i];
      if (now - icb->idle_since > KK_ICB_IDLE_NS &&
          icb != newest[util_logbase2(icb->capacity)])
         kk_icb_destroy(dev, icb);
      else
         free_icbs[kept++] = icb;
   }
   util_dynarray_resize(&dev->icbs.free, struct kk_icb *, kept);
   simple_mtx_unlock(&dev->icbs.lock);
   util_dynarray_clear(icbs);
}
