/*
 * Copyright 2026 David Andrews
 * SPDX-License-Identifier: MIT
 */

#include "kk_cmd_buffer.h"
#include "kk_shader.h"

#include "nir_builder.h"

enum target_kind {
   TARGET_GLOBAL,
   TARGET_SHARED,
   TARGET_IMAGE,
};

struct target {
   enum target_kind kind;
   nir_def *addr;
   unsigned base;
   nir_intrinsic_instr *image;
   /* The base array layer of the image's view. */
   nir_def *base_layer;
   /* Whether the shader may take one lock per subgroup (locked_group). */
   bool group;
};

static nir_def *
load_arena(nir_builder *b)
{
   nir_def *root = nir_load_buffer_ptr_kk(b, 1, 64, .binding = 0);
   return nir_build_load_global_constant(
      b, 1, 64,
      nir_iadd_imm(b, root,
                   offsetof(struct kk_root_descriptor_table, atomic64)),
      .align_mul = 8, .access = ACCESS_CAN_SPECULATE);
}

static nir_def *
load_locked_flag(nir_builder *b, unsigned class)
{
   return nir_build_load_global(b, 1, 32,
                                nir_iadd_imm(b, load_arena(b), 4u * class),
                                .align_mul = 4, .access = ACCESS_ATOMIC);
}

static nir_def *
lock_key(nir_builder *b, const struct target *t)
{
   switch (t->kind) {
   case TARGET_GLOBAL:
      return nir_u2u32(b, nir_ushr_imm(b, t->addr, 3));
   case TARGET_SHARED:
      return nir_iadd_imm(b, nir_ushr_imm(b, t->addr, 3), t->base);
   case TARGET_IMAGE: {
      /* The texel's x, y and layer in the image, not in the view: a view
       * starts at its base array layer, and every view of a texel has to
       * take the same lock. A 3D view has base layer 0, and a 2D view of a
       * 3D image has its slice as base layer.
       */
      const enum glsl_sampler_dim dim = nir_intrinsic_image_dim(t->image);
      const bool layered = nir_intrinsic_image_array(t->image) ||
                           dim == GLSL_SAMPLER_DIM_CUBE ||
                           dim == GLSL_SAMPLER_DIM_3D;
      nir_def *coord = nir_u2u32(b, t->image->src[1].ssa);
      nir_def *x = nir_channel(b, coord, 0);
      nir_def *y = dim == GLSL_SAMPLER_DIM_1D || dim == GLSL_SAMPLER_DIM_BUF
                      ? nir_imm_int(b, 0)
                      : nir_channel(b, coord, 1);
      nir_def *layer = t->base_layer;
      if (layered) {
         layer = nir_iadd(b, layer,
                          nir_channel(b, coord,
                                      dim == GLSL_SAMPLER_DIM_1D ? 1 : 2));
      }
      nir_def *key = nir_imul_imm(b, x, 0x9e3779b1u);
      key = nir_iadd(b, key, nir_imul_imm(b, y, 0x85ebca77u));
      key = nir_iadd(b, key, nir_imul_imm(b, layer, 0xc2b2ae3du));
      return nir_ixor(b, key, nir_ushr_imm(b, key, 15));
   }
   }
   UNREACHABLE("bad target");
}

static nir_def *
lock_address(nir_builder *b, const struct target *t)
{
   nir_def *key = lock_key(b, t);
   nir_def *slot = nir_iand_imm(b, key, KK_ATOMIC64_LOCK_COUNT - 1u);
   nir_def *offset =
      nir_iadd_imm(b, nir_ishl_imm(b, slot, 2), KK_ATOMIC64_LOCK_OFFSET);
   return nir_iadd(b, load_arena(b), nir_u2u64(b, offset));
}

static void
fence(nir_builder *b, const struct target *t)
{
   static const nir_variable_mode modes[] = {
      [TARGET_GLOBAL] = nir_var_mem_ssbo,
      [TARGET_SHARED] = nir_var_mem_shared,
      [TARGET_IMAGE] = nir_var_image,
   };
   nir_barrier(b, .execution_scope = SCOPE_NONE, .memory_scope = SCOPE_DEVICE,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = modes[t->kind]);
}

static nir_def *
load_value(nir_builder *b, const struct target *t)
{
   if (t->kind == TARGET_GLOBAL)
      return nir_build_load_global(b, 1, 64, t->addr, .align_mul = 8);

   if (t->kind == TARGET_IMAGE) {
      const nir_intrinsic_instr *image = t->image;
      return nir_bindless_image_load(
         b, 1, 64, image->src[0].ssa, image->src[1].ssa, nir_imm_int(b, 0),
         nir_imm_int(b, 0), .image_dim = nir_intrinsic_image_dim(image),
         .image_array = nir_intrinsic_image_array(image),
         .format = nir_intrinsic_format(image),
         .access = nir_intrinsic_access(image) & ~ACCESS_ATOMIC,
         .dest_type = nir_type_uint64);
   }

   return nir_load_shared(b, 1, 64, t->addr, .base = t->base, .align_mul = 8);
}

static void
store_value(nir_builder *b, const struct target *t, nir_def *value)
{
   if (t->kind == TARGET_GLOBAL) {
      nir_build_store_global(b, value, t->addr, .align_mul = 8,
                             .write_mask = 1);
   } else if (t->kind == TARGET_IMAGE) {
      const nir_intrinsic_instr *image = t->image;
      nir_bindless_image_store(
         b, image->src[0].ssa, image->src[1].ssa, nir_imm_int(b, 0),
         nir_pad_vec4(b, value), nir_imm_int(b, 0),
         .image_dim = nir_intrinsic_image_dim(image),
         .image_array = nir_intrinsic_image_array(image),
         .format = nir_intrinsic_format(image),
         .access = nir_intrinsic_access(image) & ~ACCESS_ATOMIC,
         .src_type = nir_type_uint64);
   } else
      nir_store_shared(b, value, t->addr, .base = t->base, .align_mul = 8,
                       .write_mask = 1);
}

static nir_def *
apply(nir_builder *b, nir_atomic_op op, nir_def *old, nir_def *data,
      nir_def *swap)
{
   switch (op) {
   case nir_atomic_op_iadd:
      return nir_iadd(b, old, data);
   case nir_atomic_op_imin:
      return nir_imin(b, old, data);
   case nir_atomic_op_umin:
      return nir_umin(b, old, data);
   case nir_atomic_op_imax:
      return nir_imax(b, old, data);
   case nir_atomic_op_umax:
      return nir_umax(b, old, data);
   case nir_atomic_op_iand:
      return nir_iand(b, old, data);
   case nir_atomic_op_ior:
      return nir_ior(b, old, data);
   case nir_atomic_op_ixor:
      return nir_ixor(b, old, data);
   case nir_atomic_op_xchg:
      return data;
   case nir_atomic_op_cmpxchg:
      return nir_bcsel(b, nir_ieq(b, old, data), swap, old);
   default:
      UNREACHABLE("not an integer atomic operation");
   }
}

static nir_def *
critical_section(nir_builder *b, const struct target *t, nir_atomic_op op,
                 nir_def *data, nir_def *swap)
{
   fence(b, t);
   nir_def *old = load_value(b, t);
   fence(b, t);
   store_value(b, t, apply(b, op, old, data, swap));
   fence(b, t);
   return old;
}

static nir_def *
try_lock(nir_builder *b, nir_def *lock)
{
   nir_def *held = nir_global_atomic_swap(b, 32, lock, nir_imm_int(b, 0),
                                          nir_imm_int(b, 1),
                                          .atomic_op = nir_atomic_op_cmpxchg);
   return nir_ieq_imm(b, held, 0);
}

static nir_def *
unlock(nir_builder *b, nir_def *lock)
{
   return nir_global_atomic(b, 32, lock, nir_imm_int(b, 0),
                            .atomic_op = nir_atomic_op_xchg);
}

static void
report_timeout(nir_builder *b)
{
   nir_global_atomic(b, 32,
                     nir_iadd_imm(b, load_arena(b), 4u * KK_ATOMIC64_TIMEOUT),
                     nir_imm_int(b, 1), .atomic_op = nir_atomic_op_xchg);
}

static nir_def *
locked_lanes(nir_builder *b, const struct target *t, nir_atomic_op op,
             nir_def *data, nir_def *swap)
{
   nir_function_impl *impl = b->impl;
   nir_variable *result =
      nir_local_variable_create(impl, glsl_uint64_t_type(), "atomic64_old");
   nir_variable *state =
      nir_local_variable_create(impl, glsl_uint_type(), "atomic64_state");
   nir_variable *spins =
      nir_local_variable_create(impl, glsl_uint_type(), "atomic64_spins");

   nir_def *lock = lock_address(b, t);
   nir_store_var(b, state, nir_imm_int(b, 0), 1);
   nir_store_var(b, spins, nir_imm_int(b, 0), 1);
   nir_store_var(b, result, nir_imm_int64(b, 0), 1);

   nir_push_loop(b);
   {
      nir_def *s = nir_load_var(b, state);
      nir_break_if(b, nir_ieq_imm(b, s, 2));
      nir_push_if(b, nir_ieq_imm(b, s, 1));
      {
         nir_store_var(b, result, critical_section(b, t, op, data, swap), 1);
         nir_store_var(b, state,
                       nir_iadd_imm(b, nir_ior_imm(b, unlock(b, lock), 1), 1),
                       1);
      }
      nir_push_else(b, NULL);
      {
         nir_def *spun = nir_load_var(b, spins);
         nir_def *enter = try_lock(b, lock);
         nir_def *timeout = nir_iand(b, nir_inot(b, enter),
                                     nir_uge_imm(b, spun, KK_ATOMIC64_SPIN_LIMIT));
         nir_store_var(b, spins, nir_iadd_imm(b, spun, 1), 1);
         nir_store_var(b, state,
                       nir_bcsel(b, enter, nir_imm_int(b, 1),
                                 nir_bcsel(b, timeout, nir_imm_int(b, 2),
                                           nir_imm_int(b, 0))),
                       1);
         nir_push_if(b, timeout);
         report_timeout(b);
         nir_pop_if(b, NULL);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_loop(b, NULL);

   return nir_load_var(b, result);
}

static nir_def *
read_first64(nir_builder *b, nir_def *v)
{
   return nir_pack_64_2x32_split(
      b, nir_read_first_invocation(b, nir_unpack_64_2x32_split_x(b, v)),
      nir_read_first_invocation(b, nir_unpack_64_2x32_split_y(b, v)));
}

static nir_def *
read_lane64(nir_builder *b, nir_def *v, nir_def *lane)
{
   return nir_pack_64_2x32_split(
      b, nir_read_invocation(b, nir_unpack_64_2x32_split_x(b, v), lane),
      nir_read_invocation(b, nir_unpack_64_2x32_split_y(b, v), lane));
}

/* 1 when every active lane has the first active lane's <v>, else 0. Built
 * from a reduction, which KK_WORKAROUND_3 also lowers votes to.
 */
static nir_def *
same32(nir_builder *b, nir_def *v)
{
   nir_def *eq = nir_b2i32(b, nir_ieq(b, v, nir_read_first_invocation(b, v)));
   return nir_reduce(b, eq, .reduction_op = nir_op_iand);
}

static nir_def *
same64(nir_builder *b, nir_def *v)
{
   return nir_iand(b, same32(b, nir_unpack_64_2x32_split_x(b, v)),
                   same32(b, nir_unpack_64_2x32_split_y(b, v)));
}

/* The address of the descriptor an image atomic's handle was loaded from, or
 * NULL. MSL holds the handle as a texture, which cannot be compared between
 * lanes; the descriptor's address names the view as well.
 */
static nir_def *
image_descriptor(const struct target *t)
{
   nir_intrinsic_instr *handle =
      nir_def_as_intrinsic_or_null(t->image->src[0].ssa);
   if (!handle || handle->intrinsic != nir_intrinsic_load_texture_handle_kk)
      return NULL;
   return handle->src[0].ssa;
}

/* Whether locked_group can run here: in an application's compute shader only
 * (the group argument of kk_nir_lower_atomic64), and for an image only when
 * its lanes can be compared through its descriptor.
 */
static bool
can_lock_group(const struct target *t)
{
   if (!t->group)
      return false;
   return t->kind != TARGET_IMAGE || image_descriptor(t);
}

/* Whether every active lane targets the first active lane's address: the
 * same buffer address, shared offset, or view and texel.
 */
static nir_def *
same_target(nir_builder *b, const struct target *t)
{
   nir_def *all;

   if (t->kind == TARGET_IMAGE) {
      all = same64(b, image_descriptor(t));
      nir_def *coord = nir_u2u32(b, t->image->src[1].ssa);
      for (unsigned i = 0; i < nir_image_intrinsic_coord_components(t->image);
           i++)
         all = nir_iand(b, all, same32(b, nir_channel(b, coord, i)));
   } else if (t->addr->bit_size == 64) {
      all = same64(b, t->addr);
   } else {
      all = same32(b, t->addr);
   }
   return nir_ine_imm(b, all, 0);
}

/* Every active lane on one address. The first lane takes the lock alone, so
 * the subgroup waits for the word once instead of once per lane: with a lock
 * per lane, the time of many invocations on one word grows with the square
 * of their number. Under the lock the lanes' operations are applied to the
 * loaded value in lane order, each lane getting the value the lanes before
 * it left, and the result is stored once: the same as the lanes taking the
 * lock one after another, for every operation.
 */
static nir_def *
locked_group(nir_builder *b, const struct target *t, nir_atomic_op op,
             nir_def *data, nir_def *swap)
{
   nir_function_impl *impl = b->impl;
   nir_variable *old = nir_local_variable_create(impl, glsl_uint64_t_type(),
                                                 "atomic64_group_old");
   nir_variable *state = nir_local_variable_create(impl, glsl_uint_type(),
                                                   "atomic64_group_state");
   nir_variable *spins = nir_local_variable_create(impl, glsl_uint_type(),
                                                   "atomic64_group_spins");
   nir_variable *value =
      nir_local_variable_create(impl, glsl_uint64_t_type(), "atomic64_value");
   nir_variable *result =
      nir_local_variable_create(impl, glsl_uint64_t_type(), "atomic64_result");
   nir_variable *lanes =
      nir_local_variable_create(impl, glsl_uint_type(), "atomic64_lanes");

   nir_def *lane = nir_load_subgroup_invocation(b);
   nir_def *first = nir_ieq(b, lane, nir_read_first_invocation(b, lane));
   nir_def *lock = lock_address(b, t);

   /* state: 0 waiting, 1 holding the lock, 2 timed out. Every lane runs the
    * loop and leaves it with the first lane's result, so the subgroup
    * operations after it do not depend on how Metal brings lanes together
    * after a loop that only some of them ran (KK_WORKAROUND_9).
    */
   nir_store_var(b, state, nir_imm_int(b, 0), 1);
   nir_store_var(b, spins, nir_imm_int(b, 0), 1);
   nir_push_loop(b);
   {
      nir_def *spun = nir_load_var(b, spins);
      nir_push_if(b, first);
      nir_def *taken = nir_b2i32(b, try_lock(b, lock));
      nir_pop_if(b, NULL);
      nir_def *got = nir_if_phi(b, taken, nir_imm_int(b, 0));
      nir_def *enter = nir_ine_imm(b, nir_read_first_invocation(b, got), 0);
      nir_def *timeout = nir_iand(b, nir_inot(b, enter),
                                  nir_uge_imm(b, spun, KK_ATOMIC64_SPIN_LIMIT));
      nir_store_var(b, spins, nir_iadd_imm(b, spun, 1), 1);
      nir_push_if(b, nir_iand(b, timeout, first));
      report_timeout(b);
      nir_pop_if(b, NULL);
      nir_store_var(b, state,
                    nir_bcsel(b, enter, nir_imm_int(b, 1),
                              nir_bcsel(b, timeout, nir_imm_int(b, 2),
                                        nir_imm_int(b, 0))),
                    1);
      nir_break_if(b, nir_ior(b, enter, timeout));
   }
   nir_pop_loop(b, NULL);

   nir_def *held = nir_ieq_imm(b, nir_load_var(b, state), 1);
   nir_store_var(b, old, nir_imm_int64(b, 0), 1);
   nir_push_if(b, nir_iand(b, first, held));
   {
      fence(b, t);
      nir_store_var(b, old, load_value(b, t), 1);
      fence(b, t);
   }
   nir_pop_if(b, NULL);

   nir_store_var(b, value, read_first64(b, nir_load_var(b, old)), 1);
   nir_store_var(b, result, nir_imm_int64(b, 0), 1);
   nir_store_var(b, lanes,
                 nir_reduce(b, nir_ishl(b, nir_imm_int(b, 1), lane),
                            .reduction_op = nir_op_ior),
                 1);
   nir_push_loop(b);
   {
      nir_def *mask = nir_load_var(b, lanes);
      nir_break_if(b, nir_ieq_imm(b, mask, 0));
      nir_def *k = nir_find_lsb(b, mask);
      nir_store_var(b, lanes, nir_iand(b, mask, nir_iadd_imm(b, mask, -1)), 1);
      nir_def *v = nir_load_var(b, value);
      nir_def *mine = nir_ieq(b, lane, k);
      nir_store_var(b, result, nir_bcsel(b, mine, v, nir_load_var(b, result)),
                    1);
      nir_store_var(b, value,
                    apply(b, op, v, read_lane64(b, data, k),
                          swap ? read_lane64(b, swap, k) : NULL),
                    1);
   }
   nir_pop_loop(b, NULL);

   nir_push_if(b, nir_iand(b, first, held));
   {
      store_value(b, t, nir_load_var(b, value));
      fence(b, t);
      unlock(b, lock);
   }
   nir_pop_if(b, NULL);

   return nir_load_var(b, result);
}

static nir_def *
locked(nir_builder *b, const struct target *t, nir_atomic_op op, nir_def *data,
       nir_def *swap)
{
   if (!can_lock_group(t))
      return locked_lanes(b, t, op, data, swap);

   nir_push_if(b, same_target(b, t));
   nir_def *group = locked_group(b, t, op, data, swap);
   nir_push_else(b, NULL);
   nir_def *lanes = locked_lanes(b, t, op, data, swap);
   nir_pop_if(b, NULL);
   return nir_if_phi(b, group, lanes);
}

static bool
is_atomic64(const nir_intrinsic_instr *intrin)
{
   switch (intrin->intrinsic) {
   case nir_intrinsic_global_atomic:
   case nir_intrinsic_global_atomic_swap:
   case nir_intrinsic_shared_atomic:
   case nir_intrinsic_shared_atomic_swap:
   case nir_intrinsic_bindless_image_atomic:
   case nir_intrinsic_bindless_image_atomic_swap:
      return intrin->def.bit_size == 64;
   case nir_intrinsic_load_global:
   case nir_intrinsic_load_global_bounded:
   case nir_intrinsic_load_global_constant_offset:
      return intrin->def.bit_size == 64 &&
             (nir_intrinsic_access(intrin) & ACCESS_ATOMIC);
   case nir_intrinsic_load_shared:
   case nir_intrinsic_bindless_image_load:
      return intrin->def.bit_size == 64 &&
             (nir_intrinsic_access(intrin) & ACCESS_ATOMIC);
   case nir_intrinsic_store_global:
   case nir_intrinsic_store_shared:
      return intrin->src[0].ssa->bit_size == 64 &&
             (nir_intrinsic_access(intrin) & ACCESS_ATOMIC);
   case nir_intrinsic_bindless_image_store:
      return intrin->src[3].ssa->bit_size == 64 &&
             (nir_intrinsic_access(intrin) & ACCESS_ATOMIC);
   default:
      return false;
   }
}

/* kk_nir_lower_descriptors puts the view's base array layer into the bits
 * above bit 0 of a 64-bit image atomic's sample source.
 */
static nir_def *
view_base_layer(nir_builder *b, const nir_intrinsic_instr *intrin)
{
   return nir_ushr_imm(b, intrin->src[2].ssa, 1);
}

static void
lower_atomic(nir_builder *b, nir_intrinsic_instr *intrin, uint8_t *locked_out,
             bool group)
{
   b->cursor = nir_before_instr(&intrin->instr);

   if (intrin->intrinsic == nir_intrinsic_load_global ||
       intrin->intrinsic == nir_intrinsic_load_global_bounded ||
       intrin->intrinsic == nir_intrinsic_load_global_constant_offset ||
       intrin->intrinsic == nir_intrinsic_load_shared ||
       intrin->intrinsic == nir_intrinsic_bindless_image_load) {
      enum target_kind kind =
         intrin->intrinsic == nir_intrinsic_load_shared ? TARGET_SHARED
         : intrin->intrinsic == nir_intrinsic_bindless_image_load
            ? TARGET_IMAGE
            : TARGET_GLOBAL;
      fence(b, &(struct target){.kind = kind});
      nir_intrinsic_set_access(intrin,
                               nir_intrinsic_access(intrin) & ~ACCESS_ATOMIC);
      return;
   }

   if (intrin->intrinsic == nir_intrinsic_store_global) {
      struct target t = {
         .kind = TARGET_GLOBAL, .addr = intrin->src[1].ssa, .group = group};
      locked(b, &t, nir_atomic_op_xchg, intrin->src[0].ssa, NULL);
      *locked_out |= BITFIELD_BIT(KK_ATOMIC64_BUFFER);
      nir_instr_remove(&intrin->instr);
      return;
   }

   if (intrin->intrinsic == nir_intrinsic_store_shared) {
      struct target t = {.kind = TARGET_SHARED,
                         .addr = intrin->src[1].ssa,
                         .base = nir_intrinsic_base(intrin),
                         .group = group};
      locked(b, &t, nir_atomic_op_xchg, intrin->src[0].ssa, NULL);
      nir_instr_remove(&intrin->instr);
      return;
   }

   if (intrin->intrinsic == nir_intrinsic_bindless_image_store) {
      struct target t = {.kind = TARGET_IMAGE,
                         .image = intrin,
                         .base_layer = view_base_layer(b, intrin),
                         .group = group};
      locked(b, &t, nir_atomic_op_xchg, nir_channel(b, intrin->src[3].ssa, 0),
             NULL);
      *locked_out |= BITFIELD_BIT(KK_ATOMIC64_IMAGE);
      nir_instr_remove(&intrin->instr);
      return;
   }

   const nir_atomic_op op = nir_intrinsic_atomic_op(intrin);

   if (intrin->intrinsic == nir_intrinsic_bindless_image_atomic ||
       intrin->intrinsic == nir_intrinsic_bindless_image_atomic_swap) {
      const bool is_swap =
         intrin->intrinsic == nir_intrinsic_bindless_image_atomic_swap;
      struct target t = {.kind = TARGET_IMAGE,
                         .image = intrin,
                         .base_layer = view_base_layer(b, intrin),
                         .group = group};
      nir_def *data = intrin->src[3].ssa;
      nir_def *swap = is_swap ? intrin->src[4].ssa : NULL;
      nir_def *atomic_texture =
         nir_i2b(b, nir_iand_imm(b, intrin->src[2].ssa, 1));
      nir_src_rewrite(&intrin->src[2], nir_imm_int(b, 0));

      nir_instr *next = nir_instr_next(&intrin->instr);
      if (next && next->type == nir_instr_type_intrinsic &&
          nir_instr_as_intrinsic(next)->intrinsic ==
             nir_intrinsic_bindless_image_fence_kk)
         nir_instr_remove(next);

      const bool native = !is_swap && nir_def_is_unused(&intrin->def) &&
                          (op == nir_atomic_op_umax || op == nir_atomic_op_umin);
      if (native) {
         nir_def *unlocked =
            nir_ieq_imm(b, load_locked_flag(b, KK_ATOMIC64_IMAGE), 0);
         nir_push_if(b, nir_iand(b, unlocked, atomic_texture));
         nir_instr_remove(&intrin->instr);
         nir_builder_instr_insert(b, &intrin->instr);
         nir_push_else(b, NULL);
         locked(b, &t, op, data, swap);
         nir_pop_if(b, NULL);
         return;
      }

      *locked_out |= BITFIELD_BIT(KK_ATOMIC64_IMAGE);
      nir_def_replace(&intrin->def, locked(b, &t, op, data, swap));
      return;
   }

   const bool is_swap = intrin->intrinsic == nir_intrinsic_global_atomic_swap ||
                        intrin->intrinsic == nir_intrinsic_shared_atomic_swap;
   const bool is_global = intrin->intrinsic == nir_intrinsic_global_atomic ||
                          intrin->intrinsic == nir_intrinsic_global_atomic_swap;

   struct target t = {
      .kind = is_global ? TARGET_GLOBAL : TARGET_SHARED,
      .addr = intrin->src[0].ssa,
      .base = is_global ? 0 : nir_intrinsic_base(intrin),
      .group = group,
   };
   nir_def *data = intrin->src[1].ssa;
   nir_def *swap = is_swap ? intrin->src[2].ssa : NULL;

   const bool native = is_global && nir_def_is_unused(&intrin->def) &&
                       (op == nir_atomic_op_umax || op == nir_atomic_op_umin);
   if (native) {
      nir_push_if(b, nir_ieq_imm(b, load_locked_flag(b, KK_ATOMIC64_BUFFER), 0));
      nir_global_atomic(b, 64, t.addr, data, .atomic_op = op);
      nir_push_else(b, NULL);
      locked(b, &t, op, data, swap);
      nir_pop_if(b, NULL);
      nir_instr_remove(&intrin->instr);
      return;
   }

   if (is_global)
      *locked_out |= BITFIELD_BIT(KK_ATOMIC64_BUFFER);

   nir_def_replace(&intrin->def, locked(b, &t, op, data, swap));
}

static bool
has_atomic64(nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                is_atomic64(nir_instr_as_intrinsic(instr)))
               return true;
         }
      }
   }
   return false;
}

/* <group>: the shader may take one lock per subgroup where its lanes allow,
 * which the caller sets for the application's compute shaders only. The
 * stages KosmicKrisp runs as kernels besides them (vertex and tessellation
 * shaders before a later stage, geometry shaders) keep a lock per lane, as
 * do vertex functions, which have no SIMD-group functions, and fragment
 * functions, where one lock per subgroup lost a few adds per draw when every
 * fragment added to one texel, for a reason not known.
 */
bool
kk_nir_lower_atomic64(nir_shader *nir, uint8_t *locked_out, bool group)
{
   if (!has_atomic64(nir))
      return false;

   bool cleaned;
   do {
      cleaned = false;
      NIR_PASS(cleaned, nir, nir_lower_vars_to_ssa);
      NIR_PASS(cleaned, nir, nir_opt_copy_prop);
      NIR_PASS(cleaned, nir, nir_opt_constant_folding);
      NIR_PASS(cleaned, nir, nir_opt_dead_cf);
      NIR_PASS(cleaned, nir, nir_opt_dce);
   } while (cleaned);

   bool progress = false;

   nir_foreach_function_impl(impl, nir) {
      struct util_dynarray atomics;
      util_dynarray_init(&atomics, NULL);

      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            if (is_atomic64(intrin))
               util_dynarray_append(&atomics, intrin);
         }
      }

      nir_builder b = nir_builder_create(impl);
      util_dynarray_foreach(&atomics, nir_intrinsic_instr *, intrin)
         lower_atomic(&b, *intrin, locked_out, group);

      if (util_dynarray_num_elements(&atomics, nir_intrinsic_instr *)) {
         progress = true;
         nir_progress(true, impl, nir_metadata_none);
      }
      util_dynarray_fini(&atomics);
   }

   return progress;
}
